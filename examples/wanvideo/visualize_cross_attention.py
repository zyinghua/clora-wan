#!/usr/bin/env python3
"""
Visualize Wan 2.1 DiT cross-attention (video patch tokens -> text tokens) for a single denoise
step, aggregated across MEGA-BLOCKS (groups of consecutive DiT blocks).

Each Wan2.1-T2V-1.3B DiT has 30 sequential blocks. We partition them into K mega-blocks of equal
stride (default: 6 mega-blocks of stride 5 → covers blocks 0..29). For each mega-block we capture
cross-attention probabilities at every DiT block in the stride and average across:
  (a) attention heads, and
  (b) the DiT blocks inside that mega-block.

Each DiT token corresponds to one (1×2×2) latent patch. For one chosen latent frame the token
grid is h×w (latent_h/2, latent_w/2). For every text key we therefore obtain K spatial heatmaps,
one per mega-block. Heatmaps can optionally be bilinearly upsampled to the video pixel
resolution (height × width) for direct alignment with frame pixels.

Outputs (in --out_dir):
  - cross_attn_stats.npz                      : per-mega-block tensors
  - cross_attn_text_reception.png             : per-mega-block normalized text-reception bars
  - cross_attn_per_text_token_megablocks.png  : rows = text tokens, cols = K mega-blocks
  - cross_attn_top_tokens_megablocks.png      : rows = top-N tokens by reception, cols = K MBs
  - cross_attn_spatial_summary.png            : entropy & max-prob spatial maps per MB

Typical use:
  python examples/wanvideo/visualize_cross_attention.py \\
    --prompt "A red sports car on a snowy road." \\
    --out_dir ./attn_viz \\
    --num_mega_blocks 6 --stride 5
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    ModelConfig,
    model_fn_wan_video,
    WanVideoUnit_PromptEmbedder,
)


def _parse_mega_blocks(
    spec: Optional[str],
    num_mega: int,
    stride: int,
    start: int,
    n_blocks: int,
) -> List[List[int]]:
    """Build mega-block partition.

    ``spec`` (overrides everything else) accepts forms like::
        "0-4|5-9|10-14|15-19|20-24|25-29"
        "0,1,2,3,4;5,6,7,8,9;..."
    Otherwise build ``num_mega`` groups of ``stride`` consecutive DiT block ids starting at
    ``start``.
    """
    if spec:
        sep = ";" if ";" in spec else "|"
        groups: List[List[int]] = []
        for part in spec.split(sep):
            part = part.strip()
            if not part:
                continue
            if "-" in part and "," not in part:
                lo, hi = part.split("-")
                groups.append(list(range(int(lo), int(hi) + 1)))
            else:
                groups.append([int(x) for x in part.split(",") if x.strip()])
        for g in groups:
            for b in g:
                if b < 0 or b >= n_blocks:
                    raise ValueError(
                        f"--mega_blocks contains DiT block {b} outside [0, {n_blocks - 1}]"
                    )
        return groups

    groups = []
    cur = start
    for _ in range(num_mega):
        end = cur + stride
        if end > n_blocks:
            raise ValueError(
                f"Auto mega-block partition runs past block {n_blocks - 1} "
                f"(start={start}, stride={stride}, num={num_mega})."
            )
        groups.append(list(range(cur, end)))
        cur = end
    return groups


def _attention_probs_from_qk(
    q: torch.Tensor, k: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Return softmax(q kᵀ / sqrt(d)) with shape [B, H, Sq, Sk] in float32.

    This reproduces what the inner ``flash_attention`` call computes for cross-attention in
    Wan: q,k are post-projection / RMSNorm and have no RoPE applied (rope is only used in
    self-attn), so vanilla scaled dot-product softmax is faithful.
    """
    qh = rearrange(q.float(), "b sq (nh dh) -> b nh sq dh", nh=num_heads)
    kh = rearrange(k.float(), "b sk (nh dh) -> b nh sk dh", nh=num_heads)
    d = qh.shape[-1]
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * (d ** -0.5)
    return F.softmax(logits, dim=-1)


def _text_reception_chunked(
    q: torch.Tensor, k: torch.Tensor, num_heads: int, chunk_size: int = 4096
) -> torch.Tensor:
    """Mean attention mass per text key across all (head, query) pairs, chunked over Sq."""
    b, sq, _ = q.shape
    recv = torch.zeros(k.shape[1], device=q.device, dtype=torch.float32)
    for st in range(0, sq, chunk_size):
        ed = min(st + chunk_size, sq)
        probs = _attention_probs_from_qk(q[:, st:ed], k, num_heads)
        recv = recv + probs.sum(dim=(0, 1, 2)).float()
    return recv / float(b * num_heads * sq)


def _latent_shape(pipe: WanVideoPipeline, height: int, width: int, num_frames: int):
    length = (num_frames - 1) // 4 + 1
    z_dim = pipe.vae.model.z_dim
    ups = pipe.vae.upsampling_factor
    return 1, z_dim, length, height // ups, width // ups


@torch.no_grad()
def _patch_grid(dit, latents: torch.Tensor) -> Tuple[int, int, int]:
    """Return (f, h, w) token grid after patch embedding (same as model_fn)."""
    x = dit.patchify(latents)
    _, _, f, h, w = x.shape
    return int(f), int(h), int(w)


class MegaBlockAccumulator:
    """Online running mean of cross-attn probabilities for one mega-block.

    Two parallel "views" are maintained, each averaged independently over heads, DiT blocks in
    the stride, and (when --aggregation full) timesteps:

      "frame" view (the chosen --frame_index slice):
        - probs_frame  [hw, Sk]  — patch→text probs for that one latent frame
        - entropy_2d   [hw]      — entropy of those distributions
        - peaked_2d    [hw]      — max prob of those distributions

      "allframes" view (computed only when ``compute_allframes=True``):
        - probs_allframes  [hw, Sk] — per-patch probs averaged over ALL f latent frames
        - entropy_allframes [hw]    — entropy averaged over all frames
        - peaked_allframes  [hw]    — max prob averaged over all frames

    Plus the full-sequence text reception:
        - recv [Sk] — mean attention mass per text key, over all (B, head, query) pairs.
    """

    def __init__(
        self,
        mb_id: int,
        dit_block_ids: List[int],
        fhw: Tuple[int, int, int],
        frame_index: int,
        compute_allframes: bool,
    ):
        self.mb_id = mb_id
        self.dit_block_ids = list(dit_block_ids)
        self.fhw = fhw
        self.frame_index = frame_index
        self.compute_allframes = compute_allframes
        f, h, w = fhw
        if frame_index < 0 or frame_index >= f:
            raise ValueError(f"frame_index {frame_index} out of range for f={f} latent frames")
        self.hw = h * w
        # frame-view running sums
        self.probs_frame_sum: Optional[torch.Tensor] = None
        self.entropy_2d_sum: Optional[torch.Tensor] = None
        self.peaked_2d_sum: Optional[torch.Tensor] = None
        # allframes-view running sums
        self.probs_allframes_sum: Optional[torch.Tensor] = None
        self.entropy_allframes_sum: Optional[torch.Tensor] = None
        self.peaked_allframes_sum: Optional[torch.Tensor] = None
        # text reception (always)
        self.recv_sum: Optional[torch.Tensor] = None
        self.count = 0

    def _accumulate_frame_only(
        self, q: torch.Tensor, k: torch.Tensor, num_heads: int, chunk_size: int
    ):
        idx0 = self.frame_index * self.hw
        idx1 = idx0 + self.hw
        sl = _attention_probs_from_qk(q[:, idx0:idx1], k, num_heads)
        probs_frame = sl[0].mean(dim=0).float().cpu()
        peaked = sl.max(dim=-1).values.mean(dim=1)[0].float().cpu()
        ent = -(sl * sl.clamp_min(1e-9).log()).sum(dim=-1).mean(dim=1)[0].float().cpu()
        del sl
        recv_full = _text_reception_chunked(q, k, num_heads, chunk_size=chunk_size).cpu()
        return probs_frame, ent, peaked, recv_full, None, None, None

    def _accumulate_all_frames(self, q: torch.Tensor, k: torch.Tensor, num_heads: int):
        f, h, w = self.fhw
        hw = self.hw
        Sk = k.shape[1]
        device = q.device
        # Per-block-call sums (over frames). All on GPU until we move to CPU at the end.
        probs_all_acc = torch.zeros(hw, Sk, dtype=torch.float32, device=device)
        peaked_all_acc = torch.zeros(hw, dtype=torch.float32, device=device)
        ent_all_acc = torch.zeros(hw, dtype=torch.float32, device=device)
        recv_acc = torch.zeros(Sk, dtype=torch.float32, device=device)
        probs_frame = peaked_frame = ent_frame = None
        for fi in range(f):
            idx0 = fi * hw
            idx1 = idx0 + hw
            sl = _attention_probs_from_qk(q[:, idx0:idx1], k, num_heads)  # [B, H, hw, Sk]
            probs_i = sl[0].mean(dim=0).float()                             # [hw, Sk]
            peaked_i = sl.max(dim=-1).values.mean(dim=1)[0].float()         # [hw]
            ent_i = -(sl * sl.clamp_min(1e-9).log()).sum(dim=-1).mean(dim=1)[0].float()
            probs_all_acc += probs_i
            peaked_all_acc += peaked_i
            ent_all_acc += ent_i
            recv_acc += sl.sum(dim=(0, 1, 2)).float()  # [Sk]; later divided by H * f * hw
            if fi == self.frame_index:
                probs_frame = probs_i.cpu()
                peaked_frame = peaked_i.cpu()
                ent_frame = ent_i.cpu()
            del sl, probs_i, peaked_i, ent_i
        # Means over frames for the allframes view
        probs_allframes = (probs_all_acc / float(f)).cpu()
        peaked_allframes = (peaked_all_acc / float(f)).cpu()
        ent_allframes = (ent_all_acc / float(f)).cpu()
        recv = (recv_acc / float(num_heads * f * hw)).cpu()
        return (
            probs_frame, ent_frame, peaked_frame,
            recv,
            probs_allframes, ent_allframes, peaked_allframes,
        )

    def add(self, q: torch.Tensor, k: torch.Tensor, num_heads: int, chunk_size: int) -> None:
        if self.compute_allframes:
            (probs_frame, ent_frame, peaked_frame,
             recv,
             probs_allframes, ent_allframes, peaked_allframes) = self._accumulate_all_frames(
                q, k, num_heads
            )
        else:
            (probs_frame, ent_frame, peaked_frame,
             recv,
             probs_allframes, ent_allframes, peaked_allframes) = self._accumulate_frame_only(
                q, k, num_heads, chunk_size
            )

        if self.count == 0:
            self.probs_frame_sum = probs_frame
            self.entropy_2d_sum = ent_frame
            self.peaked_2d_sum = peaked_frame
            self.recv_sum = recv
            if self.compute_allframes:
                self.probs_allframes_sum = probs_allframes
                self.entropy_allframes_sum = ent_allframes
                self.peaked_allframes_sum = peaked_allframes
        else:
            self.probs_frame_sum += probs_frame
            self.entropy_2d_sum += ent_frame
            self.peaked_2d_sum += peaked_frame
            self.recv_sum += recv
            if self.compute_allframes:
                self.probs_allframes_sum += probs_allframes
                self.entropy_allframes_sum += ent_allframes
                self.peaked_allframes_sum += peaked_allframes
        self.count += 1

    def finalize(self) -> Dict[str, object]:
        if self.count == 0:
            raise RuntimeError(f"Mega-block {self.mb_id} captured no DiT blocks.")
        c = float(self.count)
        out: Dict[str, object] = {
            "probs_frame": (self.probs_frame_sum / c).numpy(),
            "recv": (self.recv_sum / c).numpy(),
            "entropy_2d": (self.entropy_2d_sum / c).numpy(),
            "peaked_2d": (self.peaked_2d_sum / c).numpy(),
            "count": self.count,
            "dit_block_ids": list(self.dit_block_ids),
        }
        if self.compute_allframes:
            out["probs_allframes"] = (self.probs_allframes_sum / c).numpy()
            out["entropy_allframes"] = (self.entropy_allframes_sum / c).numpy()
            out["peaked_allframes"] = (self.peaked_allframes_sum / c).numpy()
        return out


def _build_hooks(
    dit,
    mega_blocks: List[List[int]],
    fhw: Tuple[int, int, int],
    frame_index: int,
    compute_allframes: bool,
    chunk_size: int,
):
    block_to_mb: Dict[int, int] = {}
    accs: Dict[int, MegaBlockAccumulator] = {}
    for mb_id, dit_ids in enumerate(mega_blocks):
        accs[mb_id] = MegaBlockAccumulator(
            mb_id, dit_ids, fhw, frame_index, compute_allframes
        )
        for b in dit_ids:
            if b in block_to_mb:
                raise ValueError(
                    f"DiT block {b} appears in multiple mega-blocks (MB{block_to_mb[b]} and MB{mb_id})."
                )
            block_to_mb[b] = mb_id

    handles = []

    def make_hook(bid: int):
        mb_id = block_to_mb[bid]

        def hook(module, inputs, _output):
            q, k, _v = inputs
            accs[mb_id].add(q, k, module.num_heads, chunk_size=chunk_size)

        return hook

    for bid in block_to_mb:
        if bid >= len(dit.blocks):
            raise ValueError(f"DiT block {bid} out of range [0, {len(dit.blocks) - 1}]")
        attn_mod = dit.blocks[bid].cross_attn.attn
        handles.append(attn_mod.register_forward_hook(make_hook(bid)))

    return handles, accs


def _maybe_upsample(heat_hw: np.ndarray, height: int, width: int, enable: bool) -> np.ndarray:
    if not enable:
        return heat_hw
    t = torch.from_numpy(heat_hw)[None, None].float()
    out = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def _save_figures(
    out_dir: str,
    prompt: str,
    seq_len: int,
    token_labels: List[str],
    mb_results: Dict[int, Dict[str, object]],
    mega_blocks: List[List[int]],
    fhw: Tuple[int, int, int],
    frame_index: int,
    height: int,
    width: int,
    upsample_to_pixel: bool,
    max_tokens_grid: int,
    top_n_tokens: int,
    views: List[str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped PNG figures (npz still saved).")
        return

    f, h, w = fhw
    K = len(mega_blocks)
    extent = [0, width, height, 0] if upsample_to_pixel else [0, w, h, 0]

    def mb_label(mb_id: int) -> str:
        ids = mega_blocks[mb_id]
        return f"MB{mb_id} (DiT {ids[0]}-{ids[-1]})"

    spatial_note = (
        f"upsampled to {height}×{width} pixels"
        if upsample_to_pixel
        else f"patch grid {h}×{w}"
    )

    # --- 1) Per-mega-block normalized text reception (always, single figure) ---
    fig1, axes1 = plt.subplots(
        K, 1, figsize=(max(10, 0.35 * seq_len + 4), 2.4 * K), squeeze=False
    )
    for mb_id in range(K):
        recv = mb_results[mb_id]["recv"][:seq_len]
        norm = recv / (recv.sum() + 1e-9)
        ax = axes1[mb_id, 0]
        ax.bar(np.arange(seq_len), norm, width=1.0, color="steelblue")
        ax.set_title(
            f"{mb_label(mb_id)}: text reception "
            "(mean over all spatial queries, heads, and DiT blocks in stride)"
        )
        ax.set_xticks(np.arange(seq_len))
        ax.set_xticklabels(
            [t[:10] for t in token_labels[:seq_len]], rotation=75, fontsize=6
        )
        ax.set_ylabel("normalized mass")
    fig1.suptitle(f"Prompt: {prompt[:140]!r}", fontsize=10)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "cross_attn_text_reception.png"), dpi=150)
    plt.close(fig1)

    def _emit_view(view: str) -> None:
        """Emit per-token grid (×2 normalizations), top-N tokens, and spatial summary
        for a given view ('frame' or 'allframes'). 'frame' = single chosen latent frame;
        'allframes' = per-patch maps averaged over all f latent frames."""
        if view == "frame":
            probs_key = "probs_frame"
            entropy_key = "entropy_2d"
            peaked_key = "peaked_2d"
            suffix = ""
            view_note = f"frame {frame_index}/{f - 1} latent frames"
        elif view == "allframes":
            probs_key = "probs_allframes"
            entropy_key = "entropy_allframes"
            peaked_key = "peaked_allframes"
            suffix = "_allframes"
            view_note = f"averaged over all {f} latent frames"
        else:
            raise ValueError(f"Unknown view: {view}")

        # Skip if any MB is missing this view's data (e.g. allframes when not computed).
        if any(probs_key not in mb_results[mb_id] for mb_id in range(K)):
            return

        # --- 2) Per-text-token spatial heatmaps (rows = tokens, cols = MBs) ---
        # Saved THREE times per view, all using the same probs but different scaling/data:
        #   global  : full-Sk probs, one shared vmax across all rows × MBs
        #   rownorm : full-Sk probs, per-row vmax across MBs
        #   realnorm: real-tokens-only renormalized probs (each patch's distribution is
        #             restricted to [:seq_len] and re-divided by its sum so it sums to 1
        #             over real prompt tokens — drops the padding "attention sink" mass).
        ntok = min(max_tokens_grid, seq_len)

        def _build_token_maps(use_realnorm: bool):
            maps = []
            for ti in range(ntok):
                per_mb = []
                for mb_id in range(K):
                    P = mb_results[mb_id][probs_key]  # [hw, Sk]
                    if use_realnorm:
                        P_real = P[:, :seq_len]
                        denom = P_real.sum(axis=-1, keepdims=True)
                        P_use = P_real / np.maximum(denom, 1e-12)
                    else:
                        P_use = P
                    per_mb.append(P_use[:, ti].reshape(h, w))
                maps.append(per_mb)
            return maps

        all_token_maps_full = _build_token_maps(use_realnorm=False)
        all_token_maps_real = _build_token_maps(use_realnorm=True)
        global_vmax_full = max(
            float(m.max()) for token_maps in all_token_maps_full for m in token_maps
        ) + 1e-12
        global_vmax_real = max(
            float(m.max()) for token_maps in all_token_maps_real for m in token_maps
        ) + 1e-12

        per_token_variants = (
            (all_token_maps_full, global_vmax_full, "global",
             "global vmax (faithful magnitudes, full Sk incl. padding)",
             f"cross_attn_per_text_token_megablocks{suffix}.png"),
            (all_token_maps_full, global_vmax_full, "rownorm",
             "per-row vmax (spatial-structure view, full Sk)",
             f"cross_attn_per_text_token_megablocks{suffix}_rownorm.png"),
            (all_token_maps_real, global_vmax_real, "global",
             "real-tokens-only renormalized, global vmax (padding sink removed)",
             f"cross_attn_per_text_token_megablocks{suffix}_realnorm.png"),
        )

        for token_maps_set, gvmax, mode, scale_note, fname in per_token_variants:
            fig, axes = plt.subplots(ntok, K, figsize=(2.6 * K, 2.4 * ntok), squeeze=False)
            for ti in range(ntok):
                token_maps = token_maps_set[ti]
                vmax = gvmax if mode == "global" else (
                    max(float(m.max()) for m in token_maps) + 1e-12
                )
                last_im = None
                for mb_id in range(K):
                    heat = _maybe_upsample(token_maps[mb_id], height, width, upsample_to_pixel)
                    ax = axes[ti, mb_id]
                    last_im = ax.imshow(
                        heat, cmap="viridis", vmin=0.0, vmax=vmax,
                        extent=extent, aspect="auto",
                    )
                    if ti == 0:
                        ax.set_title(mb_label(mb_id), fontsize=9)
                    if mb_id == 0:
                        ax.set_ylabel(f"{ti}: {token_labels[ti][:14]!r}", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.colorbar(last_im, ax=axes[ti, K - 1], fraction=0.04, pad=0.02)
            fig.suptitle(
                f"Cross-attn ({view_note}, {spatial_note}): "
                f"rows = text tokens, cols = mega-blocks; mean over heads & stride DiT blocks. "
                f"[{scale_note}]",
                fontsize=10,
            )
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)

        # --- 3) Top-N tokens by reception (last MB) × mega-blocks ---
        # Emitted twice: original (full-Sk probs) and *_realnorm (real-tokens-only renormalized).
        last_recv = mb_results[K - 1]["recv"][:seq_len]
        top_idx = np.argsort(-last_recv)[: min(top_n_tokens, seq_len)]
        n = len(top_idx)

        for use_realnorm, fname in (
            (False, f"cross_attn_top_tokens_megablocks{suffix}.png"),
            (True,  f"cross_attn_top_tokens_megablocks{suffix}_realnorm.png"),
        ):
            fig3, axes3 = plt.subplots(n, K, figsize=(2.6 * K, 2.4 * n), squeeze=False)
            for r, ti in enumerate(top_idx):
                token_maps = []
                for mb_id in range(K):
                    P = mb_results[mb_id][probs_key]
                    if use_realnorm:
                        P_real = P[:, :seq_len]
                        P_use = P_real / np.maximum(P_real.sum(axis=-1, keepdims=True), 1e-12)
                    else:
                        P_use = P
                    token_maps.append(P_use[:, ti].reshape(h, w))
                vmax = max(float(m.max()) for m in token_maps) + 1e-12
                last_im = None
                for mb_id in range(K):
                    heat = _maybe_upsample(token_maps[mb_id], height, width, upsample_to_pixel)
                    ax = axes3[r, mb_id]
                    last_im = ax.imshow(
                        heat, cmap="coolwarm", vmin=0.0, vmax=vmax,
                        extent=extent, aspect="auto",
                    )
                    if r == 0:
                        ax.set_title(mb_label(mb_id), fontsize=9)
                    if mb_id == 0:
                        ax.set_ylabel(f"{ti}: {token_labels[ti][:14]!r}", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.colorbar(last_im, ax=axes3[r, K - 1], fraction=0.04, pad=0.02)
            note = " [real-tokens-only renormalized]" if use_realnorm else ""
            fig3.suptitle(
                f"Top-{n} text tokens (ranked by MB{K - 1} reception); columns = mega-blocks "
                f"[{view_note}]{note}",
                fontsize=10,
            )
            fig3.tight_layout()
            fig3.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig3)

        # --- 4) Spatial sharpness summary per mega-block (entropy & max-prob) ---
        fig4, axes4 = plt.subplots(K, 2, figsize=(8, 2.6 * K), squeeze=False)
        for mb_id in range(K):
            ent = mb_results[mb_id][entropy_key].reshape(h, w)
            pk = mb_results[mb_id][peaked_key].reshape(h, w)
            ent_u = _maybe_upsample(ent, height, width, upsample_to_pixel)
            pk_u = _maybe_upsample(pk, height, width, upsample_to_pixel)
            im0 = axes4[mb_id, 0].imshow(ent_u, cmap="magma", extent=extent, aspect="auto")
            axes4[mb_id, 0].set_title(f"{mb_label(mb_id)}: entropy H over text keys")
            plt.colorbar(im0, ax=axes4[mb_id, 0], fraction=0.046)
            im1 = axes4[mb_id, 1].imshow(pk_u, cmap="viridis", extent=extent, aspect="auto")
            axes4[mb_id, 1].set_title(f"{mb_label(mb_id)}: max prob (mean over heads)")
            plt.colorbar(im1, ax=axes4[mb_id, 1], fraction=0.046)
            for c in (0, 1):
                axes4[mb_id, c].set_xticks([])
                axes4[mb_id, c].set_yticks([])
        fig4.suptitle(
            f"Spatial sharpness ({view_note}); averaged across heads & stride DiT blocks",
            fontsize=10,
        )
        fig4.tight_layout()
        fig4.savefig(
            os.path.join(out_dir, f"cross_attn_spatial_summary{suffix}.png"), dpi=150
        )
        plt.close(fig4)

    for view in views:
        _emit_view(view)


def main():
    p = argparse.ArgumentParser(
        description="Wan DiT cross-attention visualization aggregated over mega-blocks"
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./wan_cross_attn_viz")
    # Mega-block partition
    p.add_argument(
        "--num_mega_blocks", type=int, default=6,
        help="Number of mega-blocks when --mega_blocks is not provided (default 6).",
    )
    p.add_argument(
        "--stride", type=int, default=5,
        help="Number of consecutive DiT blocks per mega-block (default 5).",
    )
    p.add_argument(
        "--start_block", type=int, default=0,
        help="First DiT block id of the first mega-block.",
    )
    p.add_argument(
        "--mega_blocks", type=str, default=None,
        help="Explicit partition, e.g. '0-4|5-9|10-14|15-19|20-24|25-29' "
             "or '0,1,2,3,4;5,6,7,8,9;...'. Overrides --num_mega_blocks/--stride/--start_block.",
    )
    # Generation shape
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument(
        "--aggregation", choices=["single", "full"], default="single",
        help="'single' = one forward pass at --timestep_index (fast snapshot). "
             "'full'   = run the full denoise trajectory and average attention across all "
             "          timesteps (recommended for research-grade measurement; ~num_inference_steps× slower).",
    )
    p.add_argument(
        "--timestep_index", type=int, default=25,
        help="Used only when --aggregation single. Index into scheduler.timesteps.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_model_path", type=str, default=None)
    # Spatial / display
    p.add_argument(
        "--frame_index", type=int, default=0,
        help="Latent-frame index used for patch queries when emitting the 'frame' view (0 = first frame).",
    )
    p.add_argument(
        "--frame_view", choices=["single", "all", "both"], default="both",
        help="'single' = only the chosen --frame_index figures (cheapest, original behavior). "
             "'all'   = only the all-frames-averaged spatial figures (per-patch attention "
             "          averaged across every latent frame; better for video-level claims). "
             "'both'  = save both (default). 'all'/'both' iterate over every frame in each "
             "          DiT-block call, so they cost ~f× more attention ops than 'single'.",
    )
    p.add_argument(
        "--upsample_to_pixel", action=argparse.BooleanOptionalAction, default=True,
        help="Bilinearly upsample patch-grid heatmaps to (height, width) for pixel-aligned display.",
    )
    p.add_argument(
        "--max_tokens_grid", type=int, default=24,
        help="How many text tokens (from start of prompt) to show in the per-token × MB grid.",
    )
    p.add_argument(
        "--top_n_tokens", type=int, default=8,
        help="How many top-receiving tokens to show in the top-N × MB figure.",
    )
    p.add_argument("--chunk_size", type=int, default=4096,
                   help="Spatial chunk size for the full-sequence text reception softmax.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model_configs = [
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
            local_model_path=args.local_model_path,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu",
            local_model_path=args.local_model_path,
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="Wan2.1_VAE.pth",
            offload_device="cpu",
            local_model_path=args.local_model_path,
        ),
    ]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
    )
    dit = pipe.dit
    dit.eval()

    mega_blocks = _parse_mega_blocks(
        args.mega_blocks,
        args.num_mega_blocks,
        args.stride,
        args.start_block,
        len(dit.blocks),
    )
    print(f"DiT has {len(dit.blocks)} blocks; partitioning into {len(mega_blocks)} mega-blocks:")
    for i, g in enumerate(mega_blocks):
        print(f"  MB{i}: DiT blocks {g[0]}..{g[-1]}  (n={len(g)})")

    embedder = WanVideoUnit_PromptEmbedder()
    pipe.load_models_to_device(("text_encoder",))
    context = embedder.encode_prompt(pipe, args.prompt)

    ids, mask = pipe.tokenizer(args.prompt, return_mask=True, add_special_tokens=True)
    seq_lens = int(mask[0].gt(0).sum().item())
    ids_list = ids[0, :seq_lens].tolist()
    tok = pipe.tokenizer.tokenizer
    token_labels = [tok.convert_ids_to_tokens(int(i)) for i in ids_list]

    shape = _latent_shape(pipe, args.height, args.width, args.num_frames)
    dev = torch.device(pipe.device)
    if dev.type in ("cuda", "mps"):
        gen = torch.Generator(device=dev).manual_seed(args.seed)
    else:
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(shape, generator=gen, dtype=pipe.torch_dtype, device=pipe.device)

    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=args.sigma_shift)
    ts = pipe.scheduler.timesteps

    fhw = _patch_grid(dit, latents)
    f, h, w = fhw
    print(
        f"Latent patch grid: f={f}, h={h}, w={w}  →  per-frame token count = h*w = {h * w}; "
        f"total video tokens = {f * h * w}"
    )

    compute_allframes = args.frame_view in ("all", "both")
    handles, accs = _build_hooks(
        dit, mega_blocks, fhw,
        frame_index=args.frame_index,
        compute_allframes=compute_allframes,
        chunk_size=args.chunk_size,
    )

    if args.aggregation == "single":
        ti = max(0, min(args.timestep_index, len(ts) - 1))
        timestep = ts[ti].to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep_value = float(timestep)
        timesteps_run = [int(ti)]
        print(f"[aggregation=single] running 1 forward pass at timestep_index={ti}, value={timestep_value:.3f}")
        try:
            with torch.no_grad():
                _ = model_fn_wan_video(
                    dit=dit,
                    latents=latents,
                    timestep=timestep.unsqueeze(0),
                    context=context,
                    clip_feature=None,
                    motion_controller=None,
                    vace=None,
                    vap=None,
                    animate_adapter=None,
                )
        finally:
            for h_ in handles:
                h_.remove()
    else:
        # 'full': run the actual denoise trajectory and let the per-MB accumulator
        # average attention across (heads × stride DiT blocks × timesteps).
        ti = -1
        timestep_value = float("nan")
        timesteps_run = list(range(len(ts)))
        print(f"[aggregation=full] running {len(ts)} forward passes (full denoise trajectory)")
        pipe.load_models_to_device(pipe.in_iteration_models)
        try:
            with torch.no_grad():
                for progress_id, t in enumerate(ts):
                    timestep = t.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                    noise_pred = model_fn_wan_video(
                        dit=dit,
                        latents=latents,
                        timestep=timestep,
                        context=context,
                        clip_feature=None,
                        motion_controller=None,
                        vace=None,
                        vap=None,
                        animate_adapter=None,
                    )
                    latents = pipe.scheduler.step(noise_pred, ts[progress_id], latents)
                    if (progress_id + 1) % max(1, len(ts) // 10) == 0 or progress_id == len(ts) - 1:
                        print(f"  step {progress_id + 1}/{len(ts)} (t={float(t):.2f}) done")
        finally:
            for h_ in handles:
                h_.remove()

    mb_results = {mb_id: acc.finalize() for mb_id, acc in accs.items()}
    for mb_id, r in mb_results.items():
        print(
            f"  MB{mb_id}: averaged over {r['count']} DiT blocks "
            f"({r['dit_block_ids']})"
        )

    views = (
        ["frame"] if args.frame_view == "single"
        else ["allframes"] if args.frame_view == "all"
        else ["frame", "allframes"]
    )

    npz_path = os.path.join(args.out_dir, "cross_attn_stats.npz")
    flat = {
        "prompt": np.array(args.prompt),
        "seq_lens": np.int32(seq_lens),
        "token_labels": np.array(token_labels[:seq_lens]),
        "f": np.int32(f),
        "h": np.int32(h),
        "w": np.int32(w),
        "frame_index": np.int32(args.frame_index),
        "frame_view": np.array(args.frame_view),
        "aggregation": np.array(args.aggregation),
        "timestep_index": np.int32(ti),
        "timestep_value": np.float32(timestep_value),
        "timesteps_run": np.array(timesteps_run, dtype=np.int32),
        "num_inference_steps": np.int32(args.num_inference_steps),
        "num_mega_blocks": np.int32(len(mega_blocks)),
        "height": np.int32(args.height),
        "width": np.int32(args.width),
    }
    for mb_id, r in mb_results.items():
        flat[f"mb{mb_id}_dit_blocks"] = np.array(r["dit_block_ids"], dtype=np.int32)
        flat[f"mb{mb_id}_probs_frame"] = r["probs_frame"].astype(np.float32)  # [hw, Sk]
        flat[f"mb{mb_id}_recv"] = r["recv"].astype(np.float32)                # [Sk]
        flat[f"mb{mb_id}_entropy2d"] = r["entropy_2d"].astype(np.float32)     # [hw]
        flat[f"mb{mb_id}_peaked2d"] = r["peaked_2d"].astype(np.float32)       # [hw]
        if "probs_allframes" in r:
            flat[f"mb{mb_id}_probs_allframes"] = r["probs_allframes"].astype(np.float32)
            flat[f"mb{mb_id}_entropy_allframes"] = r["entropy_allframes"].astype(np.float32)
            flat[f"mb{mb_id}_peaked_allframes"] = r["peaked_allframes"].astype(np.float32)
    np.savez_compressed(npz_path, **flat)
    print(f"Saved {npz_path}")

    _save_figures(
        args.out_dir,
        args.prompt,
        seq_lens,
        token_labels,
        mb_results,
        mega_blocks,
        fhw,
        args.frame_index,
        args.height,
        args.width,
        args.upsample_to_pixel,
        args.max_tokens_grid,
        args.top_n_tokens,
        views,
    )
    print(f"Figures (if matplotlib available) under {args.out_dir}")


if __name__ == "__main__":
    main()
