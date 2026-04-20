#!/usr/bin/env python3
"""
Visualize Wan 2.1 DiT cross-attention (video tokens -> text tokens) for a single denoise step.

Video tokens follow patch order ``(f, h, w)`` → sequence ``f * h * w``; indices ``[frame * h*w :
(frame+1) * h*w)`` are one temporal frame of 2D patches.

By default this emphasizes **one frame’s** queries (``--frame_index 0`` = first frame): for each
text key, you get a heatmap over that frame’s ``h × w`` grid (mean over attention heads). Chunked
**full-sequence** text reception is still computed for comparison.

Typical use:
  python examples/wanvideo/visualize_cross_attention.py \\
    --prompt "A red sports car on a snowy road." \\
    --out_dir ./attn_viz \\
    --blocks 0,5,15,29

Requires matplotlib for PNG figures; raw tensors are always saved as .npz.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

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


def _parse_blocks(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _attention_probs_from_qk(
    q: torch.Tensor, k: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Return softmax(q k^T / sqrt(d)) with shape [B, H, Sq, Sk] in float32."""
    qh = rearrange(q.float(), "b sq (nh dh) -> b nh sq dh", nh=num_heads)
    kh = rearrange(k.float(), "b sk (nh dh) -> b nh sk dh", nh=num_heads)
    d = qh.shape[-1]
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * (d ** -0.5)
    return F.softmax(logits, dim=-1)


def _text_reception_chunked(
    q: torch.Tensor, k: torch.Tensor, num_heads: int, chunk_size: int = 4096
) -> np.ndarray:
    """Mean attention mass per text position, averaged over all spatial queries (chunked over Sq)."""
    b, sq, _ = q.shape
    recv = torch.zeros(k.shape[1], device=q.device, dtype=torch.float32)
    for st in range(0, sq, chunk_size):
        ed = min(st + chunk_size, sq)
        probs = _attention_probs_from_qk(q[:, st:ed], k, num_heads)
        recv = recv + probs.sum(dim=(0, 1, 2)).float()
    recv = recv / float(b * num_heads * sq)
    return recv.cpu().numpy()


def _text_reception_one_frame(
    q: torch.Tensor, k: torch.Tensor, num_heads: int, idx0: int, idx1: int
) -> Tuple[np.ndarray, torch.Tensor]:
    """Mean mass per text key using only queries in ``[idx0, idx1)``; returns (recv_np, probs_bhsk)."""
    sl = _attention_probs_from_qk(q[:, idx0:idx1], k, num_heads)
    recv = sl.mean(dim=(0, 1, 2)).float().cpu().numpy()
    return recv, sl


def _latent_shape(pipe: WanVideoPipeline, height: int, width: int, num_frames: int) -> Tuple[int, int, int, int, int]:
    length = (num_frames - 1) // 4 + 1
    z_dim = pipe.vae.model.z_dim
    ups = pipe.vae.upsampling_factor
    lh = height // ups
    lw = width // ups
    return 1, z_dim, length, lh, lw


@torch.no_grad()
def _patch_grid(dit, latents: torch.Tensor) -> Tuple[int, int, int]:
    """Return (f, h, w) token grid after patch embedding (same as model_fn)."""
    x = dit.patchify(latents)
    _, _, f, h, w = x.shape
    return int(f), int(h), int(w)


def _build_hooks(
    dit,
    block_ids: List[int],
    fhw: Tuple[int, int, int],
    frame_index: int,
    spatial_plots: bool,
    store: Dict[str, object],
    chunk_size: int = 4096,
) -> List[torch.utils.hooks.RemovableHandle]:
    f, h, w = fhw
    hw = h * w
    if frame_index < 0 or frame_index >= f:
        raise ValueError(f"frame_index {frame_index} out of range for f={f} latent frames")
    idx0 = frame_index * hw
    idx1 = (frame_index + 1) * hw

    handles = []

    def make_hook(bid: int):
        def hook(module, inputs, _output):
            q, k, _v = inputs
            nh = module.num_heads
            recv = _text_reception_chunked(q, k, nh, chunk_size=chunk_size)
            recv_frame, sl = _text_reception_one_frame(q, k, nh, idx0, idx1)
            peaked_2d = None
            entropy_2d = None
            probs_frame = None
            if spatial_plots:
                peaked = sl.max(dim=-1).values.mean(dim=1).float().cpu().numpy()
                ent = -(sl * (sl.clamp_min(1e-9)).log()).sum(dim=-1).mean(dim=1).float().cpu().numpy()
                peaked_2d = peaked.reshape(h, w)
                entropy_2d = ent.reshape(h, w)
                probs_frame = sl[0].mean(dim=0).float().cpu().numpy()
            store["per_block"][bid] = {
                "recv": recv,
                "recv_frame": recv_frame,
                "frame_index": frame_index,
                "peaked_2d": peaked_2d,
                "entropy_2d": entropy_2d,
                "probs_frame": probs_frame,
            }

        return hook

    store["per_block"] = {}
    for bid in block_ids:
        if bid < 0 or bid >= len(dit.blocks):
            raise ValueError(f"block_id {bid} out of range [0, {len(dit.blocks) - 1}]")
        attn_mod = dit.blocks[bid].cross_attn.attn
        handles.append(attn_mod.register_forward_hook(make_hook(bid)))
    return handles


def _save_figures(
    out_dir: str,
    prompt: str,
    seq_len: int,
    token_labels: List[str],
    store: Dict,
    fhw: Tuple[int, int, int],
    block_ids: List[int],
    frame_index: int,
    per_token_block_id: int,
    per_token_grid_max: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped PNG figures (npz still saved).")
        return

    f, h, w = fhw
    n = len(block_ids)
    fig_h = max(3, n * 3.0)

    # --- Text reception: full sequence vs selected frame (first frame by default) ---
    fig1, axes1 = plt.subplots(n, 2, figsize=(16, fig_h), squeeze=False)
    # With squeeze=False, n==1 already yields shape (1, 2). Do not expand_dims(., 0)
    # or axes1[0,0] becomes a length-2 ndarray of Axes and .bar() fails.
    if axes1.ndim == 1:
        axes1 = axes1.reshape(1, -1)
    for row, bid in enumerate(block_ids):
        d = store["per_block"][bid]
        recv = d["recv"][:seq_len]
        ax0 = axes1[row, 0]
        ax0.bar(np.arange(seq_len), recv / (recv.sum() + 1e-9), width=1.0, color="steelblue")
        ax0.set_title(f"block {bid}: all frames (mean over every spatial query)")
        ax0.set_xlabel("text token index")
        ax0.set_ylabel("normalized mass")

        rf = d["recv_frame"][:seq_len]
        ax1 = axes1[row, 1]
        ax1.bar(np.arange(seq_len), rf / (rf.sum() + 1e-9), width=1.0, color="darkorange")
        ax1.set_title(f"block {bid}: frame {frame_index} only (h×w patch queries)")
        ax1.set_xlabel("text token index")
        ax1.set_ylabel("normalized mass")
    fig1.suptitle(f"Prompt (trunc): {prompt[:120]!r}", fontsize=10)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "cross_attn_text_reception.png"), dpi=150)
    plt.close(fig1)

    # --- Spatial summaries on selected frame (entropy / max-over-text) ---
    sample = store["per_block"][block_ids[0]]
    if sample.get("entropy_2d") is not None and sample.get("peaked_2d") is not None:
        fig2, axes2 = plt.subplots(n, 2, figsize=(10, fig_h), squeeze=False)
        if axes2.ndim == 1:
            axes2 = axes2.reshape(1, -1)
        for row, bid in enumerate(block_ids):
            d = store["per_block"][bid]
            ent = d["entropy_2d"]
            pk = d["peaked_2d"]
            if ent is None or pk is None:
                continue
            im0 = axes2[row, 0].imshow(ent, aspect="auto", cmap="magma")
            axes2[row, 0].set_title(f"Entropy H(block {bid})")
            plt.colorbar(im0, ax=axes2[row, 0], fraction=0.046)
            im1 = axes2[row, 1].imshow(pk, aspect="auto", cmap="viridis")
            axes2[row, 1].set_title(f"Max prob (mean heads) block {bid}")
            plt.colorbar(im1, ax=axes2[row, 1], fraction=0.046)
        fig2.suptitle(
            f"Frame {frame_index} / {f} latent frames  (patch grid h={h} w={w})",
            fontsize=10,
        )
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "cross_attn_spatial_frame.png"), dpi=150)
        plt.close(fig2)

    # --- Top text tokens by *frame* reception: heatmap over that frame's patches ---
    probs_f = store["per_block"][per_token_block_id].get("probs_frame")
    if probs_f is not None:
        topk = min(8, seq_len)
        rf = store["per_block"][per_token_block_id]["recv_frame"][:seq_len]
        top_idx = np.argsort(-rf)[:topk]
        fig3, axes3 = plt.subplots(topk, 1, figsize=(10, 2.2 * topk))
        if topk == 1:
            axes3 = [axes3]
        for ax, ti in zip(axes3, top_idx):
            heat = probs_f[:, ti].reshape(h, w)
            im = ax.imshow(heat, aspect="auto", cmap="coolwarm")
            lab = token_labels[ti] if ti < len(token_labels) else str(ti)
            ax.set_title(
                f"block {per_token_block_id} frame {frame_index} | text {ti}: {lab!r} "
                f"(mean heads: patch→token weight)"
            )
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig3.suptitle(
            "Frame queries → text keys: highest frame-local reception (see per-token grid for all tokens)",
            fontsize=10,
        )
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, "cross_attn_top_tokens_spatial.png"), dpi=150)
        plt.close(fig3)

    # --- Grid: each text token → heatmap over frame (first N prompt tokens) ---
    probs_grid = store["per_block"][per_token_block_id].get("probs_frame")
    if probs_grid is not None:
        ntok = min(per_token_grid_max, seq_len)
        ncol = min(6, ntok)
        nrow = int(np.ceil(ntok / ncol))
        fig4, axes4 = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.6 * nrow), squeeze=False)
        axes4 = np.asarray(axes4)
        if axes4.ndim == 0:
            axes4 = np.array([[axes4.item()]])
        elif axes4.ndim == 1 and nrow == 1:
            axes4 = axes4.reshape(1, -1)
        elif axes4.ndim == 1 and ncol == 1:
            axes4 = axes4.reshape(-1, 1)
        for i in range(ntok):
            r, c = i // ncol, i % ncol
            ax = axes4[r, c]
            heat = probs_grid[:, i].reshape(h, w)
            im = ax.imshow(heat, aspect="auto", cmap="viridis")
            lab = token_labels[i] if i < len(token_labels) else str(i)
            ax.set_title(f"{i}: {lab[:20]!r}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for j in range(ntok, nrow * ncol):
            r, c = j // ncol, j % ncol
            axes4[r, c].axis("off")
        fig4.suptitle(
            f"Cross-attn: frame {frame_index} patch queries → each text key (block {per_token_block_id}, "
            f"mean heads; first {ntok} tokens)",
            fontsize=11,
        )
        fig4.tight_layout()
        fig4.savefig(os.path.join(out_dir, "cross_attn_per_text_token_frame.png"), dpi=150)
        plt.close(fig4)


def main():
    p = argparse.ArgumentParser(description="Wan DiT cross-attention visualization (single step)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./wan_cross_attn_viz")
    p.add_argument("--blocks", type=str, default="0,9,19,29", help="Comma-separated DiT block indices")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--timestep_index", type=int, default=25, help="Index into scheduler.timesteps after set_timesteps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_model_path", type=str, default=None, help="If set, passed to all ModelConfig entries")
    p.add_argument(
        "--spatial_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true (default), save entropy/max maps, top-token rows, and per-token grid (uses frame slice).",
    )
    p.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Which latent-frame index along time to use for patch queries (0 = first frame).",
    )
    p.add_argument(
        "--per_token_block",
        type=int,
        default=-1,
        help="DiT block for per-token heatmap grid / top-token figure; default = last entry in --blocks.",
    )
    p.add_argument(
        "--per_token_grid_max",
        type=int,
        default=24,
        help="How many prompt tokens (from the start) to include in cross_attn_per_text_token_frame.png",
    )
    p.add_argument("--chunk_size", type=int, default=4096, help="Spatial chunk size for full-sequence text reception")
    args = p.parse_args()

    block_ids = _parse_blocks(args.blocks)
    per_token_block = args.per_token_block
    if per_token_block < 0:
        per_token_block = block_ids[-1]
    elif per_token_block not in block_ids:
        raise ValueError(
            f"--per_token_block {per_token_block} must appear in --blocks {block_ids} "
            "so hooks capture that layer."
        )
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

    embedder = WanVideoUnit_PromptEmbedder()
    pipe.load_models_to_device(("text_encoder",))
    context = embedder.encode_prompt(pipe, args.prompt)

    ids, mask = pipe.tokenizer(args.prompt, return_mask=True, add_special_tokens=True)
    seq_lens = int(mask[0].gt(0).sum().item())
    ids_list = ids[0, :seq_lens].tolist()
    tok = pipe.tokenizer.tokenizer
    token_labels = [tok.convert_ids_to_tokens(int(i)) for i in ids_list]

    shape = _latent_shape(pipe, args.height, args.width, args.num_frames)
    # Generator device must match the tensor device (CUDA/MPS randn cannot use a CPU generator).
    dev = torch.device(pipe.device)
    if dev.type in ("cuda", "mps"):
        gen = torch.Generator(device=dev).manual_seed(args.seed)
    else:
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn(shape, generator=gen, dtype=pipe.torch_dtype, device=pipe.device)

    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=args.sigma_shift)
    ts = pipe.scheduler.timesteps
    ti = max(0, min(args.timestep_index, len(ts) - 1))
    timestep = ts[ti].to(dtype=pipe.torch_dtype, device=pipe.device)

    fhw = _patch_grid(dit, latents)
    store: Dict = {}
    handles = _build_hooks(
        dit,
        block_ids,
        fhw,
        frame_index=args.frame_index,
        spatial_plots=args.spatial_plots,
        store=store,
        chunk_size=args.chunk_size,
    )

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

    # Save compressed numpy bundle
    npz_path = os.path.join(args.out_dir, "cross_attn_stats.npz")
    flat = {
        "prompt": np.array(args.prompt),
        "seq_lens": np.int32(seq_lens),
        "f": np.int32(fhw[0]),
        "h": np.int32(fhw[1]),
        "w": np.int32(fhw[2]),
        "timestep_index": np.int32(ti),
        "timestep_value": np.float32(float(timestep)),
        "frame_index": np.int32(args.frame_index),
        "spatial_plots": np.bool_(args.spatial_plots),
    }
    for bid in block_ids:
        d = store["per_block"][bid]
        flat[f"block_{bid}_recv"] = d["recv"].astype(np.float32)
        flat[f"block_{bid}_recv_frame"] = d["recv_frame"].astype(np.float32)
        if d["entropy_2d"] is not None:
            flat[f"block_{bid}_entropy2d"] = d["entropy_2d"].astype(np.float32)
            flat[f"block_{bid}_peaked2d"] = d["peaked_2d"].astype(np.float32)
        if d["probs_frame"] is not None:
            flat[f"block_{bid}_probs_frame"] = d["probs_frame"].astype(np.float32)
    np.savez_compressed(npz_path, **flat)
    print(f"Saved {npz_path}")

    _save_figures(
        args.out_dir,
        args.prompt,
        seq_lens,
        token_labels,
        store,
        fhw,
        block_ids,
        args.frame_index,
        per_token_block,
        args.per_token_grid_max,
    )
    print(f"Figures (if matplotlib available) under {args.out_dir}")


if __name__ == "__main__":
    main()
