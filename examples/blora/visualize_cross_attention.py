#!/usr/bin/env python3
"""
Visualize SDXL UNet cross-attention (image patches -> text tokens) for a single denoise step.

This mirrors the Wan video cross-attention visualization but for SDXL's 2D UNet.
SDXL spatial tokens follow patch order ``(h, w)`` -> sequence ``h * w``.

BLoRA identifies two special blocks in SDXL:
  - 'content': unet.up_blocks.0.attentions.0
  - 'style':   unet.up_blocks.0.attentions.1

This script visualizes cross-attention patterns across these and other configurable blocks
to understand how text conditioning is distributed spatially.

Typical use:
  python examples/blora/visualize_cross_attention.py \\
    --prompt "A photo of a cat wearing a hat." \\
    --out_dir ./blora_attn_viz \\
    --blocks up_blocks.0.attentions.0,up_blocks.0.attentions.1,mid_block.attentions.0

Requires matplotlib for PNG figures; raw tensors are always saved as .npz.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

# Add blora-diffusers to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLORA_DIFFUSERS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "blora-diffusers")
if os.path.exists(BLORA_DIFFUSERS_PATH):
    sys.path.insert(0, os.path.join(BLORA_DIFFUSERS_PATH, "src"))

from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention


# BLoRA block definitions
BLORA_BLOCKS = {
    "content": "up_blocks.0.attentions.0",
    "style": "up_blocks.0.attentions.1",
}

# Default blocks to visualize (BLoRA blocks + mid block for comparison)
DEFAULT_BLOCKS = [
    "up_blocks.0.attentions.0",  # BLoRA content
    "up_blocks.0.attentions.1",  # BLoRA style
    "mid_block.attentions.0",    # Middle block for comparison
]


def _parse_blocks(s: str) -> List[str]:
    """Parse comma-separated block names."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            # Allow shorthand: 'content' -> 'up_blocks.0.attentions.0'
            if part in BLORA_BLOCKS:
                out.append(BLORA_BLOCKS[part])
            else:
                out.append(part)
    return out


def _get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """Get a submodule by dot-separated name."""
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _attention_probs_from_qk(
    q: torch.Tensor, k: torch.Tensor, num_heads: int, scale: float
) -> torch.Tensor:
    """Return softmax(q k^T * scale) with shape [B, H, Sq, Sk] in float32."""
    b, sq, dim = q.shape
    _, sk, _ = k.shape
    head_dim = dim // num_heads
    
    # Reshape to [B, num_heads, seq, head_dim]
    qh = q.view(b, sq, num_heads, head_dim).permute(0, 2, 1, 3).float()
    kh = k.view(b, sk, num_heads, head_dim).permute(0, 2, 1, 3).float()
    
    logits = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    return F.softmax(logits, dim=-1)


def _text_reception_full(
    q: torch.Tensor, k: torch.Tensor, num_heads: int, scale: float
) -> np.ndarray:
    """Mean attention mass per text position, averaged over all spatial queries."""
    probs = _attention_probs_from_qk(q, k, num_heads, scale)  # [B, H, Sq, Sk]
    # Average over batch, heads, and spatial queries
    recv = probs.mean(dim=(0, 1, 2)).float().cpu().numpy()
    return recv


def _compute_spatial_stats(
    probs: torch.Tensor, h: int, w: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spatial statistics from attention probabilities.
    
    Args:
        probs: [B, H, Sq, Sk] attention probabilities
        h, w: spatial dimensions (Sq = h * w)
    
    Returns:
        peaked_2d: [h, w] max attention over text tokens (mean over heads)
        entropy_2d: [h, w] entropy of attention distribution (mean over heads)
        probs_spatial: [h*w, Sk] attention probs averaged over batch and heads
    """
    # probs shape: [B, H, h*w, Sk]
    b, num_heads, sq, sk = probs.shape
    
    # Max over text keys (peaked attention)
    peaked = probs.max(dim=-1).values.mean(dim=(0, 1)).float().cpu().numpy()  # [h*w]
    peaked_2d = peaked.reshape(h, w)
    
    # Entropy of attention distribution
    ent = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1)  # [B, H, h*w]
    ent = ent.mean(dim=(0, 1)).float().cpu().numpy()  # [h*w]
    entropy_2d = ent.reshape(h, w)
    
    # Average probs over batch and heads for per-token maps
    probs_spatial = probs.mean(dim=(0, 1)).float().cpu().numpy()  # [h*w, Sk]
    
    return peaked_2d, entropy_2d, probs_spatial


class CrossAttnCaptureProcessor:
    """Attention processor that captures Q, K, V and attention probs."""
    
    def __init__(self, store: Dict, block_name: str, original_processor):
        self.store = store
        self.block_name = block_name
        self.original_processor = original_processor
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Only capture for cross-attention (encoder_hidden_states is not None)
        if encoder_hidden_states is not None:
            residual = hidden_states
            
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            else:
                batch_size = hidden_states.shape[0]
            
            # Compute Q, K, V
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # Store Q and K for later analysis
            self.store[self.block_name] = {
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "num_heads": attn.heads,
                "scale": attn.scale,
            }
        
        # Call original processor for actual computation
        return self.original_processor(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
        )


@contextmanager
def capture_cross_attention(unet, block_names: List[str], store: Dict):
    """Context manager to capture cross-attention Q, K from specified blocks."""
    original_processors = {}
    
    # Get all attention processors
    all_processors = dict(unet.attn_processors)
    
    try:
        new_processors = {}
        for name, proc in all_processors.items():
            # Check if this processor belongs to one of our target blocks
            # Processor names look like: "up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"
            matched_block = None
            for block_name in block_names:
                if block_name in name and "attn2" in name:  # attn2 is cross-attention
                    matched_block = block_name
                    break
            
            if matched_block is not None:
                original_processors[name] = proc
                new_processors[name] = CrossAttnCaptureProcessor(store, matched_block, proc)
            else:
                new_processors[name] = proc
        
        unet.set_attn_processor(new_processors)
        yield
    finally:
        # Restore original processors
        unet.set_attn_processor(all_processors)


def _infer_spatial_size(num_patches: int) -> Tuple[int, int]:
    """Infer h, w from number of spatial patches (assuming square-ish)."""
    # Try exact square first
    sqrt = int(np.sqrt(num_patches))
    if sqrt * sqrt == num_patches:
        return sqrt, sqrt
    # Otherwise find closest factors
    for h in range(sqrt, 0, -1):
        if num_patches % h == 0:
            w = num_patches // h
            return h, w
    return num_patches, 1


def _save_figures(
    out_dir: str,
    prompt: str,
    token_labels: List[str],
    store: Dict,
    block_names: List[str],
    per_token_block: str,
    per_token_grid_max: int,
    fixed_scale: bool = True,
) -> None:
    """Save visualization figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped PNG figures (npz still saved).")
        return
    
    n = len(block_names)
    if n == 0:
        print("No blocks captured, skipping figures.")
        return
    
    # Get sequence length from first captured block
    first_block = next(iter(store.values()))
    seq_len = first_block["key"].shape[1]
    
    fig_h = max(3, n * 3.0)
    
    # --- Figure 1: Text reception per block ---
    fig1, axes1 = plt.subplots(n, 1, figsize=(12, fig_h), squeeze=False)
    for row, block_name in enumerate(block_names):
        if block_name not in store:
            continue
        d = store[block_name]
        recv = _text_reception_full(d["query"], d["key"], d["num_heads"], d["scale"])
        recv = recv[:seq_len]
        
        ax = axes1[row, 0]
        ax.bar(np.arange(len(recv)), recv / (recv.sum() + 1e-9), width=1.0, color="steelblue")
        
        # Mark BLoRA designation
        blora_label = ""
        if block_name == BLORA_BLOCKS["content"]:
            blora_label = " [BLoRA: CONTENT]"
        elif block_name == BLORA_BLOCKS["style"]:
            blora_label = " [BLoRA: STYLE]"
        
        ax.set_title(f"{block_name}{blora_label}")
        ax.set_xlabel("text token index")
        ax.set_ylabel("normalized mass")
    
    fig1.suptitle(f"Cross-Attn Text Reception\nPrompt: {prompt[:80]!r}", fontsize=10)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "cross_attn_text_reception.png"), dpi=150)
    plt.close(fig1)
    
    # --- Figure 2: Spatial summaries (entropy / max-over-text) ---
    # Pre-compute stats for all blocks to determine shared scales
    spatial_data = {}
    for block_name in block_names:
        if block_name not in store:
            continue
        d = store[block_name]
        probs = _attention_probs_from_qk(d["query"], d["key"], d["num_heads"], d["scale"])
        sq = probs.shape[2]
        h, w = _infer_spatial_size(sq)
        peaked_2d, entropy_2d, _ = _compute_spatial_stats(probs, h, w)
        store[block_name]["h"] = h
        store[block_name]["w"] = w
        store[block_name]["probs"] = probs
        spatial_data[block_name] = {"entropy_2d": entropy_2d, "peaked_2d": peaked_2d}
    
    # Compute shared scales if fixed_scale
    ent_vmin, ent_vmax, pk_vmin, pk_vmax = None, None, None, None
    if fixed_scale and spatial_data:
        all_ent = [sd["entropy_2d"] for sd in spatial_data.values()]
        all_pk = [sd["peaked_2d"] for sd in spatial_data.values()]
        ent_vmin, ent_vmax = np.min(all_ent), np.max(all_ent)
        pk_vmin, pk_vmax = np.min(all_pk), np.max(all_pk)
    
    fig2, axes2 = plt.subplots(n, 2, figsize=(10, fig_h), squeeze=False)
    for row, block_name in enumerate(block_names):
        if block_name not in spatial_data:
            continue
        entropy_2d = spatial_data[block_name]["entropy_2d"]
        peaked_2d = spatial_data[block_name]["peaked_2d"]
        
        im0 = axes2[row, 0].imshow(entropy_2d, aspect="auto", cmap="magma", vmin=ent_vmin, vmax=ent_vmax)
        axes2[row, 0].set_title(f"Entropy: {block_name}")
        plt.colorbar(im0, ax=axes2[row, 0], fraction=0.046)
        
        im1 = axes2[row, 1].imshow(peaked_2d, aspect="auto", cmap="viridis", vmin=pk_vmin, vmax=pk_vmax)
        axes2[row, 1].set_title(f"Max prob: {block_name}")
        plt.colorbar(im1, ax=axes2[row, 1], fraction=0.046)
    
    fig2.suptitle(
        "Spatial Attention Statistics (entropy = spread, max = peaked)"
        + (" [shared scale]" if fixed_scale else ""),
        fontsize=10,
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "cross_attn_spatial_frame.png"), dpi=150)
    plt.close(fig2)
    
    # --- Figure 3: Top text tokens by reception (spatial heatmaps) ---
    if per_token_block in store:
        d = store[per_token_block]
        probs = d.get("probs")
        if probs is None:
            probs = _attention_probs_from_qk(d["query"], d["key"], d["num_heads"], d["scale"])
        
        h, w = d.get("h", 8), d.get("w", 8)
        probs_spatial = probs.mean(dim=(0, 1)).float().cpu().numpy()  # [h*w, seq]
        
        # Get text reception and find top tokens
        recv = _text_reception_full(d["query"], d["key"], d["num_heads"], d["scale"])
        topk = min(8, len(recv))
        top_idx = np.argsort(-recv)[:topk]
        
        # Compute shared scale across top tokens if fixed_scale
        heat_vmin, heat_vmax = None, None
        if fixed_scale:
            heats = [probs_spatial[:, ti].reshape(h, w) for ti in top_idx]
            heat_vmin, heat_vmax = np.min(heats), np.max(heats)
        
        fig3, axes3 = plt.subplots(topk, 1, figsize=(10, 2.2 * topk))
        if topk == 1:
            axes3 = [axes3]
        
        for ax, ti in zip(axes3, top_idx):
            heat = probs_spatial[:, ti].reshape(h, w)
            im = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=heat_vmin, vmax=heat_vmax)
            lab = token_labels[ti] if ti < len(token_labels) else str(ti)
            ax.set_title(f"token {ti}: {lab!r} (spatial attention)")
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        blora_label = ""
        if per_token_block == BLORA_BLOCKS["content"]:
            blora_label = " [BLoRA: CONTENT]"
        elif per_token_block == BLORA_BLOCKS["style"]:
            blora_label = " [BLoRA: STYLE]"
        
        fig3.suptitle(
            f"Top tokens by reception: {per_token_block}{blora_label}"
            + (" [shared scale]" if fixed_scale else ""),
            fontsize=10,
        )
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, "cross_attn_top_tokens_spatial.png"), dpi=150)
        plt.close(fig3)
    
    # --- Figure 4: Per-token grid ---
    if per_token_block in store:
        d = store[per_token_block]
        probs = d.get("probs")
        if probs is None:
            probs = _attention_probs_from_qk(d["query"], d["key"], d["num_heads"], d["scale"])
        
        h, w = d.get("h", 8), d.get("w", 8)
        probs_spatial = probs.mean(dim=(0, 1)).float().cpu().numpy()
        
        ntok = min(per_token_grid_max, probs_spatial.shape[1], len(token_labels))
        ncol = min(6, ntok)
        nrow = int(np.ceil(ntok / ncol))
        
        # Compute shared scale across all tokens in grid if fixed_scale
        grid_vmin, grid_vmax = None, None
        if fixed_scale:
            all_heats = [probs_spatial[:, i].reshape(h, w) for i in range(ntok)]
            grid_vmin, grid_vmax = np.min(all_heats), np.max(all_heats)
        
        fig4, axes4 = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.6 * nrow), squeeze=False)
        
        for i in range(ntok):
            r, c = i // ncol, i % ncol
            ax = axes4[r, c]
            heat = probs_spatial[:, i].reshape(h, w)
            im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=grid_vmin, vmax=grid_vmax)
            lab = token_labels[i] if i < len(token_labels) else str(i)
            ax.set_title(f"{i}: {lab[:15]!r}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        
        # Hide empty subplots
        for j in range(ntok, nrow * ncol):
            r, c = j // ncol, j % ncol
            axes4[r, c].axis("off")
        
        fig4.suptitle(
            f"Cross-attn: spatial queries → each text key ({per_token_block}, first {ntok} tokens)"
            + (" [shared scale]" if fixed_scale else ""),
            fontsize=11,
        )
        fig4.tight_layout()
        fig4.savefig(os.path.join(out_dir, "cross_attn_per_text_token_frame.png"), dpi=150)
        plt.close(fig4)


def main():
    p = argparse.ArgumentParser(description="SDXL/BLoRA cross-attention visualization")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--out_dir", type=str, default="./sdxl_cross_attn_viz")
    p.add_argument(
        "--blocks",
        type=str,
        default=",".join(DEFAULT_BLOCKS),
        help="Comma-separated UNet block names (e.g., 'up_blocks.0.attentions.0,content,style')",
    )
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument(
        "--timestep_index",
        type=int,
        default=15,
        help="Which step index to capture attention from (0 = first, -1 = last)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model ID or local path",
    )
    p.add_argument(
        "--per_token_block",
        type=str,
        default="",
        help="Block for per-token spatial heatmaps (default: last in --blocks)",
    )
    p.add_argument(
        "--per_token_grid_max",
        type=int,
        default=24,
        help="Max prompt tokens in per-token grid figure",
    )
    p.add_argument(
        "--fixed_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use shared color scale across heatmaps for direct comparison (default: True).",
    )
    args = p.parse_args()
    
    block_names = _parse_blocks(args.blocks)
    per_token_block = args.per_token_block.strip()
    if not per_token_block:
        per_token_block = block_names[-1] if block_names else BLORA_BLOCKS["style"]
    elif per_token_block in BLORA_BLOCKS:
        per_token_block = BLORA_BLOCKS[per_token_block]
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Loading SDXL pipeline from {args.model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)
    
    # Get tokenizer and encode prompt to get token labels
    tokenizer = pipe.tokenizer
    tokens = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids[0]
    attention_mask = tokens.attention_mask[0]
    seq_len = int(attention_mask.sum().item())
    token_labels = [tokenizer.decode([tid]) for tid in input_ids[:seq_len].tolist()]
    
    print(f"Prompt tokens ({seq_len}): {token_labels}")
    print(f"Capturing cross-attention from blocks: {block_names}")
    
    # Store for captured attention data
    store: Dict = {}
    captured_step = [False]
    target_step = args.timestep_index if args.timestep_index >= 0 else args.num_inference_steps + args.timestep_index
    
    # Custom callback to capture at specific step
    def capture_callback(pipe, step_index, timestep, callback_kwargs):
        if step_index == target_step and not captured_step[0]:
            captured_step[0] = True
            print(f"Capturing attention at step {step_index} (timestep={timestep:.2f})")
        return callback_kwargs
    
    # Generator
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Run inference with attention capture
    print(f"Running inference ({args.num_inference_steps} steps)...")
    with capture_cross_attention(pipe.unet, block_names, store):
        # We need to capture during a specific step, so we use a simpler approach:
        # run step by step manually, or use callback with the store already hooked
        _ = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            callback_on_step_end=capture_callback,
        )
    
    if not store:
        print("WARNING: No cross-attention data captured. Check block names.")
        print("Available attention processor names in UNet:")
        for name in pipe.unet.attn_processors.keys():
            if "attn2" in name:
                print(f"  {name}")
        return
    
    print(f"Captured {len(store)} block(s): {list(store.keys())}")
    
    # Save NPZ
    npz_path = os.path.join(args.out_dir, "cross_attn_stats.npz")
    flat = {
        "prompt": np.array(args.prompt),
        "seq_len": np.int32(seq_len),
        "timestep_index": np.int32(target_step),
        "blocks": np.array(block_names),
    }
    for block_name, d in store.items():
        safe_name = block_name.replace(".", "_")
        recv = _text_reception_full(d["query"], d["key"], d["num_heads"], d["scale"])
        flat[f"{safe_name}_recv"] = recv.astype(np.float32)
        flat[f"{safe_name}_num_heads"] = np.int32(d["num_heads"])
    
    np.savez_compressed(npz_path, **flat)
    print(f"Saved {npz_path}")
    
    # Save figures
    _save_figures(
        args.out_dir,
        args.prompt,
        token_labels,
        store,
        block_names,
        per_token_block,
        args.per_token_grid_max,
        args.fixed_scale,
    )
    print(f"Figures saved under {args.out_dir}")


if __name__ == "__main__":
    main()
