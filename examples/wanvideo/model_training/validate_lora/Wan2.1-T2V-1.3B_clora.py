"""Loads one or more LoRA checkpoints into the DiT and runs T2V.

python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py \\
    --lora ./models/train/Wan2.1-T2V-1.3B_blora_g0,2_s5/epoch-4_g0.safetensors:1.0 \\
    --lora ./models/train/Wan2.1-T2V-1.3B_blora_g0,2_s5/epoch-4_g2.safetensors:1.0 \\
    --prompt "a tiger walking on grass" \\
    --output video_clora.mp4
"""

import argparse
import torch

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_lora_spec(spec: str):
    """Parse ``path[:alpha]`` into ``(path, alpha)``."""
    if ":" in spec:
        # Right-split so Windows-like ``C:\...`` doesn't confuse us.
        path, alpha = spec.rsplit(":", 1)
        try:
            return path, float(alpha)
        except ValueError:
            return spec, 1.0
    return spec, 1.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", action="append", default=[],
                   help="LoRA checkpoint to load, format ``path[:alpha]``. "
                        "Pass multiple times to compose; each is fused additively.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE)
    p.add_argument("--output", type=str, default="video_clora.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quality", type=int, default=5)
    p.add_argument("--local_model_path", type=str, default=None,
                   help="If set, load Wan base weights from this local dir "
                        "(matches the `local_model_path` in the inference script).")
    return p.parse_args()


def main():
    args = parse_args()

    model_kwargs = {}
    if args.local_model_path:
        model_kwargs["local_model_path"] = args.local_model_path

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors",
                        **model_kwargs),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                        **model_kwargs),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                        origin_file_pattern="Wan2.1_VAE.pth",
                        **model_kwargs),
        ],
    )

    if not args.lora:
        print("[CLoRA] no --lora supplied; running base model only.")
    for spec in args.lora:
        path, alpha = parse_lora_spec(spec)
        print(f"[CLoRA] fusing LoRA {path} with alpha={alpha}")
        pipe.load_lora(pipe.dit, path, alpha=alpha)

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        tiled=True,
    )
    save_video(video, args.output, fps=args.fps, quality=args.quality)
    print(f"[CLoRA] saved -> {args.output}")


if __name__ == "__main__":
    main()
