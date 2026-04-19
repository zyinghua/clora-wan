import argparse
import os
import sys

import torch

from sdxl_injection_pipeline import SDXLPromptInjectionPipeline

# Pull the BLOCKS mapping from the repo's blora_utils if available.
BLOCKS = {
    'content': ['unet.up_blocks.0.attentions.0'],
    'style': ['unet.up_blocks.0.attentions.1'],
}


def resolve_injection_blocks(target: str):
    if target in (None, "", "none"):
        return None
    if target == "both":
        return BLOCKS["content"] + BLOCKS["style"]
    if target in BLOCKS:
        return BLOCKS[target]
    raise ValueError(f"Unknown injection target {target!r}. Choose from: content, style, both, none.")


def main():
    parser = argparse.ArgumentParser(description="SDXL inference with optional per-block prompt injection.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model id or local path for the SDXL base checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of a green bunny.",
        help="Main prompt used for the whole denoising pass.",
    )
    parser.add_argument(
        "--ablation_prompt",
        type=str,
        default="A photo of a blue bunny",
        help="Secondary prompt injected only into the cross-attention of the target block(s).",
    )
    parser.add_argument(
        "--injection_target",
        type=str,
        default="style",
        choices=["none", "content", "style", "both"],
        help="Which SDXL block(s) receive the ablation prompt (see blora_utils.BLOCKS).",
    )
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output", type=str, default="sdxl_output.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    pipe = SDXLPromptInjectionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(device)

    generator = torch.Generator(device=device)#.manual_seed(args.seed)
    generator.seed()

    injection_blocks = resolve_injection_blocks(args.injection_target)
    if injection_blocks and args.ablation_prompt is None:
        raise ValueError("--injection_target is set but --ablation_prompt was not provided.")

    images = pipe(
        prompt=args.prompt,
        ablation_prompt=args.ablation_prompt,
        injection_blocks=injection_blocks,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
    ).images

    if len(images) == 1:
        images[0].save(args.output)
        print(f"Saved image to {args.output}")
    else:
        stem, _, ext = args.output.rpartition(".")
        for i, img in enumerate(images):
            path = f"{stem}_{i}.{ext}" if stem else f"{args.output}_{i}"
            img.save(path)
            print(f"Saved image to {path}")


if __name__ == "__main__":
    main()
