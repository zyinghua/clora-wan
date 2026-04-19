"""SDXL pipeline that injects a secondary "ablation" prompt into specific UNet blocks.

The main prompt drives the whole denoising pass. For every cross-attention layer
whose processor name matches one of the `injection_blocks` patterns, the
encoder_hidden_states are swapped with the ablation prompt's embeddings. This is
how B-LoRA's content/style blocks are probed at inference time — see
`blora_utils.BLOCKS`:

    BLOCKS = {
        'content': ['unet.up_blocks.0.attentions.0'],
        'style':   ['unet.up_blocks.0.attentions.1'],
    }
"""

from __future__ import annotations

from typing import Iterable

import torch

from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0


def _strip_unet_prefix(block: str) -> str:
    return block[len("unet."):] if block.startswith("unet.") else block


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(p in name for p in patterns)


class PromptInjectionAttnProcessor:
    """Wraps AttnProcessor2_0 and overrides cross-attn encoder_hidden_states.

    Self-attention calls pass encoder_hidden_states=None, so the override is
    skipped automatically for those.
    """

    def __init__(self, state: dict):
        self._state = state
        self._inner = AttnProcessor2_0()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        injected = self._state.get("embeds") if encoder_hidden_states is not None else None
        if injected is not None:
            encoder_hidden_states = injected
        return self._inner(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            *args,
            **kwargs,
        )


class SDXLPromptInjectionPipeline(StableDiffusionXLPipeline):
    """StableDiffusionXLPipeline with per-block prompt injection."""

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        ablation_prompt=None,
        injection_blocks=None,
        negative_prompt=None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 5.0,
        **kwargs,
    ):
        do_cfg = guidance_scale > 1.0
        device = self._execution_device

        main_pos, main_neg, main_pooled_pos, main_pooled_neg = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
        )

        original_processors = None
        state = {"embeds": None}

        if ablation_prompt is not None and injection_blocks:
            patterns = [_strip_unet_prefix(b) for b in injection_blocks]

            abl_pos, abl_neg, _, _ = self.encode_prompt(
                prompt=ablation_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_cfg,
            )
            if do_cfg:
                state["embeds"] = torch.cat([abl_neg, abl_pos], dim=0)
            else:
                state["embeds"] = abl_pos

            original_processors = dict(self.unet.attn_processors)
            new_processors = {}
            for name, proc in original_processors.items():
                if _matches_any(name, patterns) and "attn2" in name:
                    new_processors[name] = PromptInjectionAttnProcessor(state)
                else:
                    new_processors[name] = proc
            self.unet.set_attn_processor(new_processors)

        try:
            return super().__call__(
                prompt_embeds=main_pos,
                negative_prompt_embeds=main_neg,
                pooled_prompt_embeds=main_pooled_pos,
                negative_pooled_prompt_embeds=main_pooled_neg,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                **kwargs,
            )
        finally:
            if original_processors is not None:
                self.unet.set_attn_processor(original_processors)
