import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


DEFAULT_TARGETS: Tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
)


@dataclass
class BlockLoRAConfig:
    block_ids: Sequence[int]
    stride: int = 1
    rank: int = 8
    alpha: Optional[int] = None
    targets: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_TARGETS))

    def __post_init__(self):
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.alpha is None:
            self.alpha = self.rank


def resolve_dit_block_indices(
    num_layers: int, block_ids: Sequence[int], stride: int = 1
) -> List[int]:
    """Expand (block_ids, stride) into the set of concrete DiT block indices."""
    idxs = []
    for b in block_ids:
        start = b * stride
        for i in range(start, min(start + stride, num_layers)):
            if i >= 0:
                idxs.append(i)
    return sorted(set(idxs))


def build_block_lora_target_regex(
    block_indices: Sequence[int], targets: Sequence[str]
) -> str:
    """Build a PEFT-compatible regex for ``LoraConfig.target_modules``.

    PEFT uses ``re.fullmatch`` against the full dotted module name (e.g.
    ``"blocks.3.self_attn.q"`` within the DiT), so we anchor with ``.*`` to
    tolerate an outer prefix in case the DiT is nested inside a pipeline.
    """
    if not block_indices:
        raise ValueError("block_indices is empty — nothing would be adapted")
    if not targets:
        raise ValueError("targets is empty — nothing would be adapted")
    block_alt = "|".join(str(i) for i in block_indices)
    target_alt = "|".join(re.escape(t) for t in targets)
    return rf".*blocks\.({block_alt})\.({target_alt})"


def build_regex_from_config(config: BlockLoRAConfig, num_layers: int) -> str:
    """Convenience: go straight from ``BlockLoRAConfig`` to a PEFT regex."""
    indices = resolve_dit_block_indices(num_layers, config.block_ids, config.stride)
    return build_block_lora_target_regex(indices, config.targets)


def filter_block_lora_state_dict(
    state_dict: dict, block_ids: Sequence[int], stride: int = 1
) -> dict:
    """Keep only LoRA tensors belonging to the given block groups."""
    prefixes = []
    for b in block_ids:
        for i in range(b * stride, (b + 1) * stride):
            prefixes.append(f"blocks.{i}.")
    return {
        k: v for k, v in state_dict.items()
        if any(p in k for p in prefixes)
    }


def scale_block_lora_state_dict(state_dict: dict, alpha: float) -> dict:
    """Scale LoRA tensors by ``alpha``. Mirrors B-LoRA's ``scale_lora``."""
    return {k: v * alpha for k, v in state_dict.items()}


def parse_block_ids_cli(spec: str) -> List[int]:
    """Parse a CLI string like ``"0,1,4"`` or ``"0-2,5"`` into a list of ints."""
    if not spec:
        return []
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))
