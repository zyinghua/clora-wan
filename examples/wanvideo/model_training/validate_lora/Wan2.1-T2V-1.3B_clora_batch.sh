#!/usr/bin/env bash

set -euo pipefail

export DIFFSYNTH_MODEL_BASE_PATH="/workspace/autodl-tmp/models/clora-wan"

# Match the training batch's CLoRA config so we resolve the same output dir.
LORA_BLOCK_IDS="1,2"
LORA_BLOCK_STRIDE="5"
EPOCH=1

LORA_BASE="/workspace/clora-wan/models/train/clora_g${LORA_BLOCK_IDS//,/-}_s${LORA_BLOCK_STRIDE}"
OUTPUT_DIR="./outputs/clora_g${LORA_BLOCK_IDS//,/-}_s${LORA_BLOCK_STRIDE}_e${EPOCH}"
mkdir -p "${OUTPUT_DIR}"

# Mirror the training batch: STEMS[i] uses PROMPTS[i] (the same prompt the LoRA was trained on).
STEMS=(
  "astronaut-moonwalk1"
  "bear-forest-walking1"
  "buffalo"
  "horse"
  "person-golf1"
  "style-motorcycling"
  "tom1"
  "oil-painting-dog-running"
  "pencil-drawing-racing-car"
  "kung-fu"
  "dog-running"
  "dog1"
  "bird1"
  "bird2"
  "person-away-from-cam"
  "person-skateboarding"
)

PROMPTS=(
  "A [a]"
  "A [b]"
  "A [c]"
  "A [d]"
  "A [e]"
  "A [f]"
  "A [g]"
  "A [h]"
  "A [i]"
  "A [j]"
  "A [k]"
  "A [l]"
  "A [m]"
  "A [n]"
  "A [o]"
  "A [p]"
)

if [[ ${#STEMS[@]} -ne ${#PROMPTS[@]} ]]; then
  echo "STEMS (${#STEMS[@]}) and PROMPTS (${#PROMPTS[@]}) length mismatch." >&2
  exit 1
fi

total="${#STEMS[@]}"
for i in "${!STEMS[@]}"; do
  stem="${STEMS[$i]}"
  prompt="${PROMPTS[$i]}"
  g1="${LORA_BASE}/${stem}/epoch-${EPOCH}_g1.safetensors"
  g2="${LORA_BASE}/${stem}/epoch-${EPOCH}_g2.safetensors"

  if [[ ! -f "${g1}" || ! -f "${g2}" ]]; then
    echo "[skip] missing LoRA file(s) for ${stem}" >&2
    [[ ! -f "${g1}" ]] && echo "  missing: ${g1}" >&2
    [[ ! -f "${g2}" ]] && echo "  missing: ${g2}" >&2
    continue
  fi

  echo "============================================================"
  echo "[$((i + 1))/${total}] ${stem}  prompt=\"${prompt}\""
  echo "============================================================"

  # 1) g1 only
  echo "  -> ${stem}_g1.mp4 (g1 only)"
  python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py \
    --lora "${g1}:1.0" \
    --prompt "${prompt}" \
    --output "${OUTPUT_DIR}/${stem}_g1.mp4"

  # 2) g2 only
  echo "  -> ${stem}_g2.mp4 (g2 only)"
  python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py \
    --lora "${g2}:1.0" \
    --prompt "${prompt}" \
    --output "${OUTPUT_DIR}/${stem}_g2.mp4"

  # 3) g1 + g2 (additively fused)
  echo "  -> ${stem}_g1g2.mp4 (g1 + g2)"
  python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py \
    --lora "${g1}:1.0" \
    --lora "${g2}:1.0" \
    --prompt "${prompt}" \
    --output "${OUTPUT_DIR}/${stem}_g1g2.mp4"
done

echo "done — outputs in ${OUTPUT_DIR}/"
