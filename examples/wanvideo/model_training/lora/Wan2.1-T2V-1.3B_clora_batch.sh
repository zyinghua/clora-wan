#!/usr/bin/env bash

set -euo pipefail

export DIFFSYNTH_MODEL_BASE_PATH="/workspace/autodl-tmp/models/clora-wan"

# Same CLoRA config applied to every video in the batch.
LORA_BLOCK_IDS="1,2"
LORA_BLOCK_STRIDE="5"

VIDEO_DIR="/workspace/clora-wan/assets/lora-videos"
OUTPUT_BASE="./models/train/clora_g${LORA_BLOCK_IDS//,/-}_s${LORA_BLOCK_STRIDE}"

# Parallel arrays: VIDEOS[i] is trained with PROMPTS[i]. Edit prompts to taste.
VIDEOS=(
  "astronaut-moonwalk1.mp4"
  "bear-forest-walking1.mp4"
  "bird1.mp4"
  "bird2.mp4"
  "buffalo.mp4"
  "dog-running.mp4"
  "dog1.mp4"
  "horse.mp4"
  "kung-fu.mp4"
  "person-golf1.mp4"
  "style-motorcycling.mp4"
  "tom1.mp4"
  "oil-painting-dog-running.mp4"
  "pencil-drawing-racing-car.mp4"
  "person-away-from-cam.mp4"
  "person-skateboarding.mp4"
  "person-walking-away-cam2.mp4"
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
  "A [q]"
)

if [[ ${#VIDEOS[@]} -ne ${#PROMPTS[@]} ]]; then
  echo "VIDEOS (${#VIDEOS[@]}) and PROMPTS (${#PROMPTS[@]}) length mismatch." >&2
  exit 1
fi

total="${#VIDEOS[@]}"
for i in "${!VIDEOS[@]}"; do
  video="${VIDEOS[$i]}"
  prompt="${PROMPTS[$i]}"
  stem="${video%.*}"
  video_path="${VIDEO_DIR}/${video}"
  output_path="${OUTPUT_BASE}/${stem}"

  if [[ ! -f "${video_path}" ]]; then
    echo "[skip] missing video: ${video_path}" >&2
    continue
  fi

  echo "============================================================"
  echo "[$((i + 1))/${total}] training on ${video}"
  echo "  prompt: ${prompt}"
  echo "  output: ${output_path}"
  echo "============================================================"

  accelerate launch examples/wanvideo/model_training/train.py \
    --dataset_base_path "" \
    --video_path "${video_path}" \
    --video_prompt "${prompt}" \
    --height 480 \
    --width 832 \
    --dataset_repeat 500 \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "${output_path}" \
    --lora_base_model "dit" \
    --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o" \
    --lora_rank 8 \
    --lora_block_ids "${LORA_BLOCK_IDS}" \
    --lora_block_stride "${LORA_BLOCK_STRIDE}"
done

echo "done — all per-video LoRAs in ${OUTPUT_BASE}/"
