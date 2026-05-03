#!/usr/bin/env bash

set -euo pipefail

export DIFFSYNTH_MODEL_BASE_PATH="/workspace/autodl-tmp/models/clora-wan"

LORA_BLOCK_IDS="1,2"
LORA_BLOCK_STRIDE="5"
OUTPUT_PATH="./models/train/Wan2.1-T2V-1.3B_clora_g${LORA_BLOCK_IDS//,/-}_s${LORA_BLOCK_STRIDE}"
VIDEO_PATH=""
VIDEO_PROMPT=""

# Multi-clip dataset defaults (used when VIDEO_PATH is empty).
DATASET_BASE_PATH="data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B"
DATASET_METADATA_PATH="data/diffsynth_example_dataset/wanvideo/Wan2.1-T2V-1.3B/metadata.csv"

if [[ -n "${VIDEO_PATH}" ]]; then
  if [[ -z "${VIDEO_PROMPT}" ]]; then
    echo "VIDEO_PATH is set but VIDEO_PROMPT is empty." >&2
    exit 1
  fi
  # Absolute paths in VIDEO_PATH bypass the join with DATASET_BASE_PATH.
  DATASET_BASE_PATH=""
  DATASET_ARGS=(--video_path "${VIDEO_PATH}" --video_prompt "${VIDEO_PROMPT}")
else
  modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.1-T2V-1.3B/*" --local_dir ./data/diffsynth_example_dataset
  DATASET_ARGS=(--dataset_metadata_path "${DATASET_METADATA_PATH}")
fi

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  "${DATASET_ARGS[@]}" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1000 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --lora_base_model "dit" \
  --lora_target_modules "self_attn.q,self_attn.k,self_attn.v,self_attn.o,cross_attn.q,cross_attn.k,cross_attn.v,cross_attn.o" \
  --lora_rank 8 \
  --lora_block_ids "${LORA_BLOCK_IDS}" \
  --lora_block_stride "${LORA_BLOCK_STRIDE}"
