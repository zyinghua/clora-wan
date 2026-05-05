#!/bin/bash
# SDXL/BLoRA Cross-Attention Visualization
#
# BLoRA identifies two special blocks:
#   - Content block: up_blocks.0.attentions.0
#   - Style block:   up_blocks.0.attentions.1
#
# This script visualizes cross-attention patterns to understand
# how text conditioning is distributed differently across these blocks.

python examples/blora/visualize_cross_attention.py \
  --prompt "A photo of a car driving on a road." \
  --out_dir ./visualizations/sdxl_cross_attn/car_driving \
  --blocks content,style,mid_block.attentions.0 \
  --per_token_block style \
  --timestep_index 15 \
  --seed 42
