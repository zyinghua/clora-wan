#!/usr/bin/env bash
#
# CLoRA Composition Test Script
# Tests various combinations of trained concepts using different block groups.
#
# Trained concepts and their placeholders:
#   [a] = astronaut-moonwalk1    (motion: moonwalking)
#   [b] = bear-forest-walking1   (subject+motion: bear walking)
#   [c] = buffalo                (subject: buffalo)
#   [d] = horse                  (subject: horse)
#   [e] = person-golf1           (motion: golf swing)
#   [f] = style-motorcycling     (style+motion: motorcycling)
#   [g] = tom1                   (subject: Tom character)
#   [h] = oil-painting-dog-running   (style: oil painting)
#   [i] = pencil-drawing-racing-car  (style: pencil sketch)
#   [j] = kung-fu                (motion: kung-fu moves)
#
# Strategy: Use g1 for one concept and g2 for another to minimize interference.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

LORA_BASE="./models/train/clora_g1-2_s5"
OUTPUT_DIR="./outputs/clora_compositions"
EPOCH="epoch-1"

mkdir -p "${OUTPUT_DIR}"

run_clora() {
    local name="$1"
    local prompt="$2"
    shift 2
    local lora_args=("$@")

    echo "============================================================"
    echo "Generating: ${name}"
    echo "Prompt: ${prompt}"
    echo "LoRAs: ${lora_args[*]}"
    echo "============================================================"

    python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py \
        "${lora_args[@]}" \
        --prompt "${prompt}" \
        --output "${OUTPUT_DIR}/${name}.mp4"

    echo "[done] ${OUTPUT_DIR}/${name}.mp4"
    echo ""
}

# =============================================================================
# 1. STYLE TRANSFER EXPERIMENTS
#    Apply artistic styles to different motions/subjects
# =============================================================================

echo ">>> STYLE TRANSFER EXPERIMENTS"

# Oil painting style [h] + bear walking motion [b]
run_clora "style_oil-painting_motion_bear-walk" \
    "A [h] in [b] motion" \
    --lora "${LORA_BASE}/oil-painting-dog-running/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/bear-forest-walking1/${EPOCH}_g1.safetensors:1.0"

# Oil painting style [h] + moonwalk motion [a]
run_clora "style_oil-painting_motion_moonwalk" \
    "A [h] doing [a]" \
    --lora "${LORA_BASE}/oil-painting-dog-running/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/astronaut-moonwalk1/${EPOCH}_g1.safetensors:1.0"

# Pencil drawing style [i] + kung-fu motion [j]
run_clora "style_pencil_motion_kungfu" \
    "A [i] performing [j]" \
    --lora "${LORA_BASE}/pencil-drawing-racing-car/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/kung-fu/${EPOCH}_g1.safetensors:1.0"

# Pencil drawing style [i] + horse subject [d]
run_clora "style_pencil_subject_horse" \
    "A [i] of [d] running" \
    --lora "${LORA_BASE}/pencil-drawing-racing-car/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/horse/${EPOCH}_g1.safetensors:1.0"

# =============================================================================
# 2. MOTION TRANSFER EXPERIMENTS
#    Apply distinctive motions to different subjects
# =============================================================================

echo ">>> MOTION TRANSFER EXPERIMENTS"

# Moonwalk motion [a] + horse subject [d]
run_clora "motion_moonwalk_subject_horse" \
    "A [d] doing [a]" \
    --lora "${LORA_BASE}/horse/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/astronaut-moonwalk1/${EPOCH}_g1.safetensors:1.0"

# Kung-fu motion [j] + buffalo subject [c]
run_clora "motion_kungfu_subject_buffalo" \
    "A [c] performing [j] moves" \
    --lora "${LORA_BASE}/buffalo/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/kung-fu/${EPOCH}_g1.safetensors:1.0"

# Golf swing motion [e] + bear subject [b]
run_clora "motion_golf_subject_bear" \
    "A [b] doing [e]" \
    --lora "${LORA_BASE}/bear-forest-walking1/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/person-golf1/${EPOCH}_g1.safetensors:1.0"

# Moonwalk motion [a] + Tom character [g]
run_clora "motion_moonwalk_subject_tom" \
    "[g] doing [a] dance" \
    --lora "${LORA_BASE}/tom1/${EPOCH}_g2.safetensors:1.0" \
    --lora "${LORA_BASE}/astronaut-moonwalk1/${EPOCH}_g1.safetensors:1.0"

# =============================================================================
# 3. SUBJECT COMPOSITION EXPERIMENTS
#    Combine different subjects/characters
# =============================================================================

echo ">>> SUBJECT COMPOSITION EXPERIMENTS"

# Horse [d] + Buffalo [c] scene
run_clora "subjects_horse_buffalo" \
    "A [d] and [c] in a field" \
    --lora "${LORA_BASE}/horse/${EPOCH}_g1.safetensors:1.0" \
    --lora "${LORA_BASE}/buffalo/${EPOCH}_g2.safetensors:1.0"

# Tom [g] + Bear [b] interaction
run_clora "subjects_tom_bear" \
    "[g] meets [b]" \
    --lora "${LORA_BASE}/tom1/${EPOCH}_g1.safetensors:1.0" \
    --lora "${LORA_BASE}/bear-forest-walking1/${EPOCH}_g2.safetensors:1.0"

# =============================================================================
# 4. STYLE + MOTION + SUBJECT (Triple composition)
#    More complex combinations using alpha blending
# =============================================================================

echo ">>> TRIPLE COMPOSITION EXPERIMENTS (with reduced alphas)"

# Oil painting [h] + horse [d] + moonwalk [a] - all at lower alpha
run_clora "triple_oil-horse-moonwalk" \
    "A [h] style [d] doing [a]" \
    --lora "${LORA_BASE}/oil-painting-dog-running/${EPOCH}_g2.safetensors:0.7" \
    --lora "${LORA_BASE}/horse/${EPOCH}_g1.safetensors:0.7" \
    --lora "${LORA_BASE}/astronaut-moonwalk1/${EPOCH}_g1.safetensors:0.5"

# Pencil [i] + Tom [g] + kung-fu [j]
run_clora "triple_pencil-tom-kungfu" \
    "A [i] of [g] doing [j]" \
    --lora "${LORA_BASE}/pencil-drawing-racing-car/${EPOCH}_g2.safetensors:0.7" \
    --lora "${LORA_BASE}/tom1/${EPOCH}_g1.safetensors:0.7" \
    --lora "${LORA_BASE}/kung-fu/${EPOCH}_g1.safetensors:0.5"

# =============================================================================
# 5. ALPHA ABLATION
#    Same composition with different alpha values
# =============================================================================

echo ">>> ALPHA ABLATION (oil painting + bear walk)"

for alpha in 0.3 0.5 0.7 1.0 1.5; do
    run_clora "ablation_alpha${alpha}_oil-bear" \
        "A [h] in [b] motion" \
        --lora "${LORA_BASE}/oil-painting-dog-running/${EPOCH}_g2.safetensors:${alpha}" \
        --lora "${LORA_BASE}/bear-forest-walking1/${EPOCH}_g1.safetensors:${alpha}"
done

# =============================================================================
# 6. SINGLE CONCEPT BASELINES
#    For comparison with compositions
# =============================================================================

echo ">>> SINGLE CONCEPT BASELINES"

run_clora "baseline_oil-painting" \
    "A [h]" \
    --lora "${LORA_BASE}/oil-painting-dog-running/${EPOCH}.safetensors:1.0"

run_clora "baseline_bear-walk" \
    "A [b]" \
    --lora "${LORA_BASE}/bear-forest-walking1/${EPOCH}.safetensors:1.0"

run_clora "baseline_moonwalk" \
    "A [a]" \
    --lora "${LORA_BASE}/astronaut-moonwalk1/${EPOCH}.safetensors:1.0"

run_clora "baseline_kungfu" \
    "A [j]" \
    --lora "${LORA_BASE}/kung-fu/${EPOCH}.safetensors:1.0"

echo "============================================================"
echo "All compositions complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "============================================================"
