#!/usr/bin/env bash

set -euo pipefail

# Block-wise CLoRA composition experiments — adapted from B-LoRA (arxiv 2403.14572).
# Established mapping for Wan T2V 1.3B (from prior probing):
#   g1 (DiT blocks 5-9)  = motion / content
#   g2 (DiT blocks 10-14) = style / content
# Content can land in either group, so we route the *other* axis (motion or style)
# to its dedicated group and put content in the leftover slot:
#   - Style transfer  "A [c] in [s] style":   content.g1 + style.g2
#   - Motion transfer "A [c] doing [m] motion": content.g2 + motion.g1
#   - Triple          "A [c] in [s] style doing [m] motion":
#       content + motion both want g1 (split alpha 0.5 each); style owns g2.

export DIFFSYNTH_MODEL_BASE_PATH="/workspace/autodl-tmp/models/clora-wan"

LORA_BLOCK_IDS="1,2"
LORA_BLOCK_STRIDE="5"
EPOCH=1
PY="examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py"
LORA_BASE="/workspace/clora-wan/models/train/clora_g${LORA_BLOCK_IDS//,/-}_s${LORA_BLOCK_STRIDE}"
OUTPUT_DIR="./outputs/clora_compose_e${EPOCH}"
mkdir -p "${OUTPUT_DIR}"

# Token map (matches training batch). Use these as placeholders in prompts.
#   [a] astronaut-moonwalk1     [b] bear-forest-walking1   [c] buffalo
#   [d] horse                   [e] person-golf1           [f] style-motorcycling*
#   [g] tom1*                   [h] oil-painting-dog-running*  [i] pencil-drawing-racing-car*
#   [j] kung-fu                 [k] dog-running            [l] dog1
#   [m] bird1                   [n] bird2                  [o] person-away-from-cam
#   [p] person-skateboarding
#   * = style-oriented training video

# Path helper: lora <stem> <joint|g1|g2>
lora() {
  case "$2" in
    joint) echo "${LORA_BASE}/$1/epoch-${EPOCH}.safetensors";;
    g1)    echo "${LORA_BASE}/$1/epoch-${EPOCH}_g1.safetensors";;
    g2)    echo "${LORA_BASE}/$1/epoch-${EPOCH}_g2.safetensors";;
  esac
}

# Run one test. Skips if any required LoRA is missing or the output already exists.
seed_counter=0
run_test() {
  local name="$1" prompt="$2"
  shift 2
  local lora_args=() spec p
  for spec in "$@"; do
    [[ -n "${spec}" ]] || continue
    p="${spec%:*}"
    if [[ ! -f "${p}" ]]; then
      echo "[skip] ${name}: missing LoRA: ${p}" >&2
      return 0
    fi
    lora_args+=(--lora "${spec}")
  done
  local out="${OUTPUT_DIR}/${name}.mp4"
  if [[ -f "${out}" ]]; then
    echo "[skip] ${name}: output exists -> ${out}"
    return 0
  fi
  seed_counter=$((seed_counter + 1))
  local seed=0
  echo "============================================================"
  echo "[${seed_counter}] ${name}  seed=${seed}"
  echo "  prompt: ${prompt}"
  echo "  loras : ${lora_args[*]:-<base only>}"
  echo "============================================================"
  python "${PY}" \
    "${lora_args[@]}" \
    --prompt "${prompt}" \
    --seed "${seed}" \
    --output "${out}"
}

# ============================================================
# B) Style transfer (content + style) — "A [c] in [s] style"
#    Routing: content_video.g1 + style_video.g2 (style lives in g2).
# ============================================================

# run_test "B1_astronaut-in-oilpaint"  "A [a] in [h] style"  \
#   "$(lora astronaut-moonwalk1 g1):1.0"  "$(lora oil-painting-dog-running g2):1.0"

# run_test "B2_astronaut-in-cartoon"   "A [a] in [g] style"  \
#   "$(lora astronaut-moonwalk1 g1):1.0"  "$(lora tom1 g2):1.0"

# run_test "B3_bear-in-pencil"  "A [b] in [i] style"  \
#   "$(lora bear-forest-walking1 g1):1.0"  "$(lora pencil-drawing-racing-car g2):1.0"

# run_test "B4_dog-in-stylemoto"  "A [k] in [f] style"  \
#   "$(lora dog-running g1):1.0"  "$(lora style-motorcycling g2):1.0"

# run_test "B5_bird-in-cartoon"  "A [m] in [g] style"  \
#   "$(lora bird1 g1):1.0"  "$(lora tom1 g2):1.0"

# # dog1 in horse style — horse clip carries a distinctive cinematic look that's
# # worth probing as a non-painterly "style" source.
# run_test "B6_dog-in-horse"  "A [l] in [d] style"  \
#   "$(lora dog1 g1):1.0"  "$(lora horse g2):1.0"

# # buffalo + oil-painting (introduces buffalo content)
# run_test "B7_buffalo-in-oilpaint"  "A [c] in [h] style"  \
#   "$(lora buffalo g1):1.0"  "$(lora oil-painting-dog-running g2):1.0"

# # person-golf + cartoon (introduces person-golf content)
# run_test "B8_golf-in-cartoon"  "A [e] in [g] style"  \
#   "$(lora person-golf1 g1):1.0"  "$(lora tom1 g2):1.0"

# # astronaut + pencil-drawing (rounds out astronaut's style sweep — already paired w/ oilpaint and cartoon)
# run_test "B9_astronaut-in-pencil"  "A [a] in [i] style"  \
#   "$(lora astronaut-moonwalk1 g1):1.0"  "$(lora pencil-drawing-racing-car g2):1.0"

# bear / buffalo in horse style — non-painterly cinematic style on different animal subjects
# run_test "B10_bear-in-horse"  "A [b] in [d] style"  \
#   "$(lora bear-forest-walking1 g1):1.0"  "$(lora horse g2):1.0"

# run_test "B11_buffalo-in-horse"  "A [c] in [d] style"  \
#   "$(lora buffalo g1):1.0"  "$(lora horse g2):1.0"

# ============================================================
# C) Motion transfer (content + motion) — "A [c] doing [m] motion"
#    Routing: content_video.g2 + motion_video.g1 (motion lives in g1, so
#    content has to fall back to g2 — the variable slot).
#    Pairings respect domain: human-human, animal-animal, plus one cross-domain.
# ============================================================

# astronaut (human) doing kung-fu (human motion)
# run_test "C1_astronaut-doing-kungfu"  "A [a] doing [j] motion"  \
#   "$(lora astronaut-moonwalk1 g2):1.0"  "$(lora kung-fu g1):1.0"

# # person-away-from-cam doing skateboarding (human-human)
# run_test "C2_person-doing-skate"  "A [o] doing [p] motion"  \
#   "$(lora person-away-from-cam g2):1.0"  "$(lora person-skateboarding g1):1.0"

# # bear doing dog-running motion (animal-animal cross-class)
# run_test "C3_bear-doing-run"  "A [b] doing [k] motion"  \
#   "$(lora bear-forest-walking1 g2):1.0"  "$(lora dog-running g1):1.0"

# # bird1 doing bird2 motion (same-class)
# run_test "C4_bird1-doing-bird2"  "A [m] doing [n] motion"  \
#   "$(lora bird1 g2):1.0"  "$(lora bird2 g1):1.0"

# # horse doing kung-fu motion (cross-domain bizarre — what does it look like?)
# run_test "C5_horse-doing-kungfu"  "A [d] doing [j] motion"  \
#   "$(lora horse g2):1.0"  "$(lora kung-fu g1):1.0"

# # buffalo doing dog-running motion (animal-animal, slow vs fast quadruped)
# run_test "C6_buffalo-doing-run"  "A [c] doing [k] motion"  \
#   "$(lora buffalo g2):1.0"  "$(lora dog-running g1):1.0"

# # person-golf doing skateboarding motion (human-human, very different motion class)
# run_test "C7_golf-doing-skate"  "A [e] doing [p] motion"  \
#   "$(lora person-golf1 g2):1.0"  "$(lora person-skateboarding g1):1.0"

# # dog1 (still) doing dog-running motion (same-species, still→active transition)
# run_test "C8_dog1-doing-run"  "A [l] doing [k] motion"  \
#   "$(lora dog1 g2):1.0"  "$(lora dog-running g1):1.0"

# # astronaut doing golf motion (human-human, walking → swinging)
# run_test "C9_astronaut-doing-golf"  "A [a] doing [e] motion"  \
#   "$(lora astronaut-moonwalk1 g2):1.0"  "$(lora person-golf1 g1):1.0"

# run_test "C10_astronaut-doing-golf-g11"  "A [a] doing [e] motion"  \
#   "$(lora astronaut-moonwalk1 g1):1.0"  "$(lora person-golf1 g1):1.0"

# ============================================================
# D) Triple composition — "A [c] in [s] style doing [m] motion"
#    With only 2 block groups, content+motion both target g1 (overlap):
#    each gets alpha 0.5 so they share the budget; style owns g2 alone.
# ============================================================

# run_test "D1_astronaut-cartoon-kungfu"  "A [a] in [g] style doing [j] motion"  \
#   "$(lora astronaut-moonwalk1 g1):1.0"  "$(lora kung-fu g1):1.0"  "$(lora tom1 g2):1.0"

# run_test "D2_bear-pencil-skate"  "A [b] in [i] style doing [p] motion"  \
#   "$(lora bear-forest-walking1 g1):1.0"  "$(lora person-skateboarding g1):1.0"  "$(lora pencil-drawing-racing-car g2):1.0"

run_test "D3_buffalo-oilpaint-run"  "A [c] in [h] style doing [k] motion"  \
  "$(lora buffalo g2):1.0"  "$(lora dog-running g1):1.0"  "$(lora oil-painting-dog-running g2):1.0"

run_test "D4_golf-oilpaint-skate"  "A [e] in [h] style doing [p] motion"  \
  "$(lora person-golf1 g1):1.0"  "$(lora person-skateboarding g1):1.0"  "$(lora oil-painting-dog-running g2):1.0"

# ============================================================
# E) Token + free-form English — does the placeholder compose with natural language?
# ============================================================
run_test "E1_astronaut-red"        "A [a] dressed in red"       "$(lora astronaut-moonwalk1 joint):1.0"
run_test "E2_bear-dancing"         "A [b] dancing"              "$(lora bear-forest-walking1 joint):1.0"
run_test "E3_lion-in-oilpaint"     "A lion in [h] style"        "$(lora oil-painting-dog-running g2):1.0"
run_test "E5_horse-doing-kungfu"   "A horse doing [j] motion"   "$(lora kung-fu g1):1.0"
run_test "E6_buffalo-underwater"   "A [c] underwater"           "$(lora buffalo joint):1.0"
run_test "E7_bear-in-horse-style"  "A bear in [d] style"        "$(lora horse g2):1.0"

# ============================================================
# G) Style fusion — load TWO style LoRAs into g2 (their deltas add) on a single content.
#    No B-LoRA analog (paper uses one style); included to probe what dual-style blending
#    looks like in Wan. Each style takes alpha 0.5 so the combined delta isn't 2x.
# ============================================================
run_test "G1_astronaut-cartoon+oilpaint"  "A [a] in [g] [h] style"  \
  "$(lora astronaut-moonwalk1 g1):1.0"  "$(lora tom1 g2):0.5"  "$(lora oil-painting-dog-running g2):0.5"

run_test "G2_bear-pencil+horse"  "A [b] in [i] [d] style"  \
  "$(lora bear-forest-walking1 g1):1.0"  "$(lora pencil-drawing-racing-car g2):0.5"  "$(lora horse g2):0.5"


echo "============================================================"
echo "done — ${seed_counter} runs, outputs in ${OUTPUT_DIR}/"
echo "============================================================"
