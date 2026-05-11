# Repo for CLoRA based on Wan 2.1 T2V 1.3B

## Paper
## Paper:
[https://github.com/zyinghua/clora-wan/tree/main/assets/CLoRA-paper.pdf](https://github.com/zyinghua/clora-wan/tree/main/assets/CLoRA-paper.pdf)


Experiments on block-level LoRA (B-LoRA-style) analysis for the Wan 2.1 T2V-1.3B video DiT, built on top of DiffSynth.

## Setup

Create a fresh conda environment and install the package in editable mode:

```bash
conda create -n clora python=3.10 -y
conda activate clora
pip install -e .
```

That's all — the `pip install -e .` step pulls in every required dependency.

## Downloading the model

Model weights are fetched automatically the first time you run the inference script. Just execute:

```bash
python examples/wanvideo/model_inference/Wan2.1-T2V-1.3B.py
```

This downloads the Wan 2.1 T2V-1.3B DiT, the UMT5-XXL text encoder, and the Wan 2.1 VAE into the path configured inside the script (`local_model_path` in the `ModelConfig` entries) and then runs a text-to-video sample. Edit that path in the script if you want the weights stored elsewhere.

#### To modify the download location, change the `local_model_path` in `model_configs` in `Wan2.1-T2V-1.3B.py`.

## Prompting Ablation

Generate videos that sweep ablation block size / id across a fixed set of content/style/motion prompt pairs:

```bash
python examples/wanvideo/model_inference/Wan2.1-T2V-1.3B_ablation.py
```

Outputs are written next to the script as `video_wan_out_<category>-<block_size>-<block_id>-<prompt>.mp4`. Edit the `ablation_tasks` list and `ablation_range` dict at the top of the file to change the prompt pairs and block sweeps.

## CLoRA training

Each shell script trains a block-scoped LoRA on the Wan DiT. Block group `i` covers DiT blocks `[i*stride, i*stride+stride)`; with the default `LORA_BLOCK_STRIDE=5`, group 1 = blocks 5-9, group 2 = blocks 10-14.

**Single video** — fits one LoRA on one clip with a placeholder prompt. Edit the variables at the top of the script (`LORA_BLOCK_IDS`, `VIDEO_PATH`, `VIDEO_PROMPT`, etc.), then run from the repo root:

```bash
bash examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B_clora.sh
```

Per epoch, the trainer writes a joint checkpoint plus per-group splits:
```
<OUTPUT_PATH>/epoch-N.safetensors        # joint
<OUTPUT_PATH>/epoch-N_g1.safetensors     # only blocks 5-9
<OUTPUT_PATH>/epoch-N_g2.safetensors     # only blocks 10-14
```

**Batch over many videos** — trains one LoRA per video, paired with corresponding prompts (`"A [a]"`, `"A [b]"`, ...):

```bash
bash examples/wanvideo/model_training/lora/Wan2.1-T2V-1.3B_clora_batch.sh
```

Edit the parallel `VIDEOS` and `PROMPTS` arrays in the script to control which clips get trained. Outputs go to `./models/train/clora_g<ids>_s<stride>/<video_stem>/`.

## CLoRA inference

**Single inference** — fuse one or more LoRA files into the DiT and generate a video. Pass each LoRA as `path[:alpha]`, repeat `--lora` to compose multiple:

```bash
python examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora.py \
  --lora path/to/epoch-1_g1.safetensors:1.0 \
  --lora path/to/epoch-1_g2.safetensors:1.0 \
  --prompt "A [a] in [h] style" \
  --output composed.mp4
```

Each `--lora` is fused additively into `pipe.dit`. Disjoint groups (e.g. `g1` + `g2`) don't conflict; same-group LoRAs blend.

**Batch over trained videos** — for every video in the training batch, runs three inferences (g1 only, g2 only, g1+g2):

```bash
bash examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora_batch.sh
```

**Compositional experiments** — runs the cross-video composition matrix (style transfer, motion transfer, triple composition, etc.) following the established block-group routing (`content.g1 + style.g2`, `content.g2 + motion.g1`):

```bash
bash examples/wanvideo/model_training/validate_lora/Wan2.1-T2V-1.3B_clora_compose_batch.sh
```

Outputs land in `./outputs/clora_compose_e<EPOCH>/`. Edit the test cases in the script to add or remove compositions.