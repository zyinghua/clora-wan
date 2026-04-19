# Repo for CLoRA based on Wan 2.1 T2V 1.3B

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