#!/bin/bash
#SBATCH --job-name=viclip_eval
#SBATCH --output=viclip_eval_%j.out
#SBATCH --error=viclip_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=3090-gcondo

# Load miniforge module and initialize conda
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

# Activate environment
conda activate clora

# Set cache directories
export MODELSCOPE_CACHE=/users/erluo/scratch/.cache/modelscope
export HF_HOME=/users/erluo/scratch/.cache/huggingface

# Navigate to project root
cd /users/erluo/scratch/clora-wan

# Install required packages if not present
pip install transformers>=4.30.0 pandas tqdm opencv-python --quiet 2>/dev/null

# Run evaluation (uses all frames from each video)
python evaluate_viclip_ablation.py \
    --ablation_dir ablation_runs2 \
    --output_dir viclip_evaluation \
    --device cuda

echo "Evaluation complete!"
