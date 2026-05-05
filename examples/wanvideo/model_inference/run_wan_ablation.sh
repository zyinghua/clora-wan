#!/bin/bash
#SBATCH --job-name=wan_ablation
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=wan_ablation_%j.out
#SBATCH --error=wan_ablation_%j.err

# 1. Load the miniforge module and initialize conda
module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh

# 2. Activate your environment
conda activate clora

# 3. Set modelscope cache to scratch (avoid /workspace permission errors)
export MODELSCOPE_CACHE=/users/erluo/scratch/.cache/modelscope

# 4. Navigate to your working directory
cd /users/erluo/scratch/clora-wan

# 5. Execute the Python script
python examples/wanvideo/model_inference/Wan2.1-T2V-1.3B_ablation.py