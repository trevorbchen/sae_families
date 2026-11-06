#!/bin/bash
#SBATCH --job-name=b0-feat-abl
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-47%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/block0_feature_ablation_16x_k32

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

. .venv/bin/activate

CHUNK_SIZE=128
FEATURE_START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
FEATURE_END=$((FEATURE_START + CHUNK_SIZE))

python block0_feature_ablation.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --data-dir data \
  --out-dir runs/block0_feature_ablation_16x_k32 \
  --feature-stats runs/block0_feature_ablation_16x_k32/feature_stats.pt \
  --feature-start "${FEATURE_START}" \
  --feature-end "${FEATURE_END}" \
  --num-images 12 \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --scales=-2,-1.5,-1,-0.5,0,0.5,1,1.5,2 \
  --streams attn,mlp \
  --features-per-forward 4 \
  --scales-per-forward 3 \
  --forward-batch-size 96 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --precision bf16 \
  --seed 2026
