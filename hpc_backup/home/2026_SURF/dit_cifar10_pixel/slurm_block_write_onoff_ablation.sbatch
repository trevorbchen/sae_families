#!/bin/bash
#SBATCH --job-name=write-onoff
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-23%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/block_writes_onoff_ablation_16x_k32

module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

. .venv/bin/activate

BLOCK_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
STREAM_ID=$((SLURM_ARRAY_TASK_ID % 2))
if [[ "${STREAM_ID}" -eq 0 ]]; then
  STREAM="attn"
else
  STREAM="mlp"
fi

python write_onoff_ablation.py \
  --block-index "${BLOCK_INDEX}" \
  --stream "${STREAM}" \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --data-dir data \
  --out-root runs/block_writes_onoff_ablation_16x_k32 \
  --num-images 12 \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --features-per-forward 8 \
  --forward-batch-size 96 \
  --precision bf16 \
  --seed 2026
