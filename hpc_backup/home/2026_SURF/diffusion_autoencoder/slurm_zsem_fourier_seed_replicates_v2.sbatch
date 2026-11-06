#!/bin/bash
#SBATCH --job-name=fourier-seeds-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fourier-seeds-v2-%A_%a.out
#SBATCH --error=logs/fourier-seeds-v2-%A_%a.err
set -euo pipefail

OBJECTIVES=(
  fourier_bands fourier_bands fourier_bands
  fourier_low fourier_low fourier_low
  fourier_mid fourier_mid fourier_mid
  fourier_high fourier_high fourier_high
)
SEEDS=(2030 2031 2032 2030 2031 2032 2030 2031 2032 2030 2031 2032)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
OUT="results/zsem_objective_seed_replicates_v2/$OBJECTIVE/seed_$SEED"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_zsem_transformed_sae.py \
  --objective "$OBJECTIVE" \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data "$DATA" \
  --out-dir "$OUT" \
  --dict-mult 32 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --reinit-every 1000 --dead-after 3000 \
  --max-rescue 256 --split-seed 2030 --seed "$SEED"
