#!/bin/bash
#SBATCH --job-name=resid-spec-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/resid-spec-smoke-%A_%a.out
#SBATCH --error=logs/resid-spec-smoke-%A_%a.err
set -euo pipefail

OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1_smoke/$OBJECTIVE"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_residual_specialist.py \
  --objective "$OBJECTIVE" \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --base-sae /resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/last.pt \
  --out-dir "$OUT" --feature-count 2048 --k 8 \
  --steps 1 --batch-size 8 --lr-warmup 1 --lambda-warmup 1 \
  --gradient-target-ratio 0.1 --calibration-examples 32 \
  --validate-every 1 --seed 2026
