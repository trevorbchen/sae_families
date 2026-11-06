#!/bin/bash
#SBATCH --job-name=resid-spec-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/resid-spec-full-%A_%a.out
#SBATCH --error=logs/resid-spec-full-%A_%a.err
set -euo pipefail

OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/$OBJECTIVE"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_residual_specialist.py \
  --objective "$OBJECTIVE" \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --base-sae /resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/last.pt \
  --out-dir "$OUT" --feature-count 2048 --k 8 \
  --steps 20000 --batch-size 32 --lr 3e-4 \
  --lr-warmup 500 --lambda-warmup 2000 \
  --gradient-target-ratio 0.1 --calibration-examples 512 \
  --mse-tolerance 0.02 --aux-coeff 0.1 --grad-clip 1.0 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed 2026
