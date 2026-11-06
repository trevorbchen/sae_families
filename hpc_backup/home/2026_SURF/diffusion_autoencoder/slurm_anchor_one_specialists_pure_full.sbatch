#!/bin/bash
#SBATCH --job-name=anchor-one-pure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/anchor-one-pure-%A_%a.out
#SBATCH --error=logs/anchor-one-pure-%A_%a.err
set -euo pipefail

OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_pure_v1"
OUT="$RUN_ROOT/anchor_plus_one/$OBJECTIVE"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/best_valid.pt"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_residual_specialist.py \
  --objective "$OBJECTIVE" --loss-mode pure_transformed \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt --base-sae "$BASE" \
  --out-dir "$OUT" --feature-count 8192 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 \
  --lr-warmup 500 --lambda-warmup 2000 \
  --gradient-target-ratio 1.0 --calibration-examples 512 \
  --mse-tolerance 0.02 --aux-coeff 0.1 --grad-clip 1.0 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed 2026
