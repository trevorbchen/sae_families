#!/bin/bash
#SBATCH --job-name=eval-resid-combined
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-resid-combined-%j.out
#SBATCH --error=logs/eval-resid-combined-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SPECIALIST_ROOT="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1"
cd "$ROOT"
mkdir -p logs "$SPECIALIST_ROOT/combined"

exec "$PY" evaluate_combined_residual_specialists.py \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --base-sae "$SPECIALIST_ROOT/base_mse_8192_k32/last.pt" \
  --specialist-root "$SPECIALIST_ROOT" \
  --out "$SPECIALIST_ROOT/combined/metrics.json" \
  --batch-size 8
