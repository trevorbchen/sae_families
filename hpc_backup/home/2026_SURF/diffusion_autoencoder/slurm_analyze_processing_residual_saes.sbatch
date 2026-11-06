#!/bin/bash
#SBATCH --job-name=processing-residual-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=logs/processing-residual-analysis-%j.out
#SBATCH --error=logs/processing-residual-analysis-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026/processing_residual_spatial"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
cd "$ROOT"
mkdir -p logs results/processing_residual_spatial/jpeg_q60 results/processing_residual_spatial/random_phase_f17

"$PY" analyze_processing_residual_sae.py \
  --residual-cache "$SCRATCH/jpeg_q60/cache.pt" --base-latent-cache "$BASE" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/jpeg_q60/saes" \
  --output-dir results/processing_residual_spatial/jpeg_q60 \
  --objectives latent_mse laplacian fourier_high --seeds 2030

exec "$PY" analyze_processing_residual_sae.py \
  --residual-cache "$SCRATCH/random_phase_f17/cache.pt" --base-latent-cache "$BASE" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/random_phase_f17/saes" \
  --output-dir results/processing_residual_spatial/random_phase_f17 \
  --objectives latent_mse fourier_high --seeds 2030
