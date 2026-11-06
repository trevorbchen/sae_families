#!/bin/bash
#SBATCH --job-name=processing-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/processing-analysis-%j.out
#SBATCH --error=logs/processing-analysis-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026/processing_feature_spatial"
cd "$ROOT"
mkdir -p logs results/processing_feature_spatial_compact/jpeg_q60 results/processing_feature_spatial_compact/random_phase_f17

"$PY" analyze_processing_feature_sae_detection.py \
  --cache "$SCRATCH/jpeg_q60/cache.pt" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/jpeg_q60/pilot_saes" \
  --output-dir results/processing_feature_spatial_compact/jpeg_q60 \
  --objectives latent_mse laplacian fourier_high --seeds 2030

exec "$PY" analyze_processing_feature_sae_detection.py \
  --cache "$SCRATCH/random_phase_f17/cache.pt" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/random_phase_f17/pilot_saes" \
  --output-dir results/processing_feature_spatial_compact/random_phase_f17 \
  --objectives latent_mse fourier_high --seeds 2030
