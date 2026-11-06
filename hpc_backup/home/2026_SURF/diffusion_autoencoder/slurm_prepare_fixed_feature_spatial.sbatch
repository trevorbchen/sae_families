#!/bin/bash
#SBATCH --job-name=fixed-feature-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fixed-feature-data-%j.out
#SBATCH --error=logs/fixed-feature-data-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs /resnick/scratch/tbchen/surf_2026/fixed_feature_spatial results/fixed_feature_spatial

exec "$PY" prepare_fixed_feature_spatial_latents.py \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --clean-latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output /resnick/scratch/tbchen/surf_2026/fixed_feature_spatial/ffhq_fixed_high.pt \
  --preview results/fixed_feature_spatial/injection_preview.png \
  --frequency-x 17 --frequency-y 17 --amplitude 3 --prevalence 0.05
