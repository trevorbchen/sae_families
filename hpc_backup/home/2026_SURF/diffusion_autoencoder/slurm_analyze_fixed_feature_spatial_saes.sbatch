#!/bin/bash
#SBATCH --job-name=fixed-feature-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fixed-feature-analysis-%j.out
#SBATCH --error=logs/fixed-feature-analysis-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/fixed_feature_spatial/analysis

exec "$PY" analyze_fixed_feature_sae_detection.py \
  --cache /resnick/scratch/tbchen/surf_2026/fixed_feature_spatial/ffhq_fixed_high.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root results/fixed_feature_spatial/saes \
  --output-dir results/fixed_feature_spatial/analysis
