#!/bin/bash
#SBATCH --job-name=residual-general-shortlist
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/residual-general-shortlist-%j.out
#SBATCH --error=logs/residual-general-shortlist-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs

exec "$PY" residual_general_feature_shortlist_gui.py \
  --checkpoint-root results/spatial_map_sae_weighted_seed_replicates/flat \
  --seed seed_2030 \
  --autoencoder results/spatial_conv_ae/last.pt \
  --latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --review-annotations /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/annotations.json \
  --choices /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/paper_selection.json \
  --port 8771 \
  --label "Residual SAE general features (lambda=1 equivalent)"
