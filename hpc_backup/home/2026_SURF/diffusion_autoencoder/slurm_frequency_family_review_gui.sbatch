#!/bin/bash
#SBATCH --job-name=freq-review-api
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/freq-review-api-%j.out
#SBATCH --error=logs/freq-review-api-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"

cd "$ROOT"
mkdir -p logs /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review

exec "$PY" residual_sae_feature_review_gui.py \
  --direct-mse /resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt \
  --grouped-objective-root results/spatial_map_sae_weighted_seed_replicates/flat \
  --grouped-objective-seed seed_2030 \
  --tandem-checkpoint /resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/runs/mse_plus_all_four_seed2030/last.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --annotations /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/annotations.json \
  --port 8813 --examples 10 --rank-examples 4096 --precompute-ranks 500 --batch-size 128
