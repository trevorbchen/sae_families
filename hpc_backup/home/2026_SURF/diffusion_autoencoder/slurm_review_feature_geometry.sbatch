#!/bin/bash
#SBATCH --job-name=review-geometry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/review-geometry-%j.out
#SBATCH --error=logs/review-geometry-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/review_feature_geometry"

cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" evaluate_review_feature_geometry.py \
  --manifest-csv results/family_sae/kimi_frequency_family_labels/analysis/feature_labels.csv \
  --output-jsonl "$OUT/feature_geometry.jsonl" \
  --output-csv "$OUT/feature_geometry.csv" \
  --direct-mse /resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt \
  --grouped-objective-root results/spatial_map_sae_weighted_seed_replicates/flat \
  --grouped-objective-seed seed_2030 \
  --tandem-checkpoint /resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/runs/mse_plus_all_four_seed2030/last.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --examples 10 --rank-examples 4096 --batch-size 128 \
  --decode-batch-size 64 --feature-batch-size 8
