#!/bin/bash
#SBATCH --job-name=plant-pure-table
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/plant-pure-table-%j.out
#SBATCH --error=logs/plant-pure-table-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"

"$PY" summarize_sparse_planted_suite.py \
  --root results/sparse_planted_residual_suite_pure
exec "$PY" build_added_feature_detection_tables.py \
  --source results/sparse_planted_residual_suite_pure/comparison.csv \
  --output results/added_feature_detection/pure_transformed \
  --pure-transformed
