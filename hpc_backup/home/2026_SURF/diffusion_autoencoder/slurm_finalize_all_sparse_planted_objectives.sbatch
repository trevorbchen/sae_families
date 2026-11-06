#!/bin/bash
#SBATCH --job-name=plant-all-table
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/plant-all-table-%j.out
#SBATCH --error=logs/plant-all-table-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"

"$PY" summarize_sparse_planted_suite.py \
  --root results/sparse_planted_residual_suite
exec "$PY" build_added_feature_detection_tables.py
