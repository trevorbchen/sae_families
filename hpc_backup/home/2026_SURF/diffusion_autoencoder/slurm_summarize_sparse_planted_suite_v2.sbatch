#!/bin/bash
#SBATCH --job-name=sparse-plant2-summary
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/sparse-plant2-summary-%j.out
#SBATCH --error=logs/sparse-plant2-summary-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
exec "$PY" summarize_sparse_planted_suite.py \
  --root results/sparse_planted_residual_suite
