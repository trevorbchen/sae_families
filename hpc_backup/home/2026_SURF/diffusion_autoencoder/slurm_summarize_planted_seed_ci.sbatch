#!/bin/bash
#SBATCH --job-name=plant-seed-ci
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --nice=10000
#SBATCH --output=logs/plant-seed-ci-%j.out
#SBATCH --error=logs/plant-seed-ci-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/family_sae/three_seed_confidence_intervals/planted_features
exec "$PY" summarize_seed_confidence_intervals.py \
  --planted-root results/sparse_planted_residual_suite_seed_ci \
  --output-dir results/family_sae/three_seed_confidence_intervals/planted_features
