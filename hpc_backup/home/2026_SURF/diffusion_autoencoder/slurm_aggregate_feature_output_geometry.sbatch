#!/bin/bash
#SBATCH --job-name=sae-geometry-agg
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/sae-geometry-agg-%j.out
#SBATCH --error=logs/sae-geometry-agg-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"

exec "$PY" aggregate_feature_output_geometry.py \
  --input-root results/family_sae/feature_output_geometry \
  --out-dir results/family_sae/feature_output_geometry/aggregate
