#!/bin/bash
#SBATCH --job-name=lap-dict-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/lap-dict-compare-%j.out
#SBATCH --error=logs/lap-dict-compare-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/laplacian_dictionary_comparison

exec "$PY" compare_laplacian_dictionary.py \
  --root "$ROOT" \
  --out-dir results/laplacian_dictionary_comparison \
  --block-size 4096
