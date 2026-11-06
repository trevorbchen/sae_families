#!/bin/bash
#SBATCH --job-name=diffae-linear-rank
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diffae-linear-rank-%j.out
#SBATCH --error=logs/diffae-linear-rank-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
"$PY" diagnose_linear_rank.py \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --out results/linear_rank_pca.json \
  --ranks 8 16 32 64 128
