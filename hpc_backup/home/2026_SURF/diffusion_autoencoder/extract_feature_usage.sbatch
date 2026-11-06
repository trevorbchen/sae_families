#!/bin/bash
#SBATCH --job-name=feat-usage-extract
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-usage-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-usage-%j.err
set -euo pipefail

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRIPT="/home/tbchen/2026 SURF/diffusion_autoencoder/extract_feature_usage_for_hard_images.py"
OUT="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/feature_usage_vector_16x_k32_lambda10.pt"

"$PY" "$SCRIPT" --out "$OUT"
