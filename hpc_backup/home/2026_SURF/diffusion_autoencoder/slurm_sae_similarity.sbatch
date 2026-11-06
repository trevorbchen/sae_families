#!/bin/bash
#SBATCH --job-name=sae-similarity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sae-similarity-%j.out
#SBATCH --error=logs/sae-similarity-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/sae_feature_similarity
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" analyze_sae_feature_similarity.py \
  --root results/zsem_sae \
  --out results/sae_feature_similarity --topk 5
