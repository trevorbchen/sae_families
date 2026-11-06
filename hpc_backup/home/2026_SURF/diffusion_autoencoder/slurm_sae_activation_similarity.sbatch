#!/bin/bash
#SBATCH --job-name=sae-act-sim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sae-act-sim-%j.out
#SBATCH --error=logs/sae-act-sim-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/sae_feature_similarity
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" analyze_sae_activation_similarity.py \
  --repo official_diffae \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --root results/zsem_sae --out results/sae_feature_similarity --batch-size 512
