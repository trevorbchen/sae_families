#!/bin/bash
#SBATCH --job-name=diffae-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diffae-smoke-%j.out
#SBATCH --error=logs/diffae-smoke-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/smoke

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
REPO="/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae"
CKPT="$REPO/checkpoints/ffhq128_autoenc_latent/last.ckpt"

"$PY" smoke_test_diffae.py \
  --repo "$REPO" \
  --checkpoint "$CKPT" \
  --image "$REPO/imgs/sandy.JPG" \
  --out-dir results/smoke \
  --steps 20
