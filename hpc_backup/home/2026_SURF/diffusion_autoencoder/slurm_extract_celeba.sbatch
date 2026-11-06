#!/bin/bash
#SBATCH --job-name=extract-celeba-zsem
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract-celeba-%j.out
#SBATCH --error=logs/extract-celeba-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_codes
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
export SSL_CERT_FILE="$("$PY" -c 'import certifi; print(certifi.where())')"
exec "$PY" extract_celeba_zsem.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --out results/zsem_codes/celeba_train.pt \
  --split train --batch-size 256
