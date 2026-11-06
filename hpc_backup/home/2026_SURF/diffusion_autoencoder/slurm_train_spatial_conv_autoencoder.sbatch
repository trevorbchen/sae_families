#!/bin/bash
#SBATCH --job-name=spatial-conv-ae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/spatial-conv-ae-%j.out
#SBATCH --error=logs/spatial-conv-ae-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
cd "$ROOT"
mkdir -p logs results/spatial_conv_ae

exec "$PY" train_spatial_conv_autoencoder.py \
  --data "$DATA" \
  --out-dir results/spatial_conv_ae \
  --latent-cache "$LATENTS" \
  --latent-channels 64 --base-channels 64 \
  --steps 50000 --batch-size 64 --lr 2e-4 --warmup 1000 \
  --weight-decay 1e-4 --eval-every 2000 --seed 2030
