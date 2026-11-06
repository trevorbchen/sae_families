#!/bin/bash
#SBATCH --job-name=spatial-cache-32k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/logs/spatial-cache-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/logs/spatial-cache-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
AE="$ROOT/results/spatial_conv_ae/last.pt"
OUT="$EXP/data/ffhq_32768_latents.pt"
mkdir -p "$EXP/logs" "$EXP/data"
cd "$ROOT"

exec "$PY" cache_spatial_autoencoder_latents.py \
  --data "$DATA" --autoencoder "$AE" --out "$OUT" \
  --batch-size 128 --split-seed 2026
