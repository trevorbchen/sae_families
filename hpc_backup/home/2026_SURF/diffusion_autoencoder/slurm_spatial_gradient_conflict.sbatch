#!/bin/bash
#SBATCH --job-name=spatial-grad-cos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/spatial-grad-cos-%j.out
#SBATCH --error=logs/spatial-grad-cos-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/spatial_objective_gradient_conflict_v1"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" analyze_spatial_objective_gradient_conflict.py \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --regular-sae /resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/last.pt \
  --output "$OUT" --examples 2048 --batch-size 32 --seed 2026
