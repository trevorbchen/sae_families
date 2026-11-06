#!/bin/bash
#SBATCH --job-name=resid-base-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=logs/resid-base-smoke-%j.out
#SBATCH --error=logs/resid-base-smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1_smoke/base"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_spatial_ae_sae.py \
  --mode regular --regular-dict-size 8192 --regular-k 32 \
  --checkpoint results/spatial_conv_ae/last.pt \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --out-dir "$OUT" --steps 1 --batch-size 8 \
  --lr 3e-4 --warmup 1 --grad-clip 1.0 --parameter-gradient-cap none \
  --aux-coeff 0.1 --validate-every 1 --checkpoint-every 1 --state-every 1 \
  --trace-flush-every 1 --skip-file-hashes --seed 2026 --split-seed 2026
