#!/bin/bash
#SBATCH --job-name=routed-all4-2034
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/logs/all-four-seed2034-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/logs/all-four-seed2034-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2"
OUT="$EXP/runs/mse_plus_all_four_seed2034"

cd "$ROOT"
mkdir -p "$EXP/logs" "$OUT"

exec "$PY" train_independent_routed_union_sae.py \
  --objectives laplacian,fourier_low,fourier_mid,fourier_high \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$OUT" \
  --mse-dict-size 8192 --mse-k 32 \
  --specialist-dict-size 8192 --specialist-k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed 2034
