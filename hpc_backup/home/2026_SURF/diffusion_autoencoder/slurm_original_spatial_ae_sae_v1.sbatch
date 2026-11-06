#!/bin/bash
#SBATCH --job-name=spatial-original-v1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/spatial-original-v1-%j.out
#SBATCH --error=logs/spatial-original-v1-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_spatial_ae_sae.py \
  --mode regular \
  --checkpoint results/spatial_conv_ae/last.pt \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --out-dir "$OUT" --steps 20000 --batch-size 32 \
  --lr 3e-4 --warmup 500 --grad-clip 1.0 --transform-weight 1.0 \
  --calibration-examples 2048 --aux-coeff 0.1 \
  --reinit-every 1000 --dead-after 3000 --max-rescue-per-group 64 \
  --validate-every 250 --checkpoint-every 1000 --state-every 500 \
  --trace-flush-every 100 --seed 2026 --split-seed 2026
