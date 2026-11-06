#!/bin/bash
#SBATCH --job-name=regular-sae-control
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/regular-sae-control-%j.out
#SBATCH --error=logs/regular-sae-control-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="results/grouped_objective_sae/regular_32x_k64_seed2026"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_objective_sae.py \
  --mode regular --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --out-dir "$OUT" --steps 20000 --batch-size 32 \
  --lr 3e-4 --warmup 500 --grad-clip 1.0 --aux-coeff 0.1 \
  --reinit-every 1000 --dead-after 3000 --max-rescue-per-group 64 \
  --seed 2026 --split-seed 2026

