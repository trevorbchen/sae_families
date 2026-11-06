#!/bin/bash
#SBATCH --job-name=constr-joint-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/constr-joint-all-%j.out
#SBATCH --error=logs/constr-joint-all-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/constrained_joint_changed_objectives_v1/all_four"
INITIAL="/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_constrained_grouped_spatial_sae.py \
  --objectives laplacian fourier_low fourier_mid fourier_high \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt --initial-sae "$INITIAL" \
  --out-dir "$OUT" --steps 20000 --mse-adapt-steps 5000 \
  --batch-size 32 --lr 1e-4 --lr-warmup 500 --transform-warmup 2000 \
  --transform-gradient-ratio 0.1 --mse-tolerance 0.01 \
  --aux-coeff 0.1 --grad-clip 1.0 --trust-every 250 \
  --geometry-every 1000 --seed 2026
