#!/bin/bash
#SBATCH --job-name=seq-resid-ab-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/seq-resid-ab-smoke-%j.out
#SBATCH --error=logs/seq-resid-ab-smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_v1"
OUT="$RUN_ROOT/anchor_plus_four_ablation_smoke_v3"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" evaluate_sequential_residual_ablations.py \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --base-sae /resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/best_valid.pt \
  --stack-root "$RUN_ROOT/anchor_plus_four" \
  --sequential-mse-control "$RUN_ROOT/anchor_plus_one/mse_control/best_feasible.pt" \
  --direct-mse-control /resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1/original_16k_k64_seed2026/best_valid.pt \
  --pure-stack-root /resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_pure_v1/anchor_plus_four \
  --mse-stack-root "$RUN_ROOT/anchor_plus_four_mse_control" \
  --out-dir "$OUT" \
  --batch-size 4 \
  --mode-chunk-size 1 \
  --max-examples 8
