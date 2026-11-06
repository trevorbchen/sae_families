#!/bin/bash
#SBATCH --job-name=grouped-proper-4xk8-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/grouped_proper_all_four_routed_4x_k8_v1/logs/smoke-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/grouped_proper_all_four_routed_4x_k8_v1/logs/smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/grouped_proper_all_four_routed_4x_k8_v1"
DATA="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/data/ffhq_32768_latents.pt"
AE="$ROOT/results/spatial_conv_ae/last.pt"
OUT="$EXP/smoke_seed2026"
mkdir -p "$EXP/logs" "$OUT"
cd "$ROOT"

exec "$PY" train_grouped_spatial_ae_sae.py \
  --mode grouped --data "$DATA" --checkpoint "$AE" --out-dir "$OUT" \
  --steps 500 --batch-size 32 --lr 3e-4 --warmup 100 \
  --loss-normalization power --lambda-mode fixed --transform-weight 1.0 \
  --parameter-gradient-cap none --grad-clip 1.0 \
  --grouped-mse-dict-size 8192 --grouped-mse-k 4 \
  --grouped-specialist-dict-size 2048 --grouped-specialist-k 1 \
  --calibration-examples 2048 --aux-coeff 0.1 \
  --reinit-every 1000 --dead-after 3000 --max-rescue-per-group 64 \
  --validate-every 100 --checkpoint-every 500 --state-every 250 \
  --trace-flush-every 100 --skip-file-hashes --seed 2026 --split-seed 2026
