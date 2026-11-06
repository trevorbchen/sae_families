#!/bin/bash
#SBATCH --job-name=bsf-stable-rank
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bsf-stable-rank-%j.out
#SBATCH --error=logs/bsf-stable-rank-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_bsf/32x_block8_top4
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" analyze_bsf_stable_rank.py \
  --bsf results/zsem_bsf/32x_block8_top4/last.pt \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --out results/zsem_bsf/32x_block8_top4/stable_rank.json \
  --batch-size 2048
