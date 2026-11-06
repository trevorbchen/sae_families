#!/bin/bash
#SBATCH --job-name=zsem-vocab-all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/zsem-vocab-all-%j.out
#SBATCH --error=logs/zsem-vocab-all-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
cd "$ROOT"
mkdir -p logs results/zsem_vocabulary_alignment_all_corrected

exec "$PY" analyze_zsem_vocabulary_alignment.py \
  --root "$ROOT" --data "$DATA" \
  --out-dir results/zsem_vocabulary_alignment_all_corrected \
  --batch-size 256 --min-active 10 --split-seed 2030
