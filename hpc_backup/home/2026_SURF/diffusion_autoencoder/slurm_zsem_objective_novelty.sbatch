#!/bin/bash
#SBATCH --job-name=zsem-objective-novelty
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/zsem-objective-novelty-%j.out
#SBATCH --error=logs/zsem-objective-novelty-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
cd "$ROOT"
mkdir -p logs results/zsem_objective_novelty

exec "$PY" analyze_zsem_objective_novelty.py \
  --root "$ROOT" --data "$DATA" \
  --out-dir results/zsem_objective_novelty \
  --batch-size 256 --match-block-size 512 --activation-block-size 256 \
  --candidate-count 8 --min-active 10 \
  --cosine-threshold 0.8 --correlation-threshold 0.5 \
  --composition-limit 2000 --seed 2030
