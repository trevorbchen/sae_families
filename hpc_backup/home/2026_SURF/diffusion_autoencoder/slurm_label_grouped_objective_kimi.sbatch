#!/bin/bash
#SBATCH --job-name=kimi-grouped-obj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kimi-grouped-obj-%j.out
#SBATCH --error=logs/kimi-grouped-obj-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"
ENDPOINT="${GUI_ENDPOINT:-http://hpc-33-13:8813}"

cd "$ROOT"
mkdir -p logs /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review

exec "$PY" label_residual_gui_kimi.py \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --out /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/kimi_grouped_objective_first200_each.jsonl \
  --end 200 --examples 5 \
  --targets \
    grouped_objective_laplacian \
    grouped_objective_fourier_low \
    grouped_objective_fourier_mid \
    grouped_objective_fourier_high
