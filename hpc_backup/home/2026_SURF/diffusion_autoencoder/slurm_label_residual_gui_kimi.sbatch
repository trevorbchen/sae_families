#!/bin/bash
#SBATCH --job-name=kimi-resid-notes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kimi-resid-notes-%j.out
#SBATCH --error=logs/kimi-resid-notes-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"

cd "$ROOT"
mkdir -p logs /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review

exec "$PY" label_residual_gui_kimi.py \
  --endpoint http://hpc-sm-03-06:8812 \
  --model "$MODEL" \
  --out /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/kimi_first200_each.jsonl \
  --end 200 --examples 5
