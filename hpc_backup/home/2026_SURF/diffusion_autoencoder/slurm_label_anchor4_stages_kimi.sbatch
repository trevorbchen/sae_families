#!/bin/bash
#SBATCH --job-name=kimi-anchor4-stages
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/kimi-anchor4-stages-%j.out
#SBATCH --error=logs/kimi-anchor4-stages-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"

cd "$ROOT"
mkdir -p logs /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review

exec "$PY" label_residual_gui_kimi.py \
  --endpoint "${GUI_ENDPOINT:?set GUI_ENDPOINT to the active review GUI}" \
  --model "$MODEL" \
  --out /resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/kimi_anchor4_first100_each_stage.jsonl \
  --end 100 --examples 5 \
  --targets \
    anchor_four_canonical:laplacian \
    anchor_four_canonical:fourier_low \
    anchor_four_canonical:fourier_mid \
    anchor_four_canonical:fourier_high
