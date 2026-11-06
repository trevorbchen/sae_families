#!/bin/bash
#SBATCH --job-name=render-write-causal-grid
#SBATCH --output=logs/render-write-causal-grid-%j.out
#SBATCH --error=logs/render-write-causal-grid-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs

source .venv/bin/activate

python render_write_class_causal_ablation_grid.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --feature-csv runs/write_class_direction_attribution_16x_k32/feature_class_specificity_zscore_filled_filtered.csv \
  --data-dir data \
  --out-dir runs/write_class_causal_ablation_grids \
  --class-id 5 \
  --t 500 \
  --set-size 100 \
  --num-images 100 \
  --nrow 10 \
  --baseline-per-class 4 \
  --precision fp32
