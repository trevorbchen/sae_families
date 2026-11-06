#!/bin/bash
#SBATCH --job-name=write-class-causal-p100
#SBATCH --output=logs/write-class-causal-p100-%j.out
#SBATCH --error=logs/write-class-causal-p100-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs

source .venv/bin/activate

python write_class_causal_ablation.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --feature-csv runs/write_class_direction_attribution_16x_k32/feature_class_specificity_zscore_filled_filtered.csv \
  --data-dir data \
  --out-root runs/write_class_causal_ablation_16x_k32 \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --set-sizes 1,2,5,10,20,50,100 \
  --num-images 12 \
  --baseline-per-class 4 \
  --precision fp32
