#!/bin/bash
#SBATCH --job-name=write-class-attr
#SBATCH --output=logs/write-class-attr-%j.out
#SBATCH --error=logs/write-class-attr-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs

source .venv/bin/activate

python write_class_direction_attribution.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --data-dir data \
  --out-root runs/write_class_direction_attribution_16x_k32 \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --blocks 0,1,2,3,4,5,6,7,8,9,10,11 \
  --streams attn,mlp \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --num-images 8 \
  --baseline-per-class 2 \
  --top-features 200 \
  --precision bf16
