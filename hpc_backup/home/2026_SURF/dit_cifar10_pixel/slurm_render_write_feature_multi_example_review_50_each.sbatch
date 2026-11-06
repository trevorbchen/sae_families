#!/bin/bash
#SBATCH --job-name=render-write-multi50
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

python render_write_feature_multi_example_grids.py \
  --selection-csv runs/feature_review_50_each/write_review_selection.csv \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --streams attn,mlp \
  --top-n 999 \
  --examples-per-feature 5 \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --out-root runs/write_feature_multi_example_grids_review_50_each \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --data-dir data \
  --num-images 12 \
  --precision fp32 \
  --seed 2026 \
  --image-size 64
