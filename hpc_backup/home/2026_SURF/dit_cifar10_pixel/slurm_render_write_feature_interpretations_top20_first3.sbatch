#!/bin/bash
#SBATCH --job-name=render-write-feats
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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

python render_write_feature_interpretations.py \
  --classes 0,1,2 \
  --streams attn,mlp \
  --top-n 20 \
  --images-per-feature 4 \
  --metrics-root runs \
  --out-root runs/write_feature_interpretation_grids_top20_first3 \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --data-dir data \
  --num-images 12 \
  --precision bf16 \
  --seed 2026
