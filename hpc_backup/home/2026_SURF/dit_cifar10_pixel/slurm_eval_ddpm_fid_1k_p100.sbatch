#!/bin/bash
#SBATCH --job-name=ddpm-fid-1k
#SBATCH --output=logs/ddpm-fid-1k-%j.out
#SBATCH --error=logs/ddpm-fid-1k-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/ddpm_fid_1k

source .venv/bin/activate

export TORCH_HOME="/central/scratch/tbchen/torch_cache"
mkdir -p "$TORCH_HOME"

python eval_ddpm_fid.py \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --data-dir data \
  --out-dir runs/ddpm_fid_1k \
  --num-samples 1000 \
  --num-real 10000 \
  --sample-batch-size 128 \
  --feature-batch-size 128 \
  --timesteps 1000 \
  --seed 2026 \
  --save-samples \
  --learn-sigma
