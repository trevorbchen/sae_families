#!/bin/bash
#SBATCH --job-name=sae-probe-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sae-probe-data-%j.out
#SBATCH --error=logs/sae-probe-data-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/sae_linear_probes/celeba_32768.pt"

cd "$ROOT"
mkdir -p logs "$(dirname "$OUT")" /resnick/scratch/tbchen/huggingface
export HF_HOME=/resnick/scratch/tbchen/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_XET=1

exec "$PY" prepare_celeba_spatial_probe_cache.py \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output "$OUT" \
  --max-images 32768 \
  --batch-size 128 \
  --seed 260818
