#!/bin/bash
#SBATCH --job-name=m512-vprobe-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/matched512-vprobe-data-%j.out
#SBATCH --error=logs/matched512-vprobe-data-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
OUT="$BASE/linear_probe_vector16x_lambda_sweep/celeba_vector_32768.pt"

cd "$ROOT"
mkdir -p logs "$(dirname "$OUT")" /resnick/scratch/tbchen/huggingface
export HF_HOME=/resnick/scratch/tbchen/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_XET=1

if [[ -s "$OUT" && -s "${OUT%.pt}.json" ]]; then
  echo "reusing $OUT"
  exit 0
fi

exec "$PY" prepare_celeba_spatial_probe_cache.py \
  --autoencoder "$BASE/autoencoders/vector/last.pt" \
  --autoencoder-type vector --output "$OUT" \
  --max-images 32768 --batch-size 128 --seed 260818
