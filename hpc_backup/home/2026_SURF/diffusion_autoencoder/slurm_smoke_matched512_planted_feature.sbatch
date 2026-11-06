#!/bin/bash
#SBATCH --job-name=m512-plant-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/m512-plant-smoke-%A_%a.out
#SBATCH --error=logs/m512-plant-smoke-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
OUT="$BASE/planted_features_smoke"

TYPES=(vector spatial)
TYPE="${TYPES[$SLURM_ARRAY_TASK_ID]}"

cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" prepare_sparse_planted_residual_latents.py \
  --feature brightness \
  --data "$DATA" \
  --clean-latent-cache "$BASE/latents/${TYPE}_ffhq_32768.pt" \
  --autoencoder "$BASE/autoencoders/${TYPE}/last.pt" \
  --autoencoder-type "$TYPE" \
  --amplitude 24 --seed 260810 \
  --output "$OUT/${TYPE}_brightness_a24.pt" \
  --preview "$OUT/${TYPE}_brightness_a24_preview.png"
