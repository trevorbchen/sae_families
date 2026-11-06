#!/bin/bash
#SBATCH --job-name=recache-spatial-conv-ae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/recache-spatial-conv-ae-%j.out
#SBATCH --error=logs/recache-spatial-conv-ae-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
AE="$ROOT/results/spatial_conv_ae/last.pt"
OUT="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
mkdir -p logs "$(dirname "$OUT")"
cd "$ROOT"

# The original scratch file at $OUT was purged by an automatic scratch-cleanup
# policy on 2026-09-06 (dir mtime; both available .snapshot copies were
# already empty for this path). Regenerating from the still-present raw
# images -- the shortlist GUI only needs *a* valid example pool to browse,
# not the exact original 16384-image subset, since annotations key off
# objective/feature/rank, not latent-cache row index.
exec "$PY" cache_spatial_autoencoder_latents.py \
  --data "$DATA" --autoencoder "$AE" --out "$OUT" \
  --batch-size 128 --split-seed 2026
