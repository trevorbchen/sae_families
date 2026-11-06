#!/bin/bash
#SBATCH --job-name=processing-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/processing-data-%A_%a.out
#SBATCH --error=logs/processing-data-%A_%a.err
set -euo pipefail

FEATURES=(jpeg random_phase)
TAGS=(jpeg_q60 random_phase_f17)
INDEX="$SLURM_ARRAY_TASK_ID"
FEATURE="${FEATURES[$INDEX]}"
TAG="${TAGS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/processing_feature_spatial/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" prepare_processing_feature_spatial_latents.py \
  --feature "$FEATURE" \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --clean-latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output "$OUT/cache.pt" --preview "$OUT/preview.png" \
  --prevalence 0.10 --jpeg-quality 60 \
  --frequency-x 17 --frequency-y 17 --amplitude 3
