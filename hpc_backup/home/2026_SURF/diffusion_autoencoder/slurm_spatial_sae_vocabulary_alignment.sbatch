#!/bin/bash
#SBATCH --job-name=spatial-vocab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/spatial-vocab-%A_%a.out
#SBATCH --error=logs/spatial-vocab-%A_%a.err
set -euo pipefail

ARCHITECTURES=(flat conv)
ARCHITECTURE="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
OUT="results/spatial_sae_vocabulary_alignment/$ARCHITECTURE"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" analyze_spatial_sae_vocabulary_alignment.py \
  --root "$ROOT" --latent-cache "$LATENTS" \
  --architecture "$ARCHITECTURE" --out-dir "$OUT" \
  --top-features 100 --min-active 10 --batch-size 64
