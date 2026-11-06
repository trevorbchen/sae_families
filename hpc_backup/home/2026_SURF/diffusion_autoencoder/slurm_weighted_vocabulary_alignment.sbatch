#!/bin/bash
#SBATCH --job-name=weighted-vocab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/weighted-vocab-%A_%a.out
#SBATCH --error=logs/weighted-vocab-%A_%a.err
set -euo pipefail

KINDS=(1d spatial conv_spatial)
KIND="${KINDS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
OUT="results/zsem_vocabulary_alignment_all_corrected/$KIND"
cd "$ROOT"
mkdir -p logs "$OUT"

if [[ "$KIND" == 1d ]]; then
  exec "$PY" analyze_zsem_vocabulary_alignment.py \
    --root "$ROOT" --data "$DATA" --out-dir "$OUT" --weighted \
    --batch-size 256 --min-active 10 --split-seed 2030
fi

ARCHITECTURE=flat
[[ "$KIND" == conv_spatial ]] && ARCHITECTURE=conv
TOP_FEATURES=100
[[ "$KIND" == spatial ]] && TOP_FEATURES=64
exec "$PY" analyze_spatial_sae_vocabulary_alignment.py \
  --root "$ROOT" --latent-cache "$LATENTS" --architecture "$ARCHITECTURE" \
  --out-dir "$OUT" --weighted --top-features "$TOP_FEATURES" \
  --min-active 10 --batch-size 64
