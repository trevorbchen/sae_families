#!/bin/bash
#SBATCH --job-name=vocab-both-plots
#SBATCH --partition=expansion
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/vocab-both-plots-%A_%a.out
#SBATCH --error=logs/vocab-both-plots-%A_%a.err
set -euo pipefail

FORMULATIONS=(
  transformed_loss transformed_loss transformed_loss
  proper_loss proper_loss proper_loss
)
KINDS=(1d spatial conv_spatial 1d spatial conv_spatial)
INDEX="$SLURM_ARRAY_TASK_ID"
FORMULATION="${FORMULATIONS[$INDEX]}"
KIND="${KINDS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="results/zsem_vocabulary_alignment_all_corrected/$FORMULATION/$KIND"
cd "$ROOT"
exec "$PY" plot_zsem_pairwise_category_matrices.py \
  --input "$OUT/pairwise_alignment.csv" --output-dir "$OUT"
