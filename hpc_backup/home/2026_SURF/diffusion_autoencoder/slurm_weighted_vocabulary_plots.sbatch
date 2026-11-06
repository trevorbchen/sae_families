#!/bin/bash
#SBATCH --job-name=weighted-vocab-plots
#SBATCH --partition=expansion
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/weighted-vocab-plots-%A_%a.out
#SBATCH --error=logs/weighted-vocab-plots-%A_%a.err
set -euo pipefail

KINDS=(1d spatial conv_spatial)
KIND="${KINDS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="results/zsem_vocabulary_alignment_all_corrected/$KIND"
cd "$ROOT"
exec "$PY" plot_zsem_pairwise_category_matrices.py \
  --input "$OUT/pairwise_alignment.csv" --output-dir "$OUT"
