#!/bin/bash
#SBATCH --job-name=spatial-vocab-post
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/spatial-vocab-post-%A_%a.out
#SBATCH --error=logs/spatial-vocab-post-%A_%a.err
set -euo pipefail

ARCHITECTURES=(flat conv)
LABELS=(spatial conv_spatial)
INDEX="$SLURM_ARRAY_TASK_ID"
ARCHITECTURE="${ARCHITECTURES[$INDEX]}"
LABEL="${LABELS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SOURCE="$ROOT/results/spatial_sae_vocabulary_alignment/$ARCHITECTURE"
OUT="$ROOT/results/zsem_vocabulary_alignment_all_corrected/$LABEL"
cd "$ROOT"
mkdir -p logs "$OUT" "$OUT/significance" "$OUT/pairwise_categories"

cp "$SOURCE/pairwise_alignment.csv" "$OUT/"
cp "$SOURCE/vocabulary_geometry.csv" "$OUT/"
cp "$SOURCE/subspace_alignment.csv" "$OUT/"
cp "$SOURCE/threshold_curves.csv" "$OUT/"
cp "$SOURCE/feature_recurrence.csv" "$OUT/"
cp "$SOURCE/summary.json" "$OUT/"

"$PY" plot_zsem_vocabulary_grid.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output "$OUT/dictionary_alignment_grid_all.png"

"$PY" analyze_zsem_alignment_significance.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/significance"

exec "$PY" plot_zsem_pairwise_category_matrices.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/pairwise_categories"
