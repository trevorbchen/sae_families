#!/bin/bash
#SBATCH --job-name=spatial-constr-vocab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/spatial-constr-vocab-%A_%a.out
#SBATCH --error=logs/spatial-constr-vocab-%A_%a.err
set -euo pipefail

ARCHITECTURES=(flat conv)
FORMULATIONS=(transformed_loss proper_loss)
INDEX="$SLURM_ARRAY_TASK_ID"
ARCHITECTURE="${ARCHITECTURES[$((INDEX / 2))]}"
FORMULATION="${FORMULATIONS[$((INDEX % 2))]}"

if [[ "$ARCHITECTURE" == "flat" ]]; then
  OUTPUT_ARCHITECTURE="spatial"
else
  OUTPUT_ARCHITECTURE="conv_spatial"
fi

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768"
LATENTS="$EXP/data/ffhq_32768_latents.pt"
CHECKPOINTS="$EXP/results/$ARCHITECTURE/$FORMULATION"
OUT="$EXP/results/family_sae/transformed_objective_constrained_sae/$FORMULATION/$OUTPUT_ARCHITECTURE"

mkdir -p "$ROOT/logs" "$OUT/pairwise_categories"
cd "$ROOT"

EXTRA=()
if [[ "$FORMULATION" == "proper_loss" ]]; then
  EXTRA+=(--weighted --include-fourier-bands)
fi

"$PY" analyze_spatial_sae_vocabulary_alignment.py \
  --root "$EXP" --checkpoint-root "$CHECKPOINTS" \
  --latent-cache "$LATENTS" --architecture "$ARCHITECTURE" \
  --out-dir "$OUT" --top-features 20 --min-active 10 --batch-size 64 \
  "${EXTRA[@]}"

"$PY" plot_zsem_vocabulary_grid.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output "$OUT/dictionary_alignment_grid_all.png"

exec "$PY" plot_zsem_pairwise_category_matrices.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/pairwise_categories"
