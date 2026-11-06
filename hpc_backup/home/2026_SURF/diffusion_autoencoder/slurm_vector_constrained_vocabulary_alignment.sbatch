#!/bin/bash
#SBATCH --job-name=vector-constrained-vocab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768/logs/vector-constrained-vocab-%A_%a.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768/logs/vector-constrained-vocab-%A_%a.err
set -euo pipefail

FORMULATIONS=(transformed_loss proper_loss)
INDEX="$SLURM_ARRAY_TASK_ID"
FORMULATION="${FORMULATIONS[$INDEX]}"
CODE="/resnick/scratch/tbchen/surf_2026/vector_ae_code"
EXP="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768"
HOME_ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
LATENTS="$EXP/data/ffhq_32768_latents.pt"
CHECKPOINTS="$EXP/results/$FORMULATION"
OUT="$EXP/results/family_sae/transformed_objective_constrained_sae/$FORMULATION/1d"
mkdir -p "$EXP/logs" "$OUT/pairwise_categories"
cd "$CODE"
export PYTHONPATH="$CODE:$HOME_ROOT"

EXTRA=()
if [[ "$FORMULATION" == "proper_loss" ]]; then
  EXTRA+=(--weighted --include-fourier-bands)
fi

"$PY" "$CODE/analyze_spatial_sae_vocabulary_alignment.py" \
  --root "$EXP" --checkpoint-root "$CHECKPOINTS" \
  --latent-cache "$LATENTS" --architecture vector \
  --out-dir "$OUT" --top-features 32 --min-active 10 --batch-size 64 \
  "${EXTRA[@]}"

"$PY" "$HOME_ROOT/plot_zsem_vocabulary_grid.py" \
  --input "$OUT/pairwise_alignment.csv" \
  --output "$OUT/dictionary_alignment_grid_all.png"

exec "$PY" "$HOME_ROOT/plot_zsem_pairwise_category_matrices.py" \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/pairwise_categories"
