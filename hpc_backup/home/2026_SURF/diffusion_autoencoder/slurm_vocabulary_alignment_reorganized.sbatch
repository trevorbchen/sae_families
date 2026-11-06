#!/bin/bash
#SBATCH --job-name=vocab-both
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=logs/vocab-both-%A_%a.out
#SBATCH --error=logs/vocab-both-%A_%a.err
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
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
OUT="results/zsem_vocabulary_alignment_all_corrected/$FORMULATION/$KIND"
cd "$ROOT"
mkdir -p logs "$OUT"

WEIGHTED=()
[[ "$FORMULATION" == proper_loss ]] && WEIGHTED=(--weighted)

if [[ "$KIND" == 1d ]]; then
  exec "$PY" analyze_zsem_vocabulary_alignment.py \
    --root "$ROOT" --data "$DATA" --out-dir "$OUT" "${WEIGHTED[@]}" \
    --batch-size 256 --min-active 10 --split-seed 2030
fi

ARCHITECTURE=flat
[[ "$KIND" == conv_spatial ]] && ARCHITECTURE=conv
TOP_FEATURES=100
if [[ "$FORMULATION" == proper_loss && "$KIND" == spatial ]]; then
  TOP_FEATURES=64
fi
exec "$PY" analyze_spatial_sae_vocabulary_alignment.py \
  --root "$ROOT" --latent-cache "$LATENTS" --architecture "$ARCHITECTURE" \
  --out-dir "$OUT" "${WEIGHTED[@]}" --top-features "$TOP_FEATURES" \
  --min-active 10 --batch-size 64
