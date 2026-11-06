#!/bin/bash
#SBATCH --job-name=m512-alignment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-4%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-alignment-%A_%a.out
#SBATCH --error=logs/matched512-alignment-%A_%a.err
set -euo pipefail

: "${ARM:?ARM must be exported by the submission script}"
RUN_MODE="${RUN_MODE:-full}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$BASE/experiments/$ARM"

case "$ARM" in
  spatial_4x_k8|spatial_16x_k32) SOURCE=spatial; ARCHITECTURE=flat ;;
  vector_4x_k8|vector_16x_k32) SOURCE=vector; ARCHITECTURE=vector ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
LATENTS="$BASE/latents/${SOURCE}_ffhq_32768.pt"
TAGS=(0p1 0p3 1p0 3p0 10p0)
if [[ "$RUN_MODE" == priority_10 ]]; then
  TAG=10p0
elif [[ "$RUN_MODE" == full ]]; then
  TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
else
  echo "unknown RUN_MODE=$RUN_MODE" >&2; exit 2
fi
CHECKPOINT_ROOT="$EXP/checkpoints/lambda_$TAG"
OUT="$EXP/alignment/lambda_$TAG"

cd "$ROOT"
mkdir -p logs "$OUT/pairwise_categories" "$CHECKPOINT_ROOT/latent_mse"
for SEED in 2030 2031 2032; do
  ln -sfn "$EXP/checkpoints/mse_control/latent_mse/seed_$SEED" \
    "$CHECKPOINT_ROOT/latent_mse/seed_$SEED"
done

"$PY" analyze_spatial_sae_vocabulary_alignment.py \
  --root "$ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --latent-cache "$LATENTS" --architecture "$ARCHITECTURE" --weighted \
  --out-dir "$OUT" --top-features 64 --min-active 10 --batch-size 64

exec "$PY" plot_cross_seed_similarity_heatmaps.py \
  --input "$OUT/pairwise_alignment.csv" \
  --output-dir "$OUT/pairwise_categories"
