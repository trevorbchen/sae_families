#!/bin/bash
#SBATCH --job-name=m512-v4-hi-geo-extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-15%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-v16-hi-geo-extra-%A_%a.out
#SBATCH --error=logs/matched512-v16-hi-geo-extra-%A_%a.err
set -euo pipefail

ARM=vector_4x_k8
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$BASE/experiments/$ARM"
AE="$BASE/autoencoders/vector/last.pt"
LATENTS="$BASE/latents/vector_ffhq_32768.pt"

SEEDS=(2031 2032)
HIGH_LAMBDAS=(30.0 100.0)
HIGH_LAMBDA_TAGS=(30p0 100p0)
OBJECTIVES=(weighted_laplacian weighted_fourier_low weighted_fourier_mid weighted_fourier_high)
DIRECTORIES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
if (( INDEX < 0 || INDEX > 15 )); then
  echo "requires array indices 0-15" >&2
  exit 2
fi
SEED_INDEX=$((INDEX % 2))
REST=$((INDEX / 2))
LAMBDA_INDEX=$((REST / 4))
OBJECTIVE_INDEX=$((REST % 4))

SEED="${SEEDS[$SEED_INDEX]}"
OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"
WEIGHT="${HIGH_LAMBDAS[$LAMBDA_INDEX]}"
TAG="${HIGH_LAMBDA_TAGS[$LAMBDA_INDEX]}"
CHECKPOINT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED/last.pt"
OUT="$EXP/geometry/lambda_$TAG/$DIRECTORY/seed_$SEED"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'arm=%s task=%s objective=%s seed=%s lambda=%s checkpoint=%s out=%s\n' \
    "$ARM" "$INDEX" "$OBJECTIVE" "$SEED" "$WEIGHT" "$CHECKPOINT" "$OUT"
  exit 0
fi

if [[ -s "$OUT/model_metrics.json" && -s "$OUT/feature_means.csv" ]]; then
  echo "reusing $OUT"
  exit 0
fi

cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" evaluate_lambda_sweep_model.py \
  --checkpoint "$CHECKPOINT" --objective "$OBJECTIVE" --seed "$SEED" \
  --lambda-value "$WEIGHT" --autoencoder "$AE" --autoencoder-type vector \
  --latent-cache "$LATENTS" --out-dir "$OUT" \
  --top-features 100 --examples-per-feature 4 --min-examples 4 \
  --multipliers -2 2
