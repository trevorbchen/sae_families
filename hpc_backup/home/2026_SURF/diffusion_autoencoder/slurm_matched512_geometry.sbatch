#!/bin/bash
#SBATCH --job-name=m512-geometry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-62%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-geometry-%A_%a.out
#SBATCH --error=logs/matched512-geometry-%A_%a.err
set -euo pipefail

: "${ARM:?ARM must be exported by the submission script}"
RUN_MODE="${RUN_MODE:-full}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$BASE/experiments/$ARM"
EXTRA=()

case "$ARM" in
  spatial_4x_k8|spatial_16x_k32) SOURCE=spatial ;;
  vector_4x_k8|vector_16x_k32) SOURCE=vector; EXTRA+=(--autoencoder-type vector) ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
AE="$BASE/autoencoders/$SOURCE/last.pt"
LATENTS="$BASE/latents/${SOURCE}_ffhq_32768.pt"
SEEDS=(2030 2031 2032)
LAMBDAS=(0.1 0.3 1.0 3.0 10.0)
LAMBDA_TAGS=(0p1 0p3 1p0 3p0 10p0)
OBJECTIVES=(weighted_laplacian weighted_fourier_low weighted_fourier_mid weighted_fourier_high)
DIRECTORIES=(laplacian fourier_low fourier_mid fourier_high)
INDEX="$SLURM_ARRAY_TASK_ID"

if [[ "$RUN_MODE" == priority_10 ]]; then
  if (( INDEX < 3 )); then
    SEED="${SEEDS[$INDEX]}"; OBJECTIVE=latent_mse; WEIGHT=0
    CHECKPOINT="$EXP/checkpoints/mse_control/latent_mse/seed_$SEED/last.pt"
    OUT="$EXP/geometry/mse_control/seed_$SEED"
  else
    SPECIALIST_INDEX=$((INDEX - 3))
    OBJECTIVE_INDEX=$((SPECIALIST_INDEX / 3))
    SEED_INDEX=$((SPECIALIST_INDEX % 3))
    SEED="${SEEDS[$SEED_INDEX]}"; OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
    DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"; WEIGHT=10.0
    CHECKPOINT="$EXP/checkpoints/lambda_10p0/$DIRECTORY/seed_$SEED/last.pt"
    OUT="$EXP/geometry/lambda_10p0/$DIRECTORY/seed_$SEED"
  fi
elif [[ "$RUN_MODE" == high_lambda_seed2030 ]]; then
  if (( INDEX < 0 || INDEX > 7 )); then
    echo "high_lambda_seed2030 requires array indices 0-7" >&2
    exit 2
  fi
  HIGH_LAMBDAS=(30.0 100.0)
  HIGH_LAMBDA_TAGS=(30p0 100p0)
  LAMBDA_INDEX=$((INDEX / 4))
  OBJECTIVE_INDEX=$((INDEX % 4))
  SEED=2030
  OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
  DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"
  WEIGHT="${HIGH_LAMBDAS[$LAMBDA_INDEX]}"
  TAG="${HIGH_LAMBDA_TAGS[$LAMBDA_INDEX]}"
  CHECKPOINT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED/last.pt"
  OUT="$EXP/geometry/lambda_$TAG/$DIRECTORY/seed_$SEED"
elif [[ "$RUN_MODE" == full ]]; then
  if (( INDEX < 3 )); then
    SEED="${SEEDS[$INDEX]}"; OBJECTIVE=latent_mse; WEIGHT=0
    CHECKPOINT="$EXP/checkpoints/mse_control/latent_mse/seed_$SEED/last.pt"
    OUT="$EXP/geometry/mse_control/seed_$SEED"
  else
    SPECIALIST_INDEX=$((INDEX - 3))
    LAMBDA_INDEX=$((SPECIALIST_INDEX / 12))
    WITHIN_LAMBDA=$((SPECIALIST_INDEX % 12))
    OBJECTIVE_INDEX=$((WITHIN_LAMBDA / 3))
    SEED_INDEX=$((WITHIN_LAMBDA % 3))
    SEED="${SEEDS[$SEED_INDEX]}"; OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
    DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"; WEIGHT="${LAMBDAS[$LAMBDA_INDEX]}"
    TAG="${LAMBDA_TAGS[$LAMBDA_INDEX]}"
    CHECKPOINT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED/last.pt"
    OUT="$EXP/geometry/lambda_$TAG/$DIRECTORY/seed_$SEED"
  fi
else
  echo "unknown RUN_MODE=$RUN_MODE" >&2; exit 2
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'arm=%s mode=%s task=%s objective=%s seed=%s lambda=%s checkpoint=%s out=%s\n' \
    "$ARM" "$RUN_MODE" "$INDEX" "$OBJECTIVE" "$SEED" "$WEIGHT" \
    "$CHECKPOINT" "$OUT"
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
  --lambda-value "$WEIGHT" --autoencoder "$AE" "${EXTRA[@]}" \
  --latent-cache "$LATENTS" --out-dir "$OUT" \
  --top-features 100 --examples-per-feature 4 --min-examples 4 \
  --multipliers -2 2
