#!/bin/bash
#SBATCH --job-name=m512-vprobe-reps-xseed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-vprobe-reps-xseed-%A_%a.out
#SBATCH --error=logs/matched512-vprobe-reps-xseed-%A_%a.err
set -euo pipefail

# Extends the linear-probe representation cache to additional SAE seeds.
# Parametrized by ARM, BASE_DIR (the linear_probe_* scratch root for that
# arm), DATA_CACHE (the celeba_*_32768.pt for that arm), and SEEDS (space-
# separated list of seeds to cache, e.g. "2031 2032" for vector_16x_k32
# which already has seed 2030, or "2030 2031 2032" for a from-scratch arm
# like spatial_16x_k32). Array size must be 29 * (number of seeds).
: "${ARM:?ARM must be exported}"
: "${BASE_DIR:?BASE_DIR must be exported}"
: "${DATA_CACHE:?DATA_CACHE must be exported}"
: "${SEEDS:?SEEDS must be exported, e.g. '2031 2032'}"
read -r -a SEED_ARRAY <<< "$SEEDS"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MATCHED="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$MATCHED/experiments/$ARM"
LAMBDAS=(0p1 0p3 1p0 3p0 10p0 30p0 100p0)
OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
PER_SEED=29
SEED_INDEX=$((INDEX / PER_SEED))
WITHIN=$((INDEX % PER_SEED))
SEED="${SEED_ARRAY[$SEED_INDEX]}"

if (( WITHIN == 0 )); then
  BASE_NAME=mse_control
  CHECKPOINT="$EXP/checkpoints/mse_control/latent_mse/seed_$SEED/last.pt"
else
  SPECIALIST=$((WITHIN - 1))
  LAMBDA="${LAMBDAS[$((SPECIALIST / 4))]}"
  OBJECTIVE="${OBJECTIVES[$((SPECIALIST % 4))]}"
  BASE_NAME="lambda_${LAMBDA}_${OBJECTIVE}"
  CHECKPOINT="$EXP/checkpoints/lambda_${LAMBDA}/${OBJECTIVE}/seed_$SEED/last.pt"
fi
NAME="${BASE_NAME}_seed${SEED}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'arm=%s seed=%s index=%s name=%s checkpoint=%s\n' "$ARM" "$SEED" "$INDEX" "$NAME" "$CHECKPOINT"
  exit 0
fi

cd "$ROOT"
mkdir -p logs "$BASE_DIR/representations"
if [[ -s "$BASE_DIR/representations/$NAME.pt" ]]; then
  echo "reusing $BASE_DIR/representations/$NAME.pt"
  exit 0
fi
exec "$PY" cache_sae_probe_representations.py \
  --data "$DATA_CACHE" \
  --output-dir "$BASE_DIR/representations" --batch-size 128 \
  --skip-raw --skip-targets --flat "$NAME=$CHECKPOINT"
