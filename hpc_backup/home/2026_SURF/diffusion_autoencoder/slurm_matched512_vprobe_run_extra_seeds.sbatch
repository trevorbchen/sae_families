#!/bin/bash
#SBATCH --job-name=m512-vprobe-run-xseed
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/matched512-vprobe-run-xseed-%A_%a.out
#SBATCH --error=logs/matched512-vprobe-run-xseed-%A_%a.err
set -euo pipefail

: "${BASE_DIR:?BASE_DIR must be exported}"
: "${SEEDS:?SEEDS must be exported, e.g. '2031 2032'}"
read -r -a SEED_ARRAY <<< "$SEEDS"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
LAMBDAS=(0p1 0p3 1p0 3p0 10p0 30p0 100p0)
OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
PER_SEED=29
SEED_INDEX=$((INDEX / PER_SEED))
WITHIN=$((INDEX % PER_SEED))
SEED="${SEED_ARRAY[$SEED_INDEX]}"

if (( WITHIN == 0 )); then
  BASE_NAME=mse_control
else
  SPECIALIST=$((WITHIN - 1))
  LAMBDA="${LAMBDAS[$((SPECIALIST / 4))]}"
  OBJECTIVE="${OBJECTIVES[$((SPECIALIST % 4))]}"
  BASE_NAME="lambda_${LAMBDA}_${OBJECTIVE}"
fi
NAME="${BASE_NAME}_seed${SEED}"

cd "$ROOT"
mkdir -p logs "$BASE_DIR/probe_runs/$NAME"
if [[ -s "$BASE_DIR/probe_runs/$NAME/results.csv" && -s "$BASE_DIR/probe_runs/$NAME/predictions.pt" ]]; then
  echo "reusing $BASE_DIR/probe_runs/$NAME"
  exit 0
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$PY" run_sae_linear_probes.py \
  --representation "$BASE_DIR/representations/$NAME.pt" \
  --targets "$BASE_DIR/representations/targets.pt" \
  --output-dir "$BASE_DIR/probe_runs/$NAME" \
  --jobs "${SLURM_CPUS_PER_TASK:-12}" --seed 260818
