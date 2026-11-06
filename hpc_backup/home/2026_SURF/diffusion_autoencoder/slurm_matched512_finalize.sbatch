#!/bin/bash
#SBATCH --job-name=m512-finalize
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/matched512-finalize-%j.out
#SBATCH --error=logs/matched512-finalize-%j.err
set -euo pipefail

: "${ARM:?ARM must be exported by the submission script}"
RUN_MODE="${RUN_MODE:-full}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/experiments/$ARM"

cd "$ROOT"
mkdir -p logs "$EXP/summary"
if [[ "$RUN_MODE" == priority_10 ]]; then
  exec "$PY" aggregate_feature_output_geometry.py \
    --experiment-root "$EXP" --lambda-tag 10p0 \
    --out-dir "$EXP/summary/feature_output_geometry/lambda_10p0"
elif [[ "$RUN_MODE" == full ]]; then
  "$PY" summarize_lambda_sweep.py \
    --experiment-root "$EXP" --out-dir "$EXP/summary" \
    --max-mean-ev-drop-pp 2.0 --max-objective-ev-drop-pp 3.0
  "$PY" plot_lambda_training_diagnostics.py \
    --experiment-root "$EXP" --out-dir "$EXP/summary/training_diagnostics"
  for TAG in 0p1 0p3 1p0 3p0 10p0; do
    "$PY" aggregate_feature_output_geometry.py \
      --experiment-root "$EXP" --lambda-tag "$TAG" \
      --out-dir "$EXP/summary/feature_output_geometry/lambda_$TAG"
  done
else
  echo "unknown RUN_MODE=$RUN_MODE" >&2
  exit 2
fi
