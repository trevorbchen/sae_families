#!/bin/bash
#SBATCH --job-name=lambda-family-final
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lambda-family-final-%j.out
#SBATCH --error=logs/lambda-family-final-%j.err
set -euo pipefail

: "${VARIANT:?VARIANT must be exported by the submission script}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_matched_32768_v1/$VARIANT"

cd "$ROOT"
mkdir -p logs "$EXP/summary"
exec "$PY" summarize_lambda_sweep.py \
  --experiment-root "$EXP" --out-dir "$EXP/summary" \
  --max-mean-ev-drop-pp 2.0 --max-objective-ev-drop-pp 3.0
