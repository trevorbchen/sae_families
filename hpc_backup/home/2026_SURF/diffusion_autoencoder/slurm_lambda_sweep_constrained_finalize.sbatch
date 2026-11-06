#!/bin/bash
#SBATCH --job-name=sae-lambda-1d-final
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sae-lambda-1d-final-%j.out
#SBATCH --error=logs/sae-lambda-1d-final-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_constrained_vector_convae_v1"

cd "$ROOT"
mkdir -p logs "$EXP/summary"
exec "$PY" summarize_lambda_sweep.py \
  --experiment-root "$EXP" --out-dir "$EXP/summary" \
  --max-mean-ev-drop-pp 2.0 --max-objective-ev-drop-pp 3.0
