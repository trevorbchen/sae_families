#!/bin/bash
#SBATCH --job-name=m512-v16-hi-fin-extra
#SBATCH --partition=expansion
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/matched512-v16-hi-fin-extra-%A_%a.out
#SBATCH --error=logs/matched512-v16-hi-fin-extra-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
ARM=vector_16x_k32
EXP="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/experiments/$ARM"
TAGS=(30p0 100p0)
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

cd "$ROOT"
mkdir -p logs "$EXP/summary/feature_output_geometry/lambda_$TAG"
exec "$PY" aggregate_feature_output_geometry.py \
  --experiment-root "$EXP" --lambda-tag "$TAG" --seeds 2030 2031 2032 \
  --out-dir "$EXP/summary/feature_output_geometry/lambda_$TAG"
