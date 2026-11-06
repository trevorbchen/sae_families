#!/bin/bash
#SBATCH --job-name=sae-lambda-geom
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-62%12
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sae-lambda-geom-%A_%a.out
#SBATCH --error=logs/sae-lambda-geom-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_flat_convae_v1"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
AE="results/spatial_conv_ae/last.pt"

SEEDS=(2030 2031 2032)
LAMBDAS=(0.1 0.3 1.0 3.0 10.0)
LAMBDA_TAGS=(0p1 0p3 1p0 3p0 10p0)
OBJECTIVES=(weighted_laplacian weighted_fourier_low weighted_fourier_mid weighted_fourier_high)
DIRECTORIES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
if (( INDEX < 3 )); then
  SEED="${SEEDS[$INDEX]}"
  OBJECTIVE="latent_mse"
  WEIGHT="0"
  CHECKPOINT="$EXP/checkpoints/mse_control/latent_mse/seed_$SEED/last.pt"
  OUT="$EXP/geometry/mse_control/seed_$SEED"
else
  SPECIALIST_INDEX=$((INDEX - 3))
  LAMBDA_INDEX=$((SPECIALIST_INDEX / 12))
  WITHIN_LAMBDA=$((SPECIALIST_INDEX % 12))
  OBJECTIVE_INDEX=$((WITHIN_LAMBDA / 3))
  SEED_INDEX=$((WITHIN_LAMBDA % 3))
  SEED="${SEEDS[$SEED_INDEX]}"
  OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
  DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"
  WEIGHT="${LAMBDAS[$LAMBDA_INDEX]}"
  TAG="${LAMBDA_TAGS[$LAMBDA_INDEX]}"
  CHECKPOINT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED/last.pt"
  OUT="$EXP/geometry/lambda_$TAG/$DIRECTORY/seed_$SEED"
fi

cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" evaluate_lambda_sweep_model.py \
  --checkpoint "$CHECKPOINT" --objective "$OBJECTIVE" --seed "$SEED" \
  --lambda-value "$WEIGHT" --autoencoder "$AE" --latent-cache "$LATENTS" \
  --out-dir "$OUT" --top-features 100 --examples-per-feature 4 \
  --min-examples 4 --multipliers -2 2
