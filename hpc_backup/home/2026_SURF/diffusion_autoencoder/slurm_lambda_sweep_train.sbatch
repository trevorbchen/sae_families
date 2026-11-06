#!/bin/bash
#SBATCH --job-name=sae-lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-47%12
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sae-lambda-%A_%a.out
#SBATCH --error=logs/sae-lambda-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_flat_convae_v1"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
AE="results/spatial_conv_ae/last.pt"

SEEDS=(2030 2031 2032)
LAMBDAS=(0.1 0.3 3.0 10.0)
LAMBDA_TAGS=(0p1 0p3 3p0 10p0)
OBJECTIVES=(weighted_laplacian weighted_fourier_low weighted_fourier_mid weighted_fourier_high)
DIRECTORIES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
LAMBDA_INDEX=$((INDEX / 12))
WITHIN_LAMBDA=$((INDEX % 12))
OBJECTIVE_INDEX=$((WITHIN_LAMBDA / 3))
SEED_INDEX=$((WITHIN_LAMBDA % 3))
SEED="${SEEDS[$SEED_INDEX]}"
OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"
WEIGHT="${LAMBDAS[$LAMBDA_INDEX]}"
TAG="${LAMBDA_TAGS[$LAMBDA_INDEX]}"
OUT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED"

cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture flat --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 4096 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight "$WEIGHT" --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed "$SEED"
