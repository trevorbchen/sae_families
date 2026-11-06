#!/bin/bash
#SBATCH --job-name=sae-lambda-1d-c8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-47%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sae-lambda-1d-c8-%A_%a.out
#SBATCH --error=logs/sae-lambda-1d-c8-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
CODE="/resnick/scratch/tbchen/surf_2026/vector_ae_code"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_constrained_vector_convae_v1"
SOURCE_EXP="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
LATENTS="$SOURCE_EXP/data/ffhq_32768_latents.pt"
AE="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment/results/vector_conv_ae/last.pt"

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

cd "$CODE"
export PYTHONPATH="$CODE:$ROOT"
mkdir -p logs "$OUT"

exec "$PY" "$CODE/train_spatial_map_sae.py" \
  --architecture flat --autoencoder-type vector --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 2048 --k 8 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight "$WEIGHT" --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 \
  --seed "$SEED" --init-seed "$SEED" --data-seed 2026
