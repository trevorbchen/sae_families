#!/bin/bash
#SBATCH --job-name=m512-detect-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/m512-detect-smoke-%j.out
#SBATCH --error=logs/m512-detect-smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$BASE/experiments/vector_16x_k32"
CHECKPOINT_ROOT="$EXP/checkpoints/lambda_1p0"
mkdir -p "$CHECKPOINT_ROOT/latent_mse"
ln -sfn "$EXP/checkpoints/mse_control/latent_mse/seed_2030" "$CHECKPOINT_ROOT/latent_mse/seed_2030"

OUT="/tmp/tbchen_smoke/detection_vector_16x_k32_lambda1p0"
cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" analyze_processing_residual_sae.py \
  --residual-cache "$BASE/planted_features_smoke/vector_brightness_a24.pt" \
  --base-latent-cache "$BASE/latents/vector_ffhq_32768.pt" \
  --autoencoder "$BASE/autoencoders/vector/last.pt" --autoencoder-type vector \
  --sae-root "$CHECKPOINT_ROOT" --output-dir "$OUT" \
  --objectives latent_mse laplacian --seeds 2030 --batch-size 64
