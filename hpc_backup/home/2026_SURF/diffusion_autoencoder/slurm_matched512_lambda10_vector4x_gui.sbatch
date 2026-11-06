#!/bin/bash
#SBATCH --job-name=m512-v4-l10-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/matched512-v4-l10-gui-%j.out
#SBATCH --error=logs/matched512-v4-l10-gui-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
REVIEW="$BASE/review/vector_4x_k8_lambda_10p0_seed_2030"

cd "$ROOT"
mkdir -p logs "$REVIEW"

exec "$PY" spatial_sae_feature_gui.py \
  --architecture flat \
  --autoencoder-type vector \
  --checkpoint-root "$BASE/experiments/vector_4x_k8/checkpoints" \
  --checkpoint-manifest "$ROOT/matched512_lambda10_vector4x_gui_manifest.json" \
  --autoencoder "$BASE/autoencoders/vector/last.pt" \
  --latent-cache "$BASE/latents/vector_ffhq_32768.pt" \
  --objectives latent_mse laplacian fourier_low fourier_mid fourier_high \
  --annotations "$REVIEW/annotations.json" \
  --port 18814 --examples 10 --rank-examples 4096 --batch-size 128 \
  --label "Vector 4x/k8 lambda=10"
