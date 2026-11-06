#!/bin/bash
#SBATCH --job-name=m512-plant-detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/m512-plant-detect-%A_%a.out
#SBATCH --error=logs/m512-plant-detect-%A_%a.err
set -euo pipefail

: "${ARM:?ARM must be exported (vector_4x_k8|vector_16x_k32|spatial_4x_k8|spatial_16x_k32)}"
: "${LAMBDA_TAG:?LAMBDA_TAG must be exported (e.g. 1p0, 30p0)}"
SEEDS="${SEEDS:-2030 2031 2032}"

TAGS=(
  brightness_a24 contrast_a0p3 hue_rotate_a0p12 gaussian_blur_s2
  unsharp_a1 edge_enhance_a32 local_gabor_a32 local_blob_a32
  sparse_impulse_a64 pixelate_b4 local_occlusion_s12
  sine_high_a0p5 sine_high_a1 sine_high_a3 sine_low_a1 sine_mid_a1
  sine_random_high_a1 checkerboard_a1 color_red_a1 jpeg_q60
)
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$BASE/experiments/$ARM"

case "$ARM" in
  spatial_4x_k8|spatial_16x_k32) TYPE=spatial ;;
  vector_4x_k8|vector_16x_k32) TYPE=vector ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

CHECKPOINT_ROOT="$EXP/checkpoints/lambda_$LAMBDA_TAG"
mkdir -p "$CHECKPOINT_ROOT/latent_mse"
for SEED in $SEEDS; do
  ln -sfn "$EXP/checkpoints/mse_control/latent_mse/seed_$SEED" \
    "$CHECKPOINT_ROOT/latent_mse/seed_$SEED"
done

OUT="$EXP/summary/feature_detection/lambda_$LAMBDA_TAG/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" analyze_processing_residual_sae.py \
  --residual-cache "$BASE/planted_features/$TYPE/$TAG/cache.pt" \
  --base-latent-cache "$BASE/latents/${TYPE}_ffhq_32768.pt" \
  --autoencoder "$BASE/autoencoders/${TYPE}/last.pt" --autoencoder-type "$TYPE" \
  --sae-root "$CHECKPOINT_ROOT" --output-dir "$OUT" \
  --objectives latent_mse laplacian fourier_low fourier_mid fourier_high \
  --seeds $SEEDS --batch-size 64
