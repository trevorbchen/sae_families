#!/bin/bash
#SBATCH --job-name=m512-plant-prep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-39%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/m512-plant-prep-%A_%a.out
#SBATCH --error=logs/m512-plant-prep-%A_%a.err
set -euo pipefail

# Generates the 20-condition planted-feature suite (matching
# added_feature_detection's TAGS) against the matched512 vector and spatial
# autoencoders, so lambda-sweep checkpoints can be scored for feature
# detection. One-time cost, shared across all arms/lambdas/seeds of each
# autoencoder type.
TAGS=(
  brightness_a24 contrast_a0p3 hue_rotate_a0p12 gaussian_blur_s2
  unsharp_a1 edge_enhance_a32 local_gabor_a32 local_blob_a32
  sparse_impulse_a64 pixelate_b4 local_occlusion_s12
  sine_high_a0p5 sine_high_a1 sine_high_a3 sine_low_a1 sine_mid_a1
  sine_random_high_a1 checkerboard_a1 color_red_a1 jpeg_q60
)
FEATURES=(
  brightness contrast hue_rotate gaussian_blur unsharp_mask edge_enhance
  local_gabor local_blob sparse_impulse pixelate local_occlusion
  sine_fixed sine_fixed sine_fixed sine_fixed sine_fixed
  sine_random_phase checkerboard color_red jpeg
)
AMPLITUDES=(24 0.3 0.12 1 1 32 32 32 64 1 1 0.5 1 3 1 1 1 1 1 1)
FREQUENCY_X=(0 0 0 0 0 0 10 0 0 0 0 17 17 17 2 8 17 17 0 0)
FREQUENCY_Y=(0 0 0 0 0 0 4 0 0 0 0 17 17 17 2 8 17 17 0 0)

TYPES=(vector spatial)
INDEX="$SLURM_ARRAY_TASK_ID"
TYPE_INDEX=$((INDEX / 20))
TAG_INDEX=$((INDEX % 20))
TYPE="${TYPES[$TYPE_INDEX]}"
TAG="${TAGS[$TAG_INDEX]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
OUT="$BASE/planted_features/$TYPE/$TAG"

cd "$ROOT"
mkdir -p logs "$OUT"
if [[ -s "$OUT/cache.pt" ]]; then
  echo "reusing $OUT/cache.pt"
  exit 0
fi
exec "$PY" prepare_sparse_planted_residual_latents.py \
  --feature "${FEATURES[$TAG_INDEX]}" --amplitude "${AMPLITUDES[$TAG_INDEX]}" \
  --frequency-x "${FREQUENCY_X[$TAG_INDEX]}" --frequency-y "${FREQUENCY_Y[$TAG_INDEX]}" \
  --prevalence 0.10 --jpeg-quality 60 \
  --data "$DATA" \
  --clean-latent-cache "$BASE/latents/${TYPE}_ffhq_32768.pt" \
  --autoencoder "$BASE/autoencoders/${TYPE}/last.pt" \
  --autoencoder-type "$TYPE" \
  --output "$OUT/cache.pt" --preview "$OUT/preview_signed_delta_x16.png"
