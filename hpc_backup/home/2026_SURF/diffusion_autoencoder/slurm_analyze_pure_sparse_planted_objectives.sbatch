#!/bin/bash
#SBATCH --job-name=plant-pure-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-19%10
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=logs/plant-pure-analysis-%A_%a.out
#SBATCH --error=logs/plant-pure-analysis-%A_%a.err
set -euo pipefail

TAGS=(
  brightness_a24 checkerboard_a1 color_red_a1 contrast_a0p3 edge_enhance_a32
  gaussian_blur_s2 hue_rotate_a0p12 jpeg_q60 local_blob_a32 local_gabor_a32
  local_occlusion_s12 pixelate_b4 sine_high_a0p5 sine_high_a1 sine_high_a3
  sine_low_a1 sine_mid_a1 sine_random_high_a1 sparse_impulse_a64 unsharp_a1
)
OBJECTIVES=(
  latent_mse laplacian fourier_bands
  fourier_low fourier_mid fourier_high
)

TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026/sparse_planted_residual_suite/$TAG"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
OUT="$ROOT/results/sparse_planted_residual_suite_pure/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" analyze_processing_residual_sae.py \
  --residual-cache "$SCRATCH/cache.pt" --base-latent-cache "$BASE" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/saes_pure" --output-dir "$OUT" \
  --objectives "${OBJECTIVES[@]}" --seeds 2030 --batch-size 64
