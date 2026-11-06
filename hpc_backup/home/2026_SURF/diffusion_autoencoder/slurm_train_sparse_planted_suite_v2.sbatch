#!/bin/bash
#SBATCH --job-name=sparse-plant2-sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --array=0-32%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sparse-plant2-sae-%A_%a.out
#SBATCH --error=logs/sparse-plant2-sae-%A_%a.err
set -euo pipefail

TAGS=(
  brightness_a24 brightness_a24 brightness_a24
  contrast_a0p3 contrast_a0p3 contrast_a0p3
  hue_rotate_a0p12 hue_rotate_a0p12 hue_rotate_a0p12
  gaussian_blur_s2 gaussian_blur_s2 gaussian_blur_s2
  unsharp_a1 unsharp_a1 unsharp_a1
  edge_enhance_a32 edge_enhance_a32 edge_enhance_a32
  local_gabor_a32 local_gabor_a32 local_gabor_a32
  local_blob_a32 local_blob_a32 local_blob_a32
  sparse_impulse_a64 sparse_impulse_a64 sparse_impulse_a64
  pixelate_b4 pixelate_b4 pixelate_b4
  local_occlusion_s12 local_occlusion_s12 local_occlusion_s12
)
OBJECTIVES=(
  latent_mse weighted_laplacian weighted_fourier_low
  latent_mse weighted_laplacian weighted_fourier_bands
  latent_mse weighted_laplacian weighted_fourier_low
  latent_mse weighted_laplacian weighted_fourier_high
  latent_mse weighted_laplacian weighted_fourier_high
  latent_mse weighted_laplacian weighted_fourier_high
  latent_mse weighted_laplacian weighted_fourier_mid
  latent_mse weighted_laplacian weighted_fourier_low
  latent_mse weighted_laplacian weighted_fourier_high
  latent_mse weighted_laplacian weighted_fourier_high
  latent_mse weighted_laplacian weighted_fourier_mid
)
INDEX="$SLURM_ARRAY_TASK_ID"
TAG="${TAGS[$INDEX]}"
OBJECTIVE="${OBJECTIVES[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sparse_planted_residual_suite/$TAG"
OUT="$RUN_ROOT/saes/$OBJECTIVE/seed_2030"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture flat --objective "$OBJECTIVE" \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --latent-cache "$RUN_ROOT/cache.pt" --base-latent-cache "$BASE" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$OUT" --feature-count 4096 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight 1.0 --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed 2030
