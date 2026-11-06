#!/bin/bash
#SBATCH --job-name=sparse-plant2-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-10%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sparse-plant2-analysis-%A_%a.out
#SBATCH --error=logs/sparse-plant2-analysis-%A_%a.err
set -euo pipefail

TAGS=(
  brightness_a24 contrast_a0p3 hue_rotate_a0p12 gaussian_blur_s2
  unsharp_a1 edge_enhance_a32 local_gabor_a32 local_blob_a32
  sparse_impulse_a64 pixelate_b4 local_occlusion_s12
)
OBJECTIVE_LIST=(
  "latent_mse weighted_laplacian weighted_fourier_low"
  "latent_mse weighted_laplacian weighted_fourier_bands"
  "latent_mse weighted_laplacian weighted_fourier_low"
  "latent_mse weighted_laplacian weighted_fourier_high"
  "latent_mse weighted_laplacian weighted_fourier_high"
  "latent_mse weighted_laplacian weighted_fourier_high"
  "latent_mse weighted_laplacian weighted_fourier_mid"
  "latent_mse weighted_laplacian weighted_fourier_low"
  "latent_mse weighted_laplacian weighted_fourier_high"
  "latent_mse weighted_laplacian weighted_fourier_high"
  "latent_mse weighted_laplacian weighted_fourier_mid"
)
INDEX="$SLURM_ARRAY_TASK_ID"
TAG="${TAGS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026/sparse_planted_residual_suite/$TAG"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
OUT="$ROOT/results/sparse_planted_residual_suite/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"
read -r -a OBJECTIVES <<< "${OBJECTIVE_LIST[$INDEX]}"

exec "$PY" analyze_processing_residual_sae.py \
  --residual-cache "$SCRATCH/cache.pt" --base-latent-cache "$BASE" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/saes" --output-dir "$OUT" \
  --objectives "${OBJECTIVES[@]}" --seeds 2030 --batch-size 64
