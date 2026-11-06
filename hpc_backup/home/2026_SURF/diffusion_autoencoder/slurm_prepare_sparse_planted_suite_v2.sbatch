#!/bin/bash
#SBATCH --job-name=sparse-plant2-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-10%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sparse-plant2-data-%A_%a.out
#SBATCH --error=logs/sparse-plant2-data-%A_%a.err
set -euo pipefail

TAGS=(
  brightness_a24 contrast_a0p3 hue_rotate_a0p12 gaussian_blur_s2
  unsharp_a1 edge_enhance_a32 local_gabor_a32 local_blob_a32
  sparse_impulse_a64 pixelate_b4 local_occlusion_s12
)
FEATURES=(
  brightness contrast hue_rotate gaussian_blur unsharp_mask edge_enhance
  local_gabor local_blob sparse_impulse pixelate local_occlusion
)
AMPLITUDES=(24 0.3 0.12 1 1 32 32 32 64 1 1)
FREQUENCY_X=(0 0 0 0 0 0 10 0 0 0 0)
FREQUENCY_Y=(0 0 0 0 0 0 4 0 0 0 0)
INDEX="$SLURM_ARRAY_TASK_ID"
TAG="${TAGS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/sparse_planted_residual_suite/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" prepare_sparse_planted_residual_latents.py \
  --feature "${FEATURES[$INDEX]}" --amplitude "${AMPLITUDES[$INDEX]}" \
  --frequency-x "${FREQUENCY_X[$INDEX]}" \
  --frequency-y "${FREQUENCY_Y[$INDEX]}" \
  --prevalence 0.10 \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --clean-latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output "$OUT/cache.pt" --preview "$OUT/preview_signed_delta_x16.png"
