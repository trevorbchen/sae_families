#!/bin/bash
#SBATCH --job-name=sparse-plant-data
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-9%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sparse-plant-data-%A_%a.out
#SBATCH --error=logs/sparse-plant-data-%A_%a.err
set -euo pipefail

TAGS=(
  sine_high_a0p5 sine_high_a1 sine_high_a3 sine_low_a1 sine_mid_a1
  sine_random_high_a1 checkerboard_a1 color_red_a1 jpeg_q60 latent_random_a0p25
)
FEATURES=(
  sine_fixed sine_fixed sine_fixed sine_fixed sine_fixed
  sine_random_phase checkerboard color_red jpeg latent_random
)
AMPLITUDES=(0.5 1 3 1 1 1 1 1 1 0.25)
FREQUENCY_X=(17 17 17 2 8 17 17 0 0 0)
FREQUENCY_Y=(17 17 17 2 8 17 17 0 0 0)
INDEX="$SLURM_ARRAY_TASK_ID"
TAG="${TAGS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/sparse_planted_residual_suite/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" prepare_sparse_planted_residual_latents.py \
  --feature "${FEATURES[$INDEX]}" \
  --amplitude "${AMPLITUDES[$INDEX]}" \
  --frequency-x "${FREQUENCY_X[$INDEX]}" \
  --frequency-y "${FREQUENCY_Y[$INDEX]}" \
  --prevalence 0.10 --jpeg-quality 60 \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --clean-latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --output "$OUT/cache.pt" --preview "$OUT/preview_signed_delta_x16.png"
