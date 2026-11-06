#!/bin/bash
#SBATCH --job-name=plant-pure-snapshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-19%10
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=logs/plant-pure-snapshot-%A_%a.out
#SBATCH --error=logs/plant-pure-snapshot-%A_%a.err
set -euo pipefail

TAGS=(
  brightness_a24 checkerboard_a1 color_red_a1 contrast_a0p3 edge_enhance_a32
  gaussian_blur_s2 hue_rotate_a0p12 jpeg_q60 local_blob_a32 local_gabor_a32
  local_occlusion_s12 pixelate_b4 sine_high_a0p5 sine_high_a1 sine_high_a3
  sine_low_a1 sine_mid_a1 sine_random_high_a1 sparse_impulse_a64 unsharp_a1
)
OBJECTIVE_ORDER=(
  latent_mse laplacian fourier_bands
  fourier_low fourier_mid fourier_high
)

TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026/sparse_planted_residual_suite/$TAG"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
OUT="$ROOT/results/sparse_planted_residual_suite_pure_partial/$TAG"
cd "$ROOT"
mkdir -p logs "$OUT"

OBJECTIVES=()
for objective in "${OBJECTIVE_ORDER[@]}"; do
  if [[ -s "$SCRATCH/saes_pure/$objective/seed_2030/last.pt" ]]; then
    OBJECTIVES+=("$objective")
  fi
done
if [[ ${#OBJECTIVES[@]} -eq 0 ]]; then
  echo "No completed pure checkpoints for $TAG"
  exit 0
fi

echo "Snapshotting $TAG: ${OBJECTIVES[*]}"
exec "$PY" analyze_processing_residual_sae.py \
  --residual-cache "$SCRATCH/cache.pt" --base-latent-cache "$BASE" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --sae-root "$SCRATCH/saes_pure" --output-dir "$OUT" \
  --objectives "${OBJECTIVES[@]}" --seeds 2030 --batch-size 64
