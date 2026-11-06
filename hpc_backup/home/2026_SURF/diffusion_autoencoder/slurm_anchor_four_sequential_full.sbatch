#!/bin/bash
#SBATCH --job-name=anchor-four-seq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/anchor-four-seq-%j.out
#SBATCH --error=logs/anchor-four-seq-%j.err
set -euo pipefail

OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_v1"
OUT_ROOT="$RUN_ROOT/anchor_plus_four"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/best_valid.pt"
cd "$ROOT"
mkdir -p logs "$OUT_ROOT"

PRIOR_ARGS=()
for STAGE in "${!OBJECTIVES[@]}"; do
  OBJECTIVE="${OBJECTIVES[$STAGE]}"
  OUT="$OUT_ROOT/stage_$((STAGE + 1))_$OBJECTIVE"
  mkdir -p "$OUT"
  "$PY" train_spatial_residual_specialist.py \
    --objective "$OBJECTIVE" \
    --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
    --autoencoder results/spatial_conv_ae/last.pt \
    --base-sae "$BASE" "${PRIOR_ARGS[@]}" \
    --out-dir "$OUT" --feature-count 2048 --k 8 \
    --steps 20000 --batch-size 32 --lr 3e-4 \
    --lr-warmup 500 --lambda-warmup 2000 \
    --gradient-target-ratio 0.1 --calibration-examples 512 \
    --mse-tolerance 0.02 --aux-coeff 0.1 --grad-clip 1.0 \
    --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
    --validate-every 250 --seed "$((2026 + STAGE))"
  PRIOR_ARGS+=(--prior-specialist "$OUT/best_feasible.pt")
done
