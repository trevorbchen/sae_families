#!/bin/bash
#SBATCH --job-name=seq-subset-retrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-15%6
#SBATCH --output=logs/seq-subset-retrain-%A_%a.out
#SBATCH --error=logs/seq-subset-retrain-%A_%a.err
set -euo pipefail

# Each array task trains one objective subset from scratch. The MSE anchor is
# intentionally shared, while every retained specialist learns the residual
# produced by the earlier specialists in that same subset.
OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_v1"
OUT_ROOT="$RUN_ROOT/retrained_subset_ablations/proper"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/best_valid.pt"
MASK="${SLURM_ARRAY_TASK_ID}"

cd "$ROOT"
mkdir -p logs "$OUT_ROOT"

NAMES=()
for INDEX in 0 1 2 3; do
  if (( MASK & (1 << INDEX) )); then
    NAMES+=("${OBJECTIVES[$INDEX]}")
  fi
done
SUBSET_NAME="$(IFS=+; printf '%s' "${NAMES[*]}")"
SUBSET_ROOT="$OUT_ROOT/mask_$(printf '%02d' "$MASK")_${SUBSET_NAME}"
mkdir -p "$SUBSET_ROOT"

printf '{"mask":%d,"objectives":"%s","feature_count_per_stage":2048,"k_per_stage":8}\n' \
  "$MASK" "$SUBSET_NAME" > "$SUBSET_ROOT/config.json"

PRIOR_ARGS=()
STAGE=0
for INDEX in 0 1 2 3; do
  if ! (( MASK & (1 << INDEX) )); then
    continue
  fi
  OBJECTIVE="${OBJECTIVES[$INDEX]}"
  STAGE=$((STAGE + 1))
  OUT="$SUBSET_ROOT/stage_${STAGE}_${OBJECTIVE}"
  mkdir -p "$OUT"
  "$PY" train_spatial_residual_specialist.py \
    --objective "$OBJECTIVE" --loss-mode anchored \
    --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
    --autoencoder results/spatial_conv_ae/last.pt \
    --base-sae "$BASE" "${PRIOR_ARGS[@]}" \
    --out-dir "$OUT" --feature-count 2048 --k 8 \
    --steps 20000 --batch-size 32 --lr 3e-4 \
    --lr-warmup 500 --lambda-warmup 2000 \
    --gradient-target-ratio 0.1 --calibration-examples 512 \
    --mse-tolerance 0.02 --aux-coeff 0.1 --grad-clip 1.0 \
    --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
    --validate-every 250 --seed "$((2026 + INDEX))"
  PRIOR_ARGS+=(--prior-specialist "$OUT/best_feasible.pt")
done
