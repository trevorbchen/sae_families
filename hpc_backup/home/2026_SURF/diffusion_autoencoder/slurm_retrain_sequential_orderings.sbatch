#!/bin/bash
#SBATCH --job-name=seq-order-retrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-23%6
#SBATCH --output=logs/seq-order-retrain-%A_%a.out
#SBATCH --error=logs/seq-order-retrain-%A_%a.err
set -euo pipefail

# All 4! orderings of the same four specialists. Every task starts from the
# same MSE anchor and retrains all four stages from scratch.
ORDERS=(
  "laplacian fourier_low fourier_mid fourier_high"
  "laplacian fourier_low fourier_high fourier_mid"
  "laplacian fourier_mid fourier_low fourier_high"
  "laplacian fourier_mid fourier_high fourier_low"
  "laplacian fourier_high fourier_low fourier_mid"
  "laplacian fourier_high fourier_mid fourier_low"
  "fourier_low laplacian fourier_mid fourier_high"
  "fourier_low laplacian fourier_high fourier_mid"
  "fourier_low fourier_mid laplacian fourier_high"
  "fourier_low fourier_mid fourier_high laplacian"
  "fourier_low fourier_high laplacian fourier_mid"
  "fourier_low fourier_high fourier_mid laplacian"
  "fourier_mid laplacian fourier_low fourier_high"
  "fourier_mid laplacian fourier_high fourier_low"
  "fourier_mid fourier_low laplacian fourier_high"
  "fourier_mid fourier_low fourier_high laplacian"
  "fourier_mid fourier_high laplacian fourier_low"
  "fourier_mid fourier_high fourier_low laplacian"
  "fourier_high laplacian fourier_low fourier_mid"
  "fourier_high laplacian fourier_mid fourier_low"
  "fourier_high fourier_low laplacian fourier_mid"
  "fourier_high fourier_low fourier_mid laplacian"
  "fourier_high fourier_mid laplacian fourier_low"
  "fourier_high fourier_mid fourier_low laplacian"
)

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_v1"
OUT_ROOT="$RUN_ROOT/retrained_orderings/proper"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/best_valid.pt"
ORDER_INDEX="$SLURM_ARRAY_TASK_ID"
read -r -a OBJECTIVES <<< "${ORDERS[$ORDER_INDEX]}"
ORDER_NAME="$(IFS=__; printf '%s' "${OBJECTIVES[*]}")"
ORDER_ROOT="$OUT_ROOT/order_$(printf '%02d' "$ORDER_INDEX")_${ORDER_NAME}"

cd "$ROOT"
mkdir -p logs "$ORDER_ROOT"

printf '{"order_index":%d,"objectives":["%s","%s","%s","%s"],"feature_count_per_stage":2048,"k_per_stage":8}\n' \
  "$ORDER_INDEX" "${OBJECTIVES[0]}" "${OBJECTIVES[1]}" \
  "${OBJECTIVES[2]}" "${OBJECTIVES[3]}" > "$ORDER_ROOT/config.json"

PRIOR_ARGS=()
for STAGE_INDEX in 0 1 2 3; do
  OBJECTIVE="${OBJECTIVES[$STAGE_INDEX]}"
  STAGE=$((STAGE_INDEX + 1))
  OUT="$ORDER_ROOT/stage_${STAGE}_${OBJECTIVE}"
  mkdir -p "$OUT"

  case "$OBJECTIVE" in
    laplacian) OBJECTIVE_SEED=2026 ;;
    fourier_low) OBJECTIVE_SEED=2027 ;;
    fourier_mid) OBJECTIVE_SEED=2028 ;;
    fourier_high) OBJECTIVE_SEED=2029 ;;
    *) printf 'Unknown objective: %s\n' "$OBJECTIVE" >&2; exit 2 ;;
  esac

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
    --validate-every 250 --seed "$OBJECTIVE_SEED"
  PRIOR_ARGS+=(--prior-specialist "$OUT/best_feasible.pt")
done
