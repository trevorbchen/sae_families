#!/bin/bash
#SBATCH --job-name=seq-pure-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/seq-pure-smoke-%A_%a.out
#SBATCH --error=logs/seq-pure-smoke-%A_%a.err
set -euo pipefail

OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/sequential_residual_specialists_pure_smoke"
BASE="/resnick/scratch/tbchen/surf_2026/spatial_residual_specialists_v1/base_mse_8192_k32/best_valid.pt"
cd "$ROOT"
mkdir -p logs "$RUN_ROOT/task_$SLURM_ARRAY_TASK_ID"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  exec "$PY" train_spatial_residual_specialist.py \
    --objective laplacian --loss-mode pure_transformed \
    --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
    --autoencoder results/spatial_conv_ae/last.pt --base-sae "$BASE" \
    --out-dir "$RUN_ROOT/task_0" --feature-count 8192 --k 32 \
    --steps 5 --batch-size 8 --lr-warmup 2 --lambda-warmup 2 \
    --gradient-target-ratio 1 --calibration-examples 16 \
    --validate-every 5 --reinit-every 0
fi

PRIOR_ARGS=()
for STAGE in "${!OBJECTIVES[@]}"; do
  OBJECTIVE="${OBJECTIVES[$STAGE]}"
  OUT="$RUN_ROOT/task_1/stage_$((STAGE + 1))_$OBJECTIVE"
  mkdir -p "$OUT"
  "$PY" train_spatial_residual_specialist.py \
    --objective "$OBJECTIVE" --loss-mode pure_transformed \
    --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
    --autoencoder results/spatial_conv_ae/last.pt \
    --base-sae "$BASE" "${PRIOR_ARGS[@]}" \
    --out-dir "$OUT" --feature-count 2048 --k 8 \
    --steps 5 --batch-size 8 --lr-warmup 2 --lambda-warmup 2 \
    --gradient-target-ratio 1 --calibration-examples 16 \
    --validate-every 5 --reinit-every 0 --seed "$((2026 + STAGE))"
  PRIOR_ARGS+=(--prior-specialist "$OUT/best_objective.pt")
done
