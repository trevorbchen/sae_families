#!/bin/bash
#SBATCH --job-name=changed-spec-repl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/changed-spec-repl-%A_%a.out
#SBATCH --error=logs/changed-spec-repl-%A_%a.err
set -euo pipefail

: "${REPL_SEED:?submit with REPL_SEED}"

OBJECTIVES=(
  laplacian fourier_low fourier_mid fourier_high
  laplacian fourier_low fourier_mid fourier_high
)
LOSS_MODES=(
  anchored anchored anchored anchored
  pure_transformed pure_transformed pure_transformed pure_transformed
)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
LOSS_MODE="${LOSS_MODES[$SLURM_ARRAY_TASK_ID]}"
if [[ "$LOSS_MODE" == "anchored" ]]; then
  TARGET_RATIO=0.1
  MODE_DIR=proper
else
  TARGET_RATIO=1.0
  MODE_DIR=pure_transformed
fi

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/changed_objective_seed_replicates"
BASE="$RUN_ROOT/seed_$REPL_SEED/base_mse_8192_k32/best_valid.pt"
OUT="$RUN_ROOT/seed_$REPL_SEED/$MODE_DIR/$OBJECTIVE"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_residual_specialist.py \
  --objective "$OBJECTIVE" --loss-mode "$LOSS_MODE" \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt --base-sae "$BASE" \
  --out-dir "$OUT" --feature-count 8192 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 \
  --lr-warmup 500 --lambda-warmup 2000 \
  --gradient-target-ratio "$TARGET_RATIO" --calibration-examples 512 \
  --mse-tolerance 0.02 --aux-coeff 0.1 --grad-clip 1.0 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed "$REPL_SEED"
