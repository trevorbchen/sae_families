#!/bin/bash
#SBATCH --job-name=spatial-param-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --output=logs/spatial-grad-smoke-%A_%a.out
#SBATCH --error=logs/spatial-grad-smoke-%A_%a.err
set -euo pipefail

MODES=(grouped grouped_pure)
NAMES=(proper_loss transformed_loss)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_gradient_only_v4_smoke/$NAME"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_spatial_ae_sae.py \
  --mode "$MODE" \
  --checkpoint results/spatial_conv_ae/last.pt \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --out-dir "$OUT" --steps 1 --batch-size 8 \
  --lr 3e-4 --warmup 1 --grad-clip 1.0 \
  --loss-normalization none --lambda-mode dynamic_gradient --gradient-target-ratio 1.0 \
  --parameter-gradient-cap initial_mse \
  --calibration-examples 64 --aux-coeff 0.1 \
  --reinit-every 1000 --dead-after 3000 --max-rescue-per-group 8 \
  --validate-every 1 --checkpoint-every 1 --state-every 1 \
  --trace-flush-every 1 --skip-file-hashes \
  --seed 2026 --split-seed 2026
