#!/bin/bash
#SBATCH --job-name=changed-mse-repl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/changed_objective_seed_replicates/logs/changed-mse-repl-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/changed_objective_seed_replicates/logs/changed-mse-repl-%j.err
set -euo pipefail

: "${REPL_SEED:?submit with REPL_SEED}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/changed_objective_seed_replicates"
BASE="$RUN_ROOT/seed_$REPL_SEED/base_mse_8192_k32/best_valid.pt"
OUT="$RUN_ROOT/seed_$REPL_SEED/proper/mse_control"
cd "$ROOT"
mkdir -p "$RUN_ROOT/logs" "$OUT"

exec "$PY" train_spatial_residual_specialist.py \
  --objective mse \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt --base-sae "$BASE" \
  --out-dir "$OUT" --feature-count 8192 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --lr-warmup 500 \
  --gradient-target-ratio 0 --calibration-examples 16 \
  --mse-tolerance 0.02 --aux-coeff 0.1 --grad-clip 1.0 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed "$REPL_SEED"
