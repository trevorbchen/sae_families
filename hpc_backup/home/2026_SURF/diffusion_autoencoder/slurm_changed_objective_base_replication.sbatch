#!/bin/bash
#SBATCH --job-name=changed-base-repl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/changed-base-repl-%j.out
#SBATCH --error=logs/changed-base-repl-%j.err
set -euo pipefail

: "${REPL_SEED:?submit with REPL_SEED}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/changed_objective_seed_replicates"
OUT="$RUN_ROOT/seed_$REPL_SEED/base_mse_8192_k32"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_spatial_ae_sae.py \
  --mode regular --regular-dict-size 8192 --regular-k 32 \
  --checkpoint results/spatial_conv_ae/last.pt \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --out-dir "$OUT" --steps 20000 --batch-size 32 \
  --lr 3e-4 --warmup 500 --grad-clip 1.0 --parameter-gradient-cap none \
  --aux-coeff 0.1 --reinit-every 1000 --dead-after 3000 \
  --max-rescue-per-group 64 --validate-every 250 \
  --checkpoint-every 5000 --state-every 1000 --trace-flush-every 100 \
  --seed "$REPL_SEED" --split-seed 2026
