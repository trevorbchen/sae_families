#!/bin/bash
#SBATCH --job-name=routed-mse-reps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/logs/mse-control-rep-%A_%a.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/logs/mse-control-rep-%A_%a.err
set -euo pipefail

SEEDS=(2030 2034)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2"
OUT="$EXP/runs/all_mse_control_5_branches_seed$SEED"

cd "$ROOT"
mkdir -p "$EXP/logs" "$OUT"

exec "$PY" train_independent_routed_union_sae.py \
  --objectives laplacian,fourier_low,fourier_mid,fourier_high \
  --all-mse-control \
  --data /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$OUT" \
  --mse-dict-size 8192 --mse-k 32 \
  --specialist-dict-size 8192 --specialist-k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed "$SEED"
