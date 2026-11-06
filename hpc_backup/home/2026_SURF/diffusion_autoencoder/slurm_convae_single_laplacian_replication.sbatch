#!/bin/bash
#SBATCH --job-name=convae-lap-repl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/convae-lap-repl-%A_%a.out
#SBATCH --error=logs/convae-lap-repl-%A_%a.err
set -euo pipefail

SEEDS=(2040 2041 2042)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
AE="results/spatial_conv_ae/last.pt"
OUT="results/convae_single_laplacian_replication/seed_$SEED"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture flat --objective weighted_laplacian \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 16384 --k 64 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight 1.0 --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed "$SEED"
