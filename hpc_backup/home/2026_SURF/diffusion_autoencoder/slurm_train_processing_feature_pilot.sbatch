#!/bin/bash
#SBATCH --job-name=processing-sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-4%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/processing-sae-%A_%a.out
#SBATCH --error=logs/processing-sae-%A_%a.err
set -euo pipefail

TAGS=(jpeg_q60 jpeg_q60 jpeg_q60 random_phase_f17 random_phase_f17)
OBJECTIVES=(latent_mse laplacian fourier_high latent_mse fourier_high)
INDEX="$SLURM_ARRAY_TASK_ID"
TAG="${TAGS[$INDEX]}"
OBJECTIVE="${OBJECTIVES[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
RUN_ROOT="/resnick/scratch/tbchen/surf_2026/processing_feature_spatial/$TAG"
OUT="$RUN_ROOT/pilot_saes/$OBJECTIVE/seed_2030"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture flat --objective "$OBJECTIVE" \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --latent-cache "$RUN_ROOT/cache.pt" \
  --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$OUT" --feature-count 4096 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed 2030
