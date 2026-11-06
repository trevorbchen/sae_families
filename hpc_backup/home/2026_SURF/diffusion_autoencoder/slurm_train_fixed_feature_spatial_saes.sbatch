#!/bin/bash
#SBATCH --job-name=fixed-feature-sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fixed-feature-sae-%A_%a.out
#SBATCH --error=logs/fixed-feature-sae-%A_%a.err
set -euo pipefail

OBJECTIVES=(latent_mse latent_mse latent_mse fourier_high fourier_high fourier_high)
SEEDS=(2030 2031 2032 2030 2031 2032)
INDEX="$SLURM_ARRAY_TASK_ID"
OBJECTIVE="${OBJECTIVES[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="results/fixed_feature_spatial/saes/$OBJECTIVE/seed_$SEED"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture flat --objective "$OBJECTIVE" \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --latent-cache /resnick/scratch/tbchen/surf_2026/fixed_feature_spatial/ffhq_fixed_high.pt \
  --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$OUT" --feature-count 4096 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed "$SEED"
