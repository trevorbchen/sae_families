#!/bin/bash
#SBATCH --job-name=spatial-map-sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/spatial-map-sae-%A_%a.out
#SBATCH --error=logs/spatial-map-sae-%A_%a.err
set -euo pipefail

ARCHITECTURES=(flat flat flat flat flat flat conv conv conv conv conv conv)
OBJECTIVES=(
  latent_mse laplacian fourier_bands fourier_low fourier_mid fourier_high
  latent_mse laplacian fourier_bands fourier_low fourier_mid fourier_high
)
ARCHITECTURE="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
AE="results/spatial_conv_ae/last.pt"
OUT="results/spatial_map_saes/$ARCHITECTURE/$OBJECTIVE"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture "$ARCHITECTURE" --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 4096 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed 2030
