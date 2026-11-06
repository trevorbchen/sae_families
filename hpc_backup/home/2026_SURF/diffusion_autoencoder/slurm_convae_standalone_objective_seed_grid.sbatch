#!/bin/bash
#SBATCH --job-name=convae-obj-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --array=0-11%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=logs/convae-obj-grid-%A_%a.out
#SBATCH --error=logs/convae-obj-grid-%A_%a.err
set -euo pipefail

OBJECTIVES=(
  latent_mse latent_mse latent_mse
  weighted_fourier_low weighted_fourier_low weighted_fourier_low
  weighted_fourier_mid weighted_fourier_mid weighted_fourier_mid
  weighted_fourier_high weighted_fourier_high weighted_fourier_high
)
SEEDS=(
  2040 2041 2042
  2040 2041 2042
  2040 2041 2042
  2040 2041 2042
)
OBJECTIVE="${OBJECTIVES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
AE="results/spatial_conv_ae/last.pt"
OUT="results/convae_standalone_objective_seed_grid/$OBJECTIVE/seed_$SEED"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture flat --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 16384 --k 64 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight 1.0 --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed "$SEED"
