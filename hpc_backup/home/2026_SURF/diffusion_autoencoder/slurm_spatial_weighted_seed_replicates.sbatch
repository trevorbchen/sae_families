#!/bin/bash
#SBATCH --job-name=spatial-weighted
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --array=0-23%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/spatial-weighted-%A_%a.out
#SBATCH --error=logs/spatial-weighted-%A_%a.err
set -euo pipefail

ARCHITECTURES=(
  flat flat flat flat flat flat flat flat flat flat flat flat
  conv conv conv conv conv conv conv conv conv conv conv conv
)
OBJECTIVES=(
  weighted_laplacian weighted_laplacian weighted_laplacian
  weighted_fourier_low weighted_fourier_low weighted_fourier_low
  weighted_fourier_mid weighted_fourier_mid weighted_fourier_mid
  weighted_fourier_high weighted_fourier_high weighted_fourier_high
  weighted_laplacian weighted_laplacian weighted_laplacian
  weighted_fourier_low weighted_fourier_low weighted_fourier_low
  weighted_fourier_mid weighted_fourier_mid weighted_fourier_mid
  weighted_fourier_high weighted_fourier_high weighted_fourier_high
)
SEEDS=(
  2030 2031 2032 2030 2031 2032 2030 2031 2032 2030 2031 2032
  2030 2031 2032 2030 2031 2032 2030 2031 2032 2030 2031 2032
)
INDEX="$SLURM_ARRAY_TASK_ID"
ARCHITECTURE="${ARCHITECTURES[$INDEX]}"
OBJECTIVE="${OBJECTIVES[$INDEX]}"
SEED="${SEEDS[$INDEX]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt"
LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"
AE="results/spatial_conv_ae/last.pt"
OUT="results/spatial_map_sae_weighted_seed_replicates/$ARCHITECTURE/$OBJECTIVE/seed_$SEED"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_spatial_map_sae.py \
  --architecture "$ARCHITECTURE" --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 4096 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight 1.0 --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 --seed "$SEED"
