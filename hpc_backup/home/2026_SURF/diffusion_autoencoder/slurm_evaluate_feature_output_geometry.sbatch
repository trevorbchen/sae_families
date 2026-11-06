#!/bin/bash
#SBATCH --job-name=sae-output-geometry
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sae-output-geometry-%A_%a.out
#SBATCH --error=logs/sae-output-geometry-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SEEDS=(2030 2031 2032)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
cd "$ROOT"
mkdir -p logs results/family_sae/feature_output_geometry

exec "$PY" evaluate_feature_output_geometry.py \
  --autoencoder results/spatial_conv_ae/last.pt \
  --latent-cache /resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt \
  --sae-root "$ROOT/results" \
  --seed "$SEED" \
  --out-dir "results/family_sae/feature_output_geometry/seed_${SEED}" \
  --top-features 500 \
  --examples-per-feature 8 \
  --min-examples 4 \
  --multipliers -2 2
