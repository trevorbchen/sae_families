#!/bin/bash
#SBATCH --job-name=m512-sprobe-setup
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/matched512-sprobe-setup-%j.out
#SBATCH --error=logs/matched512-sprobe-setup-%j.err
set -euo pipefail

# Spatial-arm analog of slurm_setup_matched512_vector_probe_representations.sbatch.
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/linear_probe_spatial16x_lambda_sweep"

cd "$ROOT"
mkdir -p logs "$BASE/representations"
exec "$PY" cache_sae_probe_representations.py \
  --data "$BASE/celeba_spatial_32768.pt" \
  --output-dir "$BASE/representations" --batch-size 128
