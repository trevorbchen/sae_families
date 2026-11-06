#!/bin/bash
#SBATCH --job-name=multifeature-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/multifeature-grid-%j.out
#SBATCH --error=logs/multifeature-grid-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/multifeature_perturbations
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" generate_multifeature_perturbations.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --sae-root results/zsem_sae \
  --out results/multifeature_perturbations --examples 12 --steps 20
