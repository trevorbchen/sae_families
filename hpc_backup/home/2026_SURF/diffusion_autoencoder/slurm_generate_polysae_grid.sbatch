#!/bin/bash
#SBATCH --job-name=poly-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/poly-grid-%j.out
#SBATCH --error=logs/poly-grid-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_sae/polynomial_grids
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" generate_polysae_grid.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --sae results/zsem_sae/polynomial_32x_k64_aux_reinit_merged/last.pt \
  --out results/zsem_sae/polynomial_grids \
  --examples 12 --steps 20
