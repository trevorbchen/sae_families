#!/bin/bash
#SBATCH --job-name=bsf-variant-grids
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bsf-variant-grids-%j.out
#SBATCH --error=logs/bsf-variant-grids-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_bsf/variant_grids
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" generate_bsf_variant_grids.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --variants-root results/zsem_bsf/variants \
  --out-dir results/zsem_bsf/variant_grids \
  --examples 12 --steps 20
