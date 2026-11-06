#!/bin/bash
# Polynomial-decoder comparison to the regular merged 32x/k64 SAE.
#SBATCH --job-name=zsem-polysae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-polysae-%j.out
#SBATCH --error=logs/zsem-polysae-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_sae
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" train_zsem_polysae.py \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --out-dir results/zsem_sae/polynomial_32x_k64_aux_reinit_merged \
  --dict-mult 32 --k 64 --aux-k 64 --aux-coeff 0.1 \
  --steps 100000 --batch-size 1024 --lr 3e-4 --warmup 2000 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256
