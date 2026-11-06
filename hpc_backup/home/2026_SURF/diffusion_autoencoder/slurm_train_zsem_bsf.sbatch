#!/bin/bash
# Same data, split, scalar dictionary budget, and active scalar budget as the
# final 32x/k32 SAE. BSF selects 4 blocks of 8 dimensions each.
#SBATCH --job-name=zsem-bsf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-bsf-%j.out
#SBATCH --error=logs/zsem-bsf-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_bsf/32x_block8_top4
exec "/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python" train_zsem_bsf.py \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --out-dir results/zsem_bsf/32x_block8_top4 \
  --dict-mult 32 --block-dim 8 --top-blocks 4 \
  --steps 100000 --batch-size 1024 --lr 3e-4 --warmup 2000
