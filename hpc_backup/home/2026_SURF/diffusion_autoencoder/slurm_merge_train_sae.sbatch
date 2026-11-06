#!/bin/bash
#SBATCH --job-name=zsem-sae-merged
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/zsem-sae-merged-%j.out
#SBATCH --error=logs/zsem-sae-merged-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_codes
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
"$PY" merge_zsem_codes.py \
  --ffhq official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --extra results/zsem_codes/celeba_train.pt \
  --out results/zsem_codes/ffhq_celeba_train.pt
exec "$PY" train_zsem_sae_final.py \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --out-dir results/zsem_sae/final_32x_k32_aux_reinit_merged \
  --dict-mult 32 --k 32 --aux-k 32 --aux-coeff 0.1 \
  --steps 100000 --batch-size 1024 --lr 3e-4 --warmup 2000 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256
