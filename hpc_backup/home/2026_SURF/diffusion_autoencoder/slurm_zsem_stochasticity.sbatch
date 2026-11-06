#!/bin/bash
#SBATCH --job-name=diffae-zsem-noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diffae-zsem-noise-%j.out
#SBATCH --error=logs/diffae-zsem-noise-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs/results
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
"$PY" diagnose_zsem_stochasticity.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --image official_diffae/imgs/sandy.JPG \
  --out results/zsem_stochasticity.json \
  --repeats 32
