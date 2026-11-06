#!/bin/bash
#SBATCH --job-name=diffae-raw-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diffae-raw-gui-%j.out
#SBATCH --error=logs/diffae-raw-gui-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs
exec "/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python" ae_feature_permutation_gui.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --port 8767 --examples 12
