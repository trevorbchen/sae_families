#!/bin/bash
#SBATCH --job-name=diffae-16xk32-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/diffae-16xk32-gui-%j.out
#SBATCH --error=logs/diffae-16xk32-gui-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
exec "/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python" feature_permutation_gui.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --sae results/zsem_sae/small_16x_k32_aux_reinit_merged/last.pt \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --port 8768 --examples 12
