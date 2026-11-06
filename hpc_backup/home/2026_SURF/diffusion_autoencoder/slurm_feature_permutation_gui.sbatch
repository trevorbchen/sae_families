#!/bin/bash
#SBATCH --job-name=diffae-feature-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/diffae-feature-gui-%j.out
#SBATCH --error=logs/diffae-feature-gui-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
exec "$PY" feature_permutation_gui.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --sae results/zsem_sae/final_32x_k32_aux_reinit_merged/last.pt \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --port 8766 \
  --examples 12
