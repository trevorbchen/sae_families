#!/bin/bash
#SBATCH --job-name=diffae-variant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --exclude=hpc-93-36
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/diffae-variant-%j.out
#SBATCH --error=logs/diffae-variant-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT_DIR="${OUT_DIR:?set OUT_DIR}" \
  "$PY" train_zsem_sae_variants.py \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --out-dir "$OUT_DIR" \
  --mode "${MODE:-topk}" \
  --norm "${NORM:-standard}" \
  --dict-mult "${DICT_MULT:-16}" \
  --k "${K:-32}" \
  --steps "${STEPS:-100000}" \
  --batch-size 1024
