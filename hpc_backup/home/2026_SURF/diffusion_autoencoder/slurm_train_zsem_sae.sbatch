#!/bin/bash
#SBATCH --job-name=diffae-zsem-sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/diffae-zsem-sae-%j.out
#SBATCH --error=logs/diffae-zsem-sae-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_sae

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT_DIR="${OUT_DIR:-results/zsem_sae/8x_k16_random}" \
DICT_MULT="${DICT_MULT:-8}" \
K="${K:-16}" \
STEPS="${STEPS:-100000}" \
"$PY" train_zsem_sae.py \
  --codes official_diffae/checkpoints/ffhq128_autoenc/latent.pkl \
  --out-dir "$OUT_DIR" \
  --dict-mult "$DICT_MULT" \
  --k "$K" \
  --steps "$STEPS" \
  --batch-size 1024
