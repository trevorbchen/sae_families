#!/bin/bash
#SBATCH --job-name=prep-fairface-ood
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prep-fairface-ood-%j.out
#SBATCH --error=logs/prep-fairface-ood-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
cd "$ROOT"
mkdir -p logs "$BASE/latents"

# The xet fast-download client fails on this compute node with a CA-cert
# error unrelated to the dataset itself; fall back to the plain HTTP
# download path in huggingface_hub instead of chasing the cert store.
export HF_HUB_DISABLE_XET=1

exec "$PY" prepare_fairface_ood_cache.py \
  --vector-autoencoder "$BASE/autoencoders/vector/last.pt" \
  --spatial-autoencoder "$BASE/autoencoders/spatial/last.pt" \
  --max-images 8192 \
  --out-vector "$BASE/latents/fairface_vector_8192.pt" \
  --out-spatial "$BASE/latents/fairface_spatial_8192.pt"
