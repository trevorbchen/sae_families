#!/bin/bash
#SBATCH --job-name=routed-vector-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/independent_routed_union_vector_v1/logs/all-four-seed2026-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/independent_routed_union_vector_v1/logs/all-four-seed2026-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/independent_routed_union_vector_v1"
OUT="$EXP/runs/mse_plus_all_four"

cd "$ROOT"
mkdir -p "$EXP/logs" "$OUT"

exec "$PY" train_independent_routed_union_sae.py \
  --objectives laplacian,fourier_low,fourier_mid,fourier_high \
  --data /resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/latents/vector_ffhq_32768.pt \
  --autoencoder /resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/autoencoders/vector/last.pt \
  --autoencoder-kind vector \
  --out-dir "$OUT" \
  --mse-dict-size 8192 --mse-k 32 \
  --specialist-dict-size 8192 --specialist-k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed 2026
