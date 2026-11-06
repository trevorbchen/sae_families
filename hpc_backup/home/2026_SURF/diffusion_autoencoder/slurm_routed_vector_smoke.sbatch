#!/bin/bash
#SBATCH --job-name=routed-vector-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/independent_routed_union_vector_v1/logs/smoke-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/independent_routed_union_vector_v1/logs/smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/independent_routed_union_vector_v1"
OUT="$EXP/smoke"

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
  --steps 30 --batch-size 32 --lr 3e-4 --warmup 5 \
  --grad-clip 1.0 --calibration-examples 64 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 10 --seed 2026 \
  --max-train-examples 512 --max-valid-examples 256
