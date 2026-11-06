#!/bin/bash
#SBATCH --job-name=zsem-xform-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/zsem-xform-smoke-%j.out
#SBATCH --error=logs/zsem-xform-smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/zsem_transformed_smoke/gradient

exec "$PY" train_zsem_transformed_sae.py \
  --objective gradient \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data /resnick/scratch/tbchen/surf_2026/zsem_objective_audit/pilot_1024.pt \
  --out-dir results/zsem_transformed_smoke/gradient \
  --dict-mult 32 --k 128 --steps 2 --batch-size 2 \
  --calibration-examples 40 --lr 3e-4 --warmup 1 --grad-clip 1.0 \
  --reinit-every 0 --seed 2030
