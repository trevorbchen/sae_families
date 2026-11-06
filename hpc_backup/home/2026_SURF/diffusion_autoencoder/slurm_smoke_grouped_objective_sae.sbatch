#!/bin/bash
#SBATCH --job-name=grouped-sae-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/grouped-sae-smoke-%j.out
#SBATCH --error=logs/grouped-sae-smoke-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="results/grouped_objective_sae/smoke_grouped"
cd "$ROOT"
mkdir -p logs "$OUT"

exec "$PY" train_grouped_objective_sae.py \
  --mode grouped --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --data /resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_16384.pt \
  --out-dir "$OUT" --steps 1 --batch-size 4 \
  --lr 3e-4 --warmup 1 --grad-clip 1.0 --transform-weight 1.0 \
  --calibration-examples 40 --aux-coeff 0.1 \
  --reinit-every 1000 --dead-after 3000 --max-rescue-per-group 4 \
  --seed 2026 --split-seed 2026

