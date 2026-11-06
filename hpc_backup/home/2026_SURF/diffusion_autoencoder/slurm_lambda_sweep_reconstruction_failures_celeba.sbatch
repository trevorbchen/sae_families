#!/bin/bash
#SBATCH --job-name=lsweep-recon-failures-celeba
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/lsweep-recon-failures-celeba-%j.out
#SBATCH --error=logs/lsweep-recon-failures-celeba-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/family_sae/lambda_sweep/reconstruction_failures

exec "$PY" evaluate_lambda_sweep_reconstruction_failures.py \
  --lambda-tag 100p0 --lambda-value 100.0 \
  --seeds 2030 2031 2032 \
  --source celeba --celeba-sample 8192 \
  --out-csv results/family_sae/lambda_sweep/reconstruction_failures/lambda_100p0_celeba_per_example_mse.csv
