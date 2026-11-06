#!/bin/bash
#SBATCH --job-name=viz-failure-clusters
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/viz-failure-clusters-%j.out
#SBATCH --error=logs/viz-failure-clusters-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/family_sae/lambda_sweep/reconstruction_failures

exec "$PY" visualize_lambda_sweep_failure_clusters.py \
  --csv results/family_sae/lambda_sweep/reconstruction_failures/lambda_100p0_celeba_per_example_mse.csv \
  --arm spatial_4x_k8 --ae-type spatial --rank-n 200 --per-objective 8 \
  --out results/family_sae/lambda_sweep/reconstruction_failures/spatial_4x_k8_failure_clusters.png
