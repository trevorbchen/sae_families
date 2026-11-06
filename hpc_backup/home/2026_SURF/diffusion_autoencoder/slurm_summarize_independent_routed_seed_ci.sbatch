#!/bin/bash
#SBATCH --job-name=routed-seed-ci
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/logs/seed-ci-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/logs/seed-ci-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
V1="/resnick/scratch/tbchen/surf_2026/independent_routed_union_v1/runs"
V2="/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/runs"
OUT="/resnick/scratch/tbchen/surf_2026/independent_routed_union_v2/seed_ci"

cd "$ROOT"
export MPLCONFIGDIR="/resnick/scratch/tbchen/surf_2026/matplotlib-cache"

exec "$PY" summarize_independent_routed_seed_ci.py \
  --pair 2026 "$V1/mse_plus_all_four/metrics.json" \
              "$V1/all_mse_control_5_branches/metrics.json" \
  --pair 2030 "$V2/mse_plus_all_four_seed2030/metrics.json" \
              "$V2/all_mse_control_5_branches_seed2030/metrics.json" \
  --pair 2034 "$V2/mse_plus_all_four_seed2034/metrics.json" \
              "$V2/all_mse_control_5_branches_seed2034/metrics.json" \
  --output-dir "$OUT"
