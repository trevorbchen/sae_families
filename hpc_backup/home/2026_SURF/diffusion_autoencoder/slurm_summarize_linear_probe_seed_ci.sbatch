#!/bin/bash
#SBATCH --job-name=probe-seed-ci
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --nice=10000
#SBATCH --output=logs/probe-seed-ci-%j.out
#SBATCH --error=logs/probe-seed-ci-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
cd "$ROOT"
mkdir -p logs results/family_sae/three_seed_confidence_intervals/linear_probe
exec "$PY" summarize_seed_confidence_intervals.py \
  --linear-runs /resnick/scratch/tbchen/surf_2026/sae_linear_probe_seed_replicates_v1/probe_runs \
  --output-dir results/family_sae/three_seed_confidence_intervals/linear_probe
