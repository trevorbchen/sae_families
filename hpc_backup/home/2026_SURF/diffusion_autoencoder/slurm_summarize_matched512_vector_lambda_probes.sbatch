#!/bin/bash
#SBATCH --job-name=m512-vprobe-sum
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/matched512-vprobe-sum-%j.out
#SBATCH --error=logs/matched512-vprobe-sum-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/linear_probe_vector16x_lambda_sweep"

cd "$ROOT"
mkdir -p logs "$BASE/summary"
exec "$PY" summarize_sae_linear_probes.py \
  --runs-dir "$BASE/probe_runs" \
  --targets "$BASE/representations/targets.pt" \
  --output-dir "$BASE/summary" --seed 260818
