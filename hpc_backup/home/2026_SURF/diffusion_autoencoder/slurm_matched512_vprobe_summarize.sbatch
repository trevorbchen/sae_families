#!/bin/bash
#SBATCH --job-name=m512-vprobe-sum
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/matched512-vprobe-sum-%j.out
#SBATCH --error=logs/matched512-vprobe-sum-%j.err
set -euo pipefail

: "${BASE_DIR:?BASE_DIR must be exported}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"

cd "$ROOT"
mkdir -p logs "$BASE_DIR/summary"
exec "$PY" summarize_sae_linear_probes.py \
  --runs-dir "$BASE_DIR/probe_runs" \
  --targets "$BASE_DIR/representations/targets.pt" \
  --output-dir "$BASE_DIR/summary" --seed 260818
