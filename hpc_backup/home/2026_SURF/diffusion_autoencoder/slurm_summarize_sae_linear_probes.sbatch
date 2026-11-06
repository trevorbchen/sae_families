#!/bin/bash
#SBATCH --job-name=sae-probe-summary
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sae-probe-summary-%j.out
#SBATCH --error=logs/sae-probe-summary-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_linear_probes"

cd "$ROOT"
mkdir -p logs "$BASE/summary"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

exec "$PY" summarize_sae_linear_probes.py \
  --runs-dir "$BASE/runs" \
  --targets "$BASE/representations/targets.pt" \
  --output-dir "$BASE/summary" \
  --seed 260818
