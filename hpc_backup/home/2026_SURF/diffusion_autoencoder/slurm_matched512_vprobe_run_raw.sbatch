#!/bin/bash
#SBATCH --job-name=m512-vprobe-run-raw
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/matched512-vprobe-run-raw-%j.out
#SBATCH --error=logs/matched512-vprobe-run-raw-%j.err
set -euo pipefail

: "${BASE_DIR:?BASE_DIR must be exported}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"

cd "$ROOT"
mkdir -p logs "$BASE_DIR/probe_runs/raw_latent"
if [[ -s "$BASE_DIR/probe_runs/raw_latent/results.csv" && -s "$BASE_DIR/probe_runs/raw_latent/predictions.pt" ]]; then
  echo "reusing $BASE_DIR/probe_runs/raw_latent"
  exit 0
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$PY" run_sae_linear_probes.py \
  --representation "$BASE_DIR/representations/raw_latent.pt" \
  --targets "$BASE_DIR/representations/targets.pt" \
  --output-dir "$BASE_DIR/probe_runs/raw_latent" \
  --jobs "${SLURM_CPUS_PER_TASK:-12}" --seed 260818
