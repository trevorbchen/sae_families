#!/bin/bash
#SBATCH --job-name=probe-run-seeds
#SBATCH --partition=expansion
#SBATCH --array=0-23%8
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/probe-run-seeds-%A_%a.out
#SBATCH --error=logs/probe-run-seeds-%A_%a.err
set -euo pipefail

SEEDS=(2026 2031 2032)
NAMES=(
  direct_mse all_mse_2branch plus_laplacian plus_fourier_low
  plus_fourier_mid plus_fourier_high all_mse_5branch tandem_all4
)
INDEX="$SLURM_ARRAY_TASK_ID"
SEED="${SEEDS[$((INDEX / 8))]}"
NAME="${NAMES[$((INDEX % 8))]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_linear_probe_seed_replicates_v1"

cd "$ROOT"
mkdir -p logs "$BASE/probe_runs/seed_$SEED/$NAME"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$PY" run_sae_linear_probes.py \
  --representation "$BASE/representations/seed_$SEED/$NAME.pt" \
  --targets "$BASE/representations/seed_$SEED/targets.pt" \
  --output-dir "$BASE/probe_runs/seed_$SEED/$NAME" \
  --jobs "${SLURM_CPUS_PER_TASK:-12}" --seed 260818
