#!/bin/bash
#SBATCH --job-name=sae-probes
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%4
#SBATCH --output=logs/sae-probes-%A_%a.out
#SBATCH --error=logs/sae-probes-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_linear_probes"
NAMES=(
  raw_latent direct_mse all_mse_2branch plus_laplacian plus_fourier_low
  plus_fourier_mid plus_fourier_high all_mse_5branch tandem_all4
  standalone_laplacian standalone_fourier_low standalone_fourier_mid
  standalone_fourier_high
)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"

cd "$ROOT"
mkdir -p logs "$BASE/runs/$NAME"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec "$PY" run_sae_linear_probes.py \
  --representation "$BASE/representations/$NAME.pt" \
  --targets "$BASE/representations/targets.pt" \
  --output-dir "$BASE/runs/$NAME" \
  --jobs "${SLURM_CPUS_PER_TASK:-12}" \
  --seed 260818
