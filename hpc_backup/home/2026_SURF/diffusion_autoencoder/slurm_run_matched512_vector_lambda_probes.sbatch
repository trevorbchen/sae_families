#!/bin/bash
#SBATCH --job-name=m512-vprobe-run
#SBATCH --partition=expansion
#SBATCH --array=0-29%8
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/matched512-vprobe-run-%A_%a.out
#SBATCH --error=logs/matched512-vprobe-run-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/linear_probe_vector16x_lambda_sweep"
LAMBDAS=(0p1 0p3 1p0 3p0 10p0 30p0 100p0)
OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
INDEX="$SLURM_ARRAY_TASK_ID"

if (( INDEX == 0 )); then
  NAME=raw_latent
elif (( INDEX == 1 )); then
  NAME=mse_control
else
  SPECIALIST=$((INDEX - 2))
  LAMBDA="${LAMBDAS[$((SPECIALIST / 4))]}"
  OBJECTIVE="${OBJECTIVES[$((SPECIALIST % 4))]}"
  NAME="lambda_${LAMBDA}_${OBJECTIVE}"
fi

cd "$ROOT"
mkdir -p logs "$BASE/probe_runs/$NAME"
if [[ -s "$BASE/probe_runs/$NAME/results.csv" && -s "$BASE/probe_runs/$NAME/predictions.pt" ]]; then
  echo "reusing $BASE/probe_runs/$NAME"
  exit 0
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$PY" run_sae_linear_probes.py \
  --representation "$BASE/representations/$NAME.pt" \
  --targets "$BASE/representations/targets.pt" \
  --output-dir "$BASE/probe_runs/$NAME" \
  --jobs "${SLURM_CPUS_PER_TASK:-12}" --seed 260818
