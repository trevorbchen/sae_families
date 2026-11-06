#!/bin/bash
#SBATCH --job-name=m512-vprobe-reps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-28%4
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-vprobe-reps-%A_%a.out
#SBATCH --error=logs/matched512-vprobe-reps-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MATCHED="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="$MATCHED/experiments/vector_16x_k32"
BASE="$MATCHED/linear_probe_vector16x_lambda_sweep"
LAMBDAS=(0p1 0p3 1p0 3p0 10p0 30p0 100p0)
OBJECTIVES=(laplacian fourier_low fourier_mid fourier_high)
INDEX="$SLURM_ARRAY_TASK_ID"

if (( INDEX == 0 )); then
  NAME=mse_control
  CHECKPOINT="$EXP/checkpoints/mse_control/latent_mse/seed_2030/last.pt"
else
  SPECIALIST=$((INDEX - 1))
  LAMBDA="${LAMBDAS[$((SPECIALIST / 4))]}"
  OBJECTIVE="${OBJECTIVES[$((SPECIALIST % 4))]}"
  NAME="lambda_${LAMBDA}_${OBJECTIVE}"
  CHECKPOINT="$EXP/checkpoints/lambda_${LAMBDA}/${OBJECTIVE}/seed_2030/last.pt"
fi

cd "$ROOT"
mkdir -p logs "$BASE/representations"
if [[ -s "$BASE/representations/$NAME.pt" ]]; then
  echo "reusing $BASE/representations/$NAME.pt"
  exit 0
fi
exec "$PY" cache_sae_probe_representations.py \
  --data "$BASE/celeba_vector_32768.pt" \
  --output-dir "$BASE/representations" --batch-size 128 \
  --skip-raw --skip-targets --flat "$NAME=$CHECKPOINT"
