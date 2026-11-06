#!/bin/bash
#SBATCH --job-name=m512-v16-hi-lam-extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-15%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/matched512-v16-hi-lam-extra-%A_%a.out
#SBATCH --error=logs/matched512-v16-hi-lam-extra-%A_%a.err
set -euo pipefail

# One-off extension of the vector_16x_k32 lambda=30/100 exploratory arm to
# seeds 2031 and 2032 (originally only seed 2030). Mirrors the
# high_lambda_seed2030 branch of slurm_matched512_sae_train.sbatch. Kept as a
# standalone script rather than editing the shared parametrized sbatch files,
# since spatial_16x_k32 training is concurrently in flight against those.
ARM=vector_16x_k32
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
EXP="$BASE/experiments/$ARM"
AE="$BASE/autoencoders/vector/last.pt"
LATENTS="$BASE/latents/vector_ffhq_32768.pt"

SEEDS=(2031 2032)
HIGH_LAMBDAS=(30.0 100.0)
HIGH_LAMBDA_TAGS=(30p0 100p0)
OBJECTIVES=(weighted_laplacian weighted_fourier_low weighted_fourier_mid weighted_fourier_high)
DIRECTORIES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
if (( INDEX < 0 || INDEX > 15 )); then
  echo "requires array indices 0-15" >&2
  exit 2
fi
SEED_INDEX=$((INDEX % 2))
REST=$((INDEX / 2))
LAMBDA_INDEX=$((REST / 4))
OBJECTIVE_INDEX=$((REST % 4))

SEED="${SEEDS[$SEED_INDEX]}"
OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"
WEIGHT="${HIGH_LAMBDAS[$LAMBDA_INDEX]}"
TAG="${HIGH_LAMBDA_TAGS[$LAMBDA_INDEX]}"
OUT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'arm=%s task=%s objective=%s seed=%s lambda=%s out=%s\n' \
    "$ARM" "$INDEX" "$OBJECTIVE" "$SEED" "$WEIGHT" "$OUT"
  exit 0
fi

if [[ -s "$OUT/last.pt" && -s "$OUT/metrics.json" ]]; then
  echo "reusing $OUT"
  exit 0
fi

cd "$ROOT"
mkdir -p logs "$OUT"
exec "$PY" train_spatial_map_sae.py \
  --architecture flat --autoencoder-type vector --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count 8192 --k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight "$WEIGHT" --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 \
  --seed "$SEED" --init-seed "$SEED" --data-seed 2026
