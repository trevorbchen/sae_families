#!/bin/bash
#SBATCH --job-name=lambda-family-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-62%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lambda-family-train-%A_%a.out
#SBATCH --error=logs/lambda-family-train-%A_%a.err
set -euo pipefail

: "${VARIANT:?VARIANT must be exported by the submission script}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
CODE="$ROOT"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_matched_32768_v1"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
EXP="$BASE/$VARIANT"
ARCHITECTURE="flat"
EXTRA=()

case "$VARIANT" in
  spatial_flat_k32)
    FEATURE_COUNT=4096; K=32
    LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/data/ffhq_32768_latents.pt"
    AE="$ROOT/results/spatial_conv_ae/last.pt"
    ;;
  spatial_flat_k8)
    FEATURE_COUNT=16384; K=8
    LATENTS="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/data/ffhq_32768_latents.pt"
    AE="$ROOT/results/spatial_conv_ae/last.pt"
    ;;
  vector_1d_k32)
    FEATURE_COUNT=4096; K=32
    CODE="/resnick/scratch/tbchen/surf_2026/vector_ae_code"
    LATENTS="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768/data/ffhq_32768_latents.pt"
    AE="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment/results/vector_conv_ae/last.pt"
    EXTRA+=(--autoencoder-type vector)
    ;;
  vector_1d_k8)
    FEATURE_COUNT=2048; K=8
    CODE="/resnick/scratch/tbchen/surf_2026/vector_ae_code"
    LATENTS="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768/data/ffhq_32768_latents.pt"
    AE="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment/results/vector_conv_ae/last.pt"
    EXTRA+=(--autoencoder-type vector)
    ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

SEEDS=(2030 2031 2032)
LAMBDAS=(0.1 0.3 1.0 3.0 10.0)
LAMBDA_TAGS=(0p1 0p3 1p0 3p0 10p0)
OBJECTIVES=(weighted_laplacian weighted_fourier_low weighted_fourier_mid weighted_fourier_high)
DIRECTORIES=(laplacian fourier_low fourier_mid fourier_high)

INDEX="$SLURM_ARRAY_TASK_ID"
if (( INDEX < 3 )); then
  SEED="${SEEDS[$INDEX]}"
  OBJECTIVE="latent_mse"
  WEIGHT=0
  OUT="$EXP/checkpoints/mse_control/latent_mse/seed_$SEED"
else
  SPECIALIST_INDEX=$((INDEX - 3))
  LAMBDA_INDEX=$((SPECIALIST_INDEX / 12))
  WITHIN_LAMBDA=$((SPECIALIST_INDEX % 12))
  OBJECTIVE_INDEX=$((WITHIN_LAMBDA / 3))
  SEED_INDEX=$((WITHIN_LAMBDA % 3))
  SEED="${SEEDS[$SEED_INDEX]}"
  OBJECTIVE="${OBJECTIVES[$OBJECTIVE_INDEX]}"
  DIRECTORY="${DIRECTORIES[$OBJECTIVE_INDEX]}"
  WEIGHT="${LAMBDAS[$LAMBDA_INDEX]}"
  TAG="${LAMBDA_TAGS[$LAMBDA_INDEX]}"
  OUT="$EXP/checkpoints/lambda_$TAG/$DIRECTORY/seed_$SEED"
fi

if [[ -s "$OUT/last.pt" ]]; then
  echo "reusing $OUT/last.pt"
  exit 0
fi

mkdir -p "$ROOT/logs" "$OUT"
cd "$CODE"
export PYTHONPATH="$CODE:$ROOT"
exec "$PY" "$CODE/train_spatial_map_sae.py" \
  --architecture "$ARCHITECTURE" "${EXTRA[@]}" --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count "$FEATURE_COUNT" --k "$K" \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight "$WEIGHT" --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 \
  --seed "$SEED" --init-seed "$SEED" --data-seed 2026
