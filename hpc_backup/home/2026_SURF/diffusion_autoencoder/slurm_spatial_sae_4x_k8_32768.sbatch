#!/bin/bash
#SBATCH --job-name=spatial-sae-4xk8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-71%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/logs/spatial-sae-%A_%a.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768/logs/spatial-sae-%A_%a.err
set -euo pipefail

PURE_OBJECTIVES=(
  latent_mse laplacian fourier_bands fourier_low fourier_mid fourier_high
)
PROPER_OBJECTIVES=(
  latent_mse weighted_laplacian weighted_fourier_bands
  weighted_fourier_low weighted_fourier_mid weighted_fourier_high
)
LABELS=(latent_mse laplacian fourier_bands fourier_low fourier_mid fourier_high)
ARCHITECTURES=(flat conv)

INDEX="$SLURM_ARRAY_TASK_ID"
ARCHITECTURE_INDEX=$((INDEX / 36))
WITHIN_ARCHITECTURE=$((INDEX % 36))
FORMULATION_INDEX=$((WITHIN_ARCHITECTURE / 18))
WITHIN_FORMULATION=$((WITHIN_ARCHITECTURE % 18))
OBJECTIVE_INDEX=$((WITHIN_FORMULATION / 3))
SEED_INDEX=$((WITHIN_FORMULATION % 3))

ARCHITECTURE="${ARCHITECTURES[$ARCHITECTURE_INDEX]}"
INIT_SEED=$((2030 + SEED_INDEX))
DATA_SEED=2026
LABEL="${LABELS[$OBJECTIVE_INDEX]}"
if (( FORMULATION_INDEX == 0 )); then
  FORMULATION="transformed_loss"
  OBJECTIVE="${PURE_OBJECTIVES[$OBJECTIVE_INDEX]}"
else
  FORMULATION="proper_loss"
  OBJECTIVE="${PROPER_OBJECTIVES[$OBJECTIVE_INDEX]}"
fi

if [[ "$ARCHITECTURE" == "flat" ]]; then
  FEATURE_COUNT=16384
else
  FEATURE_COUNT=256
fi

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/spatial_ae_experiment_4x_k8_32768"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
LATENTS="$EXP/data/ffhq_32768_latents.pt"
AE="$ROOT/results/spatial_conv_ae/last.pt"
OUT="$EXP/results/$ARCHITECTURE/$FORMULATION/$LABEL/seed_$INIT_SEED"
mkdir -p "$EXP/logs" "$OUT"
cd "$ROOT"

exec "$PY" train_spatial_map_sae.py \
  --architecture "$ARCHITECTURE" --objective "$OBJECTIVE" \
  --data "$DATA" --latent-cache "$LATENTS" --autoencoder "$AE" \
  --out-dir "$OUT" --feature-count "$FEATURE_COUNT" --k 8 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --transform-weight 1.0 --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 256 \
  --seed "$INIT_SEED" --init-seed "$INIT_SEED" --data-seed "$DATA_SEED"
