#!/bin/bash
#SBATCH --job-name=probe-sae-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --array=0-15%8
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/probe-sae-seeds-%A_%a.out
#SBATCH --error=logs/probe-sae-seeds-%A_%a.err
set -euo pipefail

SEEDS=(2031 2032)
NAMES=(
  direct_mse all_mse_2branch plus_laplacian plus_fourier_low
  plus_fourier_mid plus_fourier_high all_mse_5branch tandem_all4
)
OBJECTIVES=(
  none laplacian laplacian fourier_low fourier_mid fourier_high
  laplacian,fourier_low,fourier_mid,fourier_high
  laplacian,fourier_low,fourier_mid,fourier_high
)

INDEX="$SLURM_ARRAY_TASK_ID"
SEED="${SEEDS[$((INDEX / 8))]}"
CONFIG="$((INDEX % 8))"
NAME="${NAMES[$CONFIG]}"
OBJECTIVE="${OBJECTIVES[$CONFIG]}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
EXP="/resnick/scratch/tbchen/surf_2026/sae_linear_probe_seed_replicates_v1"
OUT="$EXP/runs/$NAME/seed_$SEED"
DATA="/resnick/scratch/tbchen/surf_2026/spatial_conv_ae/ffhq_16384_latents.pt"

cd "$ROOT"
mkdir -p logs "$OUT"

if [[ "$CONFIG" == 0 ]]; then
  [[ -s "$OUT/best_valid.pt" ]] && exit 0
  exec "$PY" train_grouped_spatial_ae_sae.py \
    --mode regular --checkpoint results/spatial_conv_ae/last.pt \
    --data "$DATA" --out-dir "$OUT" \
    --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
    --grad-clip 1.0 --transform-weight 1.0 --calibration-examples 2048 \
    --aux-coeff 0.1 --reinit-every 1000 --dead-after 3000 \
    --max-rescue-per-group 64 --validate-every 250 \
    --checkpoint-every 1000 --state-every 500 --trace-flush-every 100 \
    --seed "$SEED" --split-seed 2026
fi

[[ -s "$OUT/last.pt" ]] && exit 0
CONTROL=()
if [[ "$CONFIG" == 1 || "$CONFIG" == 6 ]]; then
  CONTROL+=(--all-mse-control)
fi
exec "$PY" train_independent_routed_union_sae.py \
  --objectives "$OBJECTIVE" "${CONTROL[@]}" \
  --data "$DATA" --autoencoder results/spatial_conv_ae/last.pt \
  --out-dir "$OUT" \
  --mse-dict-size 8192 --mse-k 32 \
  --specialist-dict-size 8192 --specialist-k 32 \
  --steps 20000 --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed "$SEED"
