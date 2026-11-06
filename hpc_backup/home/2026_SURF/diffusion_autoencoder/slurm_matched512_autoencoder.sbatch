#!/bin/bash
#SBATCH --job-name=m512-ae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/matched512-ae-%j.out
#SBATCH --error=logs/matched512-ae-%j.err
set -euo pipefail

: "${LATENT_SOURCE:?LATENT_SOURCE must be spatial or vector}"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
DATA="/resnick/scratch/tbchen/surf_2026/zsem_transformed/ffhq_32768.pt"
OUT="$BASE/autoencoders/$LATENT_SOURCE"
CACHE="$BASE/latents/${LATENT_SOURCE}_ffhq_32768.pt"

if [[ -s "$OUT/last.pt" && -s "$CACHE" ]]; then
  echo "reusing matched $LATENT_SOURCE autoencoder and cache"
  exit 0
fi

cd "$ROOT"
mkdir -p logs "$OUT" "$(dirname "$CACHE")"
if [[ "$LATENT_SOURCE" == spatial ]]; then
  exec "$PY" train_spatial_conv_autoencoder.py \
    --data "$DATA" --out-dir "$OUT" --latent-cache "$CACHE" \
    --latent-channels 8 --base-channels 64 --steps 50000 \
    --batch-size 64 --lr 2e-4 --warmup 1000 --seed 2030
elif [[ "$LATENT_SOURCE" == vector ]]; then
  exec "$PY" train_vector_conv_autoencoder.py \
    --data "$DATA" --out-dir "$OUT" --latent-cache "$CACHE" \
    --latent-dim 512 --base-channels 64 --steps 50000 \
    --batch-size 64 --lr 2e-4 --warmup 1000 --seed 2030
else
  echo "unknown LATENT_SOURCE=$LATENT_SOURCE" >&2
  exit 2
fi
