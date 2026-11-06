#!/bin/bash
#SBATCH --job-name=m512-feature-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/m512-feature-gui-%j.out
#SBATCH --error=logs/m512-feature-gui-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
: "${ARM:=vector_16x_k32}"
: "${LAMBDA_TAG:=100p0}"
: "${AE_TYPE:=vector}"
: "${SEED:=2030}"
: "${PORT:=8770}"

echo "node: $(hostname)"
echo "arm=$ARM lambda_tag=$LAMBDA_TAG autoencoder_type=$AE_TYPE seed=$SEED port=$PORT"

ANNOTATIONS="results/family_sae/lambda_sweep/$ARM/general/feature_gui_annotations_lambda_${LAMBDA_TAG}.json"
mkdir -p "$(dirname "$ANNOTATIONS")"

exec "$PY" matched512_feature_permutation_gui.py \
  --checkpoint-root "$BASE/experiments/$ARM/checkpoints/lambda_$LAMBDA_TAG" \
  --autoencoder "$BASE/autoencoders/$AE_TYPE/last.pt" \
  --autoencoder-type "$AE_TYPE" \
  --latent-cache "$BASE/latents/${AE_TYPE}_ffhq_32768.pt" \
  --annotations "$ANNOTATIONS" \
  --seed "$SEED" \
  --port "$PORT" \
  --label "$ARM lambda=$LAMBDA_TAG"
