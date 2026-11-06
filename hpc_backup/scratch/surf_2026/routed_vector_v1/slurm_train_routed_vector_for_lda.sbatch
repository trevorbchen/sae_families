#!/bin/bash
# DRAFT, not yet submitted. Trains the five-branch routed ("combined") SAE on the matched-512 VECTOR
# autoencoder latents, i.e. the same 512-d space / standardization / lambda=10 as the vector_16x_k32
# single-objective SAEs, so its per-feature vectors can be projected with Figure 11's LDA.
# (The existing routed checkpoints are on the 64x8x8 spatial AE and are NOT comparable.)
# No separate "extract" job is needed: the routed checkpoint is ~60 MB, and explore_combined_lda.py
# does the feature extraction locally from the already-local latent caches.
#
# One-time setup on the cluster (login node, no GPU):
#   mkdir -p /resnick/scratch/tbchen/surf_2026/routed_vector_v1/{code,logs,runs}
#   scp  train_independent_routed_union_vector.py  ->  .../routed_vector_v1/code/
# Smoke test first (about a minute, also tells you seconds/step for the real --time):
#   sbatch --array=0 --export=ALL,STEPS=200 --time=00:20:00 slurm_train_routed_vector_for_lda.sbatch
# Real run: sbatch slurm_train_routed_vector_for_lda.sbatch   (3 seeds, one task each)
#SBATCH --job-name=routed-vec-lda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/routed_vector_v1/logs/routed-vec-%A_%a.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/routed_vector_v1/logs/routed-vec-%A_%a.err
set -euo pipefail

SEEDS=(2030 2031 2032)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-20000}"

BASE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1"
EXP="/resnick/scratch/tbchen/surf_2026/routed_vector_v1"
CODE="$EXP/code"
ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
OUT="$EXP/runs/mse_plus_all_four_vector_seed$SEED"

mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$CODE:$ROOT"

# Same latent cache + AE as the vector_16x_k32 arm (see slurm_matched512_sae_train.sbatch).
# Dictionary/k mirror the existing routed model: MSE 8192/k32 + 4 specialists of 2048/k8.
# --transform-weight 10 = the lambda=10 used by the single-objective arm in Figure 11.
exec "$PY" "$CODE/train_independent_routed_union_vector.py" \
  --objectives laplacian,fourier_low,fourier_mid,fourier_high \
  --data "$BASE/latents/vector_ffhq_32768.pt" \
  --autoencoder "$BASE/autoencoders/vector/last.pt" \
  --out-dir "$OUT" \
  --mse-dict-size 8192 --mse-k 32 \
  --specialist-dict-size 8192 --specialist-k 32 \
  --transform-weight 10 \
  --steps "$STEPS" --batch-size 32 --lr 3e-4 --warmup 500 \
  --grad-clip 1.0 --calibration-examples 2048 \
  --reinit-every 1000 --dead-after 3000 --max-rescue 64 \
  --validate-every 250 --seed "$SEED"
