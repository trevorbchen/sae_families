#!/bin/bash
#SBATCH --job-name=zsem-obj-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/zsem-obj-audit-%j.out
#SBATCH --error=logs/zsem-obj-audit-%j.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRATCH="/resnick/scratch/tbchen/surf_2026/zsem_objective_audit"
DINO="/resnick/scratch/tbchen/models/dinov2-base"
PILOT_EXAMPLES="${PILOT_EXAMPLES:-1024}"
PREP_BATCH_SIZE="${PREP_BATCH_SIZE:-32}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-4}"
cd "$ROOT"
mkdir -p logs "$SCRATCH" results/zsem_objective_audit

PILOT_DATA="$SCRATCH/pilot_${PILOT_EXAMPLES}.pt"
if [[ ! -f "$PILOT_DATA" ]]; then
  "$PY" prepare_zsem_objective_pilot.py \
    --repo official_diffae \
    --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
    --codes results/zsem_codes/ffhq_celeba_train.pt \
    --out "$PILOT_DATA" \
    --examples "$PILOT_EXAMPLES" --batch-size "$PREP_BATCH_SIZE" \
    --steps 20 --seed 2026
fi

exec "$PY" audit_zsem_objective_geometry.py \
  --repo official_diffae \
  --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
  --sae results/zsem_sae/final_32x_k128_aux_reinit_merged/last.pt \
  --pilot-data "$PILOT_DATA" \
  --dino "$DINO" \
  --out-dir "results/zsem_objective_audit/pilot_${PILOT_EXAMPLES}" \
  --batch-size "$AUDIT_BATCH_SIZE" --seed 2027
