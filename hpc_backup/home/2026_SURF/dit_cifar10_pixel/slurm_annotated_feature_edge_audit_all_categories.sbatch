#!/bin/bash
#SBATCH --job-name=feature-edge-all
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

python audit_annotated_feature_edges.py \
  --notes-json "sae_feature_review_notes (1).json" \
  --out-root runs/annotated_feature_edge_audit_all_categories_auto \
  --include-categories outline,color,noise,background,subject,brightness \
  --max-features 120 \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --num-images 5 \
  --max-overlay-rows 6 \
  --subject-method auto \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --block0-sae-ckpt runs/sae_block0_writes_16x_k32/last.pt \
  --remaining-sae-ckpt runs/sae_remaining_block_writes_16x_k32/last.pt \
  --input-sae-ckpt runs/sae_block_inputs_16x_k32_long/last.pt \
  --middle-sae-ckpt runs/sae_middle_reads_16x_k32_long/last.pt \
  --data-dir data \
  --precision fp32 \
  --seed 2026 \
  --image-size 96
