#!/bin/bash
# Matched BSF comparison: same data, steps, scalar dictionary size, and 32
# active scalar dimensions per example.
#SBATCH --job-name=bsf-variants
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/bsf-variants-%A_%a.out
#SBATCH --error=logs/bsf-variants-%A_%a.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/zsem_bsf/variants
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
case "$SLURM_ARRAY_TASK_ID" in
  0) NAME="vanilla_d8_k4"; VARIANT=vanilla; G=8; K=4 ;;
  1) NAME="grassmannian_d8_k4"; VARIANT=grassmannian; G=8; K=4 ;;
  2) NAME="vanilla_d4_k8"; VARIANT=vanilla; G=4; K=8 ;;
  3) NAME="vanilla_d2_k16"; VARIANT=vanilla; G=2; K=16 ;;
esac
exec "$PY" train_zsem_bsf_variants.py \
  --codes results/zsem_codes/ffhq_celeba_train.pt \
  --out-dir "results/zsem_bsf/variants/$NAME" \
  --variant "$VARIANT" --dict-mult 32 --block-dim "$G" --top-blocks "$K" \
  --steps 100000 --batch-size 1024 --lr 3e-4 --warmup 2000
