#!/bin/bash
#SBATCH --job-name=kimi-tandem-200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/kimi-tandem-200-%j.out
#SBATCH --error=logs/kimi-tandem-200-%j.err
set -euo pipefail

# Kimi's vision attention briefly needs a large contiguous allocation. The GUI
# shares this GPU, so avoid losing several GiB to allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"
ENDPOINT="${GUI_ENDPOINT:-http://hpc-sm-03-09:8813}"
OUT="/resnick/scratch/tbchen/surf_2026/proper_residual_feature_review/kimi_tandem_first200.jsonl"

cd "$ROOT"
mkdir -p logs "$(dirname "$OUT")"

exec "$PY" label_residual_gui_kimi.py \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --out "$OUT" \
  --end 200 \
  --examples 5 \
  --targets tandem_canonical
