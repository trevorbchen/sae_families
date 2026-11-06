#!/bin/bash
#SBATCH --job-name=kimi-freq-strict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-29%8
#SBATCH --output=logs/kimi-freq-strict-%A_%a.out
#SBATCH --error=logs/kimi-freq-strict-%A_%a.err
set -euo pipefail

ROOT="/home/tbchen/2026 SURF/diffusion_autoencoder"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"
ENDPOINT="${GUI_ENDPOINT:?submit with --export=ALL,GUI_ENDPOINT=http://NODE:8813}"
PRIOR="/resnick/scratch/tbchen/surf_2026/frequency_family_labels_top500_v2"
OUT_ROOT="/resnick/scratch/tbchen/surf_2026/frequency_family_labels_top500_v3_strict"

SOURCES=(
  direct_mse
  grouped_objective_laplacian
  grouped_objective_fourier_low
  grouped_objective_fourier_mid
  grouped_objective_fourier_high
  tandem_canonical
)

SOURCE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
START=$((CHUNK_INDEX * 100))
END=$((START + 100))
SOURCE="${SOURCES[$SOURCE_INDEX]}"
OUT="$OUT_ROOT/${SOURCE}_${START}_${END}.jsonl"

cd "$ROOT"
mkdir -p logs "$OUT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY" label_residual_gui_kimi.py \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --out "$OUT" \
  --start "$START" --end "$END" \
  --examples 5 --max-new-tokens 520 \
  --per-subject --retry-from "$PRIOR" \
  --replace-kimi-notes \
  --targets "$SOURCE"
