#!/bin/bash
#SBATCH --job-name=kimi-strength-ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/kimi-strength-ab-%A_%a.out
#SBATCH --error=logs/kimi-strength-ab-%A_%a.err
set -euo pipefail

cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/kimi_feature_labels

MODES=(grid subjects)
MODE="${MODES[$SLURM_ARRAY_TASK_ID]}"
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"

"$PY" label_features_kimi.py \
  --repo "/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae" \
  --model "$MODEL" \
  --checkpoint "/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae/checkpoints/ffhq128_autoenc/last.ckpt" \
  --codes "/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae/checkpoints/ffhq128_autoenc/latent.pkl" \
  --kind topk \
  --sae "/home/tbchen/2026 SURF/diffusion_autoencoder/results/zsem_sae/8x_k16/last.pt" \
  --start 0 --end 16 --examples 3 \
  --multipliers -10 -4 -2 0 2 4 10 \
  --panel-mode "$MODE" \
  --out "results/kimi_feature_labels/8x_k16_7strength_${MODE}.jsonl" \
  --panel-dir "results/kimi_feature_labels/8x_k16_7strength_${MODE}_panels"
