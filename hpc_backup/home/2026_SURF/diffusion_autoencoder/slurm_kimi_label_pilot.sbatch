#!/bin/bash
#SBATCH --job-name=kimi-label-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kimi-label-pilot-%j.out
#SBATCH --error=logs/kimi-label-pilot-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs results/kimi_feature_labels
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
MODEL="/resnick/scratch/tbchen/models/Kimi-VL-A3B-Instruct"
COMMON=(--repo "/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae"
  --model "$MODEL"
  --checkpoint "/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae/checkpoints/ffhq128_autoenc/last.ckpt"
  --codes "/home/tbchen/2026 SURF/diffusion_autoencoder/official_diffae/checkpoints/ffhq128_autoenc/latent.pkl"
  --examples 3 --start 0 --end 16)
"$PY" label_features_kimi.py "${COMMON[@]}" --kind topk \
  --sae "/home/tbchen/2026 SURF/diffusion_autoencoder/results/zsem_sae/8x_k16/last.pt" \
  --out results/kimi_feature_labels/8x_k16.jsonl \
  --panel-dir results/kimi_feature_labels/8x_k16_panels
