#!/bin/bash
# Interactive BSF block-inspection GUI. H200 is requested for the largest
# currently supported GPU class; the job can be moved to another GPU if needed.
#SBATCH --job-name=bsf-feature-gui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bsf-feature-gui-%j.out
#SBATCH --error=logs/bsf-feature-gui-%j.err
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"
mkdir -p logs
PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
while true; do
  "$PY" bsf_feature_permutation_gui.py \
    --repo official_diffae \
    --checkpoint official_diffae/checkpoints/ffhq128_autoenc/last.ckpt \
    --bsf results/zsem_bsf/32x_block8_top4/last.pt \
    --codes results/zsem_codes/ffhq_celeba_train.pt \
    --port 8768 --examples 12 --batch-size 512
  status=$?
  echo "BSF GUI exited with status $status; restarting in 2 seconds" >&2
  sleep 2
done
