#!/bin/bash
#SBATCH --job-name=read-eps-freq-all
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

python analyze_read_feature_eps_frequency.py \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --sites input,middle \
  --features-per-group 20 \
  --clean-max 200 \
  --noisy-min 800 \
  --metrics-root runs \
  --out-root runs/read_feature_eps_frequency_clean_vs_noisy_all_classes \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --input-sae-ckpt runs/sae_block_inputs_16x_k32_long/last.pt \
  --middle-sae-ckpt runs/sae_middle_reads_16x_k32_long/last.pt \
  --data-dir data \
  --num-images 12 \
  --precision fp32 \
  --seed 2026
