#!/bin/bash
#SBATCH --job-name=transport-freq
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

python analyze_transport_feature_frequency.py \
  --classes 0,1,2,3,4,5,6,7,8,9 \
  --site-kinds input,middle,attn,mlp \
  --features-per-group 20 \
  --clean-source-t 100 \
  --noisy-source-t 900 \
  --target-s 0 \
  --out-root runs/transport_feature_frequency_clean_vs_noisy_all_classes \
  --ckpt runs/dit_s2_cifar10_consistency_transport/last.pt \
  --input-sae-ckpt runs/sae_transport_block_inputs_16x_k32/last.pt \
  --middle-sae-ckpt runs/sae_transport_middle_reads_16x_k32/last.pt \
  --writes-sae-ckpt runs/sae_transport_writes_16x_k32/last.pt \
  --data-dir data \
  --num-images 12 \
  --precision bf16 \
  --seed 2026
