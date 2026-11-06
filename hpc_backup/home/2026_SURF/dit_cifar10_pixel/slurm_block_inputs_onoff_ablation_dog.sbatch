#!/bin/bash
#SBATCH --job-name=dog-blk-in-onoff
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs runs/block_inputs_onoff_ablation_16x_k32_dog
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate
python block_input_onoff_ablation_dog.py   --block-index "${SLURM_ARRAY_TASK_ID}"   --ckpt runs/dit_s2_cifar10/last.pt   --sae-ckpt runs/sae_block_inputs_16x_k32_long/last.pt   --data-dir data   --out-root runs/block_inputs_onoff_ablation_16x_k32_dog   --num-images 12   --class-id 5   --t-values 0,100,200,300,400,500,600,700,800,900,999   --features-per-forward 8   --forward-batch-size 96   --precision bf16   --seed 2026
