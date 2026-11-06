#!/bin/bash
#SBATCH --job-name=cls-blk-in-common
#SBATCH --account=perona
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-57%12
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
module load cuda/11.8.0-gcc-11.3.1-i5witig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
. .venv/bin/activate

TASK_FILE="remaining_common_gpu_tasks.tsv"
TASK_LINE=$(awk -v n="$((SLURM_ARRAY_TASK_ID + 1))" 'NR == n { print }' "${TASK_FILE}")
read -r CLASS_ID CLASS_NAME BLOCK_ID <<< "${TASK_LINE}"

OUT_ROOT="runs/block_inputs_onoff_ablation_16x_k32_class_${CLASS_ID}_${CLASS_NAME}"
BLOCK_DIR="${OUT_ROOT}/block$(printf "%02d" "${BLOCK_ID}")_input"
RESULT="${BLOCK_DIR}/block$(printf "%02d" "${BLOCK_ID}")_input_onoff_ablation.pt"
mkdir -p "${OUT_ROOT}"

if [ -f "${RESULT}" ]; then
  echo "skip existing result: ${RESULT}"
  exit 0
fi

echo "task=${SLURM_ARRAY_TASK_ID} class_id=${CLASS_ID} class_name=${CLASS_NAME} block=${BLOCK_ID} out=${OUT_ROOT}"
python block_input_onoff_ablation_dog.py \
  --block-index "${BLOCK_ID}" \
  --ckpt runs/dit_s2_cifar10/last.pt \
  --sae-ckpt runs/sae_block_inputs_16x_k32_long/last.pt \
  --data-dir data \
  --out-root "${OUT_ROOT}" \
  --num-images 12 \
  --class-id "${CLASS_ID}" \
  --t-values 0,100,200,300,400,500,600,700,800,900,999 \
  --features-per-forward 8 \
  --forward-batch-size 96 \
  --precision bf16 \
  --seed 2026
