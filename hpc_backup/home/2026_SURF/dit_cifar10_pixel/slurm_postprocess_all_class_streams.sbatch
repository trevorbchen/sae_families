#!/bin/bash
#SBATCH --job-name=post-cyclic-all
#SBATCH --account=perona
#SBATCH --partition=expansion
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "/home/tbchen/2026 SURF/dit_cifar10_pixel"
mkdir -p logs
module purge
module load python/3.10.12-gcc-11.3.1-djdhgrq
. .venv/bin/activate

CLASSES=(0_airplane 1_automobile 2_bird 3_cat 4_deer 5_dog 6_frog 7_horse 8_ship 9_truck)

INPUT_ROOTS=()
MIDDLE_ROOTS=()
ATTN_ROOTS=()
MLP_ROOTS=()
for cls in "${CLASSES[@]}"; do
  if [ "${cls}" = "5_dog" ]; then
    INPUT_ROOTS+=("runs/block_inputs_onoff_ablation_16x_k32_dog")
  else
    INPUT_ROOTS+=("runs/block_inputs_onoff_ablation_16x_k32_class_${cls}")
  fi
  MIDDLE_ROOTS+=("runs/block_middle_reads_onoff_ablation_16x_k32_class_${cls}")
  ATTN_ROOTS+=("runs/block_attn_writes_onoff_ablation_16x_k32_class_${cls}")
  MLP_ROOTS+=("runs/block_mlp_writes_onoff_ablation_16x_k32_class_${cls}")
done

python postprocess_class_ablation.py --site input "${INPUT_ROOTS[@]}"
python postprocess_class_ablation.py --site middle "${MIDDLE_ROOTS[@]}"
python postprocess_class_ablation.py --site attn "${ATTN_ROOTS[@]}"
python postprocess_class_ablation.py --site mlp "${MLP_ROOTS[@]}"
