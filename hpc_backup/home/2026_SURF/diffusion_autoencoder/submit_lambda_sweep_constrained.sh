#!/bin/bash
set -euo pipefail

EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_constrained_vector_convae_v1"
SOURCE="/resnick/scratch/tbchen/surf_2026/vector_ae_experiment_4x_k8_32768/results/proper_loss"
mkdir -p "$EXP/checkpoints/mse_control/latent_mse" "$EXP/checkpoints/lambda_1p0"

for SEED in 2030 2031 2032; do
  ln -sfn "$SOURCE/latent_mse/seed_$SEED" \
    "$EXP/checkpoints/mse_control/latent_mse/seed_$SEED"
done
for OBJECTIVE in laplacian fourier_low fourier_mid fourier_high; do
  for SEED in 2030 2031 2032; do
    TARGET="$EXP/checkpoints/lambda_1p0/$OBJECTIVE/seed_$SEED"
    mkdir -p "$(dirname "$TARGET")"
    ln -sfn "$SOURCE/$OBJECTIVE/seed_$SEED" "$TARGET"
  done
done

TRAIN_JOB=$(sbatch --parsable slurm_lambda_sweep_constrained_train.sbatch)
TRAIN_JOB="${TRAIN_JOB%%;*}"
GEOMETRY_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" slurm_lambda_sweep_constrained_geometry.sbatch)
GEOMETRY_JOB="${GEOMETRY_JOB%%;*}"
ALIGNMENT_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" slurm_lambda_sweep_constrained_alignment.sbatch)
ALIGNMENT_JOB="${ALIGNMENT_JOB%%;*}"
FINAL_JOB=$(sbatch --parsable --dependency="afterok:$GEOMETRY_JOB:$ALIGNMENT_JOB" slurm_lambda_sweep_constrained_finalize.sbatch)
FINAL_JOB="${FINAL_JOB%%;*}"

printf 'train=%s\ngeometry=%s\nalignment=%s\nfinal=%s\n' \
  "$TRAIN_JOB" "$GEOMETRY_JOB" "$ALIGNMENT_JOB" "$FINAL_JOB" | tee "$EXP/submission_jobs.txt"
