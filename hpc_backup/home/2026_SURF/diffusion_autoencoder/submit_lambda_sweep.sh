#!/bin/bash
set -euo pipefail

EXP="/resnick/scratch/tbchen/surf_2026/sae_lambda_sweep_flat_convae_v1"
SOURCE="/home/tbchen/2026 SURF/diffusion_autoencoder/results/spatial_map_sae_weighted_seed_replicates/flat"
for OBJECTIVE in laplacian fourier_low fourier_mid fourier_high; do
  SOURCE_OBJECTIVE="weighted_$OBJECTIVE"
  for SEED in 2030 2031 2032; do
    TARGET="$EXP/checkpoints/lambda_1p0/$OBJECTIVE/seed_$SEED"
    mkdir -p "$(dirname "$TARGET")"
    ln -sfn "$SOURCE/$SOURCE_OBJECTIVE/seed_$SEED" "$TARGET"
  done
done

TRAIN_JOB=$(sbatch --parsable slurm_lambda_sweep_train.sbatch)
TRAIN_JOB="${TRAIN_JOB%%;*}"
GEOMETRY_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" slurm_lambda_sweep_geometry.sbatch)
GEOMETRY_JOB="${GEOMETRY_JOB%%;*}"
ALIGNMENT_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" slurm_lambda_sweep_alignment.sbatch)
ALIGNMENT_JOB="${ALIGNMENT_JOB%%;*}"
FINAL_JOB=$(sbatch --parsable --dependency="afterok:$GEOMETRY_JOB:$ALIGNMENT_JOB" slurm_lambda_sweep_finalize.sbatch)
FINAL_JOB="${FINAL_JOB%%;*}"

printf 'train=%s\ngeometry=%s\nalignment=%s\nfinal=%s\n' \
  "$TRAIN_JOB" "$GEOMETRY_JOB" "$ALIGNMENT_JOB" "$FINAL_JOB"
