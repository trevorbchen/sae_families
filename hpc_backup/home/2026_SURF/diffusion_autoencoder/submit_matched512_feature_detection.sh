#!/bin/bash
# One-shot driver: submits planted-feature cache prep, then detection runs
# for vector_4x_k8 and vector_16x_k32 across their 5 core lambdas (0.1-10,
# 3 seeds). Spatial arms and lambda 30/100 are deliberately excluded here --
# spatial isn't trained yet, and 30/100 will follow once the extra-seed
# extension (see slurm_matched512_v16_extra_seeds_*.sbatch) finishes syncing
# to 3 seeds. Re-run manually for those once ready.
set -euo pipefail
cd "/home/tbchen/2026 SURF/diffusion_autoencoder"

PREP=$(sbatch --parsable slurm_matched512_prepare_planted_features.sbatch)
echo "prep=$PREP"

MANIFEST="results/family_sae/lambda_sweep/feature_detection_submission_jobs.csv"
mkdir -p "$(dirname "$MANIFEST")"
echo "arm,lambda_tag,job_id" > "$MANIFEST"

for ARM in vector_4x_k8 vector_16x_k32; do
  for TAG in 0p1 0p3 1p0 3p0 10p0; do
    JOB=$(sbatch --parsable --dependency=afterok:$PREP \
      --export=ALL,ARM=$ARM,LAMBDA_TAG=$TAG,SEEDS="2030 2031 2032" \
      slurm_matched512_feature_detection.sbatch)
    echo "$ARM,$TAG,$JOB" | tee -a "$MANIFEST"
  done
done
