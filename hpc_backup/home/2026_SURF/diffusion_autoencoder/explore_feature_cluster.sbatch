#!/bin/bash
#SBATCH --job-name=feat-cluster-explore
#SBATCH --partition=expansion
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:16:00
#SBATCH --output=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-cluster-explore-%j.out
#SBATCH --error=/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/logs/feat-cluster-explore-%j.err
set -euo pipefail

PY="/home/tbchen/2026 SURF/dit_cifar10_pixel/.venv/bin/python"
SCRIPT="/home/tbchen/2026 SURF/diffusion_autoencoder/explore_feature_cluster_objective_comparison.py"
USAGE="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/feature_usage_vector_16x_k32_lambda10.pt"
OUTDIR="/resnick/scratch/tbchen/surf_2026/sae_matched_512_v1/tsne/figures"

"$PY" "$SCRIPT" \
  --usage "$USAGE" \
  --source ffhq_heldout \
  --out-prefix "$OUTDIR/ffhq_heldout_cluster_explore"
