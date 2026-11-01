"""Re-render the by-objective/by-seed/by-fidelity LDA panels for FFHQ/CelebA/
FairFace from the already-saved *_lda_embedding.npz caches (see
explore_feature_lda.py's docstring -- it saves these "for reuse"), with the
dataset name dropped from the in-image title (e.g. "ffhq_heldout, LDA by
objective" -> "By objective"): the dataset is already named in the LaTeX
caption, so repeating it in the plot title was redundant. Does not refit
LDA or touch the underlying embedding at all -- titles, fonts, and layout
only, reusing explore_feature_lda.py's own plot_categorical_contours /
plot_fidelity functions so the rendering logic is identical to the
original figures.
"""
import argparse
from pathlib import Path

import numpy as np

from explore_feature_lda import (
    OBJECTIVES, OBJECTIVE_COLORS, OBJECTIVE_LABELS, SEED_COLORS, SEED_LABELS,
    plot_categorical_contours, plot_fidelity,
)

parser = argparse.ArgumentParser()
parser.add_argument("--embedding", type=Path, required=True, help="*_lda_embedding.npz")
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--out-prefix", required=True)
args = parser.parse_args()

data = np.load(args.embedding)
emb2d = data["embedding"][:, :2]
objective_arr = data["objective"]
seed_arr = data["seed"]
fidelity_arr = data["fidelity"]

args.out_dir.mkdir(parents=True, exist_ok=True)
prefix = args.out_prefix

plot_categorical_contours(emb2d, objective_arr, OBJECTIVE_COLORS, OBJECTIVE_LABELS, OBJECTIVES,
                           "By objective", args.out_dir / f"{prefix}_lda_by_objective.png")
plot_categorical_contours(emb2d, seed_arr, SEED_COLORS, SEED_LABELS, list(SEED_COLORS),
                           "By seed", args.out_dir / f"{prefix}_lda_by_seed.png")
plot_fidelity(emb2d, fidelity_arr,
              "By fidelity", args.out_dir / f"{prefix}_lda_by_fidelity.png")

print(f"held_out_accuracy from cache: {float(data['held_out_accuracy']):.1%}")
