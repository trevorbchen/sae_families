"""Re-render the appendix EV-without-anchor bar chart (CVPR2026 Figure,
\\label{fig:ev-without-anchor}) from the already-computed pure_ev_table.csv,
with the same larger-font/title/legend treatment as make_fig03a.py's
with-anchor sibling -- the legacy render (aslides/make_anchor_ev_plots.py's
pure_ev_table branch) had no title, no legend, font 11, and a label/error-bar
overlap glitch at the large negative bars. Reuses make_fig03a.py's exact
color palette and bar-labeling logic; the only differences are the y-limits
(this table's bars run deeply negative) and the title.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--table", type=Path, required=True,
                     help="pure_ev_table.csv, e.g. results/family_sae/grouped_objective_sae/"
                          "standalone_objectives/convae_4096_k32/tables/pure_ev_table.csv")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
TABLE = args.table
OUT = args.output

INK = "#17202A"
TEAL = "#0F8587"
GRAY = "#6B7280"
GRID = "#D9DEE2"

df = pd.read_csv(TABLE)
x = range(len(df))
means = df["mean_ev_percent"].to_numpy()
lower = means - df["ci95_low_percent"].to_numpy()
upper = df["ci95_high_percent"].to_numpy() - means
colors = [GRAY] + [TEAL] * (len(df) - 1)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
})
fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
bars = ax.bar(x, means, color=colors, width=0.67, edgecolor="none", zorder=2)
ax.errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor=INK, elinewidth=1.6, capsize=5, capthick=1.6, zorder=3)
ax.axhline(0, color=INK, linewidth=1.1)

ylim = (-180, 75)
span = ylim[1] - ylim[0]
ci_lo = df["ci95_low_percent"].to_numpy()
ci_hi = df["ci95_high_percent"].to_numpy()
for bar, mean, lo, hi in zip(bars, means, ci_lo, ci_hi):
    # Label beyond the bar's own error whisker (above for positive bars,
    # below for negative ones) so it never overlaps the whisker -- the
    # glitch in the legacy render. The negative offset is larger than the
    # positive one so the most negative label (Fourier low) still clears
    # the x-axis category labels underneath it.
    if mean >= 0:
        y, va = hi + 0.018 * span, "bottom"
    else:
        y, va = lo - 0.05 * span, "top"
    ax.text(bar.get_x() + bar.get_width() / 2, y, f"{mean:.1f}%", ha="center", va=va, color=INK, weight="bold", fontsize=15)

labels = [name.replace("Fourier ", "Fourier\n") for name in df["objective"]]
ax.set_xticks(list(x), labels, fontsize=15)
ax.set_ylabel("Explained variance", fontsize=16)
ax.set_ylim(*ylim)
ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", length=0, pad=16)
ax.set_title("Reconstruction quality without the anchor", fontsize=16, pad=16)

from matplotlib.patches import Patch
handles = [Patch(facecolor=GRAY, label="MSE control"),
           Patch(facecolor=TEAL, label="unanchored objective")]
ax.legend(handles=handles, loc="upper right", fontsize=13, frameon=False)

fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.18)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
