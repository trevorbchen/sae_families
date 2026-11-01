"""Re-render the main-text EV-with-anchor bar chart (CVPR2026 Figure 4)
from the already-computed proper_ev_table.csv, with larger fonts, a title,
and a legend distinguishing the MSE-control reference bar from the four
anchored-objective bars -- addresses reviewer feedback that the original
render (aslides/make_anchor_ev_plots.py) was illegible at print size.
Reuses that script's exact color palette and bar-labeling logic; only the
font sizes, title, and legend are new.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--table", type=Path, required=True,
                     help="proper_ev_table.csv, e.g. results/family_sae/grouped_objective_sae/"
                          "standalone_objectives/convae_4096_k32/tables/proper_ev_table.csv")
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

ylim = (0, 60)
span = ylim[1] - ylim[0]
for bar, mean in zip(bars, means):
    y = mean + 0.018 * span
    ax.text(bar.get_x() + bar.get_width() / 2, y, f"{mean:.1f}%", ha="center", va="bottom", color=INK, weight="bold", fontsize=15)

labels = [name.replace("Fourier ", "Fourier\n") for name in df["objective"]]
ax.set_xticks(list(x), labels, fontsize=15)
ax.set_ylabel("Explained variance", fontsize=16)
ax.set_ylim(*ylim)
ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
ax.tick_params(axis="y", labelsize=14)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", length=0, pad=8)
ax.set_title("Reconstruction quality of transformed SAEs", fontsize=16, pad=38)

from matplotlib.patches import Patch
handles = [Patch(facecolor=GRAY, label="MSE control (reference)"),
           Patch(facecolor=TEAL, label="anchored objective (ours)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
          ncol=2, fontsize=13, frameon=False, handletextpad=0.5, columnspacing=1.4)

fig.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.18)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
