"""Two-panel figure for the "combining objectives into a single model" result
(CVPR2026 main text, promoted from an appendix-only single-panel chart).

Panel (a): held-out EV cost of combining all four objectives into one
five-branch SAE vs. an architecture-matched five-branch MSE-only control
(peak_ev, mean +/- 95% seed CI, n=3 seeds). Source:
results/family_sae/grouped_objective_sae/independent_routed_tandem/seed_ci/seed_summary.csv
(written by summarize_independent_routed_seed_ci.py).

Panel (b): each specialist branch's relative reduction in its own transformed
loss vs. its matched MSE-only branch, i.e. specialization survives inside the
combined model. Source:
results/family_sae/grouped_objective_sae/independent_routed_tandem/tables/specialist_vs_mse_control.csv
(written by evaluate_independent_routed_union_control.py).

Both read already-computed tables; nothing is recomputed here.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

INK = "#17202A"
TEAL = "#0F8587"
GRAY = "#6B7280"
GRID = "#D9DEE2"

OBJECTIVE_COLORS = {
    "laplacian": "#e41a1c",
    "fourier_low": "#4daf4a",
    "fourier_mid": "#ff7f00",
    "fourier_high": "#377eb8",
}
OBJECTIVE_LABELS = {
    "laplacian": "Laplacian",
    "fourier_low": "Fourier\nlow",
    "fourier_mid": "Fourier\nmid",
    "fourier_high": "Fourier\nhigh",
}
OBJECTIVE_ORDER = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")

parser = argparse.ArgumentParser()
parser.add_argument("--seed-summary", type=Path, required=True,
                     help="seed_summary.csv from summarize_independent_routed_seed_ci.py")
parser.add_argument("--specialist-vs-control", type=Path, required=True,
                     help="specialist_vs_mse_control.csv from evaluate_independent_routed_union_control.py")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
})

# ---- load panel (a): seed-mean EV, five_branch_mse vs grouped_all_four ----
seed_rows = list(csv.DictReader(open(args.seed_summary)))
peak_rows = {r["condition"]: r for r in seed_rows if r["metric"] == "peak_ev"}
control_mean = float(peak_rows["five_branch_mse"]["mean"]) * 100
control_ci = float(peak_rows["five_branch_mse"]["ci_half_width"]) * 100
combined_mean = float(peak_rows["grouped_all_four"]["mean"]) * 100
combined_ci = float(peak_rows["grouped_all_four"]["ci_half_width"]) * 100

# ---- load panel (b): per-branch relative loss reduction ----
spec_rows = {r["objective"]: r for r in csv.DictReader(open(args.specialist_vs_control))}
reductions = [float(spec_rows[o]["relative_reduction"]) * 100 for o in OBJECTIVE_ORDER]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# Panel (a)
labels_a = ["Five-branch\nMSE control", "MSE + four routed\nobjectives"]
means_a = [control_mean, combined_mean]
errs_a = [control_ci, combined_ci]
colors_a = [GRAY, TEAL]
bars_a = ax_a.bar(labels_a, means_a, yerr=errs_a, color=colors_a, width=0.55,
                   capsize=5, ecolor=INK, error_kw={"elinewidth": 1.6, "capthick": 1.6}, zorder=2)
for bar, mean in zip(bars_a, means_a):
    ax_a.text(bar.get_x() + bar.get_width() / 2, mean + 1.1, f"{mean:.2f}%",
               ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
ax_a.set_ylabel("Explained variance", fontsize=15)
ax_a.set_ylim(0, 60)
ax_a.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax_a.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.tick_params(axis="x", length=0, labelsize=13)
ax_a.set_title("Reconstruction cost of combining objectives", fontsize=15, pad=12)

# Panel (b)
xs = range(len(OBJECTIVE_ORDER))
colors_b = [OBJECTIVE_COLORS[o] for o in OBJECTIVE_ORDER]
bars_b = ax_b.bar(xs, reductions, color=colors_b, width=0.6, zorder=2)
for bar, value in zip(bars_b, reductions):
    ax_b.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}%",
               ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
ax_b.set_xticks(list(xs), [OBJECTIVE_LABELS[o] for o in OBJECTIVE_ORDER], fontsize=13)
ax_b.set_ylabel("Relative loss reduction\nvs. matched MSE branch", fontsize=15)
ax_b.set_ylim(0, 20)
ax_b.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax_b.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.tick_params(axis="x", length=0)
ax_b.set_title("Specialization survives inside the combined model", fontsize=15, pad=12)

fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.14, wspace=0.28)
fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", args.output)
print(f"panel (a): control={control_mean:.2f}% combined={combined_mean:.2f}%")
print(f"panel (b): {dict(zip(OBJECTIVE_ORDER, (f'{r:.1f}%' for r in reductions)))}")
