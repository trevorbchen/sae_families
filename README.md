# sae_families

Analysis code backing the "objective geometry in AE-space" results in
*Objective Geometry Steers Sparse Autoencoder Vocabularies*
([CVPR2026 repo](https://github.com/trevorbchen/CVPR2026)) — specifically
Figure 1 (`fig:density-contours-celeba`) and Figure 11
(`fig:feature-contours-celeba` / `fig:feature-lda-celeba`) in `main.tex`.

These scripts measure how much an SAE's training objective (MSE, Laplacian,
Fourier low/mid/high) shows up in the shared ConvAE latent space, without
any cross-SAE dictionary-index alignment — every objective/seed's SAE
reconstructs into the same 512-dim coordinate system, so no index
correspondence between independently-trained dictionaries is needed.

## Scripts

- **`explore_reconstruction_tsne.py`** — per-image, h-projected, normalized
  reconstruction-residual direction, pooled across all objectives/seeds,
  embedded with t-SNE. This is the paper's new Figure 1: a direct,
  unsupervised split by training objective.
- **`explore_feature_tsne.py`** — per-(image, active feature) causal
  ablation effect (`value_i * decoder[index_i]`, the exact closed form of
  removing that one feature's contribution from a TopK SAE's linear
  reconstruction — see Method's `anchoredTransformLoss`/proper-objective
  discussion), embedded with t-SNE. Produces a near-null result (no local
  structure) — this is deliberate, not a bug: see `explore_feature_lda.py`.
- **`explore_feature_lda.py`** — the same per-feature points as above, but
  projected with linear discriminant analysis instead of t-SNE. Recovers
  real, substantial structure (~60% held-out 5-way classification accuracy
  vs. 20% chance) that t-SNE cannot represent, because the signal is a
  global linear direction (a small, consistent per-objective mean shift
  spread across many dimensions), not a local density cluster. Figure 11b.
- **`tsne_viz_utils.py`** — shared KDE-density-contour plotting helpers used
  by the three scripts above, in place of raw scatter (unreadable at this
  point count — tens of thousands of points, no t-SNE-style repulsion in
  the LDA case).
- **`visualize_pure_cluster.py`** — decodes real images from a density-
  anchored region of a saved embedding (anchor = local peak of one
  objective's own point density, *not* k-means/PCA clustering), to show
  what the reconstructions in that region actually look like under every
  objective. Backs the paper's fidelity/sharpness-tradeoff finding
  (Laplacian sharper in-distribution, breaking down/inverting OOD) cited in
  the Figure 11 paragraph.

- **`plot_feature_detection_heatmap.py`** / **`plot_linear_probe_heatmap.py`** —
  colored table renders of the planted-recovery and linear-probe summary
  CSVs (not scatter/contour plots like the scripts above); font sizes
  bumped for legibility at print size. `plot_paper_feature_detection_heatmap_single_lambda.py`
  wraps the former to a single headline $\lambda$ for the main-text figure
  (CVPR2026 Figure 6b, `fig06_planted_recovery_heatmap_lambda10.png`).
  `plot_feature_detection_heatmap.py`'s renderer additionally got a
  visual-polish pass (softer custom red-cream-green colormap in place of
  stock `RdYlGn`, units moved from each cell into the column headers, a
  border frame and a divider under the MSE-control row) — same numbers,
  just less "sad looking" per reviewer feedback. `plot_linear_probe_heatmap.py`
  was deliberately **not** touched: its main-text single-$\lambda$ figure
  (Figure 6c, `fig07_linear_probe_heatmap_lambda10.png`) has an unresolved
  seed-aggregation data-discrepancy flagged earlier in this project and the
  author asked to leave it as published.
- **`make_fig03a.py`** — re-renders the EV-with-anchor bar chart (Figure 5)
  from `proper_ev_table.csv` with larger fonts, a title, and a legend.
- **`make_fig03b.py`** — the appendix sibling of `make_fig03a.py`
  (`fig:ev-without-anchor`, `fig03b_ev_without_anchor.png`): same palette,
  fonts, and legend treatment, pointed at `pure_ev_table.csv` instead, with
  `ylim=(-155, 65)` since these bars collapse deeply negative without the
  anchor. Also fixes a label/error-bar overlap glitch in the legacy render
  (`aslides/make_anchor_ev_plots.py`'s `pure_ev_table` branch, which had no
  title/legend either) by placing each bar's value label beyond its own CI
  whisker instead of at a fixed offset from the bar top. Verified against
  `pure_ev_table.csv` directly: MSE control 55.3%, Laplacian -23.0%, Fourier
  low -130.4%, Fourier mid -12.7%, Fourier high -20.6%.
- **`visualize_pure_cluster.py`** title cleanup — the `fig.suptitle(...)` no
  longer bakes in the raw density-anchor coordinate tuple (`anchor=(x,y)`);
  replaced with a plain "{Dataset}: images nearest the {Objective} density
  peak" plus the neighbor-composition line, and the column/row-label font
  sizes were bumped (9/8pt → 13/10pt). **Not regenerated**: figA19-A24, the
  six already-deployed gallery PNGs, since there is no recorded invocation
  (CLI args) for them anywhere in either repo's history, and
  `find_density_anchor`'s result depends on `--arm`/`--lambda-tag`/
  `--recon-seed`/`--source` matching exactly — guessing those would risk
  silently swapping in different reconstructed images. Whoever has the
  original command can rerun it with the now-fixed script; flagged to the
  paper author rather than guessed.
- Also looked for a reproducible source for figA2/A6/A8/A10 (the 12
  per-arm planted-feature-detection "full breakdown" appendix tables,
  λ-rows × objective-columns, one top-$k$ depth per image): **no script
  anywhere in either repo produces this table layout** —
  `plot_feature_detection_heatmap.py`'s current `build_table()` produces a
  different shape (rows are (λ, objective) pairs, columns are the three
  top-$k$ depths together) than what's actually deployed. These 12 images
  are effectively orphaned; left untouched rather than guessed at.
- `plot_linear_probe_heatmap.py` itself needed no code changes (already
  clean rows/columns/colors), just a plain-English `--title` instead of the
  raw arm code it was previously invoked with:
  `python3 plot_linear_probe_heatmap.py --summary-csv .../vector_16x_k32/general/linear_probe/summary.csv --output .../figA4_linear_probe_heatmap_all_lambda.png --title "Linear-probe accessibility (vector layout, large dictionary)"`
  and the `spatial_16x_k32` sibling (figA12) with `--title "Linear-probe
  accessibility (spatial layout, large dictionary)"`. Both verified to
  reproduce the previously-deployed numbers exactly (e.g. Raw ConvAE row:
  0.873/0.642/0.620/0.351).
- **`make_fig_vocabulary_similarity.py`** — re-renders the vocabulary-
  similarity heatmap (Figure 4, `fig:vocabulary-similarity-heatmap`,
  `figA18_vocabulary_similarity_heatmap.png`) as a single row of four panels
  instead of the original hand-assembled 2x2 grid, with larger fonts and a
  tighter font-to-panel-width ratio so it stays legible once LaTeX scales it
  to page width. Reads `pairwise_category_summary.csv` from
  `plot_cross_seed_similarity_heatmaps.py` directly (no recomputation);
  verified to reproduce the original's numbers exactly (e.g. MSE/MSE
  diagonal: decoder-cosine 0.529, activation-CKA 0.377, activation-
  correlation 0.279, atom-recurrence 0.047).

## Data

Expects a local mirror of the matched-512 lambda-sweep checkpoints/latents
(`autoencoders/vector/last.pt`, `experiments/<arm>/checkpoints/...`,
per-source latent caches) and the `feature_usage_vector_16x_k32_lambda10.pt`
extraction file — see the main `CVPR2026` repo / `family_sae` results tree
for provenance. All runs here use `vector_16x_k32` ($\lambda$=10), the
unconstrained arm, matching the paper's linear-probe convention.

## Example

```bash
python3 explore_reconstruction_tsne.py \
  --usage <path>/feature_usage_vector_16x_k32_lambda10.pt \
  --base <path>/sae_matched_512_v1 \
  --source celeba \
  --out-prefix out/celeba_reconstruction

python3 explore_feature_lda.py \
  --usage <path>/feature_usage_vector_16x_k32_lambda10.pt \
  --base <path>/sae_matched_512_v1 \
  --source celeba \
  --out-dir out/lda --out-prefix celeba
```
