# Three-seed confidence intervals

The earlier linear-probe confidence intervals resampled downstream tasks,
attributes, or identities from one trained SAE. They did not measure
dictionary-training variability. The earlier planted-feature suite likewise
contained only SAE seed 2030.

This replication uses SAE seeds 2026, 2031, and 2032. The data split, probe
split, optimizer settings, dictionary sizes, TopK budgets, and downstream probe
seed remain fixed. Reported 95% confidence intervals are Student-t intervals
over the three independent SAE initialization seeds.

For changed-objective comparisons, the primary interval is computed from the
within-seed paired difference against the architecture-matched MSE control:

- each MSE-plus-one-objective model versus the two-branch all-MSE control;
- the MSE-plus-four-objective model versus the five-branch all-MSE control;
- each planted-feature objective versus the latent-MSE SAE for the same planted
  feature and seed.

Expected output directory: `three_seed_confidence_intervals/`, with separate
`linear_probe/` and `planted_features/` tables.

With only three seeds, these intervals are expected to be wide. Image-level or
task-level resampling may still be reported separately, but it must not be
described as uncertainty across SAE training runs.
