# Original, Proper-Loss, and Transformed-Loss SAE Comparison

Terminology used throughout this experiment:

- **Original SAE:** ordinary latent-MSE objective.
- **Proper-loss SAE:** MSE anchor plus transformed specialist objectives
  (internally named `grouped`).
- **Transformed-loss SAE:** pure transformed specialist objectives
  (internally named `grouped_pure`).

This experiment compares a regular TopK SAE against a partitioned,
gradient-routed TopK SAE. Both use the same FFHQ examples, split, batch order,
initial encoder and decoder tensors, seed, total dictionary width, and total
TopK budget.

## Shared configuration

- Input: standardized 512-dimensional DiffAE semantic codes paired with the
  16,384-image FFHQ transformed-objective dataset.
- Dictionary width: 16,384 features.
- Total selected features: k=64.
- Seed and split seed: 2026.
- Steps: 20,000.
- Batch size: 32.
- AuxK coefficient: 0.1.
- Decoder columns are unit normalized; dead-feature rescue runs every 1,000
  steps.
- Decoder bias is fixed at zero because standardized training codes have zero
  mean. This prevents a shared bias from mixing group gradients.

## Models

The regular control uses one 16,384-feature dictionary with global k=64 and
ordinary latent MSE.

The proper-loss model allocates:

| Group | Width | k | Objective |
|---|---:|---:|---|
| MSE | 8,192 | 32 | latent MSE |
| Laplacian | 2,048 | 8 | latent MSE + normalized Laplacian loss |
| Fourier low | 2,048 | 8 | latent MSE + normalized low-band loss |
| Fourier mid | 2,048 | 8 | latent MSE + normalized mid-band loss |
| Fourier high | 2,048 | 8 | latent MSE + normalized high-band loss |

Every objective is evaluated on the same combined reconstruction. Its gradient
is then routed only to its assigned group's contribution. Groupwise TopK gives
exactly 64 selected slots in total.

The transformed-loss model keeps the same partition, initialization, and total
capacity. Its MSE group still receives latent MSE, but each specialist group
receives only its assigned normalized transformed primary loss rather than
latent MSE plus that loss. AuxK remains enabled as residual/dead-feature
support; "transformed loss" refers specifically to the routed primary
objective.

## Slurm jobs

- `393255`: one-step real DiffAE grouped smoke test.
- `393257`: regular control.
- `393264`: grouped model, dependent on successful smoke test.

- `572307`: pure-transformed grouped one-step DiffAE smoke test.
- `572310`: pure-transformed grouped 20,000-step H200 run, dependent on
  successful completion of `572307`.

The older merged-data `final_32x_k64_aux_reinit_merged` checkpoint is not the
primary control because it was trained on 232,771 merged FFHQ/CelebA codes and
therefore does not share this experiment's paired image dataset.

## Bookkeeping v2 reruns

The matched v2 reruns write to
`/resnick/scratch/tbchen/surf_2026/grouped_objective_sae_bookkeeping_v2`.
Each run records exact per-step losses, timesteps, routed and parameter-gradient
norms, clipping events and streaks, group-write norms, pairwise cancellation,
activation statistics, and resampling events. Held-out latent MSE/EV is
evaluated every 250 steps. Model checkpoints are archived every 1,000 steps,
the latest optimizer/RNG state is overwritten every 500 steps, and the best
validation checkpoint is retained.

Bookkeeping v2 Slurm jobs:

- `599706`: three-mode one-step smoke array.
- `599711`: original SAE, dependent on `599706`.
- `599712_0`: proper-loss SAE, dependent on `599706`.
- `599712_1`: transformed-loss SAE, dependent on `599706`.

All v2 jobs completed successfully. Held-out latent EV and clipping results:

| Model | Best EV (step) | Final EV | Clipped steps | Dominant clipping group |
|---|---:|---:|---:|---|
| Original | 63.96% (8,250) | 62.27% | 0 / 20,000 | none |
| Proper loss | 51.16% (4,000) | -26.38% | 19,999 / 20,000 | Laplacian |
| Transformed loss | 20.52% (1,500) | -513.01% | 20,000 / 20,000 | Laplacian |

The logging localizes the modified-objective instability to the Laplacian
specialist's gradient scale. Its gradient dominates every clipping event in
both modified models. Full traces and checkpoints remain under the v2 scratch
path above.

## Full Conv-AE latent comparison

The follow-up replaces DiffAE with the deterministic spatial Conv-AE and uses
its full `64 x 8 x 8` latent maps, not planted or processing residuals. Each map
is flattened to 4,096 dimensions for the same partitioned SAE design. The
original, proper-loss, and transformed-loss models remain matched at a 16,384
feature vocabulary and total `k=64`, with the same v2 diagnostics. Outputs are
stored under
`/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_bookkeeping_v1`.

Conv-AE Slurm jobs:

- `647921`: three-mode smoke array, completed successfully.
- `647972`: full original SAE.
- `647976`: cancelled before starting because its H200 estimate was August 19.
- `681211_0`: full proper-loss SAE on L40S.
- `681211_1`: full transformed-loss SAE on L40S.

Both L40S specialist jobs completed in about 70 minutes. Results:

| Model | Best EV (step) | Final EV | Clipped steps | Main dominant group |
|---|---:|---:|---:|---|
| Original | 50.61% (17,000) | 50.49% | 0 / 20,000 | none |
| Proper loss | 19.70% (250) | -909.24% | 19,798 / 20,000 | Fourier mid |
| Transformed loss | 11.92% (1) | -15,337.92% | 19,811 / 20,000 | MSE compensation |

Replacing DiffAE with the Conv-AE therefore does not stabilize the grouped
routed objectives. In the proper-loss run, Fourier-mid gradients dominate most
clipping events. In the transformed-loss run, specialist writes drive the
combined reconstruction so far away that the MSE group becomes the dominant
compensating gradient.

## Raw-loss, gradient-calibrated Conv-AE follow-up

The preceding runs normalized each transformed loss by calibration-set signal
power, then assigned every specialist a fixed lambda of 1.0. Similar loss
values did not imply similar gradients, and both specialist models diverged.

The first follow-up removed calibration-power loss normalization entirely. It
used the raw Laplacian or Fourier-band error and chose one fixed lambda per group:

`lambda_g = median(||grad_h L_mse||) / median(||grad_h L_g||)`

The medians were measured over 2,048 training examples at the initialized SAE,
before any optimizer step. Outputs are stored under
`/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_gradient_scaled_v2`.

- Smoke array `693446` completed successfully.
- Full proper/pure array `693466` was stopped at roughly step 500.
- Proper and pure runs independently produced effectively identical scales:
  Laplacian `23.8165`, Fourier low `0.0066399`, Fourier mid `0.352536`, and
  Fourier high `16.2885`.
- The fixed scales were initially stable, but drift caused clipping by about
  step 300. At step 500, proper had clipped 21.2% of steps and pure had clipped
  45.6%. This shows initialization-only gradient calibration is insufficient.

The v3 design therefore computes a detached lambda on every batch from the live
output gradients, with no loss-power normalization:

`lambda_g(t) = ||grad_h L_mse|| / ||grad_h L_g||`.

The raw losses are unchanged; this only controls their optimization scale.
Outputs are stored under
`/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_dynamic_gradient_v3`.

- P100 smoke array `694272` completed successfully.
- Full L40S array `694273` is queued.
- In the pure transformed smoke run, every routed specialist gradient norm was
  exactly `0.010249`, matching MSE, and the global norm was `0.268` without
  clipping.

Live output-gradient matching alone was still insufficient: by step 1,100,
the routed norms remained equal while the actual group parameter-gradient
norms ranged from about 1.1 to 7.6 because the SAE Jacobians had changed. The
v3 full array `694273` was therefore stopped.

The v4 gradient-only design retains live output-gradient matching and also caps
each partition's actual parameter-gradient norm at the MSE partition's step-1
norm. It still applies no loss-power normalization. Outputs are stored under
`/resnick/scratch/tbchen/surf_2026/grouped_spatial_conv_ae_gradient_only_v4`.

- P100 smoke array `694829` completed successfully.
- Full L40S array `694830` is running/queued.
- The proper run's parameter-gradient target is `0.0998394`.
- At step 500, all five capped group norms were at or below that target, the
  global norm was `0.223`, no steps had clipped, and validation EV was 26.63%.

Both v4 tasks completed all 20,000 steps in array `694830`, but norm control did
not stabilize learning. The cap held the global norm at `0.223` with zero
clipped steps while the uncapped group norms and dynamic lambdas grew without
bound. Thus clipping was a symptom, not the underlying failure.

| Model | Best EV (step) | Final EV | Final latent MSE |
|---|---:|---:|---:|
| Proper | 28.37% (250) | -665.64% | 7.711 |
| Pure transformed | 11.92% (1) | -118,088.15% | 1190.337 |

The parameter cap effectively imposed persistent fixed-magnitude updates even
after the raw optimization geometry became pathological. Gradient magnitude
balancing alone therefore cannot reconcile the competing routed directions.
