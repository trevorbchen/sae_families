# Grouped Objective SAE Comparison

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

The grouped model allocates:

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

The pure-transformed grouped variant keeps the same partition, initialization,
and total capacity. Its MSE group still receives latent MSE, but each specialist
group receives only its assigned normalized transformed primary loss rather
than latent MSE plus that loss. AuxK remains enabled as residual/dead-feature
support, so "pure" refers specifically to the routed primary objective.

## Slurm jobs

- `393255`: one-step real DiffAE grouped smoke test.
- `393257`: regular control.
- `393264`: grouped model, dependent on successful smoke test.

The pure-transformed smoke and full job IDs are recorded after submission.

The older merged-data `final_32x_k64_aux_reinit_merged` checkpoint is not the
primary control because it was trained on 232,771 merged FFHQ/CelebA codes and
therefore does not share this experiment's paired image dataset.
