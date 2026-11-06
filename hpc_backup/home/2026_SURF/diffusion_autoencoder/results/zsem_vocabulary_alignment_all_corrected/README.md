# Vocabulary Alignment Formulations

This directory separates two SAE objective formulations across the same three
latent architectures and three training seeds.

## `transformed_loss/`

Legacy experiments trained the non-baseline SAEs using only the normalized
Laplacian or Fourier reconstruction term. They did not retain latent MSE as a
full-rank reconstruction anchor.

Objectives: latent MSE baseline, pure Laplacian, Fourier all-bands, Fourier
low, Fourier mid, and Fourier high.

## `proper_loss/`

Corrected experiments use latent MSE plus one normalized transformed-space
term with weight 1.0.

Objectives: latent MSE baseline, MSE + Laplacian, MSE + Fourier low, MSE +
Fourier mid, and MSE + Fourier high.

## Architectures

- `1d/`: TopK SAE on the 512-dimensional DiffAE semantic code.
- `spatial/`: ordinary TopK SAE on the flattened spatial autoencoder map.
- `conv_spatial/`: convolutional TopK SAE with translation-shared atoms.

Each objective has seeds 2030, 2031, and 2032. Alignment uses held-out
activations, decoder/encoder matching, activation correlation, linear CKA,
subspace metrics, feature recurrence, and pairwise category plots. Spatial
comparisons use a fixed number of eligible active features: 100 normally and
64 for corrected flat-spatial runs because the smallest eligible dictionary
contains 72 features.
