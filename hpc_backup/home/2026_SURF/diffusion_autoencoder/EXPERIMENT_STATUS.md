# Diff-AE Experiment

This folder contains the Diffusion Autoencoder follow-up to the CIFAR-10 DiT work.

The official implementation is kept in `official_diffae/`. The first run uses the
pretrained FFHQ-128 Diff-AE checkpoint from the authors. We will first verify
encoding, DDIM reconstruction, and semantic-code extraction before training a
TopK SAE on the global `z_sem` vectors.

Results from this study belong here and should not be mixed with the existing
`dit_cifar10_pixel_hpc/` outputs.

## BSF comparison

`train_zsem_bsf.py` trains a Block-Sparse Featurizer on the same pooled
`z_sem` codes and split as the SAE. The initial comparison uses 32x scalar
dictionary size, 8-dimensional blocks, and four active blocks (32 active
scalar dimensions), so reconstruction can be compared against the final
32x/k32 TopK SAE without changing the data budget or active scalar budget.
