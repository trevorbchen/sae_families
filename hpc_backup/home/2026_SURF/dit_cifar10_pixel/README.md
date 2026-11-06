# DiT-S/2 CIFAR-10 Pixel Training

This folder trains a native pixel-space DiT-S/2 on CIFAR-10:

- image size: 32x32 RGB, no VAE
- patch size: 2
- hidden size: 384
- depth: 12
- heads: 6
- class conditioning: CIFAR-10 labels
- conditioning mechanism: adaLN-Zero

Start with:

```bash
sbatch slurm_debug.sbatch
```

After the debug run produces checkpoints and sample grids, launch the longer run:

```bash
sbatch slurm_train.sbatch
```

Outputs are written under `runs/`; Slurm logs are written under `logs/`.
