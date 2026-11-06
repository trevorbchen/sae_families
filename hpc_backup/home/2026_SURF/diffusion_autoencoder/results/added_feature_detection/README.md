# Added Feature Detection

This report compares ordinary latent-MSE SAEs against SAEs trained with 
latent MSE plus a normalized Laplacian or Fourier reconstruction term. 
The DiffAE/spatial-AE latent representation is frozen; only the SAE is retrained.

Each image-space feature is planted in 10% of examples. The reported percentage is:

```text
100 * (bias-only residual MSE - reconstructed residual MSE)
    / bias-only residual MSE
```

Top-1 is the percentage explained by the single most useful SAE feature; top-10
uses the ten most useful features jointly. Higher is better.

Weighted loss improves top-1 percentage for **17/20** planted 
conditions and top-10 percentage for **12/20** conditions.

| Condition | MSE top-1 | Best weighted loss top-1 | Change | MSE top-10 | Best weighted loss top-10 | Change |
|---|---:|---:|---:|---:|---:|---:|
| brightness_a24 | 8.27% | 16.16% (weighted_fourier_high) | +7.89 pp | 35.65% | 39.86% (weighted_fourier_bands) | +4.21 pp |
| checkerboard_a1 | 0.33% | 0.36% (weighted_fourier_low) | +0.03 pp | 8.02% | 6.83% (weighted_fourier_bands) | -1.20 pp |
| color_red_a1 | 25.59% | 31.96% (weighted_laplacian) | +6.37 pp | 44.30% | 47.11% (weighted_laplacian) | +2.82 pp |
| contrast_a0p3 | 2.00% | 6.10% (weighted_laplacian) | +4.10 pp | 26.72% | 39.06% (weighted_laplacian) | +12.33 pp |
| edge_enhance_a32 | 14.79% | 5.74% (weighted_fourier_high) | -9.05 pp | 31.81% | 27.37% (weighted_fourier_bands) | -4.44 pp |
| gaussian_blur_s2 | 16.58% | 17.01% (weighted_laplacian) | +0.44 pp | 37.97% | 34.83% (weighted_fourier_high) | -3.14 pp |
| hue_rotate_a0p12 | 10.35% | 32.89% (weighted_laplacian) | +22.54 pp | 37.81% | 50.87% (weighted_fourier_high) | +13.06 pp |
| jpeg_q60 | 0.22% | 6.90% (weighted_fourier_mid) | +6.67 pp | 0.41% | 6.90% (weighted_fourier_mid) | +6.49 pp |
| local_blob_a32 | 7.92% | 11.94% (weighted_laplacian) | +4.02 pp | 37.33% | 44.54% (weighted_fourier_bands) | +7.21 pp |
| local_gabor_a32 | 7.85% | 39.36% (weighted_fourier_low) | +31.51 pp | 54.61% | 74.65% (weighted_fourier_low) | +20.05 pp |
| local_occlusion_s12 | 18.52% | 7.28% (weighted_laplacian) | -11.24 pp | 67.16% | 71.45% (weighted_fourier_bands) | +4.29 pp |
| pixelate_b4 | 4.41% | 6.46% (weighted_fourier_high) | +2.05 pp | 9.51% | 8.58% (weighted_fourier_high) | -0.93 pp |
| sine_high_a0p5 | 20.82% | 23.79% (weighted_fourier_bands) | +2.98 pp | 31.81% | 24.06% (weighted_fourier_bands) | -7.75 pp |
| sine_high_a1 | 6.05% | 24.97% (weighted_fourier_low) | +18.92 pp | 31.09% | 29.28% (weighted_fourier_low) | -1.81 pp |
| sine_high_a3 | 6.80% | 32.98% (weighted_fourier_mid) | +26.18 pp | 35.37% | 37.54% (weighted_fourier_low) | +2.17 pp |
| sine_low_a1 | 10.32% | 14.53% (weighted_fourier_high) | +4.21 pp | 36.51% | 39.99% (weighted_fourier_low) | +3.48 pp |
| sine_mid_a1 | 8.78% | 35.97% (weighted_fourier_mid) | +27.18 pp | 33.97% | 41.38% (weighted_fourier_mid) | +7.41 pp |
| sine_random_high_a1 | 3.93% | 11.13% (weighted_fourier_bands) | +7.20 pp | 24.28% | 18.89% (weighted_laplacian) | -5.39 pp |
| sparse_impulse_a64 | 5.96% | 20.85% (weighted_fourier_low) | +14.90 pp | 37.21% | 41.36% (weighted_fourier_bands) | +4.16 pp |
| unsharp_a1 | 24.49% | 10.06% (weighted_fourier_high) | -14.42 pp | 42.71% | 41.05% (weighted_fourier_bands) | -1.65 pp |

## Files

- `feature_detection_summary.csv`: one comparison row per planted image feature.
- `all_objective_results.csv`: every objective run, including the latent control.
- `objective_by_feature_top1_percent_explained.csv`: 6 x 20 top-1 matrix.
- `objective_by_feature_top5_percent_explained.csv`: 6 x 20 top-5 matrix.
- `objective_by_feature_top10_percent_explained.csv`: 6 x 20 top-10 matrix.
- `objective_by_feature_full_sae_percent_explained.csv`: 6 x 20 full-SAE matrix.
- `latent_control.csv`: fixed random latent-direction positive control, when included.
- Source visualizations are stored in the suite results directory.

The latent-random control is excluded from the 20-feature win counts because it
was trained only with ordinary MSE and has no matched image-space objective.
