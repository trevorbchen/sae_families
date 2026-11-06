# Added Feature Detection

This report compares ordinary latent-MSE SAEs against SAEs trained with 
only a normalized Laplacian or Fourier reconstruction objective. 
The DiffAE/spatial-AE latent representation is frozen; only the SAE is retrained.

Each image-space feature is planted in 10% of examples. The reported percentage is:

```text
100 * (bias-only residual MSE - reconstructed residual MSE)
    / bias-only residual MSE
```

Top-1 is the percentage explained by the single most useful SAE feature; top-10
uses the ten most useful features jointly. Higher is better.

Pure transformed loss improves top-1 percentage for **5/20** planted 
conditions and top-10 percentage for **2/20** conditions.

| Condition | MSE top-1 | Best pure transformed loss top-1 | Change | MSE top-10 | Best pure transformed loss top-10 | Change |
|---|---:|---:|---:|---:|---:|---:|
| brightness_a24 | 9.88% | 6.02% (laplacian) | -3.86 pp | 38.09% | 16.53% (fourier_bands) | -21.57 pp |
| checkerboard_a1 | 0.42% | 1.43% (fourier_high) | +1.00 pp | 6.03% | 3.06% (fourier_bands) | -2.97 pp |
| color_red_a1 | 26.51% | 9.74% (fourier_high) | -16.77 pp | 44.04% | 11.75% (fourier_bands) | -32.30 pp |
| contrast_a0p3 | 1.46% | 7.15% (laplacian) | +5.69 pp | 22.80% | 25.50% (laplacian) | +2.70 pp |
| edge_enhance_a32 | 13.44% | 4.16% (laplacian) | -9.27 pp | 32.00% | 23.18% (fourier_bands) | -8.81 pp |
| gaussian_blur_s2 | 16.23% | 1.59% (fourier_low) | -14.64 pp | 39.66% | 3.87% (fourier_high) | -35.79 pp |
| hue_rotate_a0p12 | 11.08% | 21.05% (laplacian) | +9.97 pp | 37.19% | 36.89% (fourier_high) | -0.30 pp |
| jpeg_q60 | 0.20% | 5.83% (fourier_bands) | +5.63 pp | 0.39% | 5.83% (fourier_bands) | +5.45 pp |
| local_blob_a32 | 7.47% | 5.37% (fourier_bands) | -2.09 pp | 37.51% | 24.73% (fourier_bands) | -12.79 pp |
| local_gabor_a32 | 7.46% | 18.66% (fourier_bands) | +11.20 pp | 54.20% | 48.99% (laplacian) | -5.21 pp |

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
