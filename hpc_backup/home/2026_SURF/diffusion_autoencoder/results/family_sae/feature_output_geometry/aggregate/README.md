# SAE feature output geometry

Each row vocabulary is evaluated with four normalized image-space metrics. The edited-output view measures the geometry of the resulting image. The causal-delta view measures only `decoded(edited latent) - decoded(original latent)`. Fourier cells are fractions of non-DC power; Laplacian cells are squared discrete-Laplacian energy divided by image energy. Heatmaps show each cell relative to the mean across the five matched vocabularies for that metric.
