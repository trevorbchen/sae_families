#!/usr/bin/env python3
"""Fast runtime checks for probe task construction and SAE code extraction."""

import argparse

import numpy as np
import scipy.sparse as sp
import torch

from cache_sae_probe_representations import load_direct, load_flat, load_union
from prepare_celeba_spatial_probe_cache import controlled_images, geometry_targets
from run_sae_linear_probes import binary_probe, regression_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True)
    parser.add_argument("--direct", required=True)
    parser.add_argument("--union", required=True)
    parser.add_argument("--flat", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.rand(7, 3, 128, 128, device=device)
    labels = torch.arange(7, device=device)
    ids = torch.arange(7, device=device)
    transformed = controlled_images(image, labels, ids)
    geometry = geometry_targets(image)
    assert transformed.shape == image.shape and torch.isfinite(transformed).all()
    assert geometry.shape == (7, 8) and torch.isfinite(geometry).all()

    latent_cache = torch.load(args.latents, map_location="cpu", weights_only=False)
    raw = latent_cache["latents"][:8].to(device)
    for loader, path in (
        (load_direct, args.direct), (load_union, args.union), (load_flat, args.flat),
    ):
        encode, dimension, k, _ = loader(path, device)
        with torch.no_grad():
            values, indices = encode(raw)
        assert values.shape == indices.shape == (8, k)
        assert int(indices.min()) >= 0 and int(indices.max()) < dimension
        assert torch.isfinite(values).all()
        del encode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generator = np.random.default_rng(260818)
    x = sp.random(180, 64, density=0.1, random_state=generator, format="csr")
    x = x.astype(np.float32)
    split = np.arange(180)
    train, valid, test = split < 110, (split >= 110) & (split < 145), split >= 145
    binary = (x[:, 0].toarray().ravel() + generator.normal(0, 0.1, 180) > 0.1).astype(int)
    continuous = x[:, 1].toarray().ravel() + generator.normal(0, 0.1, 180)
    binary_result = binary_probe(
        x, binary, train, valid, test, (0.1, 1.0), (1, 4, 16), 260818
    )
    regression_result = regression_probe(x, continuous, train, valid, test, (0.1, 1.0))
    assert np.isfinite(binary_result["test_auroc"])
    assert len(binary_result["feature_efficiency"]) == 3
    assert np.isfinite(regression_result["test_r2"])
    print({
        "geometry_shape": tuple(geometry.shape),
        "binary_auroc": binary_result["test_auroc"],
        "regression_r2": regression_result["test_r2"],
    }, flush=True)


if __name__ == "__main__":
    main()
