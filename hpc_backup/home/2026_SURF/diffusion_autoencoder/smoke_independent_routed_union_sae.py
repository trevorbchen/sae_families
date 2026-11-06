#!/usr/bin/env python3
"""Verify branch-local gradients and deployable union inference."""

import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from train_independent_routed_union_sae import (
    BranchSpec,
    IndependentRoutedUnionSAE,
    load_independent_routed_union,
)


def main():
    specs = tuple(
        BranchSpec(name, objective, 8, 2)
        for name, objective in (
            ("mse", "latent_mse"),
            ("laplacian", "laplacian"),
            ("fourier_low", "fourier_low"),
            ("fourier_mid", "fourier_mid"),
            ("fourier_high", "fourier_high"),
        )
    )
    torch.manual_seed(7)
    model = IndependentRoutedUnionSAE((2, 2, 2), specs)
    inputs = torch.randn(4, 2, 2, 2)
    union, outputs = model(inputs)
    expected = torch.stack(
        [outputs[spec.name]["reconstruction"] for spec in specs]
    ).mean(0)
    torch.testing.assert_close(union, expected)

    # A branch loss must not leak into any other dictionary's parameters.
    outputs["fourier_mid"]["reconstruction"].square().mean().backward()
    for spec in specs:
        gradients = [parameter.grad for parameter in model.branches[spec.name].parameters()]
        has_gradient = any(
            gradient is not None and bool((gradient != 0).any())
            for gradient in gradients
        )
        assert has_gradient == (spec.name == "fourier_mid"), spec.name

    payload = {
        "models": {
            spec.name: model.branches[spec.name].state_dict() for spec in specs
        },
        "specs": [asdict(spec) for spec in specs],
        "combination": {
            "kind": "uniform_mean",
            "branch_weights": [1 / len(specs)] * len(specs),
        },
        "input_shape": (2, 2, 2),
    }
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "model.pt"
        torch.save(payload, checkpoint)
        loaded, _ = load_independent_routed_union(checkpoint)
        loaded_union, _ = loaded(inputs)
    torch.testing.assert_close(loaded_union, union.detach())
    print("independent routed-union smoke test passed")


if __name__ == "__main__":
    main()
