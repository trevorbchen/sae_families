#!/usr/bin/env python3
"""CPU smoke checks for grouped SAE capacity and deterministic initialization."""

import torch

from train_grouped_objective_sae import (
    GROUPED_PURE_SPECS,
    GROUPED_SPECS,
    REGULAR_SPECS,
    GroupSpec,
    PartitionedTopKSAE,
    group_objective_losses,
    routed_gradient_proxy,
)


def main():
    torch.manual_seed(2026)
    regular = PartitionedTopKSAE(512, REGULAR_SPECS)
    torch.manual_seed(2026)
    grouped = PartitionedTopKSAE(512, GROUPED_SPECS)
    assert regular.dict_size == grouped.dict_size == 16_384
    assert regular.total_k == grouped.total_k == 64
    assert torch.equal(regular.decoder, grouped.decoder)
    assert torch.equal(regular.encoder.weight, grouped.encoder.weight)
    assert torch.equal(regular.encoder.bias, grouped.encoder.bias)
    inputs = torch.randn(2, 512)
    regular_reconstruction, regular_groups = regular(inputs)
    grouped_reconstruction, grouped_groups = grouped(inputs)
    assert regular_reconstruction.shape == grouped_reconstruction.shape == (2, 512)
    assert sum(group["values"].shape[1] for group in grouped_groups.values()) == 64
    assert regular_groups["mse"]["values"].shape[1] == 64

    transformed = {
        "laplacian": torch.tensor([5.0, 7.0]),
        "fourier_low": torch.tensor([11.0, 13.0]),
        "fourier_mid": torch.tensor([17.0, 19.0]),
        "fourier_high": torch.tensor([23.0, 29.0]),
    }
    latent_loss = torch.tensor([2.0, 3.0])
    pure_losses = group_objective_losses(
        "grouped_pure", GROUPED_PURE_SPECS, latent_loss, transformed, 0.5
    )
    anchored_losses = group_objective_losses(
        "grouped", GROUPED_SPECS, latent_loss, transformed, 0.5
    )
    assert torch.equal(pure_losses["laplacian"], torch.tensor([2.5, 3.5]))
    assert torch.equal(anchored_losses["laplacian"], torch.tensor([4.5, 6.5]))

    tiny_specs = (
        GroupSpec("first", "latent_mse", 4, 1),
        GroupSpec("second", "weighted_laplacian", 4, 1),
    )
    torch.manual_seed(7)
    tiny = PartitionedTopKSAE(3, tiny_specs)
    tiny_reconstruction, tiny_groups = tiny(torch.randn(5, 3))
    for group in tiny_groups.values():
        group["contribution"].retain_grad()
    first_loss = tiny_reconstruction.square().mean()
    second_loss = (
        (tiny_reconstruction - 1).square()
        * torch.tensor([1.0, 2.0, 4.0])
    ).mean()
    first_gradient = torch.autograd.grad(
        first_loss, tiny_reconstruction, retain_graph=True
    )[0]
    second_gradient = torch.autograd.grad(
        second_loss, tiny_reconstruction, retain_graph=True
    )[0]
    proxy = routed_gradient_proxy(
        tiny,
        tiny_groups,
        {"first": first_gradient, "second": second_gradient},
    )
    proxy.backward()
    assert torch.allclose(
        tiny_groups["first"]["contribution"].grad, first_gradient
    )
    assert torch.allclose(
        tiny_groups["second"]["contribution"].grad, second_gradient
    )
    print("grouped SAE smoke checks passed")


if __name__ == "__main__":
    main()
