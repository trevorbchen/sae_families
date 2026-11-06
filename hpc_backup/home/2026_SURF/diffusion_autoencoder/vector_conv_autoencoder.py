#!/usr/bin/env python3
"""Deterministic convolutional autoencoder with a global vector bottleneck."""

import torch
import torch.nn as nn

from spatial_conv_autoencoder import DownBlock, ResidualBlock, UpBlock


class VectorConvAutoencoder(nn.Module):
    """128x128 RGB image <-> fixed-width vector, without skip paths."""

    def __init__(self, latent_dim=512, base_channels=64):
        super().__init__()
        b = base_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            ResidualBlock(b),
            DownBlock(b, b * 2),
            DownBlock(b * 2, b * 4),
            DownBlock(b * 4, b * 6),
            DownBlock(b * 6, b * 8),
            DownBlock(b * 8, b * 8),
            nn.GroupNorm(32, b * 8),
            nn.SiLU(),
        )
        self.encoder_out = nn.Linear(b * 8 * 4 * 4, latent_dim)
        self.decoder_in = nn.Linear(latent_dim, b * 8 * 4 * 4)
        self.decoder = nn.Sequential(
            ResidualBlock(b * 8),
            UpBlock(b * 8, b * 8),
            UpBlock(b * 8, b * 6),
            UpBlock(b * 6, b * 4),
            UpBlock(b * 4, b * 2),
            UpBlock(b * 2, b),
            ResidualBlock(b),
            nn.GroupNorm(32, b),
            nn.SiLU(),
            nn.Conv2d(b, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, image):
        hidden = self.encoder_conv(image)
        return self.encoder_out(hidden.flatten(1))

    def decode(self, latent):
        hidden = self.decoder_in(latent)
        hidden = hidden.view(len(latent), self.base_channels * 8, 4, 4)
        return self.decoder(hidden)

    def forward(self, image):
        return self.decode(self.encode(image))


def load_vector_autoencoder(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = VectorConvAutoencoder(
        latent_dim=config["latent_dim"],
        base_channels=config["base_channels"],
    )
    model.load_state_dict(state["model"])
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
