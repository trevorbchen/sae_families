#!/usr/bin/env python3
"""Deterministic convolutional autoencoder with a genuine spatial bottleneck."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x):
        return self.residual(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.residual(self.conv(x))


class SpatialConvAutoencoder(nn.Module):
    """128x128 RGB image <-> [latent_channels, 8, 8] map, without skip paths."""

    def __init__(self, latent_channels=64, base_channels=64):
        super().__init__()
        b = base_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            ResidualBlock(b),
            DownBlock(b, b * 2),
            DownBlock(b * 2, b * 4),
            DownBlock(b * 4, b * 6),
            DownBlock(b * 6, b * 8),
            nn.GroupNorm(32, b * 8),
            nn.SiLU(),
            nn.Conv2d(b * 8, latent_channels, 1),
        )
        self.decoder_in = nn.Conv2d(latent_channels, b * 8, 1)
        self.decoder = nn.Sequential(
            ResidualBlock(b * 8),
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
        return self.encoder(image)

    def decode(self, latent):
        return self.decoder(self.decoder_in(latent))

    def forward(self, image):
        return self.decode(self.encode(image))


def load_spatial_autoencoder(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = SpatialConvAutoencoder(
        latent_channels=config["latent_channels"],
        base_channels=config["base_channels"],
    )
    model.load_state_dict(state["model"])
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
