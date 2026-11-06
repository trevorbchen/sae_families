#!/usr/bin/env python3
"""Exercise every extended sparse planted-image transform on synthetic inputs."""

from types import SimpleNamespace

import torch

from prepare_sparse_planted_residual_latents import transform_images


FEATURES = (
    "brightness",
    "contrast",
    "hue_rotate",
    "gaussian_blur",
    "unsharp_mask",
    "edge_enhance",
    "local_gabor",
    "local_blob",
    "sparse_impulse",
    "pixelate",
    "local_occlusion",
)


def main():
    images = torch.randint(0, 256, (3, 3, 128, 128), dtype=torch.uint8)
    for feature in FEATURES:
        args = SimpleNamespace(
            feature=feature,
            jpeg_quality=60,
            amplitude=0.12 if feature == "hue_rotate" else 1.0,
            frequency_x=10,
            frequency_y=4,
            phase=0.0,
            blur_kernel=9,
            blur_sigma=2.0,
            local_sigma=0.12,
            block_size=4,
            occlusion_size=12,
        )
        transformed = transform_images(
            images, torch.arange(len(images)), None, args
        )
        delta = float((transformed - images.float()).abs().mean())
        assert transformed.shape == images.shape
        assert torch.isfinite(transformed).all()
        assert delta > 0
        print(feature, f"mean_abs_delta={delta:.6f}")


if __name__ == "__main__":
    main()
