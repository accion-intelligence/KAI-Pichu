"""Input distributions and case splits for the depthwise 7x7 convolution task.

Activations are FP16 in NCHW memory order; weights are FP16 [C, 7, 7]. Values
are drawn in FP32 and rounded to FP16 so the reference sees exactly what the
kernel receives. Shapes follow ConvNeXt-style stages and add odd spatial sizes,
so image rows are not 16-byte aligned and vectorized loads along W need
alignment handling and a tail.

This module is task-owner code. It is frozen with the benchmark but never
shown to the optimization agent.
"""
from __future__ import annotations

from typing import Callable

import torch

KERNEL = 7
Inputs = dict[str, torch.Tensor]
Shape = tuple[int, int, int, int]  # N, C, H, W
Profile = Callable[[Shape, torch.Generator, torch.device], Inputs]


def _normal(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
            *, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std + mean


def _weights(channels: int, generator: torch.Generator, device: torch.device, *, std: float) -> torch.Tensor:
    return _normal((channels, KERNEL, KERNEL), generator, device, std=std)


def unit_normal(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Standard-normal activations with trained-scale weights."""
    return {"x": _normal(shape, generator, device), "weight": _weights(shape[1], generator, device, std=0.15)}


def relu_sparse(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Post-activation inputs: roughly half the values are exactly zero."""
    return {"x": torch.relu(_normal(shape, generator, device)),
            "weight": _weights(shape[1], generator, device, std=0.15)}


def large_range(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Large activations and weights: outputs reach thousands, so FP16 accumulation overflows or loses digits."""
    return {"x": _normal(shape, generator, device, std=6.0), "weight": _weights(shape[1], generator, device, std=0.6)}


def smooth_images(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Spatially correlated inputs like feature maps: low-frequency noise upsampled to full resolution."""
    n, c, h, w = shape
    coarse = _normal((n, c, max(h // 8, 1), max(w // 8, 1)), generator, device, std=2.0)
    x = torch.nn.functional.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    return {"x": x + _normal(shape, generator, device, std=0.1), "weight": _weights(c, generator, device, std=0.15)}


PROFILES: dict[str, Profile] = {
    "unit_normal": unit_normal,
    "relu_sparse": relu_sparse,
    "large_range": large_range,
    "smooth_images": smooth_images,
}

# (profile, seed, N, C, H, W). Acceptance uses profiles, seeds and shapes that
# search never sees. 57x61 and 61x59 are odd, so image rows (W halves) are not
# 16-byte aligned and no tile size divides them; 14x14 is smaller than two halos.
SPLITS: dict[str, list[tuple[str, int, int, int, int, int]]] = {
    "smoke": [("unit_normal", 0, 2, 96, 56, 56)],
    # Search and acceptance span the same ranges: N 2..16, C 96..768 including an
    # unaligned count, spatial 14..61 including odd sizes, and all four value
    # profiles. Search holds the extremes; acceptance samples the interior with
    # other seeds, so a held-out failure indicts the candidate, not the split.
    "search": [("large_range", 11, 16, 96, 56, 56),
               ("relu_sparse", 12, 4, 512, 28, 28),
               ("smooth_images", 13, 4, 104, 61, 57),
               ("unit_normal", 14, 2, 768, 14, 14)],
    "acceptance": [("large_range", 1001, 8, 256, 56, 56),
                   ("smooth_images", 1002, 12, 160, 56, 56),
                   ("unit_normal", 1003, 4, 192, 57, 61),
                   ("relu_sparse", 1004, 4, 100, 61, 59),
                   ("unit_normal", 1005, 2, 704, 14, 14)],
}


def generate(profile: str, shape: Shape, seed: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    values = PROFILES[profile](shape, generator, device)
    return {"x": values["x"].to(torch.float16).contiguous(),
            "weight": values["weight"].to(torch.float16).contiguous()}
