"""Input distributions and case splits for the epilogue fusion task.

x and residual are FP32 [rows, cols]; bias is FP32 [cols]. Shapes vary across
cases, including odd row and column counts, so a fused kernel cannot assume
aligned rows or a tail-free element count. Values cover GELU's linear,
transition and saturated regions.

This module is task-owner code. It is frozen with the benchmark but never
shown to the optimization agent.
"""
from __future__ import annotations

from typing import Callable

import torch

Inputs = dict[str, torch.Tensor]
Shape = tuple[int, int]  # rows, cols
Profile = Callable[[Shape, torch.Generator, torch.device], Inputs]


def _normal(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
            *, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std + mean


def unit_normal(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Standard-normal activations: most values in GELU's transition region."""
    return {"x": _normal(shape, generator, device), "bias": _normal((shape[1],), generator, device, std=0.5),
            "residual": _normal(shape, generator, device)}


def wide_range(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Large magnitudes: GELU saturates to identity or zero, and erf's tails matter."""
    return {"x": _normal(shape, generator, device, std=8.0), "bias": _normal((shape[1],), generator, device, std=2.0),
            "residual": _normal(shape, generator, device, std=4.0)}


def shifted_rows(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Each row has its own offset, so different rows land in different GELU regimes."""
    offset = _normal((shape[0], 1), generator, device, std=3.0)
    return {"x": _normal(shape, generator, device) + offset, "bias": _normal((shape[1],), generator, device),
            "residual": _normal(shape, generator, device)}


def dominant_bias(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Small activations under a large per-column bias: dropping the bias kernel is very visible."""
    return {"x": _normal(shape, generator, device, std=0.1), "bias": _normal((shape[1],), generator, device, std=3.0),
            "residual": _normal(shape, generator, device, std=0.1)}


PROFILES: dict[str, Profile] = {
    "unit_normal": unit_normal,
    "wide_range": wide_range,
    "shifted_rows": shifted_rows,
    "dominant_bias": dominant_bias,
}

# (profile, seed, rows, cols). Acceptance uses profiles, seeds and shapes that
# search never sees. The odd row and column counts (2047 x 3001, 2049 x 2999) make
# rows start at unaligned addresses and leave rows * cols with a remainder for
# every vector width, so vectorized kernels need a real tail path in both splits.
SPLITS: dict[str, list[tuple[str, int, int, int]]] = {
    "smoke": [("unit_normal", 0, 1024, 2048)],
    "search": [("unit_normal", 11, 4096, 4096), ("wide_range", 12, 2047, 3001)],
    "acceptance": [("unit_normal", 1001, 4096, 4096), ("shifted_rows", 1002, 8192, 2048),
                   ("wide_range", 1003, 2049, 2999), ("dominant_bias", 1004, 1024, 8192)],
}


def generate(profile: str, shape: Shape, seed: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    return PROFILES[profile](shape, generator, device)
