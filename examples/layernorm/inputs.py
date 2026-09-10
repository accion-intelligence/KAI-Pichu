"""Input distributions for the LayerNorm task.

The shape is fixed by the baseline's 1024-column specialization, so diversity
comes from the values: per-row offsets and scales, near-constant rows, sparse
activation spikes, and affine parameters that are never the identity. Search
and acceptance draw from different profiles and seeds, so a candidate cannot
pass by matching the distribution it was tuned on.

This module is task-owner code. It is frozen with the benchmark but never
shown to the optimization agent.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

Inputs = dict[str, torch.Tensor]
Profile = Callable[[int, int, torch.Generator, torch.device], Inputs]


def _normal(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
            *, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std + mean


def _uniform(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
             low: float, high: float) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32) * (high - low) + low


def _trained_affine(cols: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Weights like a trained LayerNorm: scale around one with a few sign flips, small bias."""
    gamma = _uniform((cols,), generator, device, 0.5, 1.5)
    flipped = torch.rand((cols,), generator=generator, device=device) < 0.05
    gamma = torch.where(flipped, -gamma, gamma)
    beta = _normal((cols,), generator, device, std=0.5)
    return gamma, beta


def unit_normal(rows: int, cols: int, generator: torch.Generator, device: torch.device) -> Inputs:
    """Standard-normal activations with trained-style affine parameters."""
    x = _normal((rows, cols), generator, device)
    gamma, beta = _trained_affine(cols, generator, device)
    return {"x": x, "gamma": gamma, "beta": beta}


def shifted_rows(rows: int, cols: int, generator: torch.Generator, device: torch.device) -> Inputs:
    """Residual-stream style rows: every row has its own offset and scale."""
    offset = _normal((rows, 1), generator, device, std=3.0)
    log_scale = _uniform((rows, 1), generator, device, math.log(0.1), math.log(10.0))
    x = offset + torch.exp(log_scale) * _normal((rows, cols), generator, device)
    gamma, beta = _trained_affine(cols, generator, device)
    return {"x": x, "gamma": gamma, "beta": beta}


def low_variance(rows: int, cols: int, generator: torch.Generator, device: torch.device) -> Inputs:
    """Nearly constant rows: variance within an order of magnitude of eps stresses rsqrt precision."""
    x = _normal((rows, cols), generator, device, mean=2.0, std=0.01)
    gamma = _normal((cols,), generator, device, mean=1.0, std=0.1)
    beta = _normal((cols,), generator, device, std=0.1)
    return {"x": x, "gamma": gamma, "beta": beta}


def sparse_outliers(rows: int, cols: int, generator: torch.Generator, device: torch.device) -> Inputs:
    """Activation spikes: half a percent of entries far outside the bulk, with random sign."""
    x = _normal((rows, cols), generator, device)
    spiked = torch.rand((rows, cols), generator=generator, device=device) < 0.005
    magnitude = _uniform((rows, cols), generator, device, 20.0, 50.0)
    sign = torch.where(torch.rand((rows, cols), generator=generator, device=device) < 0.5, -1.0, 1.0)
    x = torch.where(spiked, sign * magnitude, x)
    gamma, beta = _trained_affine(cols, generator, device)
    return {"x": x, "gamma": gamma, "beta": beta}


PROFILES: dict[str, Profile] = {
    "unit_normal": unit_normal,
    "shifted_rows": shifted_rows,
    "low_variance": low_variance,
    "sparse_outliers": sparse_outliers,
}

# Acceptance uses profiles and seeds that search never sees.
SPLITS: dict[str, list[tuple[str, int]]] = {
    "smoke": [("unit_normal", 0)],
    "search": [("unit_normal", 11), ("shifted_rows", 12)],
    "acceptance": [("unit_normal", 1001), ("shifted_rows", 1002),
                   ("low_variance", 1003), ("sparse_outliers", 1004)],
}


def generate(profile: str, rows: int, cols: int, seed: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    return PROFILES[profile](rows, cols, generator, device)
