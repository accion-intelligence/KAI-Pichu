"""Input distributions and case splits for the attention forward task.

Q, K and V are FP16 tensors of shape [batch, heads, seq, head_dim]. Values are
drawn in FP32 and rounded to FP16, so the oracle sees exactly the numbers the
kernel receives. The task fixes head_dim at 64; batch, heads, seq and the
causal flag vary across cases, including sequence lengths that are not
multiples of common tile sizes. Values vary so that softmax rows are flat,
sharply peaked, offset by a large common bias, or dominated by a few
heavy-tailed values.

This module is task-owner code. It is frozen with the benchmark but never
shown to the optimization agent.
"""
from __future__ import annotations

from typing import Callable

import torch

Inputs = dict[str, torch.Tensor]
Shape = tuple[int, int, int, int]  # batch, heads, seq, head_dim
Profile = Callable[[Shape, torch.Generator, torch.device], Inputs]


def _normal(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
            *, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std + mean


def _uniform(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
             low: float, high: float) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float32) * (high - low) + low


def unit_normal(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Independent standard-normal Q, K and V: moderately flat attention rows."""
    return {name: _normal(shape, generator, device) for name in ("q", "k", "v")}


def sharp_rows(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Large query magnitudes give sharply peaked softmax rows and logits far from zero."""
    return {"q": _normal(shape, generator, device, std=3.0),
            "k": _normal(shape, generator, device),
            "v": _normal(shape, generator, device)}


def shifted_keys(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Every key of a head shares a large offset, so each row's logits carry a common
    bias that softmax must cancel numerically instead of overflowing."""
    batch, heads, _, head_dim = shape
    offset = _normal((batch, heads, 1, head_dim), generator, device, std=4.0)
    return {"q": _normal(shape, generator, device, std=1.5),
            "k": _normal(shape, generator, device) + offset,
            "v": _normal(shape, generator, device)}


def spiky_values(shape: Shape, generator: torch.Generator, device: torch.device) -> Inputs:
    """Half a percent of V entries lie far outside the bulk, so a few keys dominate some outputs."""
    v = _normal(shape, generator, device)
    spiked = torch.rand(shape, generator=generator, device=device) < 0.005
    magnitude = _uniform(shape, generator, device, 20.0, 50.0)
    sign = torch.where(torch.rand(shape, generator=generator, device=device) < 0.5, -1.0, 1.0)
    return {"q": _normal(shape, generator, device),
            "k": _normal(shape, generator, device),
            "v": torch.where(spiked, sign * magnitude, v)}


PROFILES: dict[str, Profile] = {
    "unit_normal": unit_normal,
    "sharp_rows": sharp_rows,
    "shifted_keys": shifted_keys,
    "spiky_values": spiky_values,
}

# (profile, seed, batch, heads, seq, causal). head_dim is fixed by the task.
# Acceptance uses seeds and shapes that search never sees, but search spans the same
# ranges: every value profile, both causal flags, batch 1-4, heads 16-32, seq 2048-8192
# including a length that is not a multiple of common tile sizes.
# Shapes are LLM-scale so that one invocation takes milliseconds; sub-millisecond
# kernels are dominated by launch and clock noise on most machines.
SPLITS: dict[str, list[tuple[str, int, int, int, int, bool]]] = {
    "smoke": [("unit_normal", 0, 1, 8, 2048, False)],
    "search": [("unit_normal", 11, 2, 16, 4096, False),
               ("sharp_rows", 12, 1, 32, 2600, True),
               ("shifted_keys", 13, 4, 16, 2048, True),
               ("spiky_values", 14, 1, 16, 8192, False)],
    "acceptance": [("unit_normal", 1001, 2, 16, 4096, True),
                   ("shifted_keys", 1002, 2, 16, 4096, False),
                   ("sharp_rows", 1003, 1, 32, 3000, True),
                   ("spiky_values", 1004, 4, 16, 2048, False),
                   ("unit_normal", 1005, 1, 16, 8192, True)],
}


def generate(profile: str, shape: Shape, seed: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    return {name: tensor.to(torch.float16)
            for name, tensor in PROFILES[profile](shape, generator, device).items()}
