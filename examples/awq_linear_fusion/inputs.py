"""Input generation and case splits for the AWQ INT4 linear-layer fusion task.

Weights are drawn as an FP32 matrix with per-output-channel scale variation
and a few outlier input channels, then quantized to asymmetric 4-bit values
per group of `GROUP_SIZE` input rows and packed in the AWQ layout. Activations
are FP16 with profiles that mimic LLM hidden states. Shapes follow Llama 7B
and 13B projections.

This module is task-owner code. It is frozen with the benchmark but never
shown to the optimization agent.
"""
from __future__ import annotations

from typing import Callable

import torch

GROUP_SIZE = 128
# Column i of a packed int32 is stored at nibble position NIBBLE_OF_COLUMN[i].
NIBBLE_OF_COLUMN = (0, 4, 1, 5, 2, 6, 3, 7)

Inputs = dict[str, torch.Tensor]
Shape = tuple[int, int, int]  # m, k, n
Profile = Callable[[Shape, torch.Generator, torch.device], tuple[torch.Tensor, torch.Tensor]]


def _normal(shape: tuple[int, ...], generator: torch.Generator, device: torch.device,
            *, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * std + mean


def _weights(k: int, n: int, generator: torch.Generator, device: torch.device, *, spiky: bool) -> torch.Tensor:
    """Trained-looking weights: per-output-channel scale spread, optional spiky groups."""
    channel_scale = torch.exp(_normal((1, n), generator, device, std=0.5)) * 0.02
    w = _normal((k, n), generator, device) * channel_scale
    if spiky:
        groups = k // GROUP_SIZE
        spike = torch.rand((groups, 1), generator=generator, device=device) < 0.1
        w = w.view(groups, GROUP_SIZE, n) * torch.where(spike, 6.0, 1.0).view(groups, 1, 1)
        w = w.reshape(k, n)
    return w


def unit_normal(shape: Shape, generator: torch.Generator, device: torch.device):
    m, k, n = shape
    return _normal((m, k), generator, device), _weights(k, n, generator, device, spiky=False)


def outlier_channels(shape: Shape, generator: torch.Generator, device: torch.device):
    """A few input channels carry values 20x larger, as in LLM hidden states."""
    m, k, n = shape
    x = _normal((m, k), generator, device)
    outliers = torch.rand((1, k), generator=generator, device=device) < 0.01
    x = x * torch.where(outliers, 20.0, 1.0)
    return x, _weights(k, n, generator, device, spiky=False)


def spiky_groups(shape: Shape, generator: torch.Generator, device: torch.device):
    """Some quantization groups have 6x larger weights, so per-group scales differ widely."""
    m, k, n = shape
    return _normal((m, k), generator, device), _weights(k, n, generator, device, spiky=True)


def large_activations(shape: Shape, generator: torch.Generator, device: torch.device):
    m, k, n = shape
    return _normal((m, k), generator, device, std=4.0), _weights(k, n, generator, device, spiky=False)


PROFILES: dict[str, Profile] = {
    "unit_normal": unit_normal,
    "outlier_channels": outlier_channels,
    "spiky_groups": spiky_groups,
    "large_activations": large_activations,
}


def quantize_awq(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit per-group quantization packed in the AWQ layout.

    Returns qweight int32 [K, N/8], qzeros int32 [K/G, N/8], scales fp16 [K/G, N].
    """
    k, n = w.shape
    groups = w.view(k // GROUP_SIZE, GROUP_SIZE, n)
    low = groups.amin(dim=1)
    high = groups.amax(dim=1)
    scale = torch.clamp((high - low) / 15.0, min=1e-5)
    zero = torch.clamp(torch.round(-low / scale), 0, 15)
    q = torch.clamp(torch.round(groups / scale.unsqueeze(1) + zero.unsqueeze(1)), 0, 15).to(torch.int64)
    return _pack(q.reshape(k, n)), _pack(zero.to(torch.int64)), scale.to(torch.float16)


def _pack(values: torch.Tensor) -> torch.Tensor:
    """int64 [R, C] of 4-bit values -> int32 [R, C/8] in AWQ nibble order."""
    rows, cols = values.shape
    grouped = values.view(rows, cols // 8, 8)
    shifts = torch.tensor([4 * nibble for nibble in NIBBLE_OF_COLUMN], device=values.device, dtype=torch.int64)
    words = (grouped << shifts).sum(dim=-1)
    return words.to(torch.int32)  # values are within [0, 2^32); bit pattern reinterpretation is intended


# (profile, seed, m, k, n). Search and acceptance span the same ranges: M from
# 1 to 8, K from 4096 to 11008, N from 4096 to 13824, all four activation and
# weight profiles. Acceptance changes the points and seeds inside those ranges,
# never the ranges, so a held-out failure means the candidate is wrong, not
# that it was never shown that regime. Shapes are Llama 7B (4096 / 11008) and
# 13B (5120 / 13824) projections; k is a multiple of the group size and n of
# 64, as the supplied kernel requires.
SPLITS: dict[str, list[tuple[str, int, int, int, int]]] = {
    "smoke": [("unit_normal", 0, 1, 1024, 1024)],
    "search": [("unit_normal", 11, 1, 4096, 4096),
               ("outlier_channels", 12, 4, 4096, 11008),
               ("spiky_groups", 13, 8, 11008, 4096),
               ("large_activations", 14, 2, 5120, 13824)],
    "acceptance": [("unit_normal", 1001, 1, 11008, 4096),
                   ("spiky_groups", 1002, 8, 4096, 4096),
                   ("outlier_channels", 1003, 2, 4096, 11008),
                   ("large_activations", 1004, 3, 5120, 13824)],
}


def generate(profile: str, shape: Shape, seed: int, device: torch.device) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed)
    x, w = PROFILES[profile](shape, generator, device)
    qweight, qzeros, scales = quantize_awq(w)
    return {"x": x.to(torch.float16).contiguous(), "qweight": qweight.contiguous(),
            "qzeros": qzeros.contiguous(), "scales": scales.contiguous()}
