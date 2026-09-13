"""Independent PyTorch oracle for the bias-add, GELU, residual-add epilogue.

Shares no code with the supplied kernels: it composes PyTorch operators in
FP32. Frozen with the benchmark; never shown to the optimization agent.
"""
from __future__ import annotations

import torch

# The supplied kernels and any fused replacement compute in FP32 with erf-based
# GELU; only rounding-order differences remain, so the tolerance is tight.
ATOL = 1e-5
RTOL = 1e-5


def epilogue(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """gelu(x + bias) + residual with the exact (erf) GELU, all in FP32."""
    return torch.nn.functional.gelu(x + bias, approximate="none") + residual


def without_bias(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """A pipeline that dropped the bias kernel; an invalid observation to reject."""
    return torch.nn.functional.gelu(x, approximate="none") + residual


def tanh_gelu(x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """A pipeline that swapped in the tanh GELU approximation; an invalid observation to reject."""
    return torch.nn.functional.gelu(x + bias, approximate="tanh") + residual
