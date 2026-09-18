"""Independent PyTorch reference for the AWQ INT4 linear layer.

Unpacks the AWQ layout in plain tensor code, dequantizes in FP32 and multiplies
in FP32, then rounds to FP16. Shares no code with the CUDA kernels. Frozen with
the benchmark; never shown to the optimization agent.
"""
from __future__ import annotations

import torch

# Column i of a packed int32 lives at nibble position AWQ_NIBBLE_OF_COLUMN[i]:
# nibbles 0..7 hold columns 0, 2, 4, 6, 1, 3, 5, 7.
AWQ_NIBBLE_OF_COLUMN = (0, 4, 1, 5, 2, 6, 3, 7)

# FP16 outputs versus the FP32 reference. The kernels round each dequantized
# weight to FP16 and accumulate in FP32; both effects stay well inside this.
ATOL = 1e-2
RTOL = 1e-2


def unpack(packed: torch.Tensor, order: tuple[int, ...] = AWQ_NIBBLE_OF_COLUMN) -> torch.Tensor:
    """int32 [R, C/8] -> int64 [R, C] of 4-bit values, columns in logical order."""
    shifts = torch.tensor([4 * nibble for nibble in order], device=packed.device, dtype=torch.int64)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    return ((words.unsqueeze(-1) >> shifts) & 0xF).reshape(packed.shape[0], packed.shape[1] * 8)


def dequantize(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int,
               *, order: tuple[int, ...] = AWQ_NIBBLE_OF_COLUMN, use_zeros: bool = True) -> torch.Tensor:
    """FP32 weights [K, N] = (q - zero) * scale with per-group zero and scale."""
    q = unpack(qweight, order).float()
    zeros = unpack(qzeros, order).float() if use_zeros else torch.zeros_like(unpack(qzeros, order).float())
    groups = torch.arange(q.shape[0], device=q.device) // group_size
    return (q - zeros[groups]) * scales.float()[groups]


def awq_linear(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
               group_size: int) -> torch.Tensor:
    """y[M, N] = x[M, K] · dequantize(W)[K, N], FP32 math, FP16 result."""
    return (x.float() @ dequantize(qweight, qzeros, scales, group_size)).to(x.dtype)


def natural_order_linear(x, qweight, qzeros, scales, group_size) -> torch.Tensor:
    """A kernel that ignores AWQ's interleaved nibble order; an invalid observation to reject."""
    return (x.float() @ dequantize(qweight, qzeros, scales, group_size, order=(0, 1, 2, 3, 4, 5, 6, 7))).to(x.dtype)


def no_zero_point_linear(x, qweight, qzeros, scales, group_size) -> torch.Tensor:
    """A kernel that forgets the zero point; an invalid observation to reject."""
    return (x.float() @ dequantize(qweight, qzeros, scales, group_size, use_zeros=False)).to(x.dtype)
