"""Independent reference for the depthwise 7x7 convolution.

PyTorch's conv2d on FP32 upcasts of the FP16 tensors, rounded back to FP16.
It goes through PyTorch's own convolution dispatch, not the cuDNN call the
baseline makes. Frozen with the benchmark; never shown to the agent.
"""
from __future__ import annotations

import torch

KERNEL = 7
PADDING = KERNEL // 2

# FP16 outputs are compared with the FP16-rounded FP32 reference. Both sides
# carry FP16 output rounding; 49-term FP32 accumulation-order differences are
# far below this, while FP16 accumulation is not.
ATOL = 2e-3
RTOL = 2e-3


def depthwise_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x: [N, C, H, W] FP16; weight: [C, 7, 7] FP16 -> [N, C, H, W] FP16."""
    channels = x.shape[1]
    out = torch.nn.functional.conv2d(x.float(), weight.float().view(channels, 1, KERNEL, KERNEL),
                                     padding=PADDING, groups=channels)
    return out.to(x.dtype).contiguous()


def without_padding_rows(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """A kernel that treats the border as if the image continued with its edge values (replicate
    instead of zero padding); an invalid observation the validator must reject."""
    channels = x.shape[1]
    padded = torch.nn.functional.pad(x.float(), (PADDING,) * 4, mode="replicate")
    out = torch.nn.functional.conv2d(padded, weight.float().view(channels, 1, KERNEL, KERNEL), groups=channels)
    return out.to(x.dtype).contiguous()


def flipped_kernel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """True convolution instead of cross-correlation (kernel flipped); an invalid observation."""
    return depthwise_conv(x, weight.flip(-1, -2))
