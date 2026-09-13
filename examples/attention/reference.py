"""Independent FP16 reference for scaled dot-product attention.

The reference shares no code with the fused kernels PyTorch dispatches to: it
is the plain composition of two matrix products and a softmax. Inputs and the
output are FP16, like the kernel's; intermediates are FP32, as in cuDNN's
fused attention (intermediate_data_type = FLOAT). FP16 intermediates would be
too coarse to serve as an oracle: at logit magnitude 50 the FP16 spacing is
0.03, which is a several-percent error in every softmax weight. The reference
works on a few heads at a time so the FP32 score matrix stays bounded.

Frozen with the benchmark; never shown to the optimization agent.
"""
from __future__ import annotations

import torch

# FP16 candidate outputs are compared with the FP16 reference at this tolerance.
# Each side carries about 5e-4 relative FP16 rounding, so a kernel that also
# accumulates softmax or P·V in FP16 does not fit; FP32 accumulation does.
ATOL = 1e-3
RTOL = 1e-3


HEADS_PER_CHUNK = 2  # Bounds the FP32 [heads, seq, seq] score matrix held at once.


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      *, scale: float, causal: bool) -> torch.Tensor:
    """softmax(Q Kᵀ · scale [+ causal mask]) V with FP32 intermediates, returned as FP16."""
    batch, heads, seq, _ = q.shape
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    future = torch.ones(seq, seq, dtype=torch.bool, device=q.device).triu(diagonal=1)
    for b in range(batch):
        for h in range(0, heads, HEADS_PER_CHUNK):
            block = slice(h, h + HEADS_PER_CHUNK)
            scores = torch.matmul(q[b, block].float(), k[b, block].float().transpose(-1, -2)) * scale
            if causal:
                scores.masked_fill_(future, float("-inf"))
            out[b, block] = torch.matmul(torch.softmax(scores, dim=-1), v[b, block].float()).to(q.dtype)
    return out


def uniform_attention(v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    """The output of a kernel that ignores Q and K and averages the allowed values.

    Used as an invalid observation the validator must reject.
    """
    values = v.float()
    if causal:
        counts = torch.arange(1, v.shape[2] + 1, dtype=torch.float32, device=v.device).view(1, 1, -1, 1)
        return (values.cumsum(dim=2) / counts).to(v.dtype)
    return values.mean(dim=2, keepdim=True).expand_as(values).to(v.dtype)
