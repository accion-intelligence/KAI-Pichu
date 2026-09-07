"""
Task: LayerNorm Forward — hidden_size=1024

Source: flash-attention/csrc/layer_norm/ln_fwd_kernels.cuh
Operation: z = (x - mu) / sqrt(var + eps) * gamma + beta
"""
import torch

ROWS = 32 * 512
COLS = 1024
EPS = 1e-5

FUNCTION_SIGNATURE = """\
void launch_layernorm_forward(
    const float* x,       // [rows, cols] input
    const float* gamma,   // [cols] scale weight
    const float* beta,    // [cols] bias weight
    float* z,             // [rows, cols] normalized output
    float* mu,            // [rows] per-row mean
    float* rs,            // [rows] per-row reciprocal std
    int rows,
    int cols,
    float eps,
    cudaStream_t stream
);
"""

SCALAR_ARGS = {"rows": ROWS, "cols": COLS, "eps": EPS}

TOLERANCES = {"atol": 1e-4, "rtol": 1e-4}

HAS_CUDA_REFERENCE = True
REFERENCE_EXTRA_INCLUDES = ["reference_sources/layer_norm"]


def get_inputs():
    x = torch.randn(ROWS, COLS, device='cuda', dtype=torch.float32)
    gamma = torch.ones(COLS, device='cuda', dtype=torch.float32)
    beta = torch.zeros(COLS, device='cuda', dtype=torch.float32)
    return [("x", x), ("gamma", gamma), ("beta", beta)]


def get_outputs(inputs):
    x = inputs[0][1]
    z = torch.empty_like(x)
    mu = torch.empty(ROWS, device='cuda', dtype=torch.float32)
    rs = torch.empty(ROWS, device='cuda', dtype=torch.float32)
    return [("z", z), ("mu", mu), ("rs", rs)]


def reference_fn(inputs):
    """PyTorch reference for correctness validation."""
    x = inputs[0][1]
    gamma = inputs[1][1]
    beta = inputs[2][1]
    mu = x.mean(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    rs = torch.rsqrt(var + EPS)
    x_hat = (x - mu.unsqueeze(-1)) * rs.unsqueeze(-1)
    z = x_hat * gamma + beta
    return [("z", z), ("mu", mu), ("rs", rs)]


DESCRIPTION = """\
Implement a CUDA kernel for LayerNorm forward pass.

Given input x of shape (rows, cols), weight gamma and bias beta of shape (cols,):
  mu[i] = mean(x[i, :])
  rs[i] = rsqrt(var(x[i, :]) + eps)
  z[i, j] = (x[i, j] - mu[i]) * rs[i] * gamma[j] + beta[j]

Dimensions: rows=%d, cols=%d, eps=%g
""" % (ROWS, COLS, EPS)
