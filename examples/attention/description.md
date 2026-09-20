# Attention forward, FP16, head_dim = 64

## Operation

For every batch element `b`, head `h` and query row `i`:

```
s[i, j] = scale · Σ_d Q[b, h, i, d] · K[b, h, j, d]        scale = 1 / sqrt(64)
p[i, :] = softmax(s[i, :])                                 over the allowed keys j
O[b, h, i, d] = Σ_j p[i, j] · V[b, h, j, d]
```

With `causal = 1`, key `j` is allowed only when `j <= i` (strictly-future keys
are masked, the diagonal is kept); every row has at least one allowed key. With
`causal = 0`, all `seq` keys are allowed. This is the same definition as cuDNN's
fused scaled dot-product attention in BHSD layout.

## Data

- `q`, `k`, `v`, `o`: FP16 (`__half`), contiguous, shape `[batch, heads, seq, 64]`,
  element `(b, h, i, d)` at offset `((b * heads + h) * seq + i) * 64 + d`.
- `head_dim` is always 64. `batch`, `heads` and `seq` vary between invocations;
  `seq >= 1` and is not necessarily a multiple of any tile size.
- Inputs are read-only. `o` is uninitialized on entry and must be completely
  written; nothing may be read from it.
- `workspace` is 256 MiB of scratch memory (`workspace_bytes`) with undefined
  contents on every call. It may be used freely or ignored; shapes whose scores
  exceed it must be tiled.
- Logits can be large and rows can be sharply peaked or carry a large common
  offset. Use a numerically stable softmax (subtract the row maximum or an
  equivalent running maximum). Non-finite outputs fail validation.

## Precision

The FP16 output is compared element-wise with an independent FP16 reference
that accumulates in FP32, at `atol = rtol = 1e-3`. FP16 rounding of the output
alone costs about 5e-4 relative error, so softmax and the P·V accumulation need
more precision than FP16 to pass; FP32 accumulation is sufficient.

## Entry point

`bridge.cu` includes `solution.cu` and exposes the fixed C ABI `kai_launch`.
`solution.cu` must define:

```c
int launch_attention_forward(
    const __half* q, const __half* k, const __half* v, __half* o,
    int batch, int heads, int seq, int head_dim, float scale, int causal,
    void* workspace, size_t workspace_bytes, cudaStream_t stream);
```

Return `0` on success or a negative value for unsupported arguments. Launch
all work on `stream`. The call is captured into a CUDA Graph and replayed:
do not call `cudaMalloc`/`cudaFree`, do not synchronize the device or the
stream, and do not keep host-side state between calls; use `workspace` for
scratch. The build is `nvcc -O3 -std=c++17 --use_fast_math -lineinfo` for `sm_120` with
the PyTorch/ATen headers on the include path, linked against libtorch, cuBLAS
and the CUDA runtime, so ATen operators and cuBLAS calls are available. The
baseline is PyTorch's fused `scaled_dot_product_attention`; calling it again
cannot be faster than the baseline. `solution.cu` is the only file that may
change.

## Measurement

Timing covers the GPU work between two events recorded immediately around one
`launch_attention_forward` call inside the graph. Q, K and V are evicted from
L2 before every invocation.
