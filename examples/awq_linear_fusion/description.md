# AWQ INT4 linear layer: dequantize + GEMM

## What exists

Two stages, launched back to back by `launch_awq_linear` in `solution.cu`:

```
W[K, N] = (q[K, N] - zero[K/G, N]) * scale[K/G, N]     kernels/dequantize_weights.cu   launch_dequantize_weights
y[M, N] = x[M, K] · W[K, N]                              kernels/cublas_gemm.cu          launch_fp16_gemm
```

`dequantize_weights` is AutoAWQ's own kernel (one thread per packed int32,
eight outputs per thread, FP16 arithmetic through `dequantize.cuh`). It writes
the whole FP16 weight matrix, `K · N · 2` bytes, into the workspace; cuBLAS
then reads it back. For a decode step (`M` of 1 to 8) that round trip moves
more than four times the bytes of the packed weights, and the packed weights
are read only once either way. The kernel sources are read-only reference
material; `solution.cu` is the only file that may change.

## The task

Produce the same `y` with fewer launches and less traffic. The natural target
is one kernel that reads the packed INT4 weights, zero points and scales,
dequantizes in registers and accumulates the dot products directly, never
materializing `W`. `solution.cu` may be rewritten from scratch: it need not
keep the `#include`s, the workspace or the cuBLAS call, only the
`launch_awq_linear` entry point below. It may still `#include` the files
under `kernels/` to reuse `dequantize_s4_to_fp16x2` or other pieces.

## Data layout (AWQ)

- `x`: FP16, `[M, K]`, row-major. `y`: FP16, `[M, N]`, row-major,
  uninitialized on entry, must be completely written.
- `qweight`: int32, `[K, N/8]`, row-major. Word `qweight[k][c]` packs the eight
  unsigned 4-bit values of columns `8c .. 8c+7` of input row `k`. **The nibble
  order is interleaved:** column `8c + i` is stored at nibble position
  `P[i]` with `P = [0, 4, 1, 5, 2, 6, 3, 7]`, so nibble positions 0..7 hold
  columns 0, 2, 4, 6, 1, 3, 5, 7. Value of column `8c+i` is
  `(qweight[k][c] >> (4 * P[i])) & 0xF`.
- `qzeros`: int32, `[K/G, N/8]`, packed exactly like `qweight`; one 4-bit zero
  point per group and output column.
- `scales`: FP16, `[K/G, N]`, one scale per group and output column.
- `G = group_size = 128`; input row `k` belongs to group `k / G`. `K` is a
  multiple of 128 and `N` a multiple of 64; `M` is 1 to 8; `K` and `N` vary
  between invocations.
- Dequantized weight: `W[k][n] = (q[k][n] - zero[k/G][n]) * scale[k/G][n]`.
- `workspace`: at least `K · N · 2` bytes with undefined contents on every call.
  A fused implementation may ignore it.

## Precision

`y` is compared element-wise with an FP32 PyTorch reference (dequantize and
multiply in FP32, round once to FP16) at `atol = rtol = 1e-2`. FP16 weights
with FP32 accumulation meet it; FP16 accumulation over `K` of 4096 or more does
not. Getting the nibble order or the zero point wrong fails every case.

## Entry point

`bridge.cu` includes `solution.cu` and exposes the fixed C ABI `kai_launch`.
`solution.cu` must define:

```c
int launch_awq_linear(const __half* x, const int* qweight, const int* qzeros, const __half* scales,
                      __half* y, int m, int k, int n, int group_size,
                      void* workspace, size_t workspace_bytes, cudaStream_t stream);
```

Return `0` on success or a negative value for unsupported arguments. Launch all
work on `stream`. The call is captured into a CUDA Graph and replayed: do not
allocate device memory, synchronize, or keep host-side state between calls
(a lazily created cuBLAS handle, as in the baseline, is acceptable). The build
is `nvcc -O3 -std=c++17 --use_fast_math` for `sm_120`, linked against cuBLAS
and the CUDA runtime.

## Measurement

Timing covers the GPU work between two events recorded around one
`launch_awq_linear` call inside the graph, however many kernels it launches.
The activations, packed weights, scales and zeros are evicted from L2 before
every invocation, so the workload is bound by weight traffic: at `M ≤ 8` the
packed weights (`K · N / 2` bytes) are the floor.
