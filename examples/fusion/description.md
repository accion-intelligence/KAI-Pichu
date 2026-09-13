# Epilogue fusion: bias-add, GELU, residual-add

## What exists

Three working FP32 kernels, each in its own file under `kernels/`, are launched
back to back by `launch_epilogue` in `solution.cu`:

```
y   = x + bias            kernels/bias_add.cu      launch_bias_add
z   = gelu(y)             kernels/gelu.cu          launch_gelu       (exact erf form)
out = z + residual        kernels/residual_add.cu  launch_residual_add
```

`y` and `z` live in the workspace, so the baseline reads and writes global
memory seven times per element. The kernel sources are read-only reference
material; `solution.cu` is the only file that may change.

## The task

Produce the same `out` with fewer, fused launches. A single kernel that reads
`x`, `bias` and `residual` once and writes `out` once is the natural target;
partial fusion is acceptable if it is faster. Keep the exact erf-based GELU
(`0.5 * v * (1 + erff(v / sqrt(2)))`); the tanh approximation fails validation.
Keep FP32 arithmetic.

## Data

- `x`, `residual`, `out`: FP32, contiguous, shape `[rows, cols]`, row-major.
- `bias`: FP32, shape `[cols]`.
- `rows` and `cols` vary between invocations; `cols` is not necessarily a
  multiple of 4 or 8, so vectorized loads need a scalar tail or alignment check.
- `out` is uninitialized on entry and must be completely written.
- `workspace` holds at least `2 * rows * cols * 4` bytes with undefined contents.
  A fused implementation may ignore it.

## Precision

`out` is compared element-wise with a PyTorch reference at `atol = rtol = 1e-5`.

## Entry point

`bridge.cu` includes `solution.cu` and exposes the fixed C ABI `kai_launch`.
`solution.cu` must define:

```c
int launch_epilogue(const float* x, const float* bias, const float* residual, float* out,
                    int rows, int cols, void* workspace, size_t workspace_bytes,
                    cudaStream_t stream);
```

Return `0` on success or a negative value for unsupported arguments. Launch all
work on `stream`. The call is captured into a CUDA Graph and replayed: do not
allocate device memory, synchronize, or keep host-side state between calls.
`solution.cu` may still `#include` the files under `kernels/` if it reuses parts
of them. The build is `nvcc -O3 -std=c++17 --use_fast_math` for `sm_120`,
linked against the CUDA runtime only.

## Measurement

Timing covers the GPU work between two events recorded around one
`launch_epilogue` call inside the graph, however many kernels it launches. The
inputs are evicted from L2 before every invocation, so the workload is
memory-bound and every avoided pass over `rows * cols` elements counts.
