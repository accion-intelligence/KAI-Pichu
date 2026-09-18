# Depthwise 7×7 convolution, NCHW FP16

## Operation

For every image `n`, channel `c` and output position `(i, j)`:

```
y[n, c, i, j] = Σ_{di=0..6} Σ_{dj=0..6}  x[n, c, i + di - 3, j + dj - 3] · w[c, di, dj]
```

Taps that fall outside the image contribute zero (same padding of 3, stride 1,
no dilation, no bias). Each channel has its own 7×7 filter and channels never
mix. This is cross-correlation, as in cuDNN and PyTorch; the kernel is not
flipped.

## Data

- `x`, `y`: FP16 (`__half`) in NCHW memory order. Element `(n, c, i, j)` lives
  at offset `((n * C + c) * H + i) * W + j`; the image column `j` is the
  fastest-varying index, and each `(n, c)` plane is `H * W` contiguous halves.
- `w`: FP16, shape `[C, 7, 7]`, element `(c, di, dj)` at `(c * 7 + di) * 7 + dj`.
- `N`, `C`, `H`, `W` vary between invocations. `H` and `W` can be odd, so an
  image row (`W * 2` bytes) is not a multiple of 16 and vectorized loads along
  `W` need an alignment check and a scalar tail; spatial sizes can also be
  smaller than two 3-pixel halos.
- Inputs are read-only. `y` is uninitialized on entry and must be completely
  written; nothing may be read from it.
- `workspace` is 256 MiB of scratch memory (`workspace_bytes`) with undefined
  contents on every call. It may be used or ignored.

## Precision

`y` is compared element-wise with a PyTorch FP32 reference rounded to FP16 at
`atol = rtol = 2e-3`. Accumulate the 49 taps in FP32 (or better); FP16
accumulation does not meet the tolerance on the larger-magnitude cases.

## Entry point

`bridge.cu` includes `solution.cu` and exposes the fixed C ABI `kai_launch`.
`solution.cu` must define:

```c
int launch_depthwise_conv(const __half* x, const __half* w, __half* y,
                          int n, int c, int h, int width,
                          void* workspace, size_t workspace_bytes, cudaStream_t stream);
```

Return `0` on success or a negative value for unsupported arguments. Launch all
work on `stream`. The call is captured into a CUDA Graph and replayed: do not
allocate device memory, synchronize the device or the stream, or depend on
host-side state between calls. Static caches (a library handle, a per-shape
plan) are allowed only if they are created on the first call for a shape,
which happens in eager warm-up before capture. The build is
`nvcc -O3 -std=c++17 --use_fast_math` for `sm_120`, linked against cuDNN and
the CUDA runtime; `solution.cu` is the only file that may change.

## Baseline

The baseline calls cuDNN's grouped convolution (group count = C) in NCHW and
lets `cudnnFindConvolutionForwardAlgorithmEx` choose cuDNN's fastest algorithm
for each shape. Calling cuDNN again cannot beat it. Each output needs 49 taps
from a 7×7 neighbourhood of its own plane; whether those taps are reused on
chip (registers, shared memory, L1) or fetched again from L2 decides the speed,
and at FP16 the workload sits far below both the memory and the arithmetic roof.

## Measurement

Timing covers the GPU work between two events recorded around one
`launch_depthwise_conv` call inside the graph. Activations and weights are
evicted from L2 before every invocation.
