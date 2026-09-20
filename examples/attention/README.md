# Attention forward: an FP16 task built on the benchmark SDK

FP16 scaled dot-product attention with `head_dim = 64`, defined as in cuDNN's
fused attention (BHSD layout, `scale = 1/sqrt(64)`, causal mask over
strictly-future keys, FP32 accumulation). The baseline calls PyTorch's fused
`scaled_dot_product_attention` through the ATen C++ API, so a candidate has to
beat a library kernel, not a naive loop.

## Files

| File | Role | Visible to the agent |
| --- | --- | --- |
| `baseline/solution.cu` | Editable implementation; must define `launch_attention_forward` | yes (as the editable source) |
| `bridge.cu` | Fixed C ABI `kai_launch` around the launcher | yes |
| `description.md` | Semantics, layout, precision and launch constraints | yes |
| `adapter.py` | Compiles the workspace against libtorch, captures one invocation in a CUDA Graph, validates | no |
| `inputs.py` | Value profiles and the smoke/search/acceptance splits | no |
| `reference.py` | FP64 oracle and the tolerance | no |

## Cases

Shapes and masks differ across cases; `head_dim` is the only fixed dimension.

| Split | Cases |
| --- | --- |
| smoke | 1×8×2048, full attention |
| search | 2×16×4096 full attention; 2×16×4096 causal with sharply peaked rows |
| acceptance | 2×16×4096 causal; 2×16×4096 full with large shared key offsets; 1×32×3000 causal with peaked rows; 4×16×2048 full with heavy-tailed values; 1×16×8192 causal |

Shapes are LLM-scale (the range the cuDNN frontend attention benchmark uses) so
that one invocation takes milliseconds; sub-millisecond kernels are dominated by
launch and clock noise.

The value profiles are chosen so that a kernel without a stable softmax, a
kernel that ignores the mask, a kernel that assumes tile-aligned `seq`, a kernel
that accumulates in FP16, or a kernel that specializes to one shape fails
validation. Acceptance profiles, seeds and shapes are held out from search.

## Correctness

Inputs are drawn in FP32 and rounded to FP16. The reference is the plain
composition of two matrix products and a softmax on exactly those FP16 values,
with FP32 intermediates and an FP16 result, as in cuDNN's definition; it shares
no code with the fused kernels. Outputs are compared element-wise at
`atol = rtol = 1e-3`. The
validator additionally rejects, per case, a perturbed output, a NaN, a missing
output, the result of the opposite mask, and the result of ignoring Q and K
(uniform attention over the allowed values).

## Measurement

`operator_latency_ms` is the GPU time between two external CUDA events captured
around one launch inside a CUDA Graph; the output is poisoned and a 256 MiB
buffer is written before every invocation. Every fixture carries a 256 MiB
workspace that implementations may use as scratch.

## Requirements

A CUDA GPU with PyTorch, and `nvcc` for `sm_120` (edit `options.cuda_arch` for
another architecture). Compilation includes the ATen headers and takes longer
than a plain CUDA kernel. Preflight without model calls:

```bash
python -m kai_pichu benchmark validate examples/attention/benchmark.yaml \
  --split search --checks-only --output attention-checks.json
```
