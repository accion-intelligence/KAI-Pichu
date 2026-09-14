# Epilogue fusion: a `kind: fusion` task

The user supplies working kernels; the agent's job is to fuse them, not to
invent a new operator. This example fuses a classic transformer epilogue,
`out = gelu(x + bias) + residual`, given as three independent FP32 CUDA kernels.

## How a fusion task differs

| Manifest | Effect |
| --- | --- |
| `kind: fusion` | Declares the task class. |
| `fusion.kernels` | The supplied kernels in execution order: name, source files, launcher entry, one-line semantics. |
| `fusion.intermediates` | Buffers handed between kernels that a fused version may keep on chip. |

The kernel sources are frozen with the benchmark, shown to the agent as
read-only material, and rejected as candidate edits. `implementation.files`
(here `baseline/solution.cu`) stays the only editable surface: the baseline
version launches the kernels in sequence, and a candidate replaces that with
fused launches. The agent's context carries a structured `fusion` block with
the kernel list, entries and intermediates in addition to the sources.

## Files

| File | Role | Visible to the agent |
| --- | --- | --- |
| `kernels/*.cu` | The three supplied kernels | yes, read-only |
| `baseline/solution.cu` | Editable pipeline; must define `launch_epilogue` | yes (editable) |
| `bridge.cu` | Fixed C ABI `kai_launch` around the launcher | yes |
| `description.md` | Semantics, layout, precision and launch constraints | yes |
| `adapter.py` | Compiles the workspace, captures one invocation in a CUDA Graph, validates | no |
| `inputs.py` | Value profiles and the smoke/search/acceptance splits | no |
| `reference.py` | PyTorch oracle and the tolerance | no |

## Cases

| Split | Cases |
| --- | --- |
| smoke | 1024×2048 |
| search | 4096×4096 standard normal; 2047×3001 wide range |
| acceptance | 4096×4096; 8192×2048 per-row offsets; 2049×2999 wide range; 1024×8192 dominant bias |

The odd shapes leave `rows * cols` with a remainder for every vector width and
start rows at unaligned addresses, so a fused kernel with `float4` loads needs
alignment handling and a scalar tail in search and in acceptance alike. The
`dominant_bias` profile makes a fusion that drops the bias stage fail loudly;
`wide_range` exercises GELU's saturated tails.

## Correctness

Outputs are compared with `torch.nn.functional.gelu(x + bias) + residual`
(exact erf GELU) at `atol = rtol = 1e-5`. The validator additionally rejects a
perturbed output, a NaN, a missing output, a pipeline without the bias stage,
and a pipeline using the tanh GELU approximation.

## Measurement

`operator_latency_ms` is the GPU time between two external CUDA events captured
around one launch inside a CUDA Graph, whether it launches three kernels or one.
The inputs are evicted from L2 before every invocation, so the pipeline is
memory-bound and fusion pays off directly in avoided passes.

## Requirements

A CUDA GPU with PyTorch and `nvcc` for `sm_120` (edit `options.cuda_arch` for
another architecture). Preflight without model calls:

```bash
python -m kai_core benchmark validate examples/fusion/benchmark.yaml \
  --split search --checks-only --output fusion-checks.json
```
