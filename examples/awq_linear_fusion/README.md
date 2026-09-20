# AWQ INT4 linear layer: a `kind: fusion` task on real kernels

AutoAWQ serves 4-bit LLM weights two ways. For small batches it runs a fused
kernel that dequantizes inside the matrix product; for larger inputs it runs
its `dequantize_weights` kernel to materialize FP16 weights and then calls
`torch.matmul`. This task hands the agent the second pipeline, exactly as
AutoAWQ ships it, and asks for the first: one kernel that reads the packed
INT4 weights, zero points and scales and never writes the FP16 matrix.

A recorded optimization run on this task is described in
[docs/showcase-awq-fusion.md](../../docs/showcase-awq-fusion.md).

## The supplied kernels

| File | Origin | Role |
| --- | --- | --- |
| `kernels/dequantize.cuh` | AutoAWQ (MIT), modified from NVIDIA FasterTransformer | `dequantize_s4_to_fp16x2`: eight INT4 values to four `half2` with `lop3` |
| `kernels/dequantize_weights.cu` | AutoAWQ (MIT) kernel, verbatim, plus a 15-line launcher | `W[K, N] = (q - zero) * scale` in FP16, one thread per packed int32 |
| `kernels/cublas_gemm.cu` | this repository | `y = x · W` through `cublasGemmEx`, the stand-in for `torch.matmul` |

Both kernel sources are frozen with the benchmark, shown to the agent read-only
and rejected as candidate edits. `baseline/solution.cu` launches them in
sequence through the workspace; a candidate replaces that file.

AutoAWQ's hand-written fused kernel, `gemm_forward_cuda` in `autoawq-kernels`,
consumes exactly this `[K, N/8]` layout. Build the extension for your GPU and
call it on the same fixtures to get an expert yardstick for a candidate; keep
that number out of the agent-visible files.

## Definition

- `y[M, N] = x[M, K] · W[K, N]`, `W = (q - zero) * scale` per group of 128 input
  rows. AWQ packing: eight 4-bit values per int32, nibble positions holding
  columns 0, 2, 4, 6, 1, 3, 5, 7.
- FP16 activations and output; FP32 accumulation is required to meet the
  tolerance. Reference: dequantize and multiply in FP32 with PyTorch, compared
  at `atol = rtol = 1e-2`.
- Weights are generated with per-channel scale spread and quantized here with
  asymmetric per-group min/max; the packing code is the task owner's and is
  never shown to the agent.

## Cases

| Split | Cases (M × K × N) |
| --- | --- |
| smoke | 1 × 1024 × 1024 |
| search | 1 × 4096 × 4096 standard normal; 4 × 4096 × 11008 outlier channels; 8 × 11008 × 4096 spiky groups; 2 × 5120 × 13824 large activations |
| acceptance | 1 × 11008 × 4096 standard normal; 8 × 4096 × 4096 spiky groups; 2 × 4096 × 11008 outlier channels; 3 × 5120 × 13824 large activations |

Shapes are Llama 7B and 13B projections at decode-time row counts. Search and
acceptance span the same ranges of M, K and N and the same four data profiles;
acceptance moves the points and seeds inside those ranges so that a held-out
failure indicts the candidate rather than the split design. The
`spiky_groups` profile makes per-group scales differ by 6× so a kernel that
indexes the wrong group fails; `outlier_channels` mimics LLM hidden states.

## Correctness

The validator rejects, per case, a perturbed output, a NaN, a missing output,
a kernel that unpacks nibbles in natural order, and a kernel that drops the
zero point.

## Measurement

`operator_latency_ms` is the GPU time between two external CUDA events captured
around one launch inside a CUDA Graph, whether it launches two kernels or one.
Activations, packed weights, scales and zeros are evicted from L2 before every
invocation. Every fixture carries a workspace of `K · N · 2` bytes plus slack
that the baseline uses for the FP16 matrix.

## Requirements

A CUDA GPU with PyTorch and `nvcc` for `sm_120` (edit `options.cuda_arch` for
another architecture); cuBLAS comes with the CUDA toolkit. Preflight without
model calls:

```bash
python -m kai_pichu benchmark validate examples/awq_linear_fusion/benchmark.yaml \
  --split search --checks-only --output awq-checks.json
```
