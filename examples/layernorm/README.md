# LayerNorm forward: the packaged task

FP32 LayerNorm forward at `[16384, 1024]`, epsilon `1e-5`. The performance baseline is the
FlashAttention CUDA implementation retained here. The correctness reference is an
independent PyTorch implementation; `z`, `mu` and `rs` are all checked at
`atol = rtol = 1e-4`.

No optimized candidate or experimental report is bundled. A new optimization run starts a new search.

## Which manifest to use

Two manifests ship here. **They measure different boundaries and their numbers are not
comparable.**

| Manifest | Boundary | What it includes |
|---|---|---|
| `benchmark-graph-events.yaml` | Adapter metric: two external CUDA event nodes inside one CUDA graph | The operator only. Excludes host submission, compilation and graph capture. |
| `benchmark.yaml` | SDK `cuda_event` timer | Elapsed GPU time around the invocation; may include submission-related idle gaps. Not CPU wall-clock latency. |

**Start from `benchmark-graph-events.yaml`.** It is the default manifest for optimizing this operator.

The outer-event manifest is retained for boundary comparison, not as this operator’s optimization target. Submission-related idle gaps can prevent A/A calibration from passing. This does not mean the `cuda_event` boundary is unusable in general. If you need that boundary, investigate contention, synchronization and run-to-run variability on your hardware. Choose blocks, repetitions and tolerance for the precision your task requires before candidate search. Document any changes and recalibrate; do not repeatedly relax the criterion simply to obtain a pass.

See [Measurement](../../docs/MEASUREMENT.md) for what calibration does and does not
establish.

## Requirements

A CUDA-capable NVIDIA GPU with PyTorch installed, and an `nvcc` that targets your
architecture. The supplied manifests target SM120; other architectures need a matching `-arch`.

## Preflight

From the repository root, with no model calls:

```bash
python -m kai_core benchmark validate examples/layernorm/benchmark-graph-events.yaml \
  --split search --output layernorm-preflight.json
```

This runs the task checks, the baseline correctness check, and A/A calibration. It uses the
GPU. Add `--checks-only` to run the checks without calibration.

To run the optimization loop, follow [model configuration and execution](../../docs/QUICKSTART.md#3-choose-the-models-and-budget) with `examples/layernorm/benchmark-graph-events.yaml` as the task manifest.

## Attribution

The task derives from the CUDAHercules benchmark task
`tasks/class2/general/layernorm_fwd_1024`. The CUDA headers are from
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (BSD-3-Clause).
See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
