# Depthwise 7×7 convolution against cuDNN

Large-kernel depthwise convolution is the slow stage of ConvNeXt-style
backbones and gets little optimization attention: there is no tensor-core path,
and library implementations are known to sit far below the memory roof. This
task asks the agent to beat cuDNN directly. The baseline calls cuDNN's grouped
convolution (group count = channels) and lets cuDNN pick its fastest algorithm
per shape with `cudnnFindConvolutionForwardAlgorithmEx`.

## Definition

- Forward only, 7×7 kernel, stride 1, zero padding 3, no bias, no dilation.
- Activations in NCHW FP16 (the layout in which cuDNN's depthwise path is
  fastest); weights `[C, 7, 7]` FP16; FP32 accumulation. Output FP16 in the
  same layout.
- Reference: PyTorch `conv2d` on FP32 upcasts, rounded to FP16, compared at
  `atol = rtol = 2e-3`.

## Files

| File | Role | Visible to the agent |
| --- | --- | --- |
| `baseline/solution.cu` | Editable implementation; must define `launch_depthwise_conv` (baseline: cuDNN) | yes (editable) |
| `bridge.cu` | Fixed C ABI `kai_launch` around the launcher | yes |
| `description.md` | Semantics, layout, precision and launch constraints | yes |
| `adapter.py` | Compiles against PyTorch's bundled cuDNN, captures one call in a CUDA Graph, validates | no |
| `inputs.py` | Value profiles and the smoke/search/acceptance splits | no |
| `reference.py` | PyTorch oracle and the tolerance | no |

## Cases

| Split | Cases (N × C × H × W) |
| --- | --- |
| smoke | 2×96×56×56 |
| search | 8×256×56×56 standard normal; 4×512×28×28 post-ReLU sparse |
| acceptance | 8×256×56×56 large range; 16×96×56×56 smooth feature maps; 4×192×57×61; 4×100×61×59 post-ReLU; 2×768×14×14 |

57×61 and 61×59 are odd: no tile size divides them and an image row of `W`
halves is not 16-byte aligned, so vectorized loads along `W` need an alignment
check and a tail; 14×14 is smaller than two halos. The `large_range` profile
pushes outputs into the hundreds so FP16 accumulation fails the tolerance.

## Correctness

The validator rejects, per case, a perturbed output, a NaN, a missing output,
replicate-padding at the border, and a flipped (true-convolution) kernel.

## Measurement

`operator_latency_ms` is the GPU time between two external CUDA events captured
around one launch inside a CUDA Graph. Activations and weights are evicted from
L2 before every invocation. cuDNN's algorithm search runs in the eager warm-up
before capture and is not timed.

## Requirements

A CUDA GPU with PyTorch (its pip-installed cuDNN provides headers and
libraries; set `options.cudnn_home` to use another installation) and `nvcc`
for `sm_120` (edit `options.cuda_arch` for another architecture). Preflight
without model calls:

```bash
python -m kai_core benchmark validate examples/depthwise_conv/benchmark.yaml \
  --split search --checks-only --output depthwise-checks.json
```
