# Showcase: 4.25× over cuDNN on depthwise 7×7 convolution

$1 of API calls, about 2 hours, one packaged task, no target speedup. This page
records how the run went; the [README](../README.md#showcase-1-2-hours-425-over-cudnn)
has the headline.

Large-kernel depthwise convolution is the slow stage of ConvNeXt-style backbones
and gets little optimization attention: no tensor-core path, and library
implementations sit far below the memory roof. We pointed KAI Pichu at
[the packaged task](../examples/depthwise_conv/README.md), whose baseline calls
cuDNN's grouped convolution with cuDNN's own fastest algorithm per shape, and
let it run for 13 rounds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/depthwise-showcase-dark.svg">
  <img src="assets/depthwise-showcase-light.svg" alt="Bar chart of speedup over cuDNN for the 13 evaluated candidates. Round 1 reaches 1.96×, round 2 3.52×, round 3 4.25× which stays the best; later rounds land between 3.65× and 4.20×; two candidates failed to build and were repaired in the next round." width="100%">
</picture>

## Setup

NVIDIA GeForce RTX 5070, CUDA 12.9, PyTorch 2.8 with cuDNN 9.10. Generator and
judge: `gpt-5.6-luna` through the OpenAI Responses API at reasoning effort
`xhigh`. Budget: 13 rounds, no `target_speedup`; NCU profiling on with up to
four evidence queries per diagnosis. 72 model calls, 1.4M input and 0.4M output
tokens, about 1.7 hours of loop time, about $1 as billed.

## Result

The best candidate (round 3) was rerun twice on five held-out cases that search
never saw: different value distributions, odd spatial sizes, an unaligned
channel count and a 14×14 map. Both acceptance runs passed every per-case
regression rule.

| Held-out case | cuDNN | KAI Pichu | Speedup |
| --- | --- | --- | --- |
| 8×256×56×56, large-range activations | 0.710 ms | 0.133 ms | 5.3× |
| 16×96×56×56, smooth feature maps | 0.538 ms | 0.104 ms | 5.2× |
| 4×192×57×61, odd spatial size | 0.314 ms | 0.067 ms | 4.7× |
| 4×100×61×59, post-ReLU sparse, unaligned rows | 0.180 ms | 0.045 ms | 4.0× |
| 2×768×14×14, small map | 0.056 ms | 0.037 ms | 1.5× |

Geometric-mean speedup across the five cases: 3.85× and 3.54× on the two
acceptance runs (95% intervals [3.77, 3.92] and [3.42, 3.66]). Outputs matched
the FP32 reference rounded to FP16 with zero error on every case.

## How it got there

The trajectory is the point of the tool, not the final number:

1. *Round 1 (1.96×).* The generator replaced cuDNN's generic IMPLICIT_GEMM path with a custom NCHW kernel that stages each plane's 7×7 halo in a shared-memory FP32 tile through alignment-checked `half2` loads, keeping cuDNN only for tiny shapes.
2. *Round 2 (3.52×).* The judge read the NCU profile of round 1: 88% SM throughput, 96% active warps, shared-memory read throughput near zero. Its diagnosis was that the 49-tap FMA loop was issue-bound, not memory-bound, and it asked for register blocking. The generator made each thread compute two adjacent outputs from the same shared rows, with an odd tile pitch to avoid bank conflicts.
3. *Round 3 (4.25×).* Same diagnosis, pushed further: four adjacent outputs per thread, so each filter row needs ten shared-memory reads for four dot products instead of twenty-eight, dispatched only where the wider tile fits.
4. *Rounds 4 to 13.* Nine variants of that design (half2-packed halos, warp shuffles, eight-output blocking, `cp.async` double buffering, width-specific kernels) all measured between 3.65× and 4.20×; none displaced round 3. Two rounds failed to compile on undefined identifiers and were repaired in the following round from the compiler's own error text.

The naive alternative, one thread per output with 49 global loads, already
beats cuDNN by about 1.9× on this GPU; the loop's contribution is the remaining
2.2× between that and the accepted kernel. The 14×14 case, where the wide tiles
do not fit, is where headroom remains. Every request, response, candidate,
report and profile of this run is a plain file in the run directory, in the
layout described in the [README](../README.md#what-a-run-leaves-behind).
