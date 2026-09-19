# Showcase: fusing AutoAWQ's INT4 linear layer, up to 9× over its two-stage path

One `kind: fusion` task, `gpt-5.6-luna` as generator and judge, 20 rounds, no
target speedup. This page records how the run went; the
[README](../README.md#showcase-9-on-awq-int4-decode-a-fusion-task) has the
headline.

AutoAWQ serves 4-bit LLM weights by running its `dequantize_weights` kernel
into an FP16 weight matrix and then a cuBLAS GEMM. [The packaged
task](../examples/awq_linear_fusion/README.md) hands the optimizer exactly that
pipeline, both kernels read-only, and asks for one kernel that never
materializes the weights. AutoAWQ also ships a hand-written fused kernel for
small batches, `gemm_forward_cuda`; it is the yardstick below, measured on the
same fixtures.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/awq-showcase-dark.svg">
  <img src="assets/awq-showcase-light.svg" alt="Bar chart of speedup over AutoAWQ's dequantize-then-cuBLAS path for 19 evaluated candidates. Round 1 reaches 5.59×, the best is 7.22× in round 17; round 7 regressed one case; three candidates failed to build or produced wrong results and were repaired in the next round." width="100%">
</picture>

## Setup

NVIDIA GeForce RTX 5070, CUDA 12.9, PyTorch 2.8, cuBLAS from the CUDA
toolkit. Generator and judge: `gpt-5.6-luna` through the OpenAI Responses API at
reasoning effort `xhigh`. Budget: 20 rounds (19 evaluated; one round was lost to
a malformed judge reply and resumed), no `target_speedup`; NCU profiling on,
capturing the search case where the candidate gained least. 84 model calls,
1.6M input and 0.47M output tokens, about 90 minutes of loop time.

The search split had two cases, `M = 1` and `M = 4`; the held-out acceptance
split spans `M = 1, 2, 3, 8`.

## Result

The best search candidate (round 17, 7.22× on the search cases) was frozen and
rerun twice on the four held-out cases. Times are medians of the in-graph
operator latency.

| Held-out case | AutoAWQ two-stage | KAI Pichu | Speedup | AutoAWQ fused kernel |
| --- | --- | --- | --- | --- |
| 1×11008×4096 | 0.428 ms | 0.052 ms | **8.2×** | 5.1× |
| 2×4096×11008 | 0.605 ms | 0.066 ms | **9.1×** | 7.9× |
| 3×5120×13824 | 0.959 ms | 0.125 ms | **7.7×** | 9.1× |
| 8×4096×4096 | 0.158 ms | 0.196 ms | 0.80× | 3.3× |

On `M = 1` and `M = 2` the generated kernel is faster than AutoAWQ's own
hand-written fused kernel. On `M = 8`, a shape the search split never
contained, it runs 20% slower than the two-stage baseline. Both acceptance
reruns were **accepted**: geometric-mean speedup 4.4× and 4.5× across the four
held-out cases (95% intervals [4.23, 4.55] and [4.35, 4.65]), with the `M = 8`
case listed in `acceptance.case_regressions` because its speedup interval sits
below the 5% regression margin. Outputs matched the FP32 reference within the
declared tolerance on every case.

## How it got there

1. *Round 1 (5.59×).* The generator replaced the two launches with one split-K
   fused kernel: each 128-row quantization group is one block, packed INT4
   words are dequantized in registers, activations staged in shared memory,
   FP32 partials reduced at the end. `M = 4` reached 7.8× immediately.
2. *Rounds 2 to 3 (6.60×).* The judge read the `M = 1` profile, saw grid
   occupancy too low for the short decode workload, and split each group across
   two, then four CTAs.
3. *Rounds 4 to 13.* Nine variants of the `M = 1` path: scale factored out of
   the inner loop, half2 dequantization, register prefetch, a single-warp
   design, a WMMA tensor-core path (2.16×, since one row wastes fifteen sixteenths
   of every MMA tile), block-level reductions. None displaced round 3. Two
   candidates failed to build on WMMA overloads and `cp.async` sizes and one
   produced wrong results; each was repaired in the following round from the
   compiler's or validator's own message.
4. *Round 14 (6.93×) and round 17 (7.22×).* Accumulating `Σx` and `Σq·x` per
   group and applying zero point and scale once, then a four-way unrolled
   packed-word preload that overlaps global loads with dequantization.

What the run left on the table is `M = 8`: the `M ≥ 2` kernel keeps one FP32
accumulator per row per lane, and at eight rows it becomes issue-bound and falls
below cuBLAS. Nothing in the search split measured that regime, so the judge
never saw it. Every request, response, candidate, report and profile is a plain
file in the run directory, in the layout described in the
[README](../README.md#what-a-run-leaves-behind).
