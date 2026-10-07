# Showcase: fusing AutoAWQ's INT4 linear layer, 3.4× over its two-stage path

One `kind: fusion` task, `gpt-5.6-luna` as coder and judge, 20 rounds, no
target speedup. This page records how the run went; the
[README](../README.md#showcase-awq-int4-decode-fusion-34-geometric-mean-up-to-44-per-shape)
has the headline.

AutoAWQ serves 4-bit LLM weights by running its `dequantize_weights` kernel
into an FP16 weight matrix and then a cuBLAS GEMM. [The packaged
task](../examples/awq_linear_fusion/README.md) hands the optimizer exactly that
pipeline, both kernels read-only, and asks for one kernel that never
materializes the weights. AutoAWQ also ships a hand-written fused kernel for
small batches, `gemm_forward_cuda`; it is the yardstick below, measured on the
same fixtures.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/awq-showcase-dark.svg">
  <img src="assets/awq-showcase-light.svg" alt="Bar chart of speedup over AutoAWQ's dequantize-then-cuBLAS path for 20 rounds. Round 1 lost to the baseline; round 2 reaches 1.92×, round 12 2.33×, round 15 3.25×, round 20 3.31× which is the best; five candidates failed to build or produced wrong results and were repaired in the next round." width="100%">
</picture>

## Setup

NVIDIA GeForce RTX 5070, CUDA 12.9, PyTorch 2.8, cuBLAS from the CUDA
toolkit. Coder and judge: `gpt-5.6-luna` through the OpenAI Responses API at
reasoning effort `xhigh`. Budget: 20 rounds, no `target_speedup`; Nsight
Compute profiling on, including per-instruction stall attribution with source
lines, captured each round on the search case where the candidate gained least.
49 model calls, 1.1M input and 0.39M output tokens, about 66 minutes of loop time.

The search split has four cases spanning `M = 1, 2, 4, 8` over Llama 7B and 13B
projection shapes and all four value profiles; the held-out acceptance split
samples other points in the same ranges.

## Result

The best search candidate (round 20, 3.31× on the search cases) was frozen and
rerun twice on the four held-out cases. Times are medians of the in-graph
operator latency.

| Held-out case | AutoAWQ two-stage | KAI Pichu | Speedup | AutoAWQ fused kernel |
| --- | --- | --- | --- | --- |
| 1×11008×4096 | 0.428 ms | 0.096 ms | **4.4×** | 5.1× |
| 2×4096×11008 | 0.605 ms | 0.149 ms | **4.1×** | 7.9× |
| 3×5120×13824 | 0.962 ms | 0.216 ms | **4.4×** | 9.1× |
| 8×4096×4096 | 0.158 ms | 0.084 ms | **1.9×** | 3.3× |

Both acceptance reruns were **accepted** with no case regression: geometric-mean
speedup 3.45× and 3.41× (95% intervals [3.35, 3.55] and [3.32, 3.49]). Outputs
matched the FP32 reference within the declared tolerance on every case. The
hand-written kernel remains faster on every shape, by 15% at `M = 1` and by about
2× at `M = 2` and `M = 3`: it dequantizes eight INT4 values with a handful of
`lop3` instructions and feeds tensor-core `mma.sync`, while the generated kernel
is scalar FMA throughout.

## How it got there

1. *Rounds 1 to 3 (0.71× → 2.00×).* A one-warp-per-tile fused kernel lost to the
   baseline on the `M = 8` and `M = 1` cases. The judge read the profile of the
   slowest case: 64 blocks, 2.8% active warps. Splitting K across four warps
   per block lifted every case above the baseline; a 32-column tile for `M = 1`
   doubled its grid.
2. *Rounds 4 to 12 (2.33×).* Eight rounds on `M = 8`, the weakest case. The
   judge's stall attribution pointed at the same instruction each time, the
   `__shfl_sync` broadcast of activations and packed words (long-scoreboard
   stalls, 65 to 77% of samples). A specialized 32-column kernel, a
   shared-memory transpose and an eight-warp row-parallel layout were each
   measured and discarded; a double-buffered producer/consumer pipeline over
   shared-memory tiles took `M = 8` from 1.07× to 1.97×. Two of these rounds
   produced wrong results and were repaired from the validator's message.
3. *Rounds 13 to 15 (3.25×).* The same evidence on `M = 1`: shuffle broadcasts
   replaced by shared-memory staging (one build error on a duplicate
   `extern __shared__` symbol, repaired next round), then a 16-byte `cp.async`
   double-buffered pipeline for the packed weight rows, which took `M = 1` from
   1.60× to 4.31×.
4. *Rounds 16 to 20 (3.31×).* Porting the `cp.async` pipeline to `M = 8` cost a
   build error and a race, then measured 2.15× on that case; a barrier-free
   variant regressed to 0.29×, and the final round restored round 15's `M = 8`
   design while keeping the improved `M = 1` path.

What the run left on the table is the tensor-core path: every kernel accumulates
with scalar FMA, which is why `M ≥ 2` stays about 2× behind the hand-written
kernel. Every request, response, candidate, report and profile is a plain file
in the run directory, in the layout described in the
[README](../README.md#what-a-run-leaves-behind).
