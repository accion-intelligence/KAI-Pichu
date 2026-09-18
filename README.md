# KAI Pichu

### An open-source CUDA kernel agent.

[Get started](#get-started) · [How it works](#how-it-works) · [Documentation](#documentation) · [Research](#research-and-attribution)

At Accion Intelligence, we’re building KAI to make GPU engineering accessible from a specification. **KAI Pichu is our open-source CUDA kernel agent:** it generates, debugs, profiles, and optimizes one GPU operator at a time. It is the first of three tiers: **KAI Pichu** (open, single operator), **KAI Pikachu** (open, end to end) and **KAI Raichu** (managed). The Python package and CLI are `kai_core` / `kai-core`.

Define what your operator must do, how to check it, and what performance it should beat. The agent generates a candidate, diagnoses failures, repairs the code, investigates performance, and tests the next change. You choose the models, hardware, and budget; the code and experiment records remain yours to inspect and use.

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/kai-core-workflow-dark.png">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/kai-core-workflow-light.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/kai-core-workflow-dark.svg">
  <img src="docs/assets/kai-core-workflow-light.svg" alt="CUDA kernel engineering workflow: define the task; generate and evaluate candidates; repair failures or diagnose performance with optional NCU feedback; recheck the best eligible candidate after search. Code and experiment records remain inspectable with or without acceptance." width="100%">
</picture>

## How it works

**One task. Two feedback loops. A record of the work.**

- **Generate and repair.** The generator writes the permitted implementation files. Build or correctness failures go to a diagnostic judge, which identifies an issue and proposes a focused repair for the next revision.
- **Investigate and optimize.** For valid candidates, the judge uses measured performance and, when enabled, Nsight Compute evidence to propose a bottleneck hypothesis and a concrete code change. Feedback includes the scored metric’s measured baseline and candidate values, units, and boundary. It can query captured profile details on demand before deciding.
- **Keep progress.** The loop retains the best eligible candidate, records hypotheses and results, and works within explicit round, model-call, and time budgets. Interrupted runs can resume under the original configuration.
- **Recheck the result.** After search completes, the selected candidate is rerun on the acceptance split. Source is marked `accepted` only when every required acceptance run passes.

The agent is an **engineering tool**, not a fixed benchmark or a kernel library. Your task specifies the semantics, baseline, cases, precision, and measurement boundary. A task can launch more than one CUDA kernel; the declared operator boundary is what gets evaluated.

## Get started

Use a Linux GPU environment with Python 3.10+, a CUDA toolchain, and the dependencies required by your operator. Bring your own model endpoint and GPU; NCU profiling is optional.

**1. Install from a source checkout**

From the repository root, in your Python environment:

```bash
python -m pip install -e .
python -m kai_core --help
```

<details>
<summary>Optional: check your GPU environment with the packaged depthwise convolution task</summary>

```bash
python -m kai_core benchmark validate examples/depthwise_conv/benchmark.yaml \
  --split search --output depthwise-preflight.json
```

This compiles the cuDNN baseline, runs task checks and baseline correctness on your GPU, and times the baseline, with **no model calls**. Add `--checks-only` to skip timing. It does not test your model endpoint or run the optimization loop. The task needs PyTorch (for its bundled cuDNN) and `nvcc` for your architecture; see [the example](examples/depthwise_conv/README.md).

</details>

**2. Define your operator**

Start with your own specification or existing implementation. A runnable task is a **manifest + adapter**: the contract for your operator and the code that prepares inputs, loads implementations, and checks results. The fastest route is the packaged skill for Claude Code and Codex:

```bash
python -m kai_core skill install          # into ./.claude/skills and ./.codex/skills; --scope user for ~/
```

Then tell your agent: *"Build a KAI task for the operator I describe"*. The skill walks it through the contract, the hidden inputs and oracle, the splits, the agent-visible description and validation, and asks you to approve the contract before optimization.

To author by hand, read the same manual the skill uses, export the schema and scaffold a task:

```bash
python -m kai_core benchmark guide --output task-manual.md
python -m kai_core benchmark schema --output task-schema.json
python -m kai_core benchmark init /absolute/path/to/your-task --template stateless
```

[Task setup and exact commands →](docs/QUICKSTART.md#2-define-the-task)

**3. Configure, run, and inspect**

Export the model configuration, set your endpoint and budget, then start the agent against your reviewed task. The optimizer checks the baseline before making its first generation call.

```bash
python -m kai_core config --output optimizer.yaml
# Edit optimizer.yaml: model, endpoint, key variable, budget; enable NCU if available.
python -m kai_core optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator
```

Add `--dry-run` to inspect the plan and frozen task bundle without model calls or GPU workload.

[Full setup, credentials, profiling, dry-run and resume →](docs/QUICKSTART.md)

## Showcase: $1, 2 hours, 4.25× over cuDNN

Large-kernel depthwise convolution is the slow stage of ConvNeXt-style backbones and gets little optimization attention: no tensor-core path, and library implementations sit far below the memory roof. We pointed KAI Pichu at [the packaged task](examples/depthwise_conv/README.md), whose baseline calls cuDNN's grouped convolution with cuDNN's own fastest algorithm per shape, and let it run for 13 rounds with no target speedup.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/depthwise-showcase-dark.svg">
  <img src="docs/assets/depthwise-showcase-light.svg" alt="Bar chart of speedup over cuDNN for the 13 evaluated candidates. Round 1 reaches 1.96×, round 2 3.52×, round 3 4.25× which stays the best; later rounds land between 3.65× and 4.20×; two candidates failed to build and were repaired in the next round." width="100%">
</picture>

**$1 of API calls. 2 hours. 4.25× faster than cuDNN.**

**Setup.** NVIDIA GeForce RTX 5070, CUDA 12.9, PyTorch 2.8 with cuDNN 9.10. Generator and judge: `gpt-5.6-luna` through the OpenAI Responses API at reasoning effort `xhigh`. Budget: 13 rounds, no `target_speedup`; NCU profiling on with up to four evidence queries per diagnosis.

**Result.** The best candidate (round 3) was rerun twice on five held-out cases that search never saw: different value distributions, odd spatial sizes, an unaligned channel count and a 14×14 map. Both acceptance runs passed every per-case regression rule.

| Held-out case | cuDNN | KAI Pichu | Speedup |
| --- | --- | --- | --- |
| 8×256×56×56, large-range activations | 0.710 ms | 0.133 ms | 5.3× |
| 16×96×56×56, smooth feature maps | 0.538 ms | 0.104 ms | 5.2× |
| 4×192×57×61, odd spatial size | 0.314 ms | 0.067 ms | 4.7× |
| 4×100×61×59, post-ReLU sparse, unaligned rows | 0.180 ms | 0.045 ms | 4.0× |
| 2×768×14×14, small map | 0.056 ms | 0.037 ms | 1.5× |

Geometric-mean speedup across the five cases: 3.85× and 3.54× on the two acceptance runs (95% intervals [3.77, 3.92] and [3.42, 3.66]). Outputs matched the FP32 reference rounded to FP16 with zero error on every case.

**How it got there.** The trajectory is the point of the tool, not the final number:

1. *Round 1 (1.96×).* The generator replaced cuDNN's generic IMPLICIT_GEMM path with a custom NCHW kernel that stages each plane's 7×7 halo in a shared-memory FP32 tile through alignment-checked `half2` loads, keeping cuDNN only for tiny shapes.
2. *Round 2 (3.52×).* The judge read the NCU profile of round 1: 88% SM throughput, 96% active warps, shared-memory read throughput near zero. Its diagnosis was that the 49-tap FMA loop was issue-bound, not memory-bound, and it asked for register blocking. The generator made each thread compute two adjacent outputs from the same shared rows, with an odd tile pitch to avoid bank conflicts.
3. *Round 3 (4.25×).* Same diagnosis, pushed further: four adjacent outputs per thread, so each filter row needs ten shared-memory reads for four dot products instead of twenty-eight, dispatched only where the wider tile fits.
4. *Rounds 4 to 13.* Nine variants of that design (half2-packed halos, warp shuffles, eight-output blocking, `cp.async` double buffering, width-specific kernels) all measured between 3.65× and 4.20×; none displaced round 3. Two rounds failed to compile on undefined identifiers and were repaired in the following round from the compiler's own error text.

The naive alternative, one thread per output with 49 global loads, already beats cuDNN by about 1.9× on this GPU; the agent's contribution is the remaining 2.2× between that and the accepted kernel. The 14×14 case, where the wide tiles do not fit, is where headroom remains. Every request, response, candidate, report and profile of this run is a plain file in the run directory, in the layout below.

## What a run leaves behind

**No accepted candidate does not mean no useful work.** Inspect the attempts, diagnostics, and source changes, not just the final status.

```text
runs/my-operator/
├── summary.json        # final status, report links, time and model usage
├── state.json          # history, best candidate and reserved budgets
├── bundle/             # frozen task and baseline
├── candidates/         # generated implementation snapshots
├── llm/                # model requests, responses and errors
├── reports/            # evaluation reports and process logs
├── profiles/           # hardware evidence, when profiling ran
├── best_search/        # best eligible candidate, if one exists
├── best_search.patch   # its implementation changes, if one exists
└── accepted/           # source, only after final acceptance passes
```

A successful command exit alone does not mean a speedup was accepted. `accepted` means the supplied tests and rules passed. It is not a guarantee about untested inputs, other GPUs, or application-level latency.

## Measurements you can inspect

The benchmark SDK checks correctness against your reference, probes the validator with deliberately invalid observations, and compares candidates using paired measurements with confidence intervals. Final acceptance repeats evaluation on the acceptance split; whether its cases differ from search is determined by your adapter.

The SDK controls built-in timing; adapter-defined metrics require their own boundary review. File allowlists and integrity checks help preserve the task, but **this is not a security sandbox**. [Measurement rules and execution risks →](docs/MEASUREMENT.md)

## Documentation

| You want to… | Start here |
| --- | --- |
| Define a task and run the agent | [Quickstart](docs/QUICKSTART.md) |
| Write a task, by hand or with a coding agent | [Benchmark manual](src/kai_core/benchmark/MANUAL.md) (also `kai-core benchmark guide`), installed as the `kai-benchmark` skill by `kai-core skill install` |
| Explore the packaged tasks | [Depthwise 7×7 convolution against cuDNN](examples/depthwise_conv/README.md), [FP16 attention example](examples/attention/README.md) |
| Fuse kernels you already have | [AWQ INT4 linear layer on AutoAWQ's kernels](examples/awq_linear_fusion/README.md), [epilogue fusion example](examples/fusion/README.md) (`kind: fusion`) |
| Understand the agent’s decisions and outputs | [Workflow](docs/WORKFLOW.md) |
| Understand correctness, timing, and acceptance | [Measurement](docs/MEASUREMENT.md) |

## Contributing

Bring a new operator, an interesting failure, a better diagnostic strategy, or results from another GPU. Share the task, environment, relevant logs, and code change, not just a speedup ratio. Remove credentials and proprietary material first. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Research and attribution

The agent builds on the CUDA generation and hardware-feedback workflow of **[CudaForge](https://arxiv.org/abs/2511.01884)** and **[StitchCUDA](https://icml.cc/virtual/2026/poster/64924)**. Its optimizer source history and retained MIT notices are documented in the repository. The task design and evaluation methodology draw on **[CUDAHercules](https://arxiv.org/abs/2605.08467)**.

If KAI Pichu supports your work, please cite the relevant papers:

```bibtex
@misc{zhang2025cudaforge,
  title = {CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization},
  author = {Zijian Zhang and Rong Wang and Shiyang Li and Yuebo Luo and Mingyi Hong and Caiwen Ding},
  year = {2025},
  eprint = {2511.01884},
  archivePrefix = {arXiv}
}

@inproceedings{li2026stitchcuda,
  title = {StitchCUDA: An Automated Multi-Agents End-to-End GPU Programming Framework with Rubric-based Agentic Reinforcement Learning},
  author = {Shiyang Li and Zijian Zhang and Winson Chen and Yuebo Luo and Mingyi Hong and Caiwen Ding},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026},
  url = {https://icml.cc/virtual/2026/poster/64924}
}

@misc{li2026cudahercules,
  title = {CUDAHercules: Benchmarking Hardware-Aware Expert-level CUDA Optimization for LLMs},
  author = {Shiyang Li and Zijian Zhang and Guangyan Sun and Yuebo Luo and Winson Chen and Yanzhi Wang and Mingyi Hong and Caiwen Ding},
  year = {2026},
  eprint = {2605.08467},
  archivePrefix = {arXiv}
}
```

Framework changes use Apache-2.0; third-party components retain their own licenses. See [LICENSE](LICENSE) and [source and license notices](THIRD_PARTY_NOTICES.md).

---

<sub>**KAI Pichu** (open · single operator) · **KAI Pikachu** (open · end to end) · **KAI Raichu** (managed)</sub>

<sub>Built by [Accion Intelligence](https://github.com/accion-intelligence).</sub>
