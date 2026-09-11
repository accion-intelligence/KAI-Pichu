# KAI Core Agent

### An open-source CUDA kernel agent.

[Get started](#get-started) · [How it works](#how-it-works) · [Documentation](#documentation) · [Research](#research-and-attribution)

At Accion Intelligence, we’re building KAI to make GPU engineering accessible from a specification. **KAI Core Agent is our open-source CUDA kernel agent:** it generates, debugs, profiles, and optimizes one GPU operator at a time.

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
<summary>Optional: check your GPU environment with the packaged LayerNorm task</summary>

```bash
python -m kai_core benchmark validate examples/layernorm/benchmark-graph-events.yaml \
  --split search --output layernorm-preflight.json
```

This runs task checks, baseline correctness, and A/A calibration using your GPU, with **no model calls**. Add `--checks-only` to skip calibration. It does not test your model endpoint or run the optimization loop.

Use the graph-events manifest above. The example ships a second manifest at a wider measurement boundary; the two are not comparable, and submission-related idle gaps in the wider boundary can prevent A/A calibration from passing. [Which manifest to use →](examples/layernorm/README.md#which-manifest-to-use)

</details>

**2. Define your operator**

Start with your own specification or existing implementation. A runnable task is a **manifest + adapter**: the contract for your operator and the code that prepares inputs, loads implementations, and checks results. Export the authoring guide and schema, then scaffold a task:

```bash
python -m kai_core benchmark guide --output task-authoring.md
python -m kai_core benchmark schema --output task-schema.json
python -m kai_core benchmark init /absolute/path/to/your-task --template stateless
```

Adapt the template yourself, or give your coding agent the exported files and this instruction:

> Build a task for the operator I describe, following this authoring guide and schema. Include the reference implementation, input cases, numerical requirements, editable CUDA files, and timing boundary. Show me the contract for review before optimization. Do not relax correctness requirements to make a candidate pass.

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

The benchmark SDK checks correctness against your reference, probes the validator with deliberately invalid observations, calibrates the baseline against itself, and compares candidates using paired measurements with confidence intervals. Final acceptance repeats evaluation on the acceptance split; whether its cases differ from search is determined by your adapter.

The SDK controls built-in timing; adapter-defined metrics require their own boundary review. Failed A/A calibration stops the run rather than triggering code repair. File allowlists and integrity checks help preserve the task, but **this is not a security sandbox**. [Measurement rules and execution risks →](docs/MEASUREMENT.md)

## Documentation

| You want to… | Start here |
| --- | --- |
| Define a task and run the agent | [Quickstart](docs/QUICKSTART.md) |
| Explore the packaged task | [LayerNorm example](examples/layernorm/README.md) |
| Understand the agent’s decisions and outputs | [Workflow](docs/WORKFLOW.md) |
| Understand correctness, timing, and acceptance | [Measurement](docs/MEASUREMENT.md) |

## Contributing

Bring a new operator, an interesting failure, a better diagnostic strategy, or results from another GPU. Share the task, environment, relevant logs, and code change, not just a speedup ratio. Remove credentials and proprietary material first. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Research and attribution

The agent builds on the CUDA generation and hardware-feedback workflow of **[CudaForge](https://arxiv.org/abs/2511.01884)**, co-authored by Shiyang Li. Its optimizer source history and retained MIT notices are documented in the repository. The LayerNorm task comes from **[CUDAHercules](https://arxiv.org/abs/2605.08467)** and uses a FlashAttention CUDA baseline.

If these research components support your work, please cite the relevant papers:

```bibtex
@misc{zhang2025cudaforge,
  title = {CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization},
  author = {Zijian Zhang and Rong Wang and Shiyang Li and Yuebo Luo and Mingyi Hong and Caiwen Ding},
  year = {2025},
  eprint = {2511.01884},
  archivePrefix = {arXiv}
}
```

Framework changes use Apache-2.0; third-party components retain their own licenses. See [LICENSE](LICENSE) and [source and license notices](THIRD_PARTY_NOTICES.md).

---

<sub>**KAI Core** (open · single operator) · **KAI Standard** (open · end to end) · **KAI Enterprise** (managed)</sub>

<sub>Built by [Accion Intelligence](https://github.com/accion-intelligence).</sub>
