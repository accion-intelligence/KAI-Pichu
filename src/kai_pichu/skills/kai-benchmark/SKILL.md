---
name: kai-benchmark
description: Build, validate and hand off a KAI Pichu benchmark task (benchmark.yaml + adapter.py) so kai-pichu can optimize a CUDA operator or fuse a kernel pipeline. Use when the user wants to optimize a GPU kernel or operator with KAI, define a benchmark for the KAI Benchmark SDK, wrap an existing baseline (cuDNN, cuBLAS, PyTorch, their own kernel) as a task, or create a `kind: fusion` task from kernels they already have.
---

# Build a KAI benchmark task

You are producing the contract that an optimization agent will be measured
against. The agent that later optimizes the task never sees the adapter, the
input generator or the oracle you write; it sees only the manifest, the
editable implementation files and the documents you list as agent-visible.
Write the task as if an adversary will try to pass it cheaply.

The authoritative reference is the manual shipped with the installed SDK. Read
it first, every time, so the field names and rules match the installed version:

```bash
kai-pichu benchmark guide            # the manual (or: python -m kai_pichu benchmark guide)
kai-pichu benchmark schema           # manifest JSON Schema
```

## Workflow

1. **Settle the contract with the user before writing code.** Confirm: operator
   semantics and precision; the baseline to beat (prefer the library the user
   would otherwise call); the independent reference and how the tolerance is
   justified; the input domain, alignments and value distributions; the timed
   boundary; the editable files and fixed ABI; `kind: operator` or `kind:
   fusion`. Ask about anything you cannot establish from the project. Do not
   invent tolerances, toy inputs or a target speedup.
2. **Scaffold.** `kai-pichu benchmark init /abs/path/task --template stateless`
   (or `stateful`, `command`), then study the closest packaged example under
   `examples/` of the KAI repository: `depthwise_conv` (nvcc kernel vs cuDNN,
   in-graph timing), `attention` (PyTorch baseline), `awq_linear_fusion` and
   `fusion` (fusion tasks). Reuse their adapter structure, not their numbers.
3. **Write the hidden side first: `inputs.py`, `reference.py`, `adapter.py`.**
   Input values must vary in every quantity the operator consumes and follow
   several realistic distributions. The oracle must exercise every term. Add
   invalid observations for the task's characteristic mistakes, not only NaN.
4. **Define the splits.** Search and acceptance span the same ranges of every
   dimension and the same data profiles; acceptance moves the points and seeds
   inside those ranges. Put each dimension's extremes in search. Keep search to
   a few cases so a round takes minutes.
5. **Write the agent-visible side: `description.md`, the bridge/ABI header,
   `agent_files`.** State layouts, packing, ranges, tolerance, ABI, build flags
   and the measurement boundary. Verify every numeric or structural claim with a
   computation or a probe before writing it down.
6. **Set measurement.** Single operator: adapter-owned in-graph operator time
   (CUDA graph between two external events, L2 eviction, output poisoning) as
   the objective, with the SDK's `latency_ms` as diagnostic. Plain-language
   `boundary` and `cache_policy`. GPU examples use `warmup: 10`, `blocks: 40`,
   `iterations: 10`; keep `max_case_regression` at the 0.05 default (cases below it are reported, the overall geometric mean decides).
7. **Validate.**
   ```bash
   kai-pichu benchmark validate benchmark.yaml --split smoke      --checks-only --output checks-smoke.json
   kai-pichu benchmark validate benchmark.yaml --split search     --checks-only --output checks-search.json
   kai-pichu benchmark validate benchmark.yaml --split acceptance --checks-only --output checks-acceptance.json
   kai-pichu benchmark validate benchmark.yaml --split search --output preflight.json
   ```
   Every case passes, every probe is rejected, the baseline's worst error leaves
   margin under the tolerance, a deliberately degraded implementation fails, and
   `baseline_timing` matches the declared boundary. Fix causes; never widen a
   tolerance or drop a probe to get a green result.
8. **Deliver.** The task directory, a short README for humans, and a contract
   summary for the user to approve: semantics, baseline, reference and
   tolerance, cases per split, boundary, editable files. Then:
   ```bash
   kai-pichu config --output optimizer.yaml     # set model, endpoint, key variable, budget, NCU
   kai-pichu optimize /abs/path/task/benchmark.yaml --config optimizer.yaml --output runs/task
   ```

## Rules that are not negotiable

- The adapter, inputs and oracle are never in `agent_files`; the SDK rejects the adapter there.
- Never change inputs, tolerance or the oracle because a candidate failed.
- No credentials in the manifest; no local benchmark results in the task's docs.
- Do not claim GPU validation from a CPU template run.
- Report what `validate` actually printed, including anything you could not verify.
