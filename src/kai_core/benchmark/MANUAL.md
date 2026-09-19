# KAI Benchmark SDK manual: building a task

A task is what KAI optimizes against: a **manifest** (`benchmark.yaml`) that
declares the contract and an **adapter** (`adapter.py`) that prepares inputs,
loads implementations, runs one workload and checks the result. This manual is
the complete reference for writing one, for a developer at a keyboard and for a
coding agent alike. Everything the SDK enforces is stated here; the exported
JSON Schema (`kai-core benchmark schema`) is authoritative for field types.

```
my-task/
  benchmark.yaml      manifest: contract, objective, measurement, file roles
  adapter.py          Benchmark subclass: prepare, load, run, validate, probes
  inputs.py           input synthesis (hidden from the optimization agent)
  reference.py        oracle (hidden from the optimization agent)
  description.md      what the agent may read about the task
  bridge.cu           fixed ABI the adapter calls (agent-visible)
  baseline/           the editable implementation files, as shipped
    solution.cu
  kernels/            kind: fusion only: the kernels to fuse, read-only
```

Paths in the manifest resolve relative to the manifest, never to the shell.

## 1. Establish the contract first

Settle these before writing code. Ask the user when an item cannot be
established from the project; never invent a loose tolerance, a toy input
distribution or a target speedup to finish sooner.

- **Semantics.** Exact output definition, dtypes, layouts, required side
  effects, precision (accumulate in FP32? round once to FP16?).
- **Baseline.** A runnable implementation of the same contract. Prefer the
  library the user would otherwise call (cuDNN, cuBLAS, PyTorch, an existing
  kernel); the agent has to beat it, so it must be the honest alternative.
- **Reference.** An oracle independent of the baseline and of any candidate:
  PyTorch math at higher precision, an analytic result, a trusted test suite.
- **Tolerance.** Justified by the reference's own rounding: measure the
  baseline's error against the reference and leave margin, and confirm that
  a plausible wrong implementation (lower precision, dropped stage) fails.
- **Input domain and distributions.** Shapes, ranges, alignments and the value
  distributions the real workload has. Input values are the task owner's
  decision; the framework does not prescribe or lint them.
- **Timed boundary.** Which work the objective covers and what is excluded.
- **Editable surface.** Which files the agent may rewrite, and which fixed ABI
  the adapter calls.
- **Task kind.** `operator` (optimize one implementation) or `fusion` (fuse
  kernels the user already has).

## 2. Commands

```bash
kai-core benchmark init  /abs/path/new-task --template stateless   # scaffold
kai-core benchmark schema --output task-schema.json               # manifest schema
kai-core benchmark guide  --output task-manual.md                 # this manual
kai-core benchmark validate benchmark.yaml --split smoke --checks-only --output checks.json
kai-core benchmark validate benchmark.yaml --split search --output preflight.json
kai-core benchmark run benchmark.yaml --candidate /abs/path/candidate --split search --output compare.json
```

`python -m kai_core benchmark …` is equivalent. Templates: `stateless` (a
PyTorch or Python operator), `stateful` (a resettable multi-step workload),
`command` (an external C++/CUDA program driven as a subprocess). The templates
run on CPU and teach the protocol; the packaged GPU examples under `examples/`
show real tasks: `depthwise_conv` (nvcc-built kernel against cuDNN, in-graph
timing), `attention` (FP16 SDPA against PyTorch), `awq_linear_fusion` and
`fusion` (`kind: fusion`). Output paths must be new; reports are never
overwritten.

## 3. Manifest reference

| Field | Type, default | Meaning |
| --- | --- | --- |
| `schema_version` | `1` | Manifest schema version |
| `name` | string | Task identifier, recorded in every report |
| `description` | string | Agent-visible prose: semantics, layouts, tolerance, what is measured |
| `kind` | `operator` (default) or `fusion` | See section 7 |
| `fusion` | object, required iff `kind: fusion` | `kernels` (≥2, in execution order, each `name`, `files`, `entry`, `description`) and `intermediates` (buffer names passed between kernels) |
| `adapter` | `relative/file.py:ClassName` | The `Benchmark` subclass; inside the task directory |
| `implementation.root` | path, default `.` | Directory holding the editable files as shipped (the baseline) |
| `implementation.files` | glob list | The files a candidate may replace; every build source must be listed |
| `benchmark_files` | glob list | Adapter helpers, input synthesis, oracle: frozen and fingerprinted, never shown to the agent |
| `agent_files` | glob list | Text the optimization agent may read (description, ABI headers). The adapter can never be listed |
| `objective.metric` | string, default `latency_ms` | `latency_ms` and `throughput` are SDK-owned; any other name must be returned by the adapter on every run |
| `objective.unit` | string | Unit of the metric |
| `objective.direction` | `minimize` / `maximize` | |
| `objective.scope` | `kernel` / `module` / `end_to_end` | Documents what the boundary covers |
| `objective.target_speedup` | number > 1, optional | Acceptance requires the lower confidence bound to reach it; omit to accept any confirmed improvement |
| `objective.max_case_regression` | fraction, default `0.05` | Cases whose speedup interval falls below `1/(1+r)` are listed in `acceptance.case_regressions` for review; acceptance itself is decided by the overall geometric-mean speedup |
| `measurement.timer` | `wall` (default) / `cuda_event` | SDK timer for `latency_ms`; see section 8 |
| `measurement.device` | int, default 0 | CUDA device index for `cuda_event` |
| `measurement.warmup` | int, default 5 | Untimed invocations per case and fixture before timing |
| `measurement.blocks` | int ≥ 4, default 20 | Paired ABBA blocks |
| `measurement.iterations` | int, default 3 | Timed invocations averaged per block position |
| `measurement.confidence`, `bootstrap_samples` | 0.95, 2000 | Bootstrap interval settings |
| `measurement.boundary` | string | Plain-language statement of what the timed interval contains |
| `measurement.cache_policy` | string | Plain-language statement of cache state between invocations |
| `limits` | list of `{metric, minimum?, maximum?}` | Hard bounds checked on every raw candidate invocation |
| `seed` | int, default 0 | Passed to `prepare` |
| `options` | free-form map | Task parameters: device strings, compiler, arch, tolerances, dependency paths. Never credentials |

The manifest, the adapter, `benchmark_files` and `agent_files` are frozen and
fingerprinted with the run; any change during a run invalidates it.

## 4. Adapter reference

Subclass `kai_core.benchmark.Benchmark`. Public types: `Case(id, params,
weight=1.0, work_units=None)`, `Observation(output, metrics={})`,
`Validation(passed, message="", errors={})`.

| Method | Contract |
| --- | --- |
| `cases(split) -> Iterable[Case]` | Deterministic, non-empty, unique ids, for `smoke`, `search`, `acceptance`. Put the shape coordinates in `params` |
| `prepare(case, seed) -> fixture` | Inputs plus initial state, usable by baseline and candidate alike. Called twice per case; both fixtures must digest identically |
| `fingerprint(fixture) -> str` | Default digests the whole fixture (scalars, strings, bytes, files by content, containers, NumPy, PyTorch tensors). Override only when the fixture also holds scratch, outputs or captured graphs: digest inputs and initial state only |
| `load_implementation(workspace) -> handle` | Build or load exactly the files under the given root. Baseline and candidate roots differ; both may be loaded at once |
| `run(handle, fixture) -> Observation` | One complete workload. Materialize outputs before returning. Adapter-owned metrics go in `Observation.metrics` |
| `validate(case, fixture, observation) -> Validation` | Trusted oracle over outputs and required side effects. `errors` values must be finite numbers |
| `invalid_observations(case, fixture, valid) -> Iterable[Observation]` | At least one wrong result per case that `validate` must reject: perturbed values, NaN, missing output, and the task's characteristic mistakes (wrong nibble order, dropped zero point, flipped kernel). Never mutate the live fixture or `valid` |
| `reset(handle, fixture)` | Restore state before every invocation, warmups included. Also the place to capture per-implementation CUDA graphs |
| `synchronize(handle)` | Wait for asynchronous work. Required for `wall` timing of GPU code |
| `cleanup_fixture(fixture)`, `cleanup_implementation(handle)` | Release memory, processes, libraries, also on failure |

**What the runner does with it.** Load the split's cases and check they are
deterministic. Load the baseline. For each case: prepare two fixtures and check
their digests agree; run untimed; validate; run every invalid observation
through `validate` and fail if any passes; reset and check the digest is
restored. Then time: warm up every implementation on both fixtures, and run
`blocks` blocks in ABBA/BAAB order with fixtures crossed between arms, `iterations`
timed calls per position, reset before every call. A candidate is additionally
checked untimed on every case before timing, and its `limits` on every raw call.

**Metrics.** `latency_ms` is the SDK timer's interval; `throughput` is
`work_units × 1000 / latency_ms` when `Case.work_units` is set. An adapter that
owns the objective (for example an in-graph `operator_latency_ms`) must return
it on every call, timed or not, strictly positive and finite.

**Loading Python.** `load_python_file(workspace / "solution.py")` loads a leaf
module without name or stale-bytecode collisions. It does not isolate native
extensions or CUDA contexts; use a process-backed adapter when two versions
cannot coexist.

**Native code.** Build with an explicit compiler command, keep the output under
a temporary directory owned by the handle, and raise the compiler's stderr in
the exception so the agent sees the real error. Load with `ctypes` and set
`argtypes`/`restype`. Link libraries by file name when the wheel ships an
unversioned-less name (`-l:libcudnn.so.9`), and surface a non-zero launcher
return code as a failure.

## 5. Cases and splits

- `smoke`: one small case that runs anywhere; used by `validate` by default.
- `search`: what the optimizer scores every round. **The only cases whose
  results the agent ever sees.** Their ids appear in the feedback, so ids may
  describe the shape.
- `acceptance`: held out; the frozen best candidate is rerun here twice.

**Search and acceptance must span the same ranges.** For every input dimension
and every data profile, acceptance values must lie inside the range the search
cases cover. Acceptance changes the points, seeds and value distributions inside
those ranges, never the ranges. Put the extremes of every dimension into search
(smallest and largest batch, shortest and longest reduction, aligned and
unaligned sizes) and let acceptance sample the interior. A shape or profile that
appears only in acceptance tests a regime the agent never measured, and a
failure there indicts the split design rather than the candidate.

**Inputs must vary in every quantity the operator consumes.** Constant or
degenerate inputs let a candidate skip a stage and still pass (an affine layer
with gamma = 1 and beta = 0 is indistinguishable from no affine layer). Use
several value distributions that resemble the real workload, seed them from
`Case.params`, and make the oracle exercise every term.

`weight` scales a case's contribution to the weighted geometric-mean speedup.
`work_units` enables `throughput`. Keep search small enough that a round is
minutes, not hours: each case is measured `blocks × 4 × iterations` times per
arm per round.

## 6. What the agent may see: context isolation

The optimization agent reads the manifest, the editable implementation files
and `agent_files` (plus fusion kernel files). It never reads the adapter, input
synthesis or the oracle, so it cannot specialize to the test distribution. Write
the task's inputs and oracle in a context that has not seen any candidate and is
not reused to write one.

Therefore everything the agent needs is stated in `description` or an
`agent_files` document: layouts, packing formats, tolerance, which quantities
vary between calls and over which ranges, the fixed ABI, build flags, what is
timed and how caches are treated. **Verify every claim in an agent-facing
document** with a computation or a probe before publishing it: a wrong
divisibility or alignment statement sends the agent down a false path for many
rounds. Never adjust inputs, tolerance or the oracle in response to a failing
candidate; if the task was wrong, re-author it and start a new run.

## 7. Task kinds

`kind: operator` (default): the agent optimizes `implementation.files`.

`kind: fusion`: the user supplies kernels and the agent fuses them.

- `fusion.kernels` lists the kernels in the baseline's execution order with
  `name`, `files` (frozen, agent-visible, read-only), the launcher `entry` the
  baseline calls and a one-line `description`; `fusion.intermediates` names the
  buffers handed from one kernel to the next.
- `implementation.files` is the pipeline that launches them, typically one
  `solution.cu`. The baseline launches the kernels unfused; a candidate may
  rewrite the file from scratch, keeping only the entry point the bridge calls.
  Kernel files cannot also be implementation files.
- The optimizer adds a structured `fusion` block (goal, kernels, intermediates,
  read-only files) to the agent context. Correctness still comes from the
  independent oracle, so a defective supplied kernel is caught at preflight.

## 8. Measurement

State `boundary` and `cache_policy` in plain words. Preparation, reset,
digesting, loading, warmup and validation are outside the timed interval, so any
work the metric must include happens inside `run`.

- `timer: wall`: host clock around `run` + `synchronize`. For blocking CPU code,
  subprocesses and servers.
- `timer: cuda_event`: CUDA events on PyTorch's current stream of
  `measurement.device`, with Python's cyclic GC deferred inside the interval.
  Work on other streams must join the current stream inside `run`. Host
  submission gaps are inside this interval; it is an outer GPU-timeline
  boundary, not a kernel time.
- **In-graph operator time** (the packaged examples' `operator_latency_ms`):
  the adapter captures one operator call into a CUDA graph between two external
  timing events, replays the graph in `run`, and reports the event interval as
  its own metric while the SDK's `latency_ms` remains a diagnostic. Use this
  for single operators, where the question is the kernel's own duration. Keep
  captured graphs keyed by implementation identity in the fixture, evict the
  inputs from L2 before every invocation when the real workload is cold (the
  examples write a 256 MiB buffer), and poison the output before every call so
  a kernel that skips writes is caught.

Warm-up and sampling: the GPU examples use `warmup: 10`, `blocks: 40`,
`iterations: 10`. Do not chase run-to-run noise with more samples; the paired
design absorbs ordinary jitter, and a case is only reported as a regression below the 5% `max_case_regression` margin.
Never time under a profiler, and keep the GPU otherwise idle.

Custom metrics (energy, bytes, a score) use their own names, documented
measurement method and matching `unit`/`direction`/`scope`. A proxy is not
end-to-end time.

## 9. Validate before anyone optimizes

1. `validate --checks-only` on smoke, search and acceptance. Every case must
   pass and every invalid observation must be rejected.
2. Check the tolerance margin: the baseline's worst error relative to the
   tolerance should leave room (the examples sit near half), and a deliberately
   degraded implementation (lower-precision accumulation, a dropped stage)
   must fail through the same adapter path.
3. Full `validate --split search`: the report's `baseline_timing` gives per-case
   statistics of the objective. Confirm they match the boundary you declared
   and are stable enough to compare against.
4. Optionally measure an expert hand-written implementation on the same
   fixtures as a private yardstick. Keep it out of agent-visible files.
5. `run --candidate` with a distinguishable implementation to prove the
   workspace root selects the executed code.

Fix build, oracle, reset, reproducibility and probe failures at the source. Do
not widen a tolerance or remove a probe to make a command pass.

## 10. Reports and exit codes

`validate` writes `checks` (per case, with the number of rejected probes),
`baseline_timing` (unless `--checks-only`), `status` (`checks_passed` or
`ready`), plus provenance: spec, cases, input fingerprints, source digests,
SDK version and timing environment. `run` adds `candidate_checks`, raw
`records`, `comparison` (overall and per-case speedup with intervals) and
`acceptance` (`verdict`, `overall_speedup` with its interval, `case_speedups`, `case_regressions`, `metric_limit_failures`).

| Exit | Meaning |
| --- | --- |
| 0 | Checks passed, preflight ready, or a comparison completed (target met or not) |
| 2 | Invalid configuration, build/execution/verification failure, or path error |

A completed comparison with `accepted: false` is a valid result, not a defect.

## 11. Hand-off to the optimizer

```bash
kai-core config --output optimizer.yaml      # model, endpoint, key variable, budget, NCU
kai-core optimize /abs/path/my-task/benchmark.yaml --config optimizer.yaml --output runs/my-task
kai-core optimize … --resume                 # continue an interrupted run; only the budget may change
```

The optimizer freezes the task bundle, runs the search preflight, then each
round generates one candidate, evaluates it on the search split, profiles the
latest eligible candidate (correct, faster than the baseline with confidence,
within every metric limit) on the search case where it gained least, and asks the
judge for the next change. Two eligible candidates are never ranked by the
harness; the model sees every round's per-case results and the top-scoring
candidate's source and decides what to build on. After the last round the
top-scoring eligible candidate is rerun on acceptance. The agent may change
only `implementation.files`; a requested change to inputs, oracle, tolerance or
weights is a new task version and a new run.

## 12. Deliverable checklist

- [ ] Manifest validates against the schema; `kind`, `objective`, `measurement` match the contract the user approved.
- [ ] Adapter, inputs and oracle are frozen (`benchmark_files`) and absent from `agent_files`.
- [ ] `description`/`agent_files` state every layout, range, tolerance and ABI fact the agent needs, each verified.
- [ ] Search and acceptance span the same ranges and profiles; inputs vary in every consumed quantity.
- [ ] Probes: perturbed value, NaN, missing output, and the task's characteristic mistakes, all rejected.
- [ ] Tolerance justified by the baseline's measured error with margin; a degraded implementation fails.
- [ ] `validate --checks-only` on all three splits and full `validate --split search` pass; `baseline_timing` reviewed.
- [ ] A short README for humans: what the task is, where the numbers come from, requirements, the preflight command.
- [ ] Contract shown to the user for review before `kai-core optimize`.
