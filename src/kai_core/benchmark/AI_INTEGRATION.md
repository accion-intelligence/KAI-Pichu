# Instructions for a coding agent: integrate a project with KAI Core Benchmark SDK v1

Your task is to construct a trustworthy, reproducible benchmark for the user's
existing GPU program. Use the installed `kai_core.benchmark` SDK. This guide is
tool-agnostic: it does not require a particular coding agent or LLM provider.

## 1. Establish the task before writing the adapter

Read the project entrypoints, build instructions, tests, and the user's stated
performance problem. Identify:

- The implementation source and how to build/load it from an explicit workspace.
- The representative workloads: shapes, dtypes, layouts, distributions, seeds,
  data/weight versions, concurrency, request/sequence lengths, and case weights.
- The intended metric and boundary: kernel, module, or actual application E2E;
  latency, throughput, energy, memory, or another measurable quantity.
- The correctness oracle and semantics: outputs, gradients, optimizer/recurrent
  state, RNG behavior, side effects, numerical tolerances and their justification.
- The allowed implementation changes, dependencies, target devices and memory
  constraints. Precision changes require support from the user's requirements.

Infer routine implementation details from the project. Ask the user when the
representative input distribution, correctness tolerance, or intended metric
cannot be established. Record those unknowns explicitly. Do not invent a loose
tolerance, weak oracle, unrealistic toy input distribution, or arbitrary target
speedup just to finish integration.

## 2. Inspect the installed protocol and scaffold

```bash
python -m kai_core benchmark --help
python -m kai_core benchmark schema --output /tmp/task-schema.json
python -m kai_core benchmark init /path/to/new-task --template stateful
```

Choose `stateless` for a PyTorch operator, `stateful` for a resettable sequence,
or `command` for an external C++/CUDA program. The supplied CPU examples teach
the protocol; replace their workload with the user's real one. Read the template
README and the public `Benchmark` abstract methods. Do not add task-specific behavior to KAI Core core code to
make an application-specific task work.

## 3. Define the bundle

Keep a `benchmark.yaml`, `adapter.py`, a human-readable task description, and
implementation source files (or an explicitly configured external source root).
Manifest paths resolve relative to the manifest, never to the shell's cwd.

Required manifest fields are `name`, `description`, `adapter`, `implementation`,
`objective`, and `measurement`; use `schema_version: 1`. Exported JSON Schema is
authoritative. Include every implementation build source in
`implementation.files`. Include additional adapter/oracle helpers and task
documents in `benchmark_files`; the manifest and adapter are recorded automatically.
Author the adapter, input synthesis and oracle in a context that has never seen a
candidate implementation and is never reused to write one. The optimizer enforces
the same separation at run time: its generator and judge see only `agent_files`.
Never adjust inputs, tolerances or the oracle in response to a failing candidate;
fix the candidate or, if the task is wrong, re-author the task and restart the run.
Value ranges and distributions are the task owner's decision; the framework does
not impose any.

List in `agent_files` only the text the optimization agent may read: ABI headers,
interface stubs and task descriptions. The agent never sees the adapter, the
oracle or input generation, so it cannot specialize candidates to the test
distribution. Semantics, shapes, tolerances and constraints the agent needs must
appear in the manifest description or in an `agent_files` document.
Do not include generated build artifacts or output reports in source globs.

Use `options` for task-specific paths, dtypes, device strings, data identities,
dependency versions, and other parameters. Never put API keys or credentials in
the manifest. Benchmark SDK commands do not need an LLM endpoint.

Define smoke, search, and acceptance splits. Search and acceptance may use the
same declared shape distribution, but final confirmation must collect independent
measurements after the candidate is frozen. Add boundary/tail/edge cases that are
required by the real task; don't infer coverage from an arbitrary count of cases.

## 4. Implement the adapter

Subclass `kai_core.benchmark.Benchmark` and implement:

| Method | Contract |
| --- | --- |
| `cases(split)` | Deterministic non-empty iterable of `Case(id, params, weight, work_units)`; IDs unique within the split |
| `prepare(case, seed)` | Construct inputs and initial state; no candidate implementation needed |
| `load_implementation(workspace)` | Build/load the actual files under the supplied root; support distinct baseline and candidate roots |
| `run(implementation, fixture)` | Complete one declared workload; return `Observation(output, metrics={})` |
| `validate(case, fixture, observation)` | Return `Validation(passed, message, errors={})`; check all required semantics |
| `invalid_observations(case, fixture, valid)` | At least one incorrect output per case that this verifier MUST reject |

The SDK digests the fixture itself to check that preparation is reproducible and
that reset restores the inputs. The default covers scalars, strings, bytes, file
paths (by content), nested containers, NumPy arrays and PyTorch tensors. Override
`fingerprint(fixture)` only when the fixture also carries scratch or output
buffers, or objects the default cannot digest; digest the inputs and initial
state, never uninitialized scratch.

Override `reset(implementation, fixture)` for mutable state. It runs before every
invocation, including warmups. A multi-step training/decode workload should run
the full declared sequence INSIDE `run`; SDK iterations are independently reset
repetitions, not additional tokens or optimizer steps. Include hidden optimizer,
RNG, server and module state in reset behavior. Both implementations use both
prepared fixtures within each measurement block, so buffer placement cannot be
permanently associated with one implementation. Warmup covers every pairing.
Store implementation-specific caches (such as captured CUDA graphs) on the
implementation handle or key them by implementation identity within the fixture.
Use fixture/implementation cleanup
hooks for processes and temporary resources, and clean partially acquired resources
yourself if prepare/load fails before returning.

For leaf Python files, `load_python_file(workspace / "solution.py")` avoids
module-name and stale-pyc collisions. It does not isolate package imports,
global native-extension registries, CUDA contexts, or operating-system resources.
Use an appropriate process-backed adapter when two versions cannot coexist.
Ensure both versions fit in memory; the SDK does not automatically swap servers.

Keep the oracle independent of candidate computation. Check outputs and required
state/side effects. A reference can be an immutable existing implementation,
analytic result, trusted test suite, or domain invariant with stated coverage.
Pass/fail must not depend on speed or the candidate's self-reported correctness.
Mutation probes must exercise meaningful errors (e.g. changed values, missing
outputs, wrong final state) and must not mutate the valid fixture accidentally.

## 5. Define measurement honestly

Declare `measurement.boundary` and `cache_policy` in plain language. Preparation,
reset, input hashing, loading, warmup and validation are excluded from timing.
Any preparation required by the user's metric must therefore happen INSIDE `run`.
Do not move necessary allocation, transfers, preprocessing or synchronization
outside the measured workload to improve the reported score.

Use `timer: wall` for blocking CPU/process/server work. Override `synchronize`
for asynchronous in-process GPU work. Use `timer: cuda_event` for PyTorch current-
stream timing on `measurement.device`; join other streams to this stream inside
`run`. CUDA events in the runner cannot measure a different process's GPU work.
Do not change eager/graph/concurrency behavior asymmetrically between variants.
The SDK defers automatic Python cyclic GC within CUDA event intervals, restoring
its original state afterward; wall timing preserves application GC behavior.
CPU scheduling gaps can still enter an event interval. Inspect long tails and
GPU occupancy before attributing a failed calibration to the implementation.

The SDK owns `latency_ms` and, with `Case.work_units`, `throughput`. Return custom
metrics under other names in `Observation.metrics`; document their measurement
method and set matching objective units/direction/scope. For external native
timers, consume structured output and explicitly document that timing authority
belongs to the trusted harness. A proxy score is not E2E timing. A/A calibration
of a proxy does not establish its relevance to application performance.

Do not run final timing under NSYS/NCU. Use an otherwise idle GPU and record its
software/hardware/clock conditions. Compare both variants in the same environment.
The SDK's source fingerprints and partial environment metadata do not replace
recording external data, dependency, driver, clock and server configuration.

## 6. Run the integration checks and calibration

```bash
python -m kai_core benchmark validate /path/to/task/benchmark.yaml \
  --checks-only --output /tmp/task-checks.json
python -m kai_core benchmark validate /path/to/task/benchmark.yaml \
  --split search --output /tmp/task-calibration.json
```

Use fresh output paths. Fix failed build, oracle, reset, reproducibility or mutation
checks. A checks-only pass is not calibration readiness. Full validation performs
paired A/A against the same baseline through identical adapter paths. The entire
confidence interval must fit inside the declared tolerance, overall and per case.
If it fails, inspect order effects, warmup, state, synchronization, cache policy,
resource contention and drift. Do not divide away bias or silently increase the
tolerance until the command passes. Report remaining uncertainty to the user.

Also test a deliberately wrong implementation through the complete adapter path,
and show that it fails correctness. Where practical, test distinguishable baseline
and candidate implementations to verify that `workspace` changes the executed code.
Mutation probes alone do not prove complete correctness or workspace routing.

## 7. Deliver a reviewable benchmark

Deliver the bundle, declared requirements, commands, validation/calibration JSON,
and a concise description of the workload, oracle, measurement boundary, coverage
and remaining limitations. Distinguish CPU wiring tests from actual GPU calibration.
Do not claim GPU validation if only the CPU template was exercised.

Freeze the benchmark definition before optimizing. The optimization agent may
modify declared implementation files, not the oracle, metric, workload weights,
input rules, or tolerance. A user-requested change to these creates a new task
version and requires a new baseline/calibration.

To compare an implementation:

```bash
python -m kai_core benchmark run /path/to/task/benchmark.yaml \
  --candidate /path/to/candidate --split search --output /tmp/task-comparison.json
```

Exit code 0 means the comparison completed; inspect `acceptance.accepted` for
the scoped result. Freeze the chosen candidate and independently repeat on the
acceptance workloads before integration. A missed target is a valid experimental
outcome, not permission to weaken the benchmark. Use `kai-core optimize benchmark.yaml --config optimizer.yaml --output /path/to/new-run`
to connect this manifest to KAI Core's optimization loop. It freezes the
benchmark, evaluates candidates with this SDK and independently checks the selected
candidate on the acceptance split.
