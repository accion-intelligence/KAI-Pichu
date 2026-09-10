# KAI Core Benchmark Framework v1

Status: implemented as an independent SDK and CLI, integrated with KAI Core's
optimization loop through structured reports. Direct benchmark commands
remain usable without an optimization agent or model endpoint.

## Product contract

A task owner supplies a project, representative workloads, correctness rules,
and optimization objectives. A coding agent can implement the adapter using
the [AI integration guide](../src/kai_core/benchmark/AI_INTEGRATION.md). KAI validates
the adapter, calibrates measurement, and compares implementation workspaces
without requiring an LLM.

The benchmark must be useful independently of optimization. Its definition is
versioned separately from candidate code. A task adapter is not a timer, an
optimization policy, or a model prompt.

## Boundaries

| Layer | Responsibility | Owner |
| --- | --- | --- |
| Benchmark definition | Workload distribution, initial state, oracle, measured boundary, objectives and constraints | Task owner and their coding agent |
| Implementation | Source code that computes the declared workload | Project maintainer / optimization agent |
| Measurement SDK | Lifecycle, paired ordering, timing, raw samples, calibration and statistics | KAI |
| Optimization engine | Hypotheses, edits, resource allocation and selection | KAI Core optimizer |

Correctness is checked before ranking. A task that cannot establish correctness
must report the missing oracle rather than inventing a permissive validator.
No universal numerical tolerance or guaranteed target speedup is imposed.

## Public objects

The stable import surface is `kai_core.benchmark`:

- `Benchmark`: abstract task adapter, constructed with `BenchmarkSpec`.
- `Case`: unique ID, JSON-compatible parameters, positive weight and optional
  positive work-unit count. Seeds and workload parameters must describe the
  actual distribution; a random seed alone does not establish representativeness.
- `Observation`: opaque output plus optional named numeric metrics.
- `Validation`: boolean result, explanation, and optional finite numeric errors.
- `fixture_fingerprint`: the default content digest of a prepared fixture, covering
  scalars, strings, bytes, files by content, containers, NumPy arrays and tensors.
- `json_fingerprint`: canonical SHA-256 for JSON-compatible input descriptions.
- `load_python_file`: unique leaf-module loading without stale bytecode.

`kai_core.benchmark.runner.Runner` supplies `validate(checks_only=False)` and
`run(candidate_root)`, returning serializable reports. It has no model dependency.
The manifest is strictly validated by `BenchmarkSpec`; unknown fields and
unsupported protocol versions fail. Export its JSON Schema with
`kai-core benchmark schema`. Add schema changes through explicit protocol versions.

## Adapter lifecycle

```text
read manifest → enumerate deterministic cases → record source fingerprints
  → load baseline/candidate implementations
  → check input reproducibility, correctness, reset and invalid-output probes
  → baseline A/A calibration
  → paired A/B measurements, if calibrated
  → verify source fingerprints again → record scoped acceptance → cleanup
```

For each invocation, including warmups:

```text
reset → synchronize → start timer → run → complete measured work → stop timer
      → validate output and required side effects
```

`prepare`, `reset`, source loading/building, input hashing and validation are
outside measurement. Therefore an adapter must put any preparation required
by the declared user-visible metric INSIDE `run`. Output allocation belongs
inside `run` unless the workload explicitly specifies preallocated outputs.

`run` is one complete workload. `iterations` means repeated independent
invocations, each reset separately. A sequence of 4096 decode updates belongs
inside one `run`, if that sequence is the declared workload. A returned lazy
iterator is not completed work; consume it inside `run`.

The SDK checks reset fingerprints during conformance and warmup, not directly
before timed calls: hashing GPU data would itself change cache conditions.
The SDK digests the fixture itself; an adapter overrides `fingerprint` only to
exclude scratch/output buffers or to describe objects the default cannot digest.
The digest must cover actual input values/layout and mutable initial state.
Preparation is repeated to verify deterministic reconstruction. Reset hidden
model/optimizer/RNG/server state as well as state stored in the fixture.

Each prepared fixture and successfully loaded implementation has a cleanup
callback, including when verification or measurement raises. If loading or
preparation fails partway through, that method must clean up its partial resources.

## Execution backends and scope

v1 has two timers:

- `wall`: monotonic host elapsed time, with adapter `synchronize` before and
  after the call. A blocking subprocess is supported. GPU adapters must supply
  a real completion barrier if their call is asynchronous.
- `cuda_event`: PyTorch CUDA events on the configured device's current stream.
  PyTorch is optional and imported only when using this timer (or an adapter
  that imports it). Auxiliary streams must join the measured stream inside
  `run`. This cannot time another process's GPU work. Match CUDA Graph/eager
  behavior to deployment; the SDK does not silently capture graphs.

`scope` is `kernel`, `module`, or `end_to_end`. These labels are declarations,
not automatic verification of boundaries. CUDA events around a Python launch
can include GPU idle gaps caused by host dispatch. In particular, tiny kernels
may require a graph-backed adapter or a trusted native timing harness to obtain
the desired precision. Increasing iterations does not remove systematic bias.

External programs may expose native metrics through structured JSON, parsed by
the trusted adapter into `Observation.metrics`. Such metrics need a distinct
name (for example `kernel_ms`), declared units, and a documented timing method.
The SDK clock remains available as `latency_ms` and cannot be overwritten.

Real persistent-server and multi-device tasks need adapters with explicit
lifecycle/completion behavior. v1 does not supply a remote worker, automatic
GPU ownership, server isolation, or resource budgeting. Both in-process
implementations/fixtures must fit in memory; use an appropriately serialized
process-backed adapter for packages or servers that cannot coexist.

## Objectives and statistics

The primary objective declares `metric`, `unit`, `direction` and `scope`.
`latency_ms` is SDK-owned and minimized. `throughput` is SDK-owned and maximized,
computed as `Case.work_units / seconds`; the owner declares the work-unit meaning.
Other metrics are measured by trusted adapter code and must be finite; objective
values must be strictly positive.

An invocation's metrics are stored unchanged. Iterations within an arm are
averaged. Each block runs ABBA or BAAB, alternating blocks, with case order
shuffled by the fixed seed. Both implementations use both independently prepared
fixtures within each block: `A0 B0 B1 A1`, then `B1 A1 A0 B0`. Warmup covers every
implementation/fixture pairing and each record identifies its `fixture_slot`.
This prevents a persistent buffer placement effect from becoming an apparent
implementation speedup. Adapters must reset fixtures when switching implementations.
Each case's block log-speedup is the mean A log
metric minus the mean B log metric (reversed for maximize). Case log-speedups
are weighted to obtain the suite score. Reported speedups are geometric means
over blocks. v1 deliberately supports one explicit aggregation method,
`weighted_geomean_speedup`; it never averages heterogeneous units or silently
relabels proxy estimates as E2E.

CUDA event timing defers Python's automatic cyclic garbage collection between
event submissions and completion, restoring the prior GC state even on failure.
This prevents harness allocations from injecting a GC pause into the GPU event
interval. Wall timing preserves the application's GC policy. This does not
eliminate operating-system scheduling gaps or other resource interference.

Confidence intervals use deterministic percentile bootstrap of whole paired
blocks, jointly across cases. They assume sufficiently independent blocks.
They do not model cross-process/session drift. The intervals are pointwise,
not a simultaneous multiple-case guarantee.

A/A uses the SAME loaded baseline implementation in both arms, through the
same adapter path with independently prepared fixtures. The overall interval
and each case interval must fit wholly inside `1 ± calibration_tolerance`.
Checking only that the interval contains 1 would admit uninformative noise.
Defaults: 20 blocks, 3 independently reset calls per arm, 5 warmups, 2000
bootstrap draws, 95% confidence, and ±0.5% A/A tolerance. These are starting
settings, not a certification of accuracy on all workloads.

For A/B, the lower speedup bound must meet `target_speedup`, or exceed 1 if
there is no target. Each case must meet its regression constraint using the
lower bound. For minimize, a maximum fractional latency regression `r` means
speedup ≥ `1/(1+r)`; for maximize it means speedup ≥ `1-r`. Additional metric
limits apply to EVERY raw candidate invocation, not just its average.

`accepted` describes this split and measurement session only. Freeze the chosen
candidate and run acceptance on independently collected data/sessions before
release. Search-time winner selection is not independent confirmation. A
target miss is not proof that no optimization exists.

## Provenance and trust

Reports include the full spec, case definitions and fingerprint, input
fingerprints, source inventories/digests, SDK source digest/version, Python/host
metadata, available timer/device metadata, raw A/A and A/B samples, intervals,
constraints, and verdict. Any declared source/benchmark change during execution
invalidates the result. There is no automatic correction by dividing away A/A bias.

`agent_files` names the text files exposed to the optimization agent (interfaces,
descriptions). It excludes the adapter, oracle and input generation by design, and
the manifest rejects an `agent_files` entry that matches the adapter. Input synthesis
belongs to a context that never sees candidate code; the framework does not prescribe
value ranges or distributions.
Declare all adapter/helper files in `benchmark_files` (manifest and adapter are
automatic), all implementation build sources in `implementation.files`, and
external dataset/weight content hashes through input fingerprints. Explicitly
record relevant dependency/driver/server versions in task options or task
documentation. The SDK cannot inventory arbitrary imports, validate a dishonest
input fingerprint, prove that an adapter loads the supplied workspace, or infer
a scientific oracle. Mutation probes detect some vacuous validators, not all.

Adapters and implementations run with the invoking user's privileges. Unique
module names, file digests and separate roots are not OS isolation. v1 does not
claim adversarial evaluation integrity. Trusted benchmark code should be kept
outside the optimization agent's editable scope; changes to the benchmark mean
a new task version and a new baseline.

## Extension and migration

Keep hardware paths, data paths and application-specific settings in manifests
and adapter options. The SDK must not depend on TC-GNN, Qwen, PyG, specific
shape sets, or a particular LLM. Examples are ordinary adapters.

KAI Core consumes this protocol directly through `kai-core optimize`.
The original KAI command/regex workflow is not a KAI Core dependency. Adapters
can wrap existing executables, but must expose meaningful output verification
and measurement boundaries. The optimizer consumes structured SDK JSON reports,
not terminal summaries or an implementation's self-reported speedup.
