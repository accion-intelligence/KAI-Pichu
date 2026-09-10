# What a measurement establishes

[← README](../README.md)

The agent’s measurement system supports the engineering loop. It answers whether a candidate passes the **declared task** and improves the **declared metric** under the measured conditions. It does not prove universal correctness or isolate hostile code.

## Correctness is task-defined

The adapter owns input preparation, the reference, tolerances, result validation, state reset and synchronization. It must provide deliberately invalid observations that its validator rejects. These probes expose some weak validators; they are not a proof of complete coverage.

The engineer also defines the cases returned for each split. An acceptance split is a separate execution path, but it is only meaningfully held out if the adapter supplies appropriate distinct cases.

## Establish the measurement boundary

The manifest names the objective and its scope, units, direction, boundary and cache policy. SDK timers are `wall` and `cuda_event`; adapters can expose additional metrics with explicit semantics. Compilation, fixture preparation, reset and correctness checks are outside the ordinary timed invocation. Synchronization and custom metric behavior must match the declared boundary.

The LayerNorm example’s `graph_events` option implements its operator metric inside the adapter. It is not a third SDK timer enum. Its operator latency excludes host submission, compilation and graph capture; it must not be presented as end-to-end application latency.

The example therefore ships two manifests. `benchmark-graph-events.yaml` uses that adapter metric; `benchmark.yaml` uses the SDK `cuda_event` timer, an outer GPU-timeline boundary that can include submission-related idle gaps. It is not a CPU wall-clock measurement. The two boundaries answer different questions; do not compare their numbers as if they measured the same work. Use `benchmark-graph-events.yaml` for optimizing this operator. The outer-event manifest is retained for boundary comparison; submission-related idle gaps can prevent A/A calibration from passing. This does not establish that `cuda_event` is unsuitable in general. Calibrate whichever boundary you choose on your own hardware.

## Calibrate, then compare

Baseline-vs-baseline A/A measurements must satisfy the configured tolerance both overall and per case. An unsuccessful calibration means the run does not meet that precision criterion; it does not by itself prove that every possible larger gain would be noise.

The runner interleaves A/B observations in alternating ABBA/BAAB blocks and crosses fixture slots. This reduces particular order and allocation biases; it does not eliminate all interference. The protocol uses weighted geometric-mean speedup and percentile-bootstrap confidence intervals. The current implementation does not remove outliers.

Output poisoning and a 256 MiB cache flush belong to the packaged LayerNorm adapter, not to every task.

## Search and acceptance are different

The loop tracks eligible search improvements. On normal search completion, it freezes the selected candidate and runs the configured acceptance repetitions. All repetitions must pass the declared target, regression and metric-limit rules for the final status to be `accepted`.

A/A instability stops the optimizer. A candidate can remain useful as a search artifact without becoming an accepted result. Inspect `summary.json`, not just the command’s exit code.

## Integrity checks are not a sandbox

The optimizer limits generated edits to declared implementation files and checks frozen task/source fingerprints. It invokes evaluation in a separate process, but the implementation and adapter execute within the evaluator’s environment. The runner explicitly treats adapters as trusted local code.

Before/after fingerprints detect observable differences at the check points; they cannot guarantee detection of every transient mutation that is restored between checks. Process management is not OS-level isolation from malicious code. Use an execution environment appropriate to the trust you place in the task and generated implementation.

Source: [benchmark API](https://github.com/accion-intelligence/KAI-Light/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_light/benchmark/api.py), [schema](https://github.com/accion-intelligence/KAI-Light/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_light/benchmark/models.py), [runner](https://github.com/accion-intelligence/KAI-Light/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_light/benchmark/runner.py).
