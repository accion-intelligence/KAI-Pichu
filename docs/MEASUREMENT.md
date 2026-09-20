# What a measurement establishes

[← README](../README.md)

The agent’s measurement system supports the engineering loop. It answers whether a candidate passes the **declared task** and improves the **declared metric** under the measured conditions. It does not prove universal correctness or isolate hostile code.

## Correctness is task-defined

The adapter owns input preparation, the reference, tolerances, result validation, state reset and synchronization. It must provide deliberately invalid observations that its validator rejects. These probes expose some weak validators; they are not a proof of complete coverage.

The engineer also defines the cases returned for each split. An acceptance split is a separate execution path, but it is only meaningfully held out if the adapter supplies distinct cases, and it is only a fair test if those cases stay inside the ranges the search cases span: acceptance changes the points, seeds and value distributions, not the range of any input dimension. A shape that appears only in acceptance tests a regime the agent never measured.

## Establish the measurement boundary

The manifest names the objective and its scope, units, direction, boundary and cache policy. SDK timers are `wall` and `cuda_event`; adapters can expose additional metrics with explicit semantics. Compilation, fixture preparation, reset and correctness checks are outside the ordinary timed invocation. Synchronization and custom metric behavior must match the declared boundary.

The packaged examples implement their `operator_latency_ms` metric inside the adapter with the `graph_events` option: two external CUDA events recorded inside a CUDA Graph around exactly one operator call. It is not a third SDK timer enum. That latency excludes host submission, compilation and graph capture, and it must not be presented as end-to-end application latency. The SDK's own `cuda_event` timer measures an outer GPU-timeline boundary that can include submission-related idle gaps; the two boundaries answer different questions, so do not compare their numbers as if they measured the same work.

## Time the baseline, then compare

Preflight times the baseline alone on every case and reports per-case statistics, so the task owner sees what the boundary measures before any model call. There is no baseline-versus-baseline stability gate: the confidence intervals of the paired A/B comparison and the acceptance rules are the precision guard.

The runner interleaves A/B observations in alternating ABBA/BAAB blocks and crosses fixture slots. This reduces particular order and allocation biases; it does not eliminate all interference. The protocol uses weighted geometric-mean speedup and percentile-bootstrap confidence intervals. The current implementation does not remove outliers.

Output poisoning and a 256 MiB cache flush belong to the packaged example adapters, not to every task.

## Search and acceptance are different

The loop records every eligible candidate and builds on the latest one. On normal search completion, it freezes the eligible candidate with the highest overall speedup and runs the configured acceptance repetitions. All repetitions must pass the declared target and metric-limit rules for the final status to be `accepted`; every case's speedup is reported beside the geometric mean, and cases below the regression margin are listed for review.

A candidate can remain useful as a search artifact without becoming an accepted result. Inspect `summary.json`, not just the command’s exit code.

## Integrity checks are not a sandbox

The optimizer limits generated edits to declared implementation files and checks frozen task/source fingerprints. It invokes evaluation in a separate process, but the implementation and adapter execute within the evaluator’s environment. The runner explicitly treats adapters as trusted local code.

Before/after fingerprints detect observable differences at the check points; they cannot guarantee detection of every transient mutation that is restored between checks. Process management is not OS-level isolation from malicious code. Use an execution environment appropriate to the trust you place in the task and generated implementation.

Source: [benchmark API](../src/kai_pichu/benchmark/api.py), [schema](../src/kai_pichu/benchmark/models.py), [runner](../src/kai_pichu/benchmark/runner.py).
