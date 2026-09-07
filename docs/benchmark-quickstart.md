# Benchmark SDK quickstart

Use Python 3.10+ in your project's existing environment:

```bash
python -m pip install -e .
python -m kai_light benchmark --help
python -m kai_light benchmark schema --output /tmp/kai-benchmark-schema.json
python -m kai_light benchmark guide --output /tmp/KAI_BENCHMARK_INTEGRATION.md
```

Give the generated guide to your coding agent, together with your project
requirements and representative workloads. The SDK itself needs Pydantic and
PyYAML. It does not install PyTorch, CUDA, vLLM, models, or profiler tools.

## Start from an example

```bash
python -m kai_light benchmark init /tmp/my-benchmark --template stateful
python -m kai_light benchmark validate /tmp/my-benchmark/benchmark.yaml \
  --checks-only --output /tmp/my-checks.json
```

`stateful` is a dependency-free recurrence example. `stateless` demonstrates a
PyTorch operator (CPU by default, CUDA settings documented in its README).
`command` builds and calls an external C++/CUDA executable (CPU compiler by
default). All are editable teaching examples, not representative production
benchmarks. Their `README.md` explains the lifecycle and measurement boundary.

Adapt the adapter and manifest to your project. Implement the entire task
definition, including error probes, workload splits, source inventories, and
state reset. `--checks-only` verifies wiring but explicitly leaves `ready=false`.

## Calibrate and compare

```bash
python -m kai_light benchmark validate /tmp/my-benchmark/benchmark.yaml \
  --split search --output /tmp/my-calibration.json

# The candidate root contains the files listed in implementation.files.
python -m kai_light benchmark run /tmp/my-benchmark/benchmark.yaml \
  --candidate /path/to/candidate --split search --output /tmp/my-comparison.json
```

`run` checks both implementations and performs fresh baseline A/A calibration
before measuring A/B. An unstable A/A prevents candidate performance acceptance.
Inspect `calibration`, `comparison`, `acceptance`, and raw `records` in the report.
Do not silently relax calibration tolerance merely to get a green result.

All output paths must be new. The CLI refuses to overwrite a previous report
or scaffold into an existing directory. Error reports include a structured
failure message. Exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | Conformance checks passed, calibration ready, or a valid A/B comparison completed |
| 1 | Calibration was unstable; no accepted optimization |
| 2 | Invalid configuration, build/execution/verification failure, or path error |

Exit 0 for `run` does NOT mean the performance target was met. Check
`acceptance.accepted`. Even that result is scoped to the selected split/session.
After freezing a promising candidate, independently rerun `--split acceptance`
and check the real application's E2E metric before integrating an optimization
initially ranked by a kernel/module proxy.

## Custom metrics

Return task-specific observations from the adapter:

```python
return Observation(output=result, metrics={"energy_j": joules, "peak_bytes": peak})
```

Declare the primary metric and optional raw-sample limits:

```yaml
objective:
  metric: energy_j
  unit: J
  direction: minimize
  scope: end_to_end
  target_speedup: 1.10
  max_case_regression: 0.01
limits:
  - metric: peak_bytes
    maximum: 8000000000
```

The adapter must measure custom quantities honestly and document their units
and boundaries. `latency_ms` and `throughput` are reserved SDK metrics.

For lifecycle details, limitations and statistical definitions, see the
[framework contract](benchmark-framework.md). Pass the finished manifest to
`kai-light optimize` to run the optimization loop.
