# Contributing to KAI Core Agent

[← README](README.md)

Help the agent do better CUDA engineering: support a new operator, improve a diagnosis, reduce wasted experiments, or make a workflow easier to use.

## Where to contribute

- **Operator tasks:** useful specifications, reference implementations, representative cases, and explicit measurement boundaries. Start with the [task-authoring workflow](docs/QUICKSTART.md#2-define-the-task).
- **Generation and diagnosis:** reproducible build failures, missed bottlenecks, better repair strategies, or more effective use of hardware feedback. See the [agent workflow](docs/WORKFLOW.md).
- **Execution and usability:** endpoint compatibility, profiling integration, interruption/resume behavior, setup instructions, and clearer errors.
- **Measurement quality:** incorrect validators, unstable timing, boundary mistakes, or missing regression coverage. Explain the impact on the engineering decision; see [measurement rules](docs/MEASUREMENT.md).

## Report a problem or experiment

Include the source commit; GPU, driver, CUDA, Python and task dependencies; model/provider configuration without secrets; the command and budget; expected versus observed behavior; and the smallest task or logs needed to reproduce it.

For performance reports, include the baseline, shapes, dtype, correctness rules, timing boundary, confidence intervals, and whether the result comes from search or final acceptance. Share unsuccessful attempts when they explain a useful failure mode. A speedup ratio alone is not sufficient.

Do not upload credentials, proprietary source, customer inputs, or unsanitized model exchanges. Review `llm/`, profiles and process logs before sharing: these may contain task and source context.

## Before submitting a change

1. Keep the change focused and explain the engineering behavior it improves. Discuss broad architecture changes in an issue first.
2. Add regression coverage for affected behavior; use the test commands and dependencies in the current source checkout. Record exactly what you ran and what you could not test.
3. For task changes, validate the contract before spending model calls:

   ```bash
   python -m kai_core benchmark validate /absolute/path/to/your-task/benchmark.yaml \
     --checks-only --output task-checks.json
   python -m kai_core benchmark validate /absolute/path/to/your-task/benchmark.yaml \
     --split search --output task-preflight.json
   ```

   Checks may execute GPU workloads. The second command also calibrates the baseline. Use new output paths when rerunning; neither command exercises the model-driven optimization loop.

4. Update documentation when commands, task contracts, configuration or outputs change. State required hardware and model costs for live tests.
5. Preserve upstream attribution and license notices. Identify the origin and license of contributed third-party code.

Generated code and task adapters execute in your environment; the framework is not a security sandbox. Avoid posting exploitable security details or sensitive logs publicly. If no private reporting channel is published, ask a maintainer for one before disclosing details.
