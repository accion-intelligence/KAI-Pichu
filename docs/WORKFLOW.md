# Inside the CUDA kernel agent

[← README](../README.md)

The agent separates **writing a candidate**, **deciding what to change next**, and **measuring what actually happened**. Generator and judge are model roles; they may use the same model. The benchmark SDK supplies execution and measurement, not an LLM opinion about correctness or speed.

## The task comes first

The engineer defines the operator contract and supplies a runnable baseline, inputs, validator, objective, permitted source files, and measurement rules. Task authoring can be assisted by a coding agent using the packaged guide, schema, and templates. The optimizer itself consumes the resulting manifest and adapter.

A run freezes the task and baseline into its own workspace. Before generating CUDA, it executes the baseline checks on the search split, plus A/A calibration if the manifest enables it. A failure here is a task or measurement problem, not evidence that generated code needs repair.

## The first candidate

The generator receives the task description, current implementation source, hardware context from preflight, and editable-file list. It returns a concrete hypothesis and complete replacement contents for the files it changes. The workspace constructs and evaluates a candidate snapshot. It does not apply generated changes to the original source checkout.

## Feedback loop 1: diagnose and repair

If the current candidate cannot complete evaluation—for example, because it fails to build or fails correctness—the judge receives the current source and actual error feedback. It identifies one critical issue and a minimal fix hint. The generator uses this strategy to create the next candidate.

This path does not treat a broken candidate as a performance optimization target. Conversely, when calibration is enabled, A/A instability stops the run rather than triggering a code-repair loop.

## Feedback loop 2: investigate and optimize

For valid candidates, the loop starts from the best eligible candidate when one exists. Otherwise it uses the current candidate. The judge receives the source, measurements, and recent hypotheses/results.

After a completed A/B evaluation, feedback includes the scored metric’s mean, median, minimum, maximum, and sample count for each case and arm, together with its unit, direction, scope, and boundary. Other recorded metrics are identified as unscored. The generator receives this feedback alongside the judge’s strategy. These descriptive statistics do not replace the paired-block speedup or its confidence interval; reports without A/B measurements do not include this summary.

With profiling enabled, NCU captures the selected implementation and case. The judge starts with a compact evidence overview. It can ask a focused question and query captured metrics, rules, or source-level details, subject to available reader capabilities and query budgets. These queries inspect the existing report; they do not launch new captures to obtain missing counters.

The judge produces one bottleneck hypothesis, optimization method, and modification plan. The generator applies the plan, and the benchmark evaluates the new candidate. Missing profiling is reported as missing evidence, not invented hardware data. Profiling is optional; ordinary measured feedback still supports the optimization loop.

## Progress and stopping

A candidate is retained as the best search result only if the implementation is runnable, the relevant constraints pass, the overall speedup interval’s lower bound exceeds 1, and its point estimate improves on the previous best.

Rounds, model calls, elapsed time, and evaluation timeouts are bounded. Model calls and rounds are reserved before execution so interruptions do not silently provide a fresh budget. Resume requires the same configuration and a verifiable frozen task; it advances past an interrupted reserved round.

The current loop runs the configured search rounds, then independently evaluates the best eligible candidate on the acceptance split for the configured number of repeats. Search improvement and final acceptance are separate states. If the time or call budget stops the run first, acceptance is not implied.

## Outputs belong to the engineer

Candidates, hypotheses, requests, responses, measurements, optional profiles, and run state remain available for inspection. The best eligible search candidate produces a source export and patch. Successful final acceptance additionally produces `accepted/`.

The engineer decides whether those results solve the intended problem and how to integrate the implementation. The agent does not claim application-level gains from an isolated operator result.

## Source map

The diagram groups behavior for readability; it is not a deployment diagram or a security boundary.

| Diagram / behavior | Source at the reviewed commit |
| --- | --- |
| Generation, branching, best candidate, budgets, resume, acceptance | [OptimizationLoop](https://github.com/accion-intelligence/KAI-Core/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_core/optimizer/engine.py) |
| Repair diagnosis, performance strategy, on-demand profile questions | [Prompts and strategy contract](https://github.com/accion-intelligence/KAI-Core/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_core/optimizer/prompts.py) |
| Evaluation processes, feedback, capture and report queries | [Evaluator](https://github.com/accion-intelligence/KAI-Core/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_core/evaluator.py) |
| Input preparation, validation, paired measurement, A/A | [Benchmark runner](https://github.com/accion-intelligence/KAI-Core/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_core/benchmark/runner.py) |
| Model, budget, resource and profiling configuration | [Configuration](https://github.com/accion-intelligence/KAI-Core/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_core/config.py) |
