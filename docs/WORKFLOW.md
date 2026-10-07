# Inside the CUDA kernel agent

[← README](../README.md)

The agent separates **writing a candidate**, **deciding what to change next**, and **measuring what actually happened**. Coder and judge are model roles; they may use the same model. The benchmark SDK supplies execution and measurement, not an LLM opinion about correctness or speed.

## The task comes first

The engineer defines the operator contract and supplies a runnable baseline, inputs, validator, objective, permitted source files, and measurement rules. Task authoring can be done by hand from the packaged manual, schema and templates, or by a coding agent through the packaged `kai-benchmark` skill (`kai-pichu skill install`). The optimizer itself consumes the resulting manifest and adapter.

A run freezes the task and baseline into its own workspace. Before generating CUDA, it executes the baseline checks on the search split and times the baseline. A failure here is a task or measurement problem, not evidence that generated code needs repair.

## The first candidate

A reply that the model's output limit cuts off is not discarded: the loop sends the text written so far back and asks the model to continue from that point, up to `max_continuations` times, and splices the pieces into one reply; if they do not form a valid object, the round takes the ordinary repair path. A reply that fails for another reason (empty, filtered) goes straight to repair, and a transport fault is retried within the call's time budget. The coder receives the task description, current implementation source, hardware context from preflight, the editable-file list, and the round history: a table of the recent rounds with overall and per-case speedups, each round's hypothesis and the judge diagnosis behind it, plus a per-case table of the best speedup, when it was reached and how many rounds it has stalled. The judge receives the same history alongside the measurements and profile, and chooses the starting point: it can read the saved code of any past round (`read_candidate`, sharing the evidence budget with profile queries) and name the round the coder should modify (`base_round`); by default the coder builds on the latest eligible candidate. Every round's code stays under `candidates/` for the run's lifetime. When the previous round's candidate completed but did not replace the best, the judge also receives that attempt's measurements and a profile overview of it, taken on the case where it fell furthest behind the best, so a failed idea is diagnosed from its own evidence rather than proposed again from the best candidate's profile. It returns a concrete hypothesis and complete replacement contents for the files it changes. The workspace constructs and evaluates a candidate snapshot. It does not apply generated changes to the original source checkout.

## Feedback loop 1: diagnose and repair

If the current candidate cannot complete evaluation—for example, because it fails to build or fails correctness—the judge receives the current source and actual error feedback. It identifies one critical issue and a minimal fix hint. The coder uses this strategy to create the next candidate.

This path does not treat a broken candidate as a performance optimization target.

## Feedback loop 2: investigate and optimize

For valid candidates, the loop starts from the latest candidate that passed the eligibility rules (correct, a confirmed gain over the baseline overall, no metric-limit failure; per-case regressions are reported, not vetoed) when one exists; otherwise it uses the current candidate. The harness never ranks two eligible candidates against each other: a real gain on one case would be lost to measurement jitter on another. When an earlier candidate holds the highest overall speedup, its measurements and source travel with the context so the model can carry that code forward. The judge receives the source, measurements, and recent hypotheses/results.

After a completed A/B evaluation, feedback includes the scored metric’s mean, median, minimum, maximum, and sample count for each case and arm, together with its unit, direction, scope, and boundary. Other recorded metrics are identified as unscored. The coder receives this feedback alongside the judge’s strategy. These descriptive statistics do not replace the paired-block speedup or its confidence interval; reports without A/B measurements do not include this summary.

With profiling enabled, NCU captures the current implementation on every search case each round (or a rotating `profile.cases_per_round` of them), since a candidate may dispatch different code by shape and the harness does not rank cases by importance. The judge receives one overview per case, with its launches, headline metrics, rule count and a stall summary of the hottest instructions with source lines, and addresses each query to a `case_id` or to `all` cases side by side. The judge starts with a compact evidence overview. It can ask a focused question and query captured metrics, rules, or source-level details, subject to available reader capabilities and query budgets. These queries inspect the existing report; they do not launch new captures to obtain missing counters.

The judge produces one bottleneck hypothesis, optimization method, and modification plan. The coder applies the plan, and the benchmark evaluates the new candidate. Missing profiling is reported as missing evidence, not invented hardware data. Profiling is optional; ordinary measured feedback still supports the optimization loop.

## Progress and stopping

A candidate is retained as the best search result only if the implementation is runnable, the relevant constraints pass, the overall speedup interval’s lower bound exceeds 1, and its point estimate improves on the previous best.

Rounds, model calls, elapsed time, and evaluation timeouts are bounded. Model calls and rounds are reserved before execution so interruptions do not silently provide a fresh budget. Resume requires the same configuration and a verifiable frozen task; it advances past an interrupted reserved round.

The current loop runs the configured search rounds, then independently evaluates the best eligible candidate on the acceptance split for the configured number of repeats. Search improvement and final acceptance are separate states. If the time or call budget stops the run first, acceptance is not implied.

## Outputs belong to the engineer

Candidates, hypotheses, requests, responses, measurements, optional profiles, and run state remain available for inspection. The best eligible search candidate produces a source export and patch. Successful final acceptance additionally produces `accepted/`.

The engineer decides whether those results solve the intended problem and how to integrate the implementation. The agent does not claim application-level gains from an isolated operator result.

## Source map

The diagram groups behavior for readability; it is not a deployment diagram or a security boundary.

| Diagram / behavior | Source in this repository |
| --- | --- |
| Generation, branching, best candidate, budgets, resume, acceptance | [OptimizationLoop](../src/kai_pichu/optimizer/engine.py) |
| Repair diagnosis, performance strategy, on-demand profile questions | [Prompts and strategy contract](../src/kai_pichu/optimizer/prompts.py) |
| Evaluation processes, feedback, capture and report queries | [Evaluator](../src/kai_pichu/evaluator.py) |
| Input preparation, validation, baseline timing, paired measurement | [Benchmark runner](../src/kai_pichu/benchmark/runner.py) |
| Model, budget, resource and profiling configuration | [Configuration](../src/kai_pichu/config.py) |
