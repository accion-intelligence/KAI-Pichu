# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KAI Pichu (`kai_pichu` package, `kai-pichu` CLI) is an agent harness for optimizing one CUDA operator (or fusing one kernel pipeline) at a time. It has two halves that must stay separate:

- **Benchmark SDK** (`src/kai_pichu/benchmark/`): owns operator semantics, inputs, oracle, tolerances, timing and acceptance. No timers, ranking or model calls belong in task adapters (`benchmark/api.py`).
- **Optimization loop** (`src/kai_pichu/optimizer/`): decides what implementation to propose next, using an LLM as coder and judge. It consumes only structured SDK reports; it never measures anything itself.

## Commands

```bash
python -m pip install -r requirements.txt   # deps + pytest + editable install (editable install is required, see below)
python -m pytest -q                          # full suite (CPU only; CI also installs CPU torch)
python -m pytest tests/test_optimizer.py -q -k <name>   # single test
python -m pip wheel --no-deps --no-build-isolation . -w dist   # CI packaging check

# Benchmark SDK (no model calls)
python -m kai_pichu benchmark validate <task>/benchmark.yaml --checks-only --output checks.json
python -m kai_pichu benchmark validate <task>/benchmark.yaml --split search --output preflight.json
python -m kai_pichu benchmark run|init|schema|guide ...

# Optimization loop
python -m kai_pichu config --output optimizer.yaml
python -m kai_pichu optimize <task>/benchmark.yaml --config optimizer.yaml --output runs/<name> [--dry-run|--resume]

# Read-only live dashboard over a run dir or a dir of runs (stdlib HTTP server)
python -m kai_pichu dashboard runs --port 8765

# Query an existing NCU report offline
python -m kai_pichu profile <report> --request '{"operation":"catalog","query":"memory"}'
```

`scripts/run_with_env.py --env-file .env --key-env <VAR> --gpu <id> -- <cli args>` runs the CLI with one dotenv key and a pinned GPU. `configs/*_smoke.yaml` are live smoke configs (cost money/GPU time). Report/output paths are written exclusively — reruns need new output paths.

## Architecture

- **Subprocess boundary.** Evaluation, tests and NCU profiling run in separate `python -m kai_pichu benchmark validate|run` and `python -m kai_pichu.profile_worker` processes (`evaluator.py`, `process.py`), each in its own process group with a deadline. This is why the package must be importable (editable install). `python -m kai_pichu benchmark ...` is dispatched in `cli.py` before argparse so it never imports the optimizer/LLM code (a test enforces this).
- **Task = manifest + adapter.** `benchmark.yaml` (pydantic model in `benchmark/models.py`, unknown fields rejected) points to `adapter.py:Task`, a `Benchmark` subclass with `cases/prepare/reset/load_implementation/run/validate` and mutation probes. Splits: `smoke/search/acceptance`. `kind: fusion` tasks expose read-only existing kernels. Templates live in `benchmark/templates/{stateless,stateful,command}`; `examples/` contains real tasks (`vector_add` is the CPU one).
- **Measurement** (`benchmark/runner.py`, `statistics.py`, `timing.py`): interleaved paired ABBA blocks, geometric-mean speedup with bootstrap CIs, per-case speedups. Task/source fingerprints (`fingerprints.py`) are checked around measurement; mutating the task invalidates the result.
- **Optimizer** (`optimizer/engine.py` `OptimizationLoop`): freezes task + baseline into `<output>/bundle`, preflights the baseline, then loops generate → evaluate → (repair judge | NCU profile + optimization judge with bounded profile queries) → generate. Candidates are JSON maps of complete file replacements restricted to the manifest's editable files. Eligible = correct, overall CI lower bound > 1, metric limits pass; the harness never ranks two eligible candidates beyond that. After search, the best candidate is rerun on the acceptance split `acceptance_repeats` times. Rounds and model calls are reserved before execution and checkpointed (`state.json`) for `--resume`. Prompts in `optimizer/prompts.py`; history tables in `history.py`; truncated-reply continuation in `continuation.py`.
- **Model transport** (`models.py`): `chat_completions`, `responses`, `anthropic` (optional extra), and `replay` (recorded text for offline control-flow tests — use this in tests, not real endpoints).
- **Dashboard** (`dashboard.py` + packaged `dashboard.html`): derives everything from files the loop already writes (request without response = call in flight, `.log` without `.process.json` = evaluation in flight, profile dir without `profile.json` = capture in flight; model calls get their round from loop order). It never writes into a run. If you change what or when the engine writes, check `tests/test_dashboard.py`.
- **Roofline** (`roofline.py`): with `profile.roofline` (default on) every NCU capture also collects NCU's roofline counters, `profile.json` stores the operator's point, and the loop captures the baseline once per search case after preflight (`profiles/baseline/<case>`). These are records only; keep them out of judge context. The dashboard's headroom map uses the baseline captures and each capture's Speed-of-Light metrics from `metrics.csv`; it does not plot the roofline (executed FLOP differs between implementations of one operator, so it misranks them).
- **Profiling** (`profiling.py`, `ncu_reader.py`, `profile_csv.py`, `ncu_compat/`): optional; missing profiles must be reported as unavailable, never invented. See `src/kai_pichu/PROFILE_GUIDE.md`.
- **Packaged docs are code.** `benchmark/MANUAL.md` (served by `benchmark guide`), `skills/kai-benchmark/SKILL.md` (installed by `skill install`), templates and `PROFILE_GUIDE.md` ship as package data and are checked by tests (e.g. templates must pass conformance). Update them with behavior changes, plus `docs/` (`WORKFLOW.md`, `MEASUREMENT.md`, `architecture.md`, `QUICKSTART.md`).

## Invariants to preserve

- The adapter, input generation and oracle are never shown to the optimizing model; only the manifest, editable sources and `agent_files` are.
- Candidate file maps reject new files, absolute paths, traversal and benchmark edits before materialization.
- Search and acceptance results stay separate; acceptance feedback is never used for retuning.
- Preserve upstream MIT attribution (`licenses/`, `THIRD_PARTY_NOTICES.md`) when touching optimizer code derived from CudaForge.
- `.env` holds real API keys and `runs/` holds model exchanges; don't print, commit or share them.
