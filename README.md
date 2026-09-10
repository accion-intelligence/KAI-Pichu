# KAI-light

KAI-light optimizes **one GPU operator at a time** against a benchmark you own.
It combines a benchmark SDK with an agent loop for code generation, correctness
diagnosis, repair and performance optimization.

Define your inputs, correctness oracle and optimization metric once. The agent
edits implementation files; the SDK determines whether each candidate is correct,
whether measurement is calibrated, and whether the final candidate meets the goal.

## Install

Python 3.10+ on Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
kai-light --help
```

[requirements.txt](requirements.txt) pins the framework dependencies and installs
`kai_light` itself in editable mode. The editable install is not optional:
evaluations, tests and the NCU worker run as separate `python -m kai_light`
processes, which fail with an unhelpful `benchmark_error` if the package is
only on the current process's `sys.path`.
Install a CUDA-compatible PyTorch and CUDA toolkit for your workload separately.
The framework itself needs only Pydantic and PyYAML. CPU tests, help and dry runs
do not require a GPU, a model server, or the parent KAI repository.
Nsight Compute is optional; its executable and counter permissions must already
be available if profiling is enabled. The framework never invokes sudo.
Optional NCU report reading adds metric descriptions, NVIDIA rule findings,
and source/SASS queries. Without it, profiling retains structured CSV evidence.
See the [profiling guide](src/kai_light/PROFILE_GUIDE.md), also available from
an installed wheel through `kai-light profile --guide`.

## Define a benchmark

Give your coding agent the SDK instructions, then adapt a template or existing task:

```bash
kai-light benchmark guide
kai-light benchmark init /tmp/my-operator --template stateless
kai-light benchmark validate /tmp/my-operator/benchmark.yaml \
  --checks-only --output /tmp/operator-checks.json
```

The manifest declares input cases, editable implementation files, correctness,
metric scope, warmup/cache policy and acceptance criteria. The adapter supplies
the task semantics. Templates cover PyTorch, stateful code and native C++/CUDA.
For KAI SDK v1 tasks, change imports from `kai.benchmark` to
`kai_light.benchmark`; the manifest schema and adapter methods are unchanged.

The [LayerNorm example](examples/layernorm/README.md)
preserves the real `16384 x 1024` FP32 task, original CUDA baseline, independent
PyTorch oracle and all three outputs. It targets SM120. Its default manifest,
`benchmark-graph-events.yaml`, scores the operator's own GPU time between two CUDA
events captured inside the graph. The alternative `benchmark.yaml` times the graph
replay from the host with outer events, which includes submission gaps and is kept
for boundary comparison only. For a single operator, use the in-graph boundary.

## Run the optimizer

Copy [configs/optimizer.example.yaml](configs/optimizer.example.yaml) and set
your model and chat-completions-compatible endpoint. API keys come only from
the named environment variable, not a checked-in config. A live run checks that
this variable is set before creating the run directory or starting GPU preflight;
set `api_key_env: ""` for a local endpoint that needs no key. Generator and judge
may use the same model or separate configurations.
For OpenAI's Responses API, set `provider: responses`; see
[the GPT-5.6 Luna smoke config](configs/gpt56_luna_smoke.yaml). It uses
`max_output_tokens`, disables server-side response storage, and leaves temperature
unset while requesting high reasoning effort.
An installed wheel can also export the template with
`kai-light config --output optimizer.yaml` or its schema with `--schema`.

The command below loads one key from a dotenv file, selects one physical GPU,
and writes to a new output directory:

```bash
python scripts/run_with_env.py --env-file /path/to/.env --gpu GPU-YOUR-UUID -- \
  optimize examples/layernorm/benchmark-graph-events.yaml \
  --config configs/gpt56_luna_smoke.yaml --output runs/my-layernorm
```

With your own config and environment variables, the same task runs as:

```bash
export KAI_LIGHT_API_KEY=...
export CUDA_VISIBLE_DEVICES=0  # choose an idle physical GPU

kai-light optimize examples/layernorm/benchmark-graph-events.yaml \
  --config configs/optimizer.example.yaml --output runs/first --dry-run

# Inspect runs/first/plan.json, then execute the same frozen task:
kai-light optimize examples/layernorm/benchmark-graph-events.yaml \
  --config configs/optimizer.example.yaml --output runs/first --resume
```

For immediate execution, omit `--dry-run` on a new output directory.
`python -m kai_light` exposes the same commands as `kai-light`.

Each run contains:

- `bundle/`: frozen benchmark, oracle/helpers and baseline sources.
- `candidates/`: separate workspaces containing only declared implementation files.
- `llm/`: model requests, responses and token usage; no authorization headers.
- `reports/`: raw SDK results, process logs and GPU occupancy observations.
- `profiles/`: optional NCU feedback bound to the actual candidate and case.
- `state.json`: atomic checkpoint and charged budgets.
- `best_search/` and `best_search.patch`: the best supported search improvement.
- `accepted/`: created only when every configured independent acceptance run passes.
- `summary.json`: final status, acceptance result and artifact references.

The loop first validates and calibrates the baseline. Incorrect candidates enter
the judge-and-repair path; correct candidates enter the judge-and-optimize
path. Hardware feedback profiles the exact source being discussed. A/A failures
or resource conflicts pause the run instead of asking the model to repair code.
The performance judge starts with a small NCU overview, then gathers evidence on
demand: pose a diagnostic question, discover relevant counters, read selected
fields, and inspect rules or source instructions when needed. Paged results include
a ready-to-use next query. Responses preserve units, missing-data explanations
and report identity; repeated reads reuse cached data. Query budgets leave model
calls for the remaining optimization rounds.

The default GPU policy checks for other compute processes before, during and
after each evaluation. It kills only its own evaluation process group on a
conflict or timeout. This is polling, **not a GPU reservation**: coordinate device
ownership on shared machines. `resources.gpu_device` explicitly marks GPU-backed
external-command tasks; in-process CUDA tasks are detected from their timer/device.

## Resume and limits

Use the same config and `--resume` after an interruption, failed model request,
resource conflict or unstable calibration. The frozen benchmark and saved
candidates must be unchanged. Rounds and model calls are reserved before work;
an interrupted round is skipped on resume rather than silently retried. Budgets
include previous elapsed time, evaluations and model calls. Completed runs need
a new output directory. Resuming final validation starts a new acceptance set;
earlier reports remain on disk and do not disappear from the experiment record.

Exit code 0 means the run completed normally or a dry run was prepared; inspect
`summary.json.final_accepted` before treating a patch as accepted. Exit 1 indicates
an interrupted/blocked/failed run; exit 2 indicates invocation or configuration
failure. A valid optimization search may end with `no_improvement` or `not_accepted`.

Adapters and generated native/Python implementations execute as trusted local
code. Source allowlists and fingerprints protect the experiment contract but
are **not an OS security sandbox**. Compilation, model-call and overall budgets
are bounded; arbitrary untrusted tasks need an external isolation layer.

## Development and source provenance

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

See [architecture](docs/architecture.md), [benchmark contract](docs/benchmark-framework.md),
and [AI integration instructions](src/kai_light/benchmark/AI_INTEGRATION.md).

KAI-light draws inspiration from [VeloQ](https://github.com/lucifer1004/veloq).

Framework code is distributed under Apache 2.0; components retain their stated
licenses. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records source and
license information. No benchmark dataset, model checkpoint, API credential or
prior private run is needed for installation.
