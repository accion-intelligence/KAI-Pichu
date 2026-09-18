# Run the agent on your operator

[← README](../README.md)

This guide matches source commit `f7047172a5078ef43de6d915185d86ed21707fb5`. The product is KAI Pichu; the current Python module remains `kai_core`. Run commands from a source checkout. This document set does not contain the executable source.

## 1. Prepare the environment

Use a Linux environment with Python 3.10+. GPU execution needs your NVIDIA driver, a compatible CUDA toolkit/compiler, and the dependencies used by the task adapter. PyTorch is not installed automatically by the framework. Nsight Compute is optional and needs permission to access performance counters.

On shared machines, coordinate exclusive access to the selected GPU. The framework checks for competing compute processes; it does not reserve the device.

In an activated Python environment, from the repository root:

```bash
python -m pip install -e .
python -m kai_core --help
```

Core runtime dependencies are Pydantic 2 and PyYAML. Installation, exported guides, schema, and dry-run do not require model calls. GPU workload checks do require the corresponding GPU environment.

## 2. Define the task

A task consists of a **manifest** (semantics, objective, files, measurement settings) and an **adapter** (input preparation, implementation loading, execution, correctness checks). The benchmark is the executable contract for your operator, not a required public leaderboard.

Export the machine-readable schema and authoring instructions:

```bash
python -m kai_core benchmark guide --output task-authoring.md
python -m kai_core benchmark schema --output task-schema.json
```

Use these with your coding agent and your operator description. You decide the input domain, output semantics, baseline, tolerances, timed boundary, and editable files. Review the generated contract before optimization.

To scaffold the adapter yourself, choose a suitable template and a new directory:

```bash
python -m kai_core benchmark init /absolute/path/to/your-task --template stateless
```

Available templates: `stateless`, `stateful`, and `command` (external-command workloads, including native implementations). Replace the template workload and validator with your task. A template is a starting structure, not a validator for an arbitrary new operator.

Important API contracts:

- `cases(split)` supplies the cases for `smoke`, `search`, or `acceptance`.
- `prepare(case, seed)` creates a reproducible fixture. The SDK digests it automatically; override `fingerprint(fixture)` only to exclude scratch or output buffers.
- `agent_files` in the manifest lists the interface and description files the optimization agent may read. The adapter, the reference and input generation are never shown to the agent.
- `load_implementation(workspace)` loads the baseline or candidate implementation.
- `run(implementation, fixture)` returns an `Observation`.
- `validate(case, fixture, observation)` returns a `Validation`.
- `invalid_observations(case, fixture, valid)` supplies invalid observations the validator must reject.
- `reset(...)` and `synchronize(...)` define repeatable execution and completion where required.

Use the generated guide and [actual API](https://github.com/accion-intelligence/KAI-Core/blob/f7047172a5078ef43de6d915185d86ed21707fb5/src/kai_core/benchmark/api.py) for full signatures. Measurement fields live under `measurement`; the SDK timer options are `wall` and `cuda_event`. Example-specific metric implementations are not additional SDK timer options.

Check conformance before making model calls:

```bash
python -m kai_core benchmark validate /absolute/path/to/your-task/benchmark.yaml \
  --checks-only --output task-checks.json
```

This executes the task’s checks, which may require a GPU, without timing. For a complete baseline preflight on the search cases (including A/A calibration if the manifest enables it):

```bash
python -m kai_core benchmark validate /absolute/path/to/your-task/benchmark.yaml \
  --split search --output task-preflight.json
```

The optimizer also runs a preflight before its first generation call. Use new report paths on repeated manual checks: reports are not overwritten.

## 3. Choose the models and budget

```bash
python -m kai_core config --output optimizer.yaml
```

Edit the exported file:

| Setting | What to choose |
| --- | --- |
| `generator.provider` | `chat_completions` or `responses` for OpenAI-shaped endpoints; `anthropic` for Claude via the official SDK (`pip install 'kai-core[anthropic]'`, see `configs/claude_opus5_smoke.yaml`) |
| `generator.model`, `generator.base_url` | Your model identifier and compatible API base URL |
| `generator.api_key_env` | The name of the environment variable holding your key |
| `judge` | Optional separate model configuration; omit to reuse the generator model |
| `budget` | Rounds, model calls, total time, per-evaluation timeout, acceptance repeats |
| `profile.enabled` | Set `true` to collect NCU evidence when your environment supports it |

The exported default key-variable name is `KAI_CORE_API_KEY`. Set it securely in your environment, or change `api_key_env` to an existing key-variable name. For a local endpoint that needs no authentication, set `api_key_env: ""`. Do not place secrets in the task or commit them to source control.

Alternatively, from the source checkout, use the bundled launcher to load the key from a dotenv file and select a physical GPU by UUID. With your configured `optimizer.yaml` and a new output directory:

```bash
python scripts/run_with_env.py --env-file /path/to/.env --gpu GPU-YOUR-UUID -- \
  optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator
```

Generation and judge calls send task context and implementation source to your configured model endpoint. Select an endpoint appropriate for the confidentiality of your code. Endpoint compatibility does not guarantee that a model can successfully optimize your operator.

The tool uses your compute and model account. Round/call/time limits bound the workflow; they are not a dollar-denominated billing guarantee. NCU diagnostic queries also consume judge calls. Leave sufficient time for final acceptance.

## 4. Inspect the plan, then run

Optional dry-run, using a new output directory:

```bash
python -m kai_core optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator --dry-run
```

Inspect `plan.json` and the frozen task bundle. Dry-run makes no model calls and does not execute the GPU workload; it does not prove that the task compiles or passes validation.

To execute that planned run, use the same configuration and directory:

```bash
python -m kai_core optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator --resume
```

Or skip dry-run and start a fresh run with the same command **without** `--resume`, using a new output directory.

An interrupted run can resume with the original configuration; only the `budget` section (rounds, model calls, seconds) may differ, and such changes are recorded in `state.json` and `summary.json`. Reserved calls and interrupted rounds remain charged to the budget; the loop does not replay unknown calls. Completed runs cannot be resumed. Start a new run to change the task or configuration.

## 5. Inspect the work

| Path | Contents / availability |
| --- | --- |
| `summary.json` | Final status, `final_accepted`, report references, elapsed time and model usage |
| `state.json` | Candidate history, current/best state and reserved budgets |
| `bundle/` | Frozen task and baseline |
| `candidates/` | Generated implementation snapshots |
| `llm/` | Model requests, responses and errors |
| `reports/` | Evaluation reports and process logs |
| `profiles/` | NCU capture and query records, when profiling ran |
| `best_search/`, `best_search.patch` | Best eligible search candidate, when one exists |
| `accepted/` | Selected source, only after successful final acceptance |

`accepted` means the supplied tests and acceptance rules passed. It is not a guarantee about untested inputs, another GPU, or an application’s end-to-end latency.

## Optional: inspect NCU evidence directly

```bash
python -m kai_core profile --guide
python -m kai_core profile --dependencies
python -m kai_core profile --schema
```

NCU capture and the optional report-reader integration are distinct. Structured CSV evidence remains available without the reader; richer rules and source-level queries depend on installed capabilities. Querying a saved report cannot recover counters that were not captured.

Run generated code only in an environment suitable for executing it. The framework’s process management and integrity checks are not a security sandbox.
