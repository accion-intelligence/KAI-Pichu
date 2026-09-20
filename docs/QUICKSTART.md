# Run the agent on your operator

[← README](../README.md)

This guide describes the code in this repository. Run the commands from a source checkout after `python -m pip install -e .`; the CLI is `kai-pichu`, and `python -m kai_pichu` is equivalent.

## 1. Prepare the environment

Use a Linux environment with Python 3.10+. GPU execution needs your NVIDIA driver, a compatible CUDA toolkit/compiler, and the dependencies used by the task adapter. PyTorch is not installed automatically by the framework. Nsight Compute is optional and needs permission to access performance counters.

On shared machines, coordinate exclusive access to the selected GPU. The framework checks for competing compute processes; it does not reserve the device.

In an activated Python environment, from the repository root:

```bash
python -m pip install -e .
python -m kai_pichu --help
```

Core runtime dependencies are Pydantic 2 and PyYAML. Installation, exported guides, schema, and dry-run do not require model calls. GPU workload checks do require the corresponding GPU environment.

## 2. Define the task

A task consists of a **manifest** (semantics, objective, files, measurement settings) and an **adapter** (input preparation, implementation loading, execution, correctness checks). The benchmark is the executable contract for your operator, not a required public leaderboard.

With a coding agent (Claude Code or Codex), install the packaged skill and describe your operator:

```bash
python -m kai_pichu skill install            # ./.claude/skills and ./.codex/skills; add --scope user for your home directory
```

The skill follows the benchmark manual shipped with the SDK. You decide the input domain, output semantics, baseline, tolerances, timed boundary, and editable files; the agent asks when it cannot establish them and shows you the contract for review before optimization. To read the manual yourself:

```bash
python -m kai_pichu benchmark guide --output task-manual.md
python -m kai_pichu benchmark schema --output task-schema.json
```

To scaffold the adapter yourself, choose a suitable template and a new directory:

```bash
python -m kai_pichu benchmark init /absolute/path/to/your-task --template stateless
```

Available templates: `stateless`, `stateful`, and `command` (external-command workloads, including native implementations). Replace the template workload and validator with your task. A template is a starting structure, not a validator for an arbitrary new operator.

The adapter protocol (`cases`, `prepare`, `load_implementation`, `run`, `validate`, `invalid_observations`, `reset`, `synchronize`, the optional `fingerprint`) and every manifest field are specified in the manual; [`api.py`](../src/kai_pichu/benchmark/api.py) has the signatures. The SDK timer options are `wall` and `cuda_event`; example-specific metric implementations are not additional SDK timers.

Check conformance before making model calls:

```bash
python -m kai_pichu benchmark validate /absolute/path/to/your-task/benchmark.yaml \
  --checks-only --output task-checks.json
```

This executes the task’s checks, which may require a GPU, without timing. For a complete baseline preflight on the search cases, which also times the baseline:

```bash
python -m kai_pichu benchmark validate /absolute/path/to/your-task/benchmark.yaml \
  --split search --output task-preflight.json
```

The optimizer also runs a preflight before its first generation call. Use new report paths on repeated manual checks: reports are not overwritten.

## 3. Choose the models and budget

```bash
python -m kai_pichu config --output optimizer.yaml
```

Edit the exported file:

| Setting | What to choose |
| --- | --- |
| `generator.provider` | `chat_completions` or `responses` for OpenAI-shaped endpoints; `anthropic` for Claude via the official SDK (`pip install 'kai-pichu[anthropic]'`, see `configs/claude_opus5_smoke.yaml`) |
| `generator.model`, `generator.base_url` | Your model identifier and compatible API base URL |
| `generator.api_key_env` | The name of the environment variable holding your key |
| `judge` | Optional separate model configuration; omit to reuse the generator model |
| `budget` | Rounds, model calls, total time, per-evaluation timeout, acceptance repeats |
| `generator.max_continuations` | When a reply is cut off by `max_tokens`, the loop asks the model to continue it up to this many times (default 2) and splices the pieces; each continuation is a model call |
| `profile.enabled` | Set `true` to collect NCU evidence when your environment supports it |
| `profile.case_id` | Pin the search case NCU captures; by default it follows the case with the lowest measured speedup |

The exported default key-variable name is `KAI_PICHU_API_KEY`. Set it securely in your environment, or change `api_key_env` to an existing key-variable name. For a local endpoint that needs no authentication, set `api_key_env: ""`. Do not place secrets in the task or commit them to source control.

Alternatively, from the source checkout, use the bundled launcher to load the key from a dotenv file and select a physical GPU by UUID. With your configured `optimizer.yaml` and a new output directory:

```bash
python scripts/run_with_env.py --env-file /path/to/.env --gpu GPU-YOUR-UUID -- \
  optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator
```

Generation and judge calls send task context and implementation source to your configured model endpoint. Select an endpoint appropriate for the confidentiality of your code. Endpoint compatibility does not guarantee that a model can successfully optimize your operator.

The tool uses your compute and model account. Round/call/time limits bound the workflow; they are not a dollar-denominated billing guarantee. NCU diagnostic queries also consume judge calls. Leave sufficient time for final acceptance.

## 4. Run

Start the loop against your reviewed task, with a new output directory:

```bash
python -m kai_pichu optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator
```

The loop freezes the task and baseline into the run directory, runs the search preflight on the GPU, and makes its first model call only after the baseline passes. It prints a status line at the end; `summary.json` holds the result.

An interrupted run resumes with the same command plus `--resume`. Only the `budget` section (rounds, model calls, seconds) may differ from the original configuration; such changes are recorded in `state.json` and `summary.json`. Reserved calls and interrupted rounds remain charged to the budget, and completed runs cannot be resumed: start a new run to change the task or configuration.

```bash
python -m kai_pichu optimize /absolute/path/to/your-task/benchmark.yaml \
  --config optimizer.yaml --output runs/my-operator --resume
```

To inspect the plan and the frozen task bundle without model calls or GPU work, add `--dry-run` (new output directory). It writes `plan.json` and the bundle; it does not prove that the task compiles or passes validation. A dry-run directory can then be executed with `--resume`.

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
python -m kai_pichu profile --guide
python -m kai_pichu profile --dependencies
python -m kai_pichu profile --schema
```

NCU capture and the optional report-reader integration are distinct. Structured CSV evidence remains available without the reader; richer rules and source-level queries depend on installed capabilities. Querying a saved report cannot recover counters that were not captured.

Run generated code only in an environment suitable for executing it. The framework’s process management and integrity checks are not a security sandbox.
