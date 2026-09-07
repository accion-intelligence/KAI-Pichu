# Validation status

The implementation has been checked locally on Linux with Python 3.13.9.
These checks validate the integration and failure handling; they do not establish
GPU speedups or a model's optimization quality.

| Check | Result and scope |
| --- | --- |
| `python -m pytest -q` in KAI-light | 139 CPU-friendly tests passed (one opt-in GPU test skipped): SDK contracts, replay loop, repair, independent acceptance, source integrity, resume/model budgets, HTTP transport, process deadlines, GPU guard, capture binding, report queries/cache identity, CSV fallback, launch-specific SASS, final-message parsing and command-alias handling. On-demand coverage includes slim overview, selected fields, executable pagination after size reduction, missing-counter suggestions, invalid requests and four evidence exchanges without starving later optimization rounds. |
| NCU on-demand report queries | 26 queries passed against existing LayerNorm and BF16 GEMM reports through the existing external reader: rules, catalog pagination, field selection, actual-name suggestions, source attribution and launch-specific SASS. Overview size dropped from 14,674/16,169 to 6,260/6,347 serialized characters; the same ten metric rows dropped from 6,409/6,431 to 4,217/4,239 characters with identity, values and units preserved. GEMM retained nine source rows. A missing LayerNorm counter and absent timed warp samples remained explicit gaps. No new GPU capture or live model run was made for the on-demand change; the external reader and capture implementation remain unchanged. |
| Original KAI regression suite | 292 tests passed; the original optimizer and SDK remain available. |
| Installed wheel outside the source checkout | CLI help, exported config, packaged AI guide and a complete replay loop with two independent SDK acceptance processes passed. The synthetic cost is a test fixture, not a latency claim. |
| Live LayerNorm example | GPT-5.6 Luna optimized the retained LayerNorm-1024 task; see [the experiment report](live-smoke-20260907.md) for outcomes and scope. |
| Live OpenAI Responses API | GPT-5.6 Luna connectivity and actual generator/judge requests tested with the explicitly loaded dotenv key. Chat Completions retains local mock coverage. |
| Live NCU / measurement boundary | Real candidate/case profiles collected. During integration, a GPU regression confirmed that a 50 ms host pause affects the outer timer but not the graph event objective. The current opt-in test targets the retained LayerNorm example; that retargeting has only received CPU/dry-run checks. |

To reproduce the CPU checks and build the distribution:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

The GitHub Actions workflow declares Python 3.10 and 3.12 jobs. Those remote jobs
have not been run or published from this checkout. Keep raw reports, failed calibrations and acceptance failures alongside
successful runs when reporting results. The live report distinguishes fixed-input
acceptance from additional checks of input updates during graph replay.
