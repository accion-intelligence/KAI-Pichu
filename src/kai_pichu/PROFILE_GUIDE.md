# NCU evidence for one operator

KAI Pichu profiles the selected search case with the same candidate shown to
the optimization judge. Compilation, input setup, reset, warmup and validation
remain outside the profiler range. One operator can launch several kernels:
retain the launch row ID when interpreting every metric. Profiling never supplies
the benchmark speedup or acceptance score.

## Enable collection

Set `profile.enabled: true`. Nsight Compute must be installed and usable by the
current user. The default capture requests `SpeedOfLight`, `LaunchStats`,
`Occupancy`, `MemoryWorkloadAnalysis`, `SchedulerStats`, `WarpStateStats`,
`ComputeWorkloadAnalysis` (per-pipe utilization: FMA, ALU, LSU, tensor, shared
memory) and `InstructionStats` (executed instruction counts and mix).
Together the last two tell an issue-bound kernel from a memory-bound one.
These configurable sections provide initial compute, memory, resource and
scheduler evidence. A nonempty `profile.metrics` list overrides sections,
preserving existing explicit-metric configurations.

Each capture saves `capture.ncu-rep`, `metrics.csv`, `workload.json`, the process
log and `profile.json`. CSV values use NCU base units. The packaged
`kai-ncu-reader` opens the report; `profile.report_reader` can select another
executable with the same contract. See [reader setup](PROFILE_DEPENDENCIES.md),
also available with `kai-pichu profile --dependencies`.

The reader supplies metric descriptions and units, NVIDIA rule findings, and
per-instruction counters, warp-stall samples and SASS/PTX where the capture
collected them (the default capture does). It needs NVIDIA's installed
`ncu_report` Python API; set `profile.ncu_report_dir` to its `extras/python`
directory if discovery fails. A packaged compatibility adapter handles absent
timed-warp APIs on older NCU and maps nonfinite values to null.

If the report reader is unavailable, CSV supplies a structured metric catalog
and values. Descriptions, rules and instruction attribution are explicitly
unavailable in this fallback. Missing descriptions are never guessed.

## Agent protocol

The optimization judge starts with one small overview per search case: a launch
inventory, up to 12 actual metric families, selected headline measurements,
metric/rule counts, a stall summary, available operations, diagnostic routes
and the query JSON Schema. Every query names the `case_id` it addresses, or
`all` to compare the cases side by side. Full rule
findings and counter details stay out of the initial prompt. The reader
may load/cache full launch details to build this overview. Use `launches` to
select other kernels from a multi-launch operator.

Gather evidence around one diagnostic question at a time. For example, use
`rules` to identify a possible occupancy issue, search `catalog` for the relevant
resource counters, read their values with `metrics`, then ask for source/SASS
evidence if it can locate the cause. This is a possible route, not a required
sequence: stop as soon as the evidence supports a concrete experiment.

When `profile_query.enabled` is true, the judge may return:

```json
{
  "action": "query_profile",
  "question": "Which collected counters can test the memory-traffic hypothesis?",
  "queries": [
    {"operation": "catalog", "row_id": "launch:0", "query": "memory", "limit": 10}
  ]
}
```

KAI executes these requests against the stage's report, saves the question and
results, appends them to the context, and calls the judge again. `question` is
recommended and accepts nonblank text up to 1000 characters; it remains
optional for existing integrations. Every model response is charged to
`budget.llm_calls`; query time counts against the wall budget.
`profile.query_rounds` (default 4) limits exchanges per diagnosis and
`profile.queries_per_round` (default 3) limits each batch. The loop reserves two
calls for the current final judgment/generation and two for each later optimization
round before allowing optional evidence requests. `profile_query.remaining_rounds`
reflects both this call budget and the per-diagnosis cap. The judge can finish
immediately when the initial evidence is sufficient; neither limit is a quota.

After discovering names, retrieve measurements:

```json
{
  "action": "query_profile",
  "question": "How much DRAM traffic did the selected launch generate?",
  "queries": [
    {"operation": "metrics", "row_id": "launch:0", "counter": "dram__bytes*", "fields": ["value"]}
  ]
}
```

The glob is an example; confirm that the actual capture carries those counters.
Catalog search matches words in names and descriptions. Multiple words must
all match; start with one concept instead of combining alternative concepts.
`counter` accepts one exact name or glob. Empty searches can return
`auxiliary.suggested_queries`: separate-word searches, or real captured names
similar to a missing exact counter. Name similarity does not establish semantic
equivalence; read the descriptions. Missing counters remain missing, with no
automatic substitution or invented values.

`fields` optionally selects top-level row columns. For example, `["value"]`
omits repeated descriptions after discovery while retaining metric name, value,
unit, status and row identity. Source `counters` and auxiliary attribution totals,
skipped counters and warnings remain present, subject to the marked size limits
below. Unknown fields produce an error listing available columns; a successful
projection also lists them in `auxiliary.available_fields`. Expressions and nested
paths are unsupported. Omit `fields` to request all columns again.

When a result has another page, copy `data.next_query` into the next batch. It
retains the filters, row ID and field selection and advances past the actual
returned rows even when the size cap shortened a page. `next_offset` remains
available for existing clients; `next_query` is null at the end. Read more pages
only if they help answer the diagnostic question.

Once evidence is sufficient or queries are disabled, return the existing final
judgment with exactly `bottleneck`, `optimization_method`, and `modification_plan`
string fields. Requesting queries after the limit produces `model_error`; the
loop does not manufacture a diagnosis or exceed the call budget.

## Query SDK and CLI

```python
from pathlib import Path
from kai_pichu.profiling import ProfileReport

report = ProfileReport(Path("capture.ncu-rep"), csv_path=Path("metrics.csv"))
catalog = report.query({"operation": "catalog", "query": "register"})
values = report.query({
    "operation": "metrics", "row_id": "launch:0",
    "counter": "launch__registers_per_thread", "fields": ["value"],
})
if catalog["status"] == "available" and catalog["data"]["next_query"]:
    next_page = report.query(catalog["data"]["next_query"])
```

```bash
kai-pichu profile --guide
kai-pichu profile --schema
kai-pichu profile capture.ncu-rep
kai-pichu profile capture.ncu-rep --request '{"operation":"catalog","query":"stall"}'
kai-pichu profile capture.ncu-rep --request '{"operation":"rules","row_id":"launch:0"}'
```

These commands read existing captures. Options include `--csv`, `--report-reader`,
`--ncu`, `--ncu-report-dir`, `--timeout`, and `--output`.

| Operation | Use |
| --- | --- |
| `launches` | Discover launch IDs and kernel identity. |
| `catalog` | Search/page real metric names, descriptions, types and units. |
| `metrics` | Retrieve counters, keeping zero and missing values distinct. |
| `rules` | Read NVIDIA findings, focus metrics and estimated speedups as hypotheses. |
| `source-metrics` | Attribute a discovered per-PC counter by `line`, `sass`, or `file`; optional file/line filters. |
| `warp-stalls` | Read timed warp-sample counts by `line`, `sass`, or `reason`. |
| `disasm` | Page `view: sass`, `ptx`, or `source` to inspect instructions and correlation. |

Source attribution needs suitable counters and compiler line info. Add
`SourceCounters` to `profile.sections` when that evidence is needed, and compile
with line information. Timed warp samples depend on NCU and capture settings.
SASS/PTX also requires CUDA disassembly tools and binary data in the report.
The SASS view selects the launch's function before pagination; a missing symbol
is an explicit error. PTX and source indexes retain cubin scope, identified by
`auxiliary.evidence_scope`; they may include other functions in the same binary.
Empty results do not prove a stall, conflict, or instruction is absent.

## Evidence contract and limits

Results carry `schema_version: 1`, `status`, `identity`, `backend`,
`data: {count, total_matched, rows, offset, next_offset, next_query, auxiliary}`, warnings
and an artifact directory. Failed queries carry an explicit error. Optimizer
identity includes the implementation fingerprint and worker case metadata;
standalone readers retain the report path and input hashes.

Each query defaults to 20 rows, with a configurable `profile.max_rows` cap
(default 30) and `profile.max_result_chars` serialized-result limit (default
16000). Field selection happens before size reduction so irrelevant columns need
not consume that budget. These are per-result limits: requested results accumulate
within a diagnosis, and this feature does not impose a global model token budget.
`max_context_chars` limits the source/benchmark context, not accumulated NCU evidence.

Preserve `auxiliary.skipped_counters`, attribution totals and warnings. Nested
omissions are marked; full reader responses remain in the query artifacts.
An oversized single result returns `result_too_large` and asks for a narrower
query. Missing values are null; measured zero remains zero. Percentages, rates,
totals and sampling counts have different meanings. Avoid summing unrelated
counters or treating sampling reasons as exact elapsed-time fractions.

Successful report queries are cached by report/CSV content hashes, executable
identity, reader settings and query arguments. If an open reader's input changes,
its queries fail explicitly. Reopen a new reader to use the changed report.
Queries never rerun the GPU workload. Requests/results are saved under each
optimizer capture; resume retains reserved model charges.

This catalog describes **already collected evidence**. Device-wide collection
capability discovery and model-triggered recapture are not implemented here.
Use the installed NCU section/metric inventory to configure a new capture when
evidence is missing. NVIDIA rules provide hypotheses; correctness, measured
A/B and independent acceptance decide whether an optimization works.
