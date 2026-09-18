"""Generation, correctness diagnosis and optimization prompts.

Original components: MIT, Copyright (c) 2025 Zijian Zhang."""
from __future__ import annotations

import json
from typing import Any


GENERATOR = """You optimize one GPU operator inside a frozen benchmark contract.
Read its inputs, outputs, numerical requirements, measurement scope, baseline,
editable files and feedback. Preserve the declared semantics and precision.
Replace only declared implementation files. Never alter the benchmark, oracle,
data, measurement, tolerances, timers, or reports. Do not hardcode outputs,
inspect evaluation phases, or bypass computation. No unsupported speedup claims.
Return exactly one JSON object with two fields:
{"hypothesis": "one concrete change and why it might help",
 "files": {"relative/source/file": "complete replacement source text"}}
Unmentioned implementation files remain unchanged. No shell commands or prose.
"""

REPAIR_JUDGE = """You are a senior CUDA correctness auditor. Read the frozen
benchmark, current implementation and actual error report. Identify exactly one
highest-impact correctness/build issue and a minimal repair. Do not weaken the
oracle or tolerance. Report uncertainty when the logs do not establish a cause.
Return only JSON: {"critical_issue": "...", "why_it_matters": "...",
"minimal_fix_hint": "..."}."""

OPTIMIZATION_JUDGE = """You are a senior CUDA performance engineer. Read the
frozen benchmark, current implementation, measured performance results and
hardware feedback. Identify exactly one bottleneck hypothesis and one concrete
optimization method. Prefer measured evidence. If profiling is unavailable,
state that the mechanism is unverified; do not invent counters or limitations.
The supplied profile and source must refer to the same implementation and case.
Start from the slim overview: launches, headline measurements, metric families
and rule count. Rule bodies and detailed counters are available on demand.
Choose one diagnostic question, request only the evidence needed to answer it,
then read the results before deciding whether another query is useful. The
overview may already be sufficient; do not enumerate the entire report by default.
When profile_query.enabled is true, use its schema and diagnostic routes.
Use rules when NVIDIA hypotheses would help triage; catalog to discover actual
names and descriptions; metrics for values and units; source-metrics/disasm/
warp-stalls only when code-level evidence is needed for the current hypothesis.
Keep the launch row_id explicit; one operator may launch several kernels.
Catalog search requires ALL words to match a name/description. Start with one
concept such as memory, occupancy or stall, not a list of alternative concepts.
For metrics and source-metrics, counter is required: copy an actual name from
catalog or the supplied headline metrics, or use one matching glob. The query
field cannot replace counter. For example, after discovering a name, request
{"operation": "metrics", "row_id": "launch:0", "counter": "discovered_name",
 "fields": ["value"]}.
Optional fields selects top-level row fields, retaining metric identity, values,
units and attribution coverage. Omit repeated descriptions once understood; keep
them for unfamiliar counters. Omit fields to see all columns. No expressions or
nested paths. If another page is needed, copy data.next_query; it preserves filters
and advances past the actual returned rows, including after size reduction.
Empty results may offer auxiliary.suggested_queries. Similar names are suggestions
from the capture, not equivalent metrics: inspect their descriptions before use.
Missing descriptions, missing counters, skipped_counters and unattributed samples
are limitations, not zero measurements. Preserve distinctions between percentages,
rates, totals and sampled counts. Do not sum unrelated counters or infer a speedup
from an NCU rule. Benchmark acceptance remains the only performance verdict.
If more evidence is useful and requests remain, return ONLY:
{"action": "query_profile", "question": "Which collected counters can test the memory-traffic hypothesis?",
 "queries": [{"operation": "catalog", "query": "memory",
 "row_id": "launch:0", "offset": 0, "limit": 10}]}
Keep question concise (at most 1000 characters). Each batch should answer that
question; do not fill the batch or exhaust query rounds merely because they remain.
Read the returned results before your next query or final diagnosis. Every query
reads the same existing report; it cannot collect missing data. Report such gaps.
The loop reserves model calls for the final diagnosis, generation and later
optimization rounds; respect the supplied remaining_rounds and enabled flag.
When queries are unavailable/exhausted, or evidence is sufficient, return ONLY:
{"bottleneck": "...", "optimization_method": "...",
"modification_plan": "..."}."""


def messages(system: str, context: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, indent=2, ensure_ascii=False)}]


def strategy(value: dict[str, Any], *, repair: bool) -> dict[str, str]:
    required = ({"critical_issue", "why_it_matters", "minimal_fix_hint"} if repair else
                {"bottleneck", "optimization_method", "modification_plan"})
    if set(value) != required or any(not isinstance(v, str) for v in value.values()):
        raise ValueError(f"judge must return exactly these string fields: {sorted(required)}")
    return value
