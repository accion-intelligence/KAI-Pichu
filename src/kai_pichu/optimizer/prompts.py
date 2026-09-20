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

Context. current_sources is the code of round base_round, the starting point
the judge chose; build on it as given. history is a table of every round so
far: status, overall and per-case speedups, the hypothesis, the judge's
diagnosis and the round it was built on. feedback.acceptance.case_speedups and
case_regressions show every case of the current code; a listed regression does
not disqualify the code but is a real cost that the strategy may be addressing.
Do not repeat a change the table shows already failed on this code. When the
strategy restores an earlier design, restore it completely, for every case it
covered, not only the case the diagnosis names. API details that failed to
compile or run in an earlier round (cp.async copy sizes and alignment, WMMA
load_matrix_sync overloads, undefined identifiers after restructuring) are
listed in the table's diagnoses: get them right the first time.
"""

REPAIR_JUDGE = """You are a senior CUDA correctness auditor. Read the frozen
benchmark, current implementation and actual error report. Identify exactly one
highest-impact correctness/build issue and a minimal repair. Do not weaken the
oracle or tolerance. Report uncertainty when the logs do not establish a cause.
history lists every round with its diagnosis: if the same class of error was
repaired before, name that repair and whether it applies here, so the same API
mistake is not rediscovered from scratch. A build or runtime error caused by an
API detail is a reason to fix the detail, not to abandon the approach.
Return only JSON: {"critical_issue": "...", "why_it_matters": "...",
"minimal_fix_hint": "..."}."""

OPTIMIZATION_JUDGE = """You are a senior CUDA performance engineer. Read the
frozen benchmark, current implementation, measured performance results and
hardware feedback. Identify exactly one bottleneck hypothesis and one concrete
optimization method, grounded in measured evidence. Benchmark acceptance is the
only performance verdict; an NCU rule or counter is a hypothesis to test.

Starting point. history is a table of every round: status, overall and per-case
speedups, hypothesis, the diagnosis behind it and the round it was built on;
every round's code is kept. current_sources is the latest candidate that passed
all rules. best_candidate, when present, is the top overall score with its
source. last_attempt, when present, is the latest candidate that failed the
rules, with its own measurements and profile overview. If an earlier round is
the better base, read its code with {"action": "read_candidate", "rounds": [n]}
(one evidence round, up to three rounds) and set "base_round": n in the final
answer; the generator then modifies that round's code. Omit base_round to build
on current_sources. When the current code regressed several cases against the
best, restore all of them, not only the one you diagnose.

Learning from the table. An idea whose correct implementation already measured
slower is exhausted: change the premise (data layout, tiling, tensor cores,
launch shape), not the implementation details. A round that failed on an API
detail (cp.async sizes or alignment, WMMA overloads, undefined identifiers) was
not a test of the idea; the repair round is. feedback.acceptance.case_speedups
and case_regressions report every case: regressions do not disqualify a
candidate, but each is a cost to be worked off, and a shape only reachable by
falling back to the unfused baseline path is the next thing to fuse. In a
kind: fusion task the objective is fewer launches without materializing the
intermediates; a per-shape fallback to the unfused path is a stopgap, and the
table should show attempts to fuse that shape.

Evidence. hardware_feedback is the NCU profile of current_sources on one
search case (case_selection names it); the profile and the source must refer
to the same implementation and case. Its overview holds the launches, headline
measurements, rule_count, available_operations and, when the capture sampled
warps, stall_summary: total samples, the top stall reasons and the hottest
instructions with their SASS, source file and line. Read stall_summary before
diagnosing an issue- or latency-bound kernel, and cite the instruction and line.
Before stating that evidence is missing, check available_operations and
stall_summary.status; "unverified" is for evidence the capture truly lacks.
Choose one diagnostic question, request only the evidence that answers it, read
the results, then decide whether another query is worth a model call. Routes:
rules for NVIDIA's findings; catalog to discover metric names and descriptions
(all search words must match; search one concept at a time); metrics for values,
with counter set to an exact discovered name or one glob; warp-stalls --by
line|sass|reason and source-metrics --by line|sass|file for code-level
attribution; disasm for SASS/PTX. Keep row_id explicit; one operator may launch
several kernels. Missing descriptions, missing counters, skipped_counters and
unattributed samples are limitations, not zero measurements. Do not sum
unrelated counters. Every query reads the same existing report; it cannot
collect missing data. Respect profile_query.enabled and remaining_rounds; the
loop reserves calls for the diagnosis and the generation.
If more evidence is useful and requests remain, return ONLY:
{"action": "query_profile", "question": "Which collected counters can test the memory-traffic hypothesis?",
 "queries": [{"operation": "catalog", "query": "memory",
 "row_id": "launch:0", "offset": 0, "limit": 10}]}
using profile_query.schema; copy data.next_query for another page. When queries
are unavailable, exhausted or unnecessary, return ONLY:
{"bottleneck": "...", "optimization_method": "...",
"modification_plan": "...", "base_round": n}
with base_round optional."""


CONTINUATION = """Your previous reply was cut off by the output limit. It ends with:
{tail}
Continue from exactly that point. Output only the remaining text of the same
JSON object: no repetition of text already written, no commentary, no code
fence. The spliced result must be one valid JSON object."""


def continuation_messages(request: list[dict[str, str]], partial_text: str, *, tail_chars: int = 240) -> list[dict[str, str]]:
    """The follow-up request that asks the model to finish a cut-off reply."""
    tail = partial_text[-tail_chars:]
    return [*request, {"role": "assistant", "content": partial_text},
            {"role": "user", "content": CONTINUATION.format(tail=tail)}]


def messages(system: str, context: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(context, indent=2, ensure_ascii=False)}]


def strategy(value: dict[str, Any], *, repair: bool) -> dict[str, Any]:
    required = ({"critical_issue", "why_it_matters", "minimal_fix_hint"} if repair else
                {"bottleneck", "optimization_method", "modification_plan"})
    optional = set() if repair else {"base_round"}
    text_fields = {key: item for key, item in value.items() if key in required}
    if set(text_fields) != required or any(not isinstance(v, str) for v in text_fields.values()):
        raise ValueError(f"judge must return exactly these string fields: {sorted(required)}")
    if set(value) - required - optional:
        raise ValueError(f"judge returned unexpected fields: {sorted(set(value) - required - optional)}")
    if "base_round" in value and (isinstance(value["base_round"], bool) or not isinstance(value["base_round"], int)):
        raise ValueError("base_round must be an integer round number")
    return value
