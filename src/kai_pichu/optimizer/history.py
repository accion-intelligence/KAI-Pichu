"""Round history as tables the judge and generator can read at a glance.

The loop already keeps every round's feedback in the run state. This module
turns that state into two Markdown tables: one row per recent round with its
overall and per-case speedups, the hypothesis that produced it and the judge's
diagnosis behind it; and one row per search case with its best speedup, when it
was reached and how long it has stalled. Both are data derived from measured
results, not instructions.
"""
from __future__ import annotations

from typing import Any

RECENT_ROUNDS = 10
# A per-case gain smaller than this is measurement jitter, not an improvement worth dating.
IMPROVEMENT_TOLERANCE = 0.01


def case_speedups(metrics: dict[str, Any]) -> dict[str, float]:
    """Per-case speedup of a completed candidate from the medians of the scored metric."""
    objective = metrics.get("objective") or {}
    minimize = objective.get("direction", "minimize") == "minimize"
    speedups: dict[str, float] = {}
    for case, arms in (objective.get("per_case") or {}).items():
        baseline, candidate = arms["baseline"]["median"], arms["candidate"]["median"]
        if baseline > 0 and candidate > 0:
            speedups[case] = baseline / candidate if minimize else candidate / baseline
    return speedups


def weakest_case(metrics: dict[str, Any]) -> str | None:
    """The search case where a completed candidate gained least over the baseline.

    A candidate that is fast on one shape and slow on another needs profiling
    evidence from the slow one.
    """
    gains = case_speedups(metrics)
    return min(gains, key=gains.get) if gains else None


def case_lost_most(attempt: dict[str, Any], reference: dict[str, Any]) -> str | None:
    """The case where a completed attempt fell furthest behind the reference candidate.

    The judge needs to know why the attempt lost, and the profile that answers
    that question is the one taken where the loss was largest.
    """
    attempt_gains, reference_gains = case_speedups(attempt), case_speedups(reference)
    ratios = {case: attempt_gains[case] / reference_gains[case] for case in attempt_gains if reference_gains.get(case)}
    return min(ratios, key=ratios.get) if ratios else weakest_case(attempt)


def history_tables(history: list[dict[str, Any]]) -> str:
    """Markdown tables of the recent rounds and of every case's progress across all rounds."""
    if not history:
        return "No rounds evaluated yet."
    cases = _case_order(history)
    return "\n\n".join(part for part in (
        _rounds_table(history[-RECENT_ROUNDS:], cases), _cases_table(history, cases)) if part)


def _case_order(history: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in history:
        for case in case_speedups(row["metrics"]):
            seen.setdefault(case, None)
    return list(seen)


def _rounds_table(rows: list[dict[str, Any]], cases: list[str]) -> str:
    header = ["round", "built on", "status", "overall", *cases, "hypothesis", "judge diagnosis"]
    lines = [_row(header), _row(["---"] * len(header))]
    for row in rows:
        metrics = row["metrics"]
        gains = case_speedups(metrics)
        status = metrics.get("status", "unknown")
        regressions = metrics.get("acceptance", {}).get("case_regressions")
        if status == "completed" and regressions:
            status = "completed, regression on " + ", ".join(regressions)
        base = row.get("base_round")
        lines.append(_row([
            str(row["id"] + 1), str(base) if base else "-", status, _speedup(row.get("score")),
            *(_speedup(gains.get(case)) for case in cases),
            _cell(row.get("hypothesis")), _cell(_diagnosis_text(row.get("diagnosis"))),
        ]))
    return "Recent rounds (speedup over baseline; one row per candidate):\n\n" + "\n".join(lines)


def _cases_table(history: list[dict[str, Any]], cases: list[str]) -> str:
    if not cases:
        return ""
    header = ["case", "best speedup", "reached in round", "rounds since improvement", "latest completed"]
    lines = [_row(header), _row(["---"] * len(header))]
    last_round = history[-1]["id"] + 1
    for case in cases:
        values = [(row["id"] + 1, case_speedups(row["metrics"]).get(case)) for row in history]
        values = [(round_id, value) for round_id, value in values if value is not None]
        best = max(value for _, value in values)
        # The round that first came within the tolerance of the best: later jitter is not progress.
        best_round = next(round_id for round_id, value in values if value >= best * (1 - IMPROVEMENT_TOLERANCE))
        latest = values[-1][1]
        lines.append(_row([case, _speedup(best), str(best_round), str(last_round - best_round), _speedup(latest)]))
    return "Per-case progress over all rounds:\n\n" + "\n".join(lines)


def _diagnosis_text(diagnosis: dict[str, str] | None) -> str:
    if not diagnosis:
        return ""
    # Optimization diagnoses carry a bottleneck and a method; repair diagnoses an issue and a fix hint.
    parts = [diagnosis.get(key) for key in ("bottleneck", "optimization_method", "critical_issue", "minimal_fix_hint")]
    return " / ".join(part for part in parts if part)


def _speedup(value: float | None) -> str:
    return f"{value:.2f}x" if isinstance(value, (int, float)) else "-"


def _cell(text: str | None) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"
