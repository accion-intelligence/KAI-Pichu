"""Round history as tables the judge and coder can read at a glance.

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
# A case whose 95% interval is wider than this fraction of its estimate is marked noisy.
NOISE_TOLERANCE = 0.15


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


def case_intervals(metrics: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Per-case 95% speedup intervals from the acceptance block, when the report carried them."""
    intervals = {}
    for case, entry in ((metrics.get("acceptance") or {}).get("case_speedups") or {}).items():
        bounds = entry.get("interval") if isinstance(entry, dict) else None
        if bounds and len(bounds) == 2:
            intervals[case] = (float(bounds[0]), float(bounds[1]))
    return intervals


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


def _cell_speedup(value: float | None, interval: tuple[float, float] | None) -> str:
    """`2.45x [2.31, 2.60]`, with `~` when the interval is too wide to tell neighbours apart."""
    if value is None:
        return "-"
    if interval is None:
        return f"{value:.2f}x"
    low, high = interval
    noisy = " ~" if value and (high - low) / value > NOISE_TOLERANCE else ""
    return f"{value:.2f}x [{low:.2f}, {high:.2f}]{noisy}"


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
        intervals = case_intervals(metrics)
        status = metrics.get("status", "unknown")
        regressions = metrics.get("acceptance", {}).get("case_regressions")
        if status == "completed" and regressions:
            status = "completed, regression on " + ", ".join(regressions)
        base = row.get("base_round")
        lines.append(_row([
            str(row["id"] + 1), str(base) if base else "-", status, _speedup(row.get("score")),
            *(_cell_speedup(gains.get(case), intervals.get(case)) for case in cases),
            _cell(row.get("hypothesis")), _cell(_diagnosis_text(row.get("diagnosis"))),
        ]))
    return ("Recent rounds (speedup over baseline with its 95% interval; `~` marks an interval wider than "
            f"{int(NOISE_TOLERANCE * 100)}% of the estimate, where neighbouring values are not distinguishable):\n\n"
            + "\n".join(lines))


def _cases_table(history: list[dict[str, Any]], cases: list[str]) -> str:
    if not cases:
        return ""
    header = ["case", "best speedup", "reached in round", "rounds since improvement", "latest completed"]
    lines = [_row(header), _row(["---"] * len(header))]
    last_round = history[-1]["id"] + 1
    for case in cases:
        values = [(row["id"] + 1, case_speedups(row["metrics"]).get(case), case_intervals(row["metrics"]).get(case))
                  for row in history]
        values = [(round_id, value, interval) for round_id, value, interval in values if value is not None]
        best, best_round, best_high = None, None, None
        for round_id, value, interval in values:
            # An improvement must clear the previous best's interval (or the tolerance when
            # no interval is known); later jitter inside the interval is not progress.
            bar = best_high if best_high is not None else (best * (1 + IMPROVEMENT_TOLERANCE) if best else None)
            if best is None or value > bar:
                best, best_round, best_high = value, round_id, (interval[1] if interval else None)
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
