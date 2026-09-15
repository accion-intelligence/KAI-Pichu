"""Read a completed or interrupted run directory. Never executes a workload.

The loop's selection rule lives in optimizer.engine._round. Reconstructing why a
candidate was not selected otherwise means reading that source and correlating it
with state.json by hand, which is what this module does once, in one place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Round:
    """One entry of state.json history, with the loop's own reasoning applied."""

    def __init__(self, row: dict[str, Any], phase: str, previous_best: float | None):
        self.id = row["id"]
        self.phase = phase
        self.hypothesis = row.get("hypothesis") or ""
        self.workspace = row.get("workspace", "")
        self.score = row.get("score")
        metrics = row.get("metrics") or {}
        self.status = metrics.get("status", "unknown")
        self.error = " ".join(part for part in (metrics.get("error_type"), metrics.get("message")) if part)
        acceptance = metrics.get("acceptance") or {}
        self.verdict = acceptance.get("verdict")
        comparison = metrics.get("comparison") or {}
        self.interval = (comparison.get("overall") or {}).get("interval")
        self.reasons = self._reasons(acceptance, previous_best)
        self.selected = not self.reasons and self.status == "completed"

    def _reasons(self, acceptance: dict[str, Any], previous_best: float | None) -> list[str]:
        """Why engine._round did not promote this candidate to best, in its order."""
        if self.status != "completed":
            return [f"not runnable ({self.status})"]
        reasons = []
        cases = acceptance.get("unconfirmed_case_constraints") or []
        if cases:
            reasons.append(f"per-case regression unconfirmed: {', '.join(map(str, cases[:4]))}")
        if acceptance.get("metric_limit_failure_count"):
            reasons.append(f"{acceptance['metric_limit_failure_count']} metric-limit failure(s)")
        if self.interval is None:
            reasons.append("no comparison interval")
        elif self.interval[0] <= 1.0:
            reasons.append(f"interval lower bound {self.interval[0]:.4f} is not above 1.0")
        floor = 1.0 if previous_best is None else previous_best
        if self.score is not None and self.score <= floor:
            reasons.append(f"speedup {self.score:.4f} does not beat {floor:.4f}")
        return reasons


def load(root: Path) -> dict[str, Any]:
    state_path = root / "state.json"
    if not state_path.is_file():
        raise ValueError(f"not a run directory (no state.json): {root}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    history = state.get("history") or []
    rounds, best = [], None
    for index, row in enumerate(history):
        # engine._round: a round repairs when the PREVIOUS candidate did not complete.
        previous = history[index - 1].get("metrics", {}).get("status") if index else None
        phase = "seed" if index == 0 else "repair" if previous != "completed" else "optimization"
        entry = Round(row, phase, best)
        if entry.selected and entry.score is not None:
            best = entry.score
        rounds.append(entry)
    return {"root": root, "state": state, "summary": summary, "rounds": rounds}


def _cell(value: Any, width: int) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def overview(run: dict[str, Any]) -> str:
    state, summary, rounds = run["state"], run["summary"], run["rounds"]
    best = state.get("best") or {}
    lines = [f"Run:     {run['root']}",
             f"Status:  {state.get('status', 'unknown')}"
             f"    accepted={summary.get('final_accepted', False)}",
             f"Rounds:  {len(rounds)} recorded"
             f"    model calls: {state.get('llm_calls', 0)}"
             f"    evaluations: {state.get('evaluations', 0)}"
             f"    elapsed: {state.get('elapsed_seconds', 0):.1f}s"]
    if state.get("message"):
        lines.append(f"Message: {state['message']}")
    if best:
        score = best.get("score")
        lines.append(f"Best:    round {best.get('id')}"
                     + (f"    speedup {score:.4f}" if isinstance(score, (int, float)) else ""))
    else:
        lines.append("Best:    none selected")
    lines += ["", f"{'Round':<6}{'Phase':<14}{'Status':<12}{'Speedup':>9}  {'Interval':<20}{'Best':<6}Why not / note"]
    for entry in rounds:
        interval = ("-" if entry.interval is None
                    else f"[{entry.interval[0]:.4f}, {entry.interval[1]:.4f}]")
        score = "-" if not isinstance(entry.score, (int, float)) else f"{entry.score:.4f}"
        note = entry.error if entry.status != "completed" else "; ".join(entry.reasons) or "selected as best"
        lines.append(f"{entry.id:<6}{entry.phase:<14}{_cell(entry.status, 11):<12}{score:>9}  "
                     f"{interval:<20}{'yes' if entry.selected else 'no':<6}{_cell(note, 46)}")
    reports = state.get("acceptance_reports") or []
    if reports:
        passed = sum(1 for row in reports if (row.get("acceptance") or {}).get("accepted"))
        lines += ["", f"Acceptance: {passed}/{len(reports)} repetition(s) passed"]
    usage = [row for row in (state.get("usage") or []) if row]
    if usage:
        totals: dict[str, int] = {}
        for row in usage:
            for key, value in row.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        if totals:
            lines.append("Usage:      " + "  ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    return "\n".join(lines)


def detail(run: dict[str, Any], number: int) -> str:
    entry = next((row for row in run["rounds"] if row.id == number), None)
    if entry is None:
        raise ValueError(f"round {number} is not in this run's history")
    root = run["root"]
    lines = [f"Round {entry.id} ({entry.phase})",
             f"Hypothesis: {entry.hypothesis or '(none recorded)'}",
             f"Status:     {entry.status}" + (f"    verdict: {entry.verdict}" if entry.verdict else ""),
             f"Selected:   {'yes' if entry.selected else 'no'}"]
    if entry.reasons:
        lines += ["Why not:"] + [f"  - {reason}" for reason in entry.reasons]
    if entry.error:
        lines += ["Error:", "  " + entry.error[:2000]]
    strategy = _strategy(root, entry.id)
    if strategy:
        lines += ["Judge strategy:"] + [f"  {key}: {value}" for key, value in strategy.items()]
    if entry.workspace:
        lines += ["", f"Candidate: {entry.workspace}"]
    metrics = next((row.get("metrics") or {} for row in run["state"]["history"] if row["id"] == number), {})
    if metrics.get("report_path"):
        lines.append(f"Report:    {metrics['report_path']}")
    return "\n".join(lines)


def _strategy(root: Path, number: int) -> dict[str, str]:
    """The judge decision that shaped a round, recovered from the request record.

    The loop writes the strategy into the generator's request context, so the
    decision is readable without re-running or re-querying any model.
    """
    keys = {"critical_issue", "why_it_matters", "minimal_fix_hint",
            "bottleneck", "optimization_method", "modification_plan"}
    for path in sorted((root / "llm").glob("call-*.request.json")):
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
            if request.get("role") != "generator":
                continue
            context = json.loads(request["messages"][1]["content"])
        except (OSError, ValueError, KeyError, IndexError):
            continue
        strategy = context.get("strategy")
        if isinstance(strategy, dict) and set(strategy) <= keys:
            if _round_of(context) == number:
                return {key: str(value) for key, value in strategy.items()}
    return {}


def _round_of(context: dict[str, Any]) -> int | None:
    """Recover a request's round from the history it carried.

    A request's history holds the rounds before it, so the highest id in it is
    the previous round. Call numbering cannot be used instead: a resumed run
    skips an interrupted round, so call order and round order can diverge.
    """
    history = context.get("history")
    if not isinstance(history, list):
        return None
    if not history:
        return 0
    latest = max((row.get("round", -1) for row in history if isinstance(row, dict)), default=-1)
    return latest + 1 if latest >= 0 else None
