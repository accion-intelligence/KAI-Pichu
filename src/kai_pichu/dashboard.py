"""Read-only live view of optimization runs.

The dashboard never writes into a run directory. Every state it shows is
derived from files the loop already records: ``state.json`` and ``config.json``,
``llm/call-*.{request,response,error}.json``, ``reports/*.{log,process.json,json}``
and ``profiles/round-*/``. A request without a response is a model call in
flight; an evaluation log without its process record is a measurement in
flight; a profile directory without ``profile.json`` is a capture in flight.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from .io import parse_object
from .profile_csv import read_csv

TERMINAL = ("accepted", "not_accepted", "no_improvement", "planned", "error", "invalidated",
            "interrupted", "budget_exhausted", "model_error", "benchmark_error", "resource_busy", "resource_error")
REPORT = re.compile(r"^(\d{4})-(preflight|round-(\d{4})|acceptance-(\d+))\.log$")
CALL = re.compile(r"^call-(\d{4})\.request\.json$")
STEP_LIMIT = 400

_cache: dict[Path, tuple[int, int, Any]] = {}


def _role(name: Any) -> str:
    """Runs recorded before the coder role was renamed call it generator."""
    return "coder" if name in (None, "generator") else str(name)


def _role_calls(calls: dict[str, int] | None) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, count in (calls or {}).items():
        result[_role(name)] = result.get(_role(name), 0) + count
    return result


def _load(path: Path) -> Any:
    """Parse a JSON or YAML file once per version; None while it is missing or still being written."""
    try:
        stat = path.stat()
    except OSError:
        return None
    hit = _cache.get(path)
    if hit and hit[:2] == (stat.st_mtime_ns, stat.st_size):
        return hit[2]
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError):
        return None
    _cache[path] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def _judge_note(body: dict[str, Any]) -> dict[str, Any] | None:
    """The judge's conclusion in one reply: a performance bottleneck, or the critical issue behind a failure.

    Profile queries and candidate reads are intermediate steps (the query record
    carries the question); a cut-off reply that only parses once spliced has none.
    """
    try:
        reply = parse_object(str(body.get("text") or ""))
    except ValueError:
        return None
    fields = {"bottleneck": ("bottleneck", "optimization_method", "modification_plan"),
              "critical issue": ("critical_issue", "why_it_matters", "minimal_fix_hint")}
    for label, keys in fields.items():
        if isinstance(reply.get(keys[0]), str) and reply[keys[0]].strip():
            base = reply.get("base_round")
            return {"label": label, "diagnosis": {key: reply[key] for key in keys if isinstance(reply.get(key), str)},
                    "base_round": base if isinstance(base, int) and not isinstance(base, bool) else None}
    return None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def is_live(root: Path) -> bool | None:
    """Whether a process holds the run's flock; None when /proc/locks cannot answer."""
    try:
        stat = (root / ".run.lock").stat()
        locks = Path("/proc/locks").read_text()
    except OSError:
        return None
    key = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}"
    return any(line.split()[1:2] == ["FLOCK"] and key in line.split() for line in locks.splitlines())


def _clip(text: Any, limit: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _tokens(usage: dict[str, Any]) -> tuple[int, int]:
    usage = usage or {}
    return (int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            int(usage.get("output_tokens") or usage.get("completion_tokens") or 0))


def _eligible(metrics: dict[str, Any]) -> bool:
    """The loop's own rule: completed, no metric-limit failure, overall interval above 1."""
    acceptance = metrics.get("acceptance") or {}
    interval = (metrics.get("comparison") or {}).get("overall", {}).get("interval") or [0, 0]
    return metrics.get("status") == "completed" and not acceptance.get("metric_limit_failure_count") and interval[0] > 1.0


def _phase(history: list[dict[str, Any]], index: int) -> str:
    if index == 0:
        return "seed"
    previous = next((row for row in history if row["id"] == index - 1), None)
    if previous is None:
        return "optimization"
    return "optimization" if previous["metrics"].get("status") == "completed" else "repair"


def _cases(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    speedups = (metrics.get("acceptance") or {}).get("case_speedups") or {}
    per_case = (metrics.get("objective") or {}).get("per_case") or {}
    return [{"case": name, "speedup": value.get("speedup"), "interval": value.get("interval"),
             "baseline": (per_case.get(name) or {}).get("baseline", {}).get("median"),
             "candidate": (per_case.get(name) or {}).get("candidate", {}).get("median")}
            for name, value in sorted(speedups.items())]


def _steps(root: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every recorded model call, evaluation, capture and profile query as a timed step."""
    steps: list[dict[str, Any]] = []
    for request_path in sorted((root / "llm").glob("call-*.request.json")):
        match = CALL.match(request_path.name)
        request = _load(request_path) if match else None
        if not match or not isinstance(request, dict):
            continue
        stem = request_path.name[:-len(".request.json")]
        step = {"kind": "model", "actor": _role(request.get("role")), "call": int(match.group(1)),
                "continuation": bool(request.get("continuation")), "start": _mtime(request_path), "end": None,
                "status": "running", "detail": ""}
        response, error = root / "llm" / f"{stem}.response.json", root / "llm" / f"{stem}.error.json"
        if response.exists():
            body = _load(response) or {}
            tokens_in, tokens_out = _tokens(body.get("usage", {}))
            step.update(end=_mtime(response), status="done", tokens_in=tokens_in, tokens_out=tokens_out,
                        finish=body.get("finish_reason"))
            if step["actor"] == "judge":
                step["note"] = _judge_note(body)
        elif error.exists():
            body = _load(error) or {}
            step.update(end=_mtime(error), status="failed",
                        detail=_clip(f"{body.get('error_type', 'error')}: {body.get('message', '')}", 600))
        steps.append(step)
    for log in sorted((root / "reports").glob("*.log")):
        match = REPORT.match(log.name)
        if not match:
            continue
        label = match.group(2)
        stage = "preflight" if label == "preflight" else "acceptance" if match.group(4) is not None else "evaluate"
        step = {"kind": "evaluate", "actor": "sdk", "stage": stage, "label": label,
                "round": int(match.group(3)) if match.group(3) else None,
                "repeat": int(match.group(4)) if match.group(4) else None,
                "start": None, "end": None, "status": "running", "detail": ""}
        process_path = log.with_suffix(".process.json")
        process = _load(process_path)
        if isinstance(process, dict):
            end = _mtime(process_path)
            step.update(end=end, start=end - float(process.get("elapsed_seconds") or 0.0))
            report = _load(log.with_suffix(".json"))
            if process.get("status") != "completed" or not isinstance(report, dict):
                step.update(status="failed", detail=_clip(process.get("message") or process.get("status"), 600))
            else:
                speedup = (report.get("comparison") or {}).get("overall", {}).get("speedup")
                verdict = (report.get("acceptance") or {}).get("verdict")
                ok = report.get("status") in ("completed", "ready")
                step.update(status="done" if ok else "failed", report_status=report.get("status"),
                            speedup=speedup, verdict=verdict,
                            detail=_clip(report.get("message") or report.get("error_type") or "", 600) if not ok else "")
        steps.append(step)
    for round_dir in sorted((root / "profiles").glob("round-*")):
        name = round_dir.name
        match = re.match(r"^round-(\d{4})(-attempt)?$", name)
        if not match or not round_dir.is_dir():
            continue
        index, attempt = int(match.group(1)), bool(match.group(2))
        capture_dirs = [round_dir] if attempt or (round_dir / "process.log").exists() else \
            [child for child in sorted(round_dir.iterdir()) if child.is_dir()]
        for capture in capture_dirs:
            steps.append(_capture_step(capture, index, attempt=attempt, case=None if capture == round_dir else capture.name))
        for query_path in sorted(round_dir.glob("queries-*.json")):
            record = _load(query_path) or {}
            operations = [str(r.get("operation")) for r in record.get("requests") or [] if isinstance(r, dict)]
            moment = _mtime(query_path)
            steps.append({"kind": "query", "actor": "ncu", "round": index, "start": moment, "end": moment,
                          "status": "done", "queries": len(operations),
                          "detail": _clip(record.get("question") or ", ".join(operations), 600),
                          "operations": operations})
    for capture in sorted((root / "profiles" / "baseline").glob("*")):
        if capture.is_dir():
            steps.append(_capture_step(capture, None, case=capture.name, baseline=True))
    # In-flight steps are dated by the previous completed step: the loop is sequential.
    steps.sort(key=lambda s: (s["start"] if s["start"] is not None else s.get("end") or float("inf")))
    finished = sorted(s["end"] for s in steps if s["end"] is not None)
    for step in steps:
        if step["end"] is None and step["start"] is None:
            step["start"] = finished[-1] if finished else time.time()
    # The loop is sequential, so a step ends before the next begins. End times are
    # recorded file times; a start inferred from a process's elapsed time is not
    # exact enough to order an evaluation after the model call that preceded it.
    # An unfinished step that something else followed was abandoned (an interrupted
    # run that was resumed): only the newest unfinished step can still be running.
    newest = max((max(s["start"], s["end"] or s["start"]) for s in steps), default=0.0)
    for step in steps:
        if step["end"] is None and step["start"] < newest and any(
                o is not step and max(o["start"], o["end"] or o["start"]) > step["start"] for o in steps):
            step["status"] = "stopped"
    steps.sort(key=lambda s: (s["end"] if s["end"] is not None else s["start"] if s["status"] == "stopped" else float("inf"),
                              s["start"]))
    _assign_rounds(steps, state)
    return steps


def _capture_step(capture: Path, index: int | None, *, case: str | None, attempt: bool = False,
                  baseline: bool = False) -> dict[str, Any]:
    result_path = capture / "profile.json"
    result = _load(result_path)
    step = {"kind": "profile", "actor": "ncu", "round": index, "attempt": attempt, "baseline": baseline,
            "case": case, "start": None, "end": None, "status": "running", "detail": ""}
    if isinstance(result, dict):
        end = _mtime(result_path)
        elapsed = float((result.get("process") or {}).get("elapsed_seconds") or 0.0)
        step.update(end=end, start=end - elapsed, status="done" if result.get("status") == "profiled" else "failed",
                    case=case or (result.get("case_selection") or {}).get("case_id"))
    else:
        step["start"] = _mtime(capture)
    return step


def _assign_rounds(steps: list[dict[str, Any]], state: dict[str, Any]) -> None:
    """Model calls carry no round; the sequential loop order supplies it.

    Captures and evaluations name their round, and a finished evaluation closes
    it. Within a round every judge call precedes the one fresh coder call,
    so a judge or fresh coder call after it starts the next round: the
    previous reply never reached evaluation (an invalid reply is still a round).
    """
    current, generated = 0, False
    for step in steps:
        if step.get("baseline"):
            continue
        if step["kind"] in ("profile", "query"):
            if step["round"] != current:
                current, generated = step["round"], False
        elif step["kind"] == "evaluate":
            if step["stage"] == "evaluate" and step["status"] != "running":
                current, generated = step["round"] + 1, False
        else:
            fresh = step["actor"] == "judge" or not step["continuation"]
            if generated and fresh:
                current, generated = current + 1, False
            if step["actor"] == "coder" and not step["continuation"]:
                generated = True
            step["round"] = current


def _describe(step: dict[str, Any], phases: dict[int, str]) -> str:
    kind = step["kind"]
    if kind == "model":
        phase = phases.get(step.get("round"), "optimization")
        if step["actor"] == "judge":
            text = "diagnose the failure" if phase == "repair" else "diagnose the bottleneck"
        else:
            text = "write the first candidate" if phase == "seed" else "write the repair" if phase == "repair" else "write the next candidate"
        if step["continuation"]:
            text = "continue a cut-off reply"
        elif step["actor"] == "judge" and step["status"] == "done":
            note = step.get("note")
            text = ("asked for evidence" if note is None else "decided the repair" if note["label"] == "critical issue"
                    else "decided the bottleneck" + (f" · builds on r{note['base_round']}" if note["base_round"] else ""))
        if step["status"] == "done" and step.get("tokens_out"):
            text += f" · {step['tokens_out']:,} tok out"
        if step["status"] == "failed":
            text += f" · {step['detail']}"
        return text
    if kind == "evaluate":
        if step["status"] == "running":
            if step["stage"] == "preflight":
                return "check and time the baseline"
            if step["stage"] == "acceptance":
                return f"held-out rerun {step['repeat'] + 1}"
            return "check correctness, paired A/B timing"
        name = {"preflight": "baseline", "acceptance": f"acceptance {(step.get('repeat') or 0) + 1}"}.get(
            step["stage"], f"round {(step.get('round') or 0) + 1}")
        if step["status"] == "failed":
            return f"{name} {step.get('report_status') or 'failed'} · {step['detail']}".strip(" ·")
        if step["stage"] == "preflight":
            return "baseline checked and timed"
        if step.get("speedup") is None:
            return f"{name} {step.get('report_status')}"
        verdict = f" · {step.get('verdict')}" if step["stage"] == "acceptance" and step.get("verdict") else ""
        return f"{name} {step['speedup']:.3f}× vs baseline{verdict}"
    if kind == "profile":
        where = step.get("case") or "first case"
        prefix = "profile baseline" if step.get("baseline") else "profile last attempt" if step.get("attempt") else "profile"
        return f"{prefix} {where}" + ("" if step["status"] != "failed" else " · capture failed")
    return f"query ×{step['queries']}: {step['detail']}"


SOL_METRICS = {"compute": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
               "memory": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
               "dram": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"}
_sol_cache: dict[Path, tuple[int, dict[str, Any] | None]] = {}


def speed_of_light(csv_path: Path) -> dict[str, Any] | None:
    """NCU's Speed-of-Light utilization for the profiled operator, as % of the GPU's peaks.

    Compute is SM throughput; memory is the busiest memory unit (L1, L2 or
    DRAM), as NCU's own SOL table reports them. An operator with several
    launches is averaged by each launch's GPU time.
    """
    try:
        stamp = csv_path.stat().st_mtime_ns
    except OSError:
        return None
    hit = _sol_cache.get(csv_path)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        launches = read_csv(csv_path)
    except (OSError, ValueError):
        launches = []
    totals = dict.fromkeys(SOL_METRICS, 0.0)
    seconds = 0.0
    for launch in launches:
        values = {m["name"]: m["value"] for m in launch["metrics"] if isinstance(m["value"], (int, float))}
        weight = values.get("gpu__time_duration.sum")
        if not weight or any(name not in values for name in SOL_METRICS.values()):
            continue
        seconds += weight
        for key, name in SOL_METRICS.items():
            totals[key] += values[name] * weight
    result = None if seconds <= 0 else {**{key: total / seconds for key, total in totals.items()}, "launches": len(launches)}
    _sol_cache[csv_path] = (stamp, result)
    return result


def _headroom(root: Path, state: dict[str, Any], profile_enabled: bool) -> dict[str, Any]:
    """Speed-of-Light utilization per search case: the baseline next to the current best candidate.

    The baseline is captured once after preflight; the best candidate when a
    later round profiles it as its anchor, matched by the implementation
    fingerprint every capture records. Utilization is not speed (redundant work
    keeps units busy too), so the page shows it beside each case's speedup.
    """
    profiles = root / "profiles"
    best_row = state.get("best") or {}
    baseline: dict[str, dict[str, Any]] = {}
    best: dict[str, dict[str, Any]] = {}
    captured = False
    for path in sorted(profiles.glob("baseline/*/profile.json")):
        captured = True
        record = _load(path)
        if isinstance(record, dict) and record.get("status") == "profiled":
            sol = speed_of_light(path.parent / "metrics.csv")
            if sol:
                baseline[(record.get("case_selection") or {}).get("case_id") or path.parent.name] = sol
    for path in sorted(profiles.glob("round-*/*/profile.json")) + sorted(profiles.glob("round-*/profile.json")):
        captured = True
        record = _load(path)
        if (best_row and isinstance(record, dict) and record.get("status") == "profiled"
                and record.get("implementation_fingerprint") == best_row.get("fingerprint")):
            case = (record.get("case_selection") or {}).get("case_id")
            sol = speed_of_light(path.parent / "metrics.csv")
            if case and sol:
                best[case] = sol
    speedups = {row["case"]: row["speedup"] for row in _cases(best_row.get("metrics") or {})}
    cases = [{"case": case, "baseline": baseline.get(case), "best": best.get(case), "speedup": speedups.get(case)}
             for case in sorted(set(baseline) | set(best))]
    if not profile_enabled and not captured:
        note = "profiling is off for this run"
    elif not captured:
        note = "waiting for the first capture"
    elif not baseline:
        note = "this run did not capture the baseline; runs started now do"
    elif best_row and not best:
        note = f"best round {best_row['id'] + 1} is captured when the next round profiles it"
    else:
        note = None
    return {"cases": cases, "note": note, "best_round": best_row["id"] + 1 if best_row else None}


def _progress(steps: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each round's result at the moment it was measured, with the best eligible score up to then.

    Time is wall-clock seconds since the run's first recorded step, so an
    interruption shows as a flat stretch; tokens are those spent by then.
    """
    if not steps:
        return []
    origin = min(step["start"] for step in steps)
    ends: dict[int, float] = {}
    for step in steps:
        if step["kind"] in ("model", "evaluate") and step.get("round") is not None and step["end"] is not None \
                and step.get("stage", "evaluate") == "evaluate":
            ends[step["round"]] = max(ends.get(step["round"], 0.0), step["end"])
    calls = sorted((step["end"], step.get("tokens_in", 0) + step.get("tokens_out", 0))
                   for step in steps if step["kind"] == "model" and step["end"] is not None)
    series, best = [], None
    for row in history:
        at = ends.get(row["id"])
        if at is None:
            continue
        metrics = row.get("metrics") or {}
        eligible = _eligible(metrics)
        if eligible and row.get("score") is not None:
            best = max(best or 0.0, row["score"])
        series.append({"round": row["id"] + 1, "t": at - origin, "tokens": sum(n for end, n in calls if end <= at),
                       "score": row.get("score"), "status": metrics.get("status"), "eligible": eligible, "best": best})
    return series


def _activity(steps: list[dict[str, Any]], live: bool | None, status: str) -> dict[str, Any] | None:
    running = [s for s in steps if s["status"] == "running"]
    if running and live is not False:
        step = running[-1]
        stage = {"model": step["actor"], "evaluate": step.get("stage", "evaluate"),
                 "profile": "baseline profile" if step.get("baseline") else "profile"}[step["kind"]]
        index = step.get("round")
        return {"stage": stage, "actor": step["actor"], "round": None if index is None else index + 1, "since": step["start"]}
    if live or (live is None and status == "running"):
        return {"stage": "harness", "actor": "harness", "round": None,
                "since": max((s["end"] for s in steps if s["end"]), default=None)}
    return None


def snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    state = _load(root / "state.json") or {}
    config = _load(root / "config.json") or {}
    manifest = _load(root / "bundle" / "benchmark.yaml") or {}
    summary = _load(root / "summary.json")
    live = is_live(root)
    status = state.get("status", "created")
    now = time.time()
    history = state.get("history") or []
    budget = config.get("budget") or {}
    next_round = state.get("next_round", 0)
    phases = {row["id"]: _phase(history, row["id"]) for row in history}
    if next_round and (next_round - 1) not in phases:
        phases[next_round - 1] = _phase(history, next_round - 1)
    steps = _steps(root, state)
    if live is False:
        for step in steps:  # no process holds the run: nothing recorded as in flight is still running
            if step["status"] == "running":
                step["status"] = "stopped"
    for step in steps:
        step["text"] = _describe(step, phases)

    best_id = (state.get("best") or {}).get("id")
    eligible_id = (state.get("latest_eligible") or {}).get("id")
    queries: dict[int, int] = {}
    calls: dict[int, dict[str, int]] = {}
    for step in steps:
        if step["kind"] == "query":
            queries[step["round"]] = queries.get(step["round"], 0) + step["queries"]
        if step["kind"] == "model" and step.get("round") is not None:
            calls.setdefault(step["round"], {"coder": 0, "judge": 0})[step["actor"]] += 1
    rounds = []
    for row in history:
        metrics = row.get("metrics") or {}
        comparison = (metrics.get("comparison") or {}).get("overall") or {}
        rounds.append({
            "round": row["id"] + 1, "phase": phases[row["id"]], "status": metrics.get("status"),
            "score": row.get("score"), "interval": comparison.get("interval"),
            "eligible": _eligible(metrics), "best": row["id"] == best_id, "anchor": row["id"] == eligible_id,
            "base_round": row.get("base_round"), "hypothesis": row.get("hypothesis"),
            "diagnosis": row.get("diagnosis"), "cases": _cases(metrics),
            "error": _clip(f"{metrics.get('error_type', '')}: {metrics.get('message', '')}", 600)
            if metrics.get("status") != "completed" else None,
            "regressions": (metrics.get("acceptance") or {}).get("case_regressions") or [],
            "queries": queries.get(row["id"], 0), "calls": calls.get(row["id"], {}),
        })

    usage = state.get("usage") or []
    tokens_in = sum(_tokens(u)[0] for u in usage)
    tokens_out = sum(_tokens(u)[1] for u in usage)
    elapsed = float(state.get("elapsed_seconds") or 0.0)
    state_saved = _mtime(root / "state.json")
    if live and state_saved:
        elapsed += max(0.0, now - state_saved)

    coder = config.get("coder") or config.get("generator") or {}
    judge = config.get("judge") or coder

    def model(entry: dict[str, Any]) -> dict[str, Any]:
        extra = entry.get("extra_body") or {}
        effort = (extra.get("reasoning") or {}).get("effort") or (extra.get("thinking") or {}).get("type")
        return {"model": entry.get("model"), "provider": entry.get("provider"), "effort": effort}

    objective = manifest.get("objective") or {}
    device = next(((row.get("metrics") or {}).get("timing_environment", {}).get("device_name")
                   for row in reversed(history) if (row.get("metrics") or {}).get("timing_environment")), None)
    if device is None:
        preflight = next(iter(sorted((root / "reports").glob("0000-preflight.json"))), None)
        device = ((_load(preflight) or {}).get("timing_environment") or {}).get("device_name") if preflight else None
    best = state.get("best")
    acceptance_runs = [{"accepted": bool((r.get("acceptance") or {}).get("accepted")),
                        "speedup": (r.get("acceptance") or {}).get("overall_speedup"),
                        "interval": (r.get("acceptance") or {}).get("overall_interval"),
                        "cases": _cases(r)} for r in state.get("acceptance_reports") or []]
    activity = _activity(steps, live, status)
    stale = status == "running" and live is False
    return {
        "name": root.name, "path": str(root), "generated_at": now, "live": live, "stale": stale,
        "status": status, "message": state.get("message") or (summary or {}).get("message", ""),
        "task": {"name": manifest.get("name") or root.name, "kind": manifest.get("kind") or "optimize",
                 "description": _clip(manifest.get("description"), 400), "metric": objective.get("metric"),
                 "unit": objective.get("unit"), "direction": objective.get("direction", "minimize"),
                 "scope": objective.get("scope"), "target_speedup": objective.get("target_speedup"),
                 "files": (manifest.get("implementation") or {}).get("files") or [], "device": device},
        "models": {"coder": model(coder), "judge": model(judge), "shared": not config.get("judge")},
        "profile_enabled": bool((config.get("profile") or {}).get("enabled")),
        "budget": {"rounds": budget.get("rounds"), "llm_calls": budget.get("llm_calls"),
                   "seconds": budget.get("seconds"), "acceptance_repeats": budget.get("acceptance_repeats")},
        "used": {"rounds": next_round, "completed_rounds": len(history), "llm_calls": state.get("llm_calls", 0),
                 "role_calls": _role_calls(state.get("role_calls")), "evaluations": state.get("evaluations", 0),
                 "seconds": elapsed, "tokens_in": tokens_in, "tokens_out": tokens_out},
        "current_round": next_round if status == "running" and next_round > len(history) else None,
        "current_phase": phases.get(next_round - 1) if next_round else "seed",
        "activity": activity,
        "best": None if not best else {"round": best["id"] + 1, "score": best.get("score"),
                                       "interval": ((best.get("metrics") or {}).get("comparison") or {}).get("overall", {}).get("interval"),
                                       "hypothesis": best.get("hypothesis"), "cases": _cases(best.get("metrics") or {})},
        "anchor_round": None if eligible_id is None else eligible_id + 1,
        "acceptance": acceptance_runs,
        "live_advice": next(({"round": step["round"] + 1, "diagnosis": step["note"]["diagnosis"],
                              "base_round": step["note"]["base_round"]}
                             for step in reversed(steps) if step.get("note") and step.get("round") == next_round - 1
                             and next_round > len(history)), None),
        "headroom": _headroom(root, state, bool((config.get("profile") or {}).get("enabled"))),
        "progress": _progress(steps, history),
        "rounds": rounds,
        "steps": [{**{key: step.get(key) for key in ("kind", "actor", "stage", "start", "end", "status", "text", "tokens_out")},
                   "round": None if step.get("round") is None else step["round"] + 1}  # 1-based, as the loop prints
                  for step in steps[-STEP_LIMIT:]],
    }


def discover(path: Path) -> dict[str, Path]:
    """A run directory itself, or every run directly under a parent directory."""
    path = path.expanduser().resolve()
    if (path / "state.json").exists():
        return {path.name: path}
    return {child.name: child for child in sorted(path.iterdir()) if (child / "state.json").exists()} if path.is_dir() else {}


def listing(path: Path) -> list[dict[str, Any]]:
    rows = []
    for name, root in discover(path).items():
        state = _load(root / "state.json") or {}
        budget = (_load(root / "config.json") or {}).get("budget") or {}
        live = is_live(root)
        rows.append({"name": name, "status": state.get("status"), "live": live,
                     "stale": state.get("status") == "running" and live is False,
                     "rounds": state.get("next_round", 0), "budget_rounds": budget.get("rounds"),
                     "best": (state.get("best") or {}).get("score"), "updated": _mtime(root / "state.json")})
    rows.sort(key=lambda row: (not row["live"], -(row["updated"] or 0)))
    return rows


PAGE = Path(__file__).with_name("dashboard.html")


def make_handler(path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, value: Any) -> None:
            self._send(code, json.dumps(value, allow_nan=False, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json(200, listing(path))
            elif url.path == "/api/run":
                name = parse_qs(url.query).get("name", [""])[0]
                runs = discover(path)  # only names found on disk are served; no path is built from the query
                if name not in runs:
                    self._json(404, {"error": f"unknown run {name!r}"})
                else:
                    self._json(200, snapshot(runs[name]))
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def serve(path: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    if not discover(path):
        raise ValueError(f"no run directory (with state.json) at or directly under {path}")
    server = ThreadingHTTPServer((host, port), make_handler(path))
    print(f"KAI Pichu dashboard: http://{host}:{server.server_port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
