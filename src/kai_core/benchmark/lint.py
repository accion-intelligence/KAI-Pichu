"""Static contract checks for a task bundle. Never runs the workload.

validate() executes prepare/run/validate for every case, so finding a contract
mistake currently requires the device the task targets. lint imports the adapter
and reads its case descriptors — the same trust model as the rest of the SDK,
since adapters are task-owner code — but never calls prepare, run,
load_implementation or any timer. It is therefore usable on a machine with no
GPU, which is where a task is usually authored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .api import Benchmark
from .loading import file_inventory, load_adapter
from .models import BenchmarkSpec, Case, load_spec

SDK_METRICS = ("latency_ms", "throughput")


class Finding:
    def __init__(self, level: str, code: str, message: str):
        self.level, self.code, self.message = level, code, message

    def __str__(self) -> str:
        return f"{self.level.upper():<7} {self.code:<28} {self.message}"


def _error(code: str, message: str) -> Finding:
    return Finding("error", code, message)


def _warn(code: str, message: str) -> Finding:
    return Finding("warning", code, message)


def _note(code: str, message: str) -> Finding:
    """Something lint cannot decide. Never a defect, so it must not read as one."""
    return Finding("note", code, message)


def _patterns(root: Path, patterns: list[str], code: str, label: str) -> tuple[list[str], list[Finding]]:
    """Resolve globs one at a time so an unmatched pattern is named, not fatal."""
    names, findings = [], []
    for pattern in patterns:
        try:
            names.extend(file_inventory(root, [pattern]))
        except (OSError, ValueError) as error:
            findings.append(_error(code, f"{label} pattern {pattern!r}: {error}"))
    return names, findings


def _cases(task: Benchmark, split: str) -> tuple[list[Case], list[Finding]]:
    try:
        first = list(task.cases(split))
        second = list(task.cases(split))
    except Exception as error:  # noqa: BLE001 - adapter code, any failure is a finding
        return [], [_error("cases.raises", f"cases({split!r}) raised {type(error).__name__}: {error}")]
    findings = []
    if not first:
        findings.append(_error("cases.empty", f"cases({split!r}) returned no cases"))
    if any(not isinstance(case, Case) for case in first):
        return [], findings + [_error("cases.type", f"cases({split!r}) must yield Case objects")]
    ids = [case.id for case in first]
    if len(set(ids)) != len(ids):
        findings.append(_error("cases.duplicate_id", f"cases({split!r}) has duplicate case ids"))
    if [case.model_dump() for case in first] != [case.model_dump() for case in second]:
        findings.append(_error("cases.nondeterministic", f"cases({split!r}) is not deterministic"))
    return first, findings


def lint(manifest: Path) -> list[Finding]:
    """Contract findings for one task, worst first. An empty list means no finding."""
    manifest = Path(manifest).expanduser().resolve(strict=True)
    try:
        spec: BenchmarkSpec = load_spec(manifest)
    except Exception as error:  # noqa: BLE001 - schema violations are the primary finding
        return [_error("manifest.invalid", f"{type(error).__name__}: {error}")]

    findings: list[Finding] = []
    directory = manifest.parent

    # Declared files must exist. Each pattern is reported separately so the author
    # learns which one is wrong rather than only that something is.
    root = (directory / spec.implementation.root)
    if not root.is_dir():
        findings.append(_error("implementation.root", f"implementation root does not exist: {root}"))
    else:
        _, problems = _patterns(root, spec.implementation.files, "implementation.files", "implementation.files")
        findings += problems
    for label, patterns in (("benchmark_files", spec.benchmark_files), ("agent_files", spec.agent_files)):
        _, problems = _patterns(directory, patterns, f"{label}.unmatched", label)
        findings += problems

    # BenchmarkSpec.check_adapter already rejects an adapter listed in agent_files,
    # so that arrives here as manifest.invalid. What it does not check is whether
    # the agent-visible files are readable text, which Workspace.create only
    # discovers once a run starts.
    for name in _patterns(directory, spec.agent_files, "agent_files.unmatched", "agent_files")[0]:
        try:
            (directory / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            findings.append(_error("agent_files.not_text", f"agent_files entry is not UTF-8 text: {name}"))

    # measurement.device drives the CUDA event timer; options.device is the
    # adapter convention. Disagreement is caught today only at evaluation time.
    option = str(spec.options.get("device", "")).strip()
    if option.startswith("cuda"):
        _, _, suffix = option.partition(":")
        if suffix and not suffix.isdigit():
            findings.append(_error("options.device", f"options.device must look like cuda or cuda:N, got {option!r}"))
        elif (int(suffix) if suffix else 0) != spec.measurement.device:
            findings.append(_error("options.device", f"options.device {option!r} does not match "
                                                     f"measurement.device {spec.measurement.device}"))

    if spec.fusion is not None:
        kernels = set(_patterns(directory, spec.fusion.file_patterns(), "fusion.unmatched", "fusion")[0])
        editable = set(_patterns(root, spec.implementation.files, "implementation.files", "implementation.files")[0]) if root.is_dir() else set()
        overlap = {name for name in kernels if name in editable}
        if overlap:
            findings.append(_error("fusion.overlap", f"fusion kernels are also implementation files: {sorted(overlap)}"))

    try:
        task = load_adapter(manifest, spec)
    except ModuleNotFoundError as error:
        # A missing dependency is an environment gap, not a task defect. Say so,
        # and say which checks were skipped, rather than reporting a broken task.
        findings.append(_warn("adapter.dependency", f"adapter needs {error.name!r}, which is not installed; "
                                                    "case and split checks were skipped"))
        return _ordered(findings)
    except Exception as error:  # noqa: BLE001 - adapter import/instantiation is the check
        findings.append(_error("adapter.load", f"{type(error).__name__}: {error}"))
        return findings

    splits = {}
    for split in ("smoke", "search", "acceptance"):
        cases, problems = _cases(task, split)
        findings += problems
        splits[split] = [case.id for case in cases]
    # A held-out split is only held out if its cases differ; docs/MEASUREMENT.md
    # states this, and nothing checks it.
    if splits["search"] and splits["search"] == splits["acceptance"]:
        findings.append(_warn("splits.identical", "search and acceptance use identical case ids; "
                                                  "acceptance is a rerun, not a held-out check"))
    if spec.objective.metric not in SDK_METRICS:
        findings.append(_note("objective.adapter_metric", f"objective {spec.objective.metric!r} is adapter-owned; "
                                                          "confirming run() emits it needs validate on the device"))
    return _ordered(findings)


ORDER = {"error": 0, "warning": 1, "note": 2}


def _ordered(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda finding: ORDER.get(finding.level, 9))


def render(findings: Iterable[Finding]) -> str:
    rows = list(findings)
    tail = "Contract checks passed; run benchmark validate on the target device to execute the task."
    if not rows:
        return f"No findings. {tail}"
    counts = {level: sum(1 for row in rows if row.level == level) for level in ORDER}
    body = "\n".join(str(row) for row in rows)
    summary = ", ".join(f"{counts[level]} {level}(s)" for level in ORDER if counts[level])
    return f"{body}\n\n{summary}." + ("" if counts["error"] else f" {tail}")
