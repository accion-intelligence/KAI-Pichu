from __future__ import annotations

import json
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any

from .benchmark.loading import file_inventory
from .benchmark.api import json_fingerprint
from .benchmark.models import BenchmarkSpec
from .config import OptimizeConfig
from .io import write_json
from .process import run_process
from .profiling import ProfileReport
from .workspace import Workspace


def cuda_device_index(spec: BenchmarkSpec) -> int | None:
    """Logical CUDA device of an in-process GPU task, or None for CPU/external workloads.

    measurement.device drives the CUDA event timer; options.device is the adapter
    convention for the workload's device. They must agree so the GPU guard never
    watches a different card than the one being measured.
    """
    declared = spec.measurement.device
    option = str(spec.options.get("device", "")).strip()
    option_index: int | None = None
    if option == "cuda" or option.startswith("cuda:"):
        _, _, suffix = option.partition(":")
        if suffix and not suffix.isdigit():
            raise ValueError(f"options.device must look like cuda or cuda:N, got {option!r}")
        option_index = int(suffix) if suffix else 0
    if option_index is not None and option_index != declared:
        raise ValueError(f"options.device {option!r} does not match measurement.device {declared}; "
                         "declare the same logical CUDA device in both places")
    if spec.measurement.timer == "cuda_event" or option_index is not None:
        return declared
    return None


def _sample_statistics(values: list[float]) -> dict[str, float | int]:
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "samples": len(values)}


def objective_measurements(report: dict[str, Any]) -> dict[str, Any] | None:
    """Measured values of the scored metric, per case and arm, from the paired A/B records.

    The speedup interval alone hides what was measured. When the benchmark's
    objective is an adapter metric such as an in-graph operator time, the model
    must see that metric's own values, not only the SDK's outer latency or NCU
    replay durations, which have different boundaries and are never scored.
    """
    records = report.get("records")
    spec = report.get("spec")
    if not records or not spec:
        return None
    objective = spec["objective"]
    metric = objective["metric"]
    arm_labels = {"a": "baseline", "b": "candidate"}

    samples_by_case: dict[str, dict[str, list[dict[str, float]]]] = {}
    for row in records:
        arms = samples_by_case.setdefault(row["case_id"], {label: [] for label in arm_labels.values()})
        arms[arm_labels[row["arm"]]].extend(row["samples"])
    per_case = {
        case_id: {label: _sample_statistics([float(sample[metric]) for sample in samples])
                  for label, samples in arms.items()}
        for case_id, arms in sorted(samples_by_case.items())
    }
    recorded_metrics = {name for row in records for sample in row["samples"] for name in sample}
    return {"metric": metric, "unit": objective["unit"], "direction": objective["direction"],
            "scope": objective["scope"], "boundary": spec["measurement"]["boundary"],
            "per_case": per_case, "unscored_recorded_metrics": sorted(recorded_metrics - {metric})}


def feedback(report: dict[str, Any]) -> dict[str, Any]:
    result = {key: report[key] for key in (
        "status", "error_type", "message", "stdout", "stderr", "acceptance",
        "checks", "candidate_checks", "timing_environment", "process_status", "report_path",
    ) if key in report}
    objective = objective_measurements(report)
    if objective is not None:
        result["objective"] = objective
    for section in ("calibration", "comparison"):
        if section not in report:
            continue
        data = report[section]
        summary = data.get("summary", data)
        result[section] = {key: data[key] for key in ("passed", "failing_cases", "tolerance") if key in data}
        if "overall" in summary:
            result[section]["overall"] = {key: summary["overall"][key] for key in ("speedup", "interval")}
    return result


class Evaluator:
    def __init__(self, workspace: Workspace, config: OptimizeConfig):
        self.workspace = workspace
        self.config = config
        self._profiles: dict[str, ProfileReport] = {}
        self.gpu_device = config.resources.gpu_device
        in_process_gpu = cuda_device_index(workspace.spec)
        if in_process_gpu is not None:
            # The in-process workload, the CUDA event timer and the GPU guard
            # must all refer to the same logical device.
            if self.gpu_device is not None and self.gpu_device != in_process_gpu:
                raise ValueError(f"resources.gpu_device={self.gpu_device} conflicts with the benchmark's "
                                 f"CUDA device {in_process_gpu}; use the same logical index or omit gpu_device")
            self.gpu_device = in_process_gpu

    def evaluate(self, tag: str, *, candidate: Path | None, split: str, timeout: float) -> dict[str, Any]:
        workspace = self.workspace
        output = workspace.root / "reports" / f"{tag}.json"
        command = [sys.executable, "-m", "kai_light", "benchmark", "validate" if candidate is None else "run",
                   str(workspace.manifest), "--split", split, "--output", str(output)]
        if candidate is not None:
            command += ["--candidate", str(candidate)]
        execution = run_process(command, cwd=workspace.root, log=output.with_suffix(".log"),
                                timeout=timeout, resources=self.config.resources, gpu_device=self.gpu_device)
        write_json(output.with_suffix(".process.json"), execution)
        if execution["status"] != "completed":
            # Never promote a child report invalidated by resource contention.
            return {"status": execution["status"], "message": execution.get("message", execution.get("log_tail", "")),
                    "process_status": execution["status"]}
        if not output.exists():
            return {"status": "error", "error_type": "SubprocessCrashed", "message": execution["log_tail"]}
        report = json.loads(output.read_text())
        if execution["returncode"] not in (0, 1, 2):
            return {"status": "error", "error_type": "SubprocessCrashed", "message": execution["log_tail"]}
        report["report_path"] = str(output.relative_to(workspace.root))
        return report

    def profile(self, tag: str, candidate: Path, *, timeout: float) -> dict[str, Any]:
        settings = self.config.profile
        if not settings.enabled:
            return {"status": "disabled", "message": "No NCU measurements are available."}
        if self.gpu_device is None:
            return {"status": "unavailable", "message": "NCU requires an explicitly configured GPU workload."}
        ncu = shutil.which(settings.ncu)
        if not ncu:
            return {"status": "unavailable", "message": "ncu executable not found"}
        output = self.workspace.root / "profiles" / tag
        output.mkdir(parents=True, exist_ok=False)
        csv = output / "metrics.csv"
        metadata = output / "workload.json"
        report_path = output / "capture.ncu-rep"
        started = time.monotonic()
        fingerprint = json_fingerprint(file_inventory(candidate, self.workspace.spec.implementation.files))
        command = [ncu, "--csv", "--page=raw", "--kernel-name-base=demangled",
                   "--target-processes=all", "--profile-from-start=off", "--replay-mode=kernel",
                   "--print-units=base", "--print-fp", "--log-file=" + str(csv),
                   "--export=" + str(report_path)]
        if settings.metrics:
            command.append("--metrics=" + ",".join(settings.metrics))
        else:
            command.extend("--section=" + section for section in settings.sections)
        command += [sys.executable, "-m", "kai_light.profile_worker", str(self.workspace.manifest),
                    "--candidate", str(candidate), "--output", str(metadata)]
        if settings.case_id:
            command += ["--case", settings.case_id]
        execution = run_process(command, cwd=self.workspace.root, log=output / "process.log", timeout=timeout,
                                resources=self.config.resources, gpu_device=self.gpu_device)
        result = {"status": "profiled" if execution["status"] == "completed" and execution["returncode"] == 0 else "profile_failed",
                  "implementation_fingerprint": fingerprint, "process": execution,
                  "profile_id": tag, "report_path": str(report_path), "csv_path": str(csv),
                  "capture": {"metrics": settings.metrics, "sections": [] if settings.metrics else settings.sections}}
        if metadata.exists():
            result["workload"] = json.loads(metadata.read_text())
        if json_fingerprint(file_inventory(candidate, self.workspace.spec.implementation.files)) != fingerprint:
            raise ValueError("profiled candidate source changed")
        if result["status"] == "profiled":
            try:
                reader = ProfileReport(report_path, csv_path=csv, settings=settings,
                    identity={"implementation_fingerprint": fingerprint, "workload": result.get("workload", {})})
                self._profiles[tag] = reader
                remaining = max(0, timeout - (time.monotonic() - started))
                result["evidence"] = reader.overview(timeout=remaining)
                result["query_available"] = result["evidence"]["status"] == "available"
            except (OSError, ValueError, RuntimeError) as error:
                result["evidence"] = {"status": "unavailable", "message": str(error)}
        write_json(output / "profile.json", result)
        return result

    def query_profile(self, profile_id: str, request: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        reader = self._profiles.get(profile_id)
        if reader is None:
            return {"status": "unavailable", "error": {"message": "no queryable profile for this optimization stage"}}
        return reader.query(request, timeout=timeout)
