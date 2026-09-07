"""Bounded, discoverable queries over one immutable operator profile.

An optional external reader supplies metric descriptions, NVIDIA rules and
source/SASS evidence; CSV remains a usable fallback.
"""
from __future__ import annotations

from collections import Counter
from difflib import get_close_matches
from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Literal

from pydantic import Field, model_validator

from .config import ConfigModel, ProfileConfig, Resources
from .io import parse_object, write_json
from .process import run_process
from .profile_csv import read_csv


DIAGNOSTICS = [
    {"question": "What was collected?", "operation": "catalog", "query": "",
     "purpose": "Discover actual metric names, descriptions and units; paginate before assuming a name."},
    {"question": "What did NVIDIA flag?", "operation": "rules",
     "purpose": "Rules are hypotheses to verify with metrics and benchmark remeasurement."},
    {"question": "Memory traffic, cache efficiency or bank conflicts?", "operation": "catalog", "query": "memory",
     "purpose": "Search descriptions; also try cache, sector, conflict or dram. Read discovered metrics next."},
    {"question": "Registers or occupancy limiting concurrency?", "operation": "catalog", "query": "occupancy",
     "purpose": "Discover occupancy/resource metrics; also try register or shared."},
    {"question": "Why are warps not issuing?", "operation": "catalog", "query": "stall",
     "purpose": "Compare measured stall reasons with eligible warps and scheduler evidence."},
    {"question": "Which code or instructions contribute?", "operation": "source-metrics",
     "purpose": "Use a discovered per-PC counter; needs captured source counters and line information. "
                "disasm reads SASS/PTX; warp-stalls needs timed warp samples. Missing evidence is not zero."},
]


class ProfileQuery(ConfigModel):
    operation: Literal["launches", "catalog", "metrics", "rules", "source-metrics", "warp-stalls", "disasm"]
    row_id: str = Field(default="launch:0", pattern=r"^launch:[0-9]+$")
    query: str = Field(default="", max_length=300, description="Case-insensitive words in metric name/description")
    counter: str | None = Field(default=None, max_length=300,
                               description="Required for metrics/source-metrics: exact discovered name or one glob; no comma lists")
    offset: int = Field(default=0, ge=0, le=1000000)
    limit: int = Field(default=20, ge=1, le=100)
    by: Literal["line", "sass", "file", "reason"] = "line"
    file: str | None = Field(default=None, max_length=1000)
    line: int | None = Field(default=None, ge=1)
    view: Literal["sass", "ptx", "source"] = "sass"
    fields: list[str] | None = Field(default=None, min_length=1, max_length=20,
        description="Optional top-level row fields. Metric names/values/units, row identity and attribution coverage are retained.")

    @model_validator(mode="after")
    def validate_query(self) -> ProfileQuery:
        if self.operation in {"metrics", "source-metrics"} and not self.counter:
            raise ValueError("discover a counter with catalog, then supply its name in the required counter field; "
                             "query search words cannot replace counter for metrics/source-metrics")
        if self.counter is not None and (not self.counter.strip() or "," in self.counter):
            raise ValueError("counter must be one nonempty exact name or glob")
        if self.operation == "source-metrics" and self.by == "reason":
            raise ValueError("source-metrics supports line, sass or file")
        if self.operation == "warp-stalls" and self.by == "file":
            raise ValueError("warp-stalls supports line, sass or reason")
        if self.line is not None and (not self.file or self.operation != "source-metrics"):
            raise ValueError("line requires source-metrics and file")
        if self.fields and any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field) for field in self.fields):
            raise ValueError("fields must contain top-level field names, not expressions or nested paths")
        return self


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded(value: Any, *, strings: int = 1200, items: int = 30) -> Any:
    """Keep diagnostic metadata, explicitly marking nested omissions."""
    if isinstance(value, str):
        return value if len(value) <= strings else value[:strings] + f" <{len(value) - strings} characters omitted>"
    if isinstance(value, list):
        rows = [bounded(row, strings=strings, items=items) for row in value[:items]]
        return rows if len(value) <= items else {"rows": rows, "total_count": len(value), "omitted": len(value) - items}
    if isinstance(value, dict):
        return {key: bounded(item, strings=strings, items=items) for key, item in value.items()}
    return value


def reader_diagnostic(value: Any) -> Any:
    """Use product terminology in diagnostics; raw reader output stays on disk."""
    if isinstance(value, str):
        # Cache paths are opaque to the agent. Mark an omitted path rather than
        # creating a fictitious path by renaming a directory inside it.
        value = re.sub(r"""[^\s"'<>]*[/\\][^\s"'<>]*kai-ncu-reader[^\s"'<>]*|[^\s"'<>]*\.kai-ncu-reader[^\s"'<>]*""",
                       "[reader artifact]", value, flags=re.IGNORECASE)
        value = re.sub("KAI_LIGHT_REPORT_READER_DIR", "profile.ncu_report_dir", value, flags=re.IGNORECASE)
        return re.sub("kai-ncu-reader", "NCU report reader", value, flags=re.IGNORECASE)
    if isinstance(value, list):
        return [reader_diagnostic(item) for item in value]
    if isinstance(value, dict):
        return {key: reader_diagnostic(item) for key, item in value.items()}
    return value


class ProfileReport:
    """Public SDK for querying an existing report; never launches a GPU workload."""
    def __init__(self, report: Path, *, csv_path: Path | None = None,
                 settings: ProfileConfig | None = None, identity: dict[str, Any] | None = None):
        self.report = report.resolve()
        self.csv = csv_path.resolve() if csv_path is not None else None
        self.settings = settings or ProfileConfig()
        self.inputs = {str(path): file_hash(path) for path in (self.report, self.csv) if path and path.is_file()}
        if not self.inputs:
            raise ValueError("no NCU report or CSV exists")
        self.identity = {**(identity or {}), "report_path": str(self.report), "input_sha256": self.inputs}
        self.binary = shutil.which(self.settings.report_reader or "kai-ncu-reader")
        self.backend = "ncu_report" if self.binary and self.report.is_file() else "csv"
        self.cache = self.report.parent / (self.report.name + ".kai-light")
        self.warnings: list[str] = []
        self.launches: list[dict[str, Any]] | None = None
        self.details: dict[str, dict[str, Any]] = {}
        self.deadline = 0.0

    def _remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("profile query time budget exhausted")
        return remaining

    def _check_inputs(self) -> None:
        if any(not Path(path).is_file() or file_hash(Path(path)) != digest for path, digest in self.inputs.items()):
            raise ValueError("profile inputs changed; collect or open a new report")

    def _reader_query(self, verb: str, args: list[str]) -> dict[str, Any]:
        if not self.binary:
            raise ValueError("NCU report reader is unavailable; see dependency setup or set profile.report_reader")
        binary = Path(self.binary).resolve()
        key = hashlib.sha256(json.dumps({"inputs": self.inputs, "verb": verb, "args": args,
            "binary": str(binary), "mtime": binary.stat().st_mtime_ns,
            "ncu": self.settings.ncu, "ncu_report_dir": self.settings.ncu_report_dir,
            "adapter_version": 1}, sort_keys=True).encode()).hexdigest()
        saved = self.cache / f"{key}.json"
        if saved.is_file():
            return parse_object(saved.read_text())
        self.cache.mkdir(parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix="query-", dir=self.cache))
        stdout = attempt / "stdout.json"
        env = dict(os.environ)
        env["KAI_LIGHT_REPORT_READER_DIR"] = str(Path(__file__).parent / "ncu_compat")
        env["KAI_LIGHT_NCU"] = self.settings.ncu
        if self.settings.ncu_report_dir:
            env["KAI_LIGHT_NCU_REPORT_DIR"] = self.settings.ncu_report_dir
        result = run_process([self.binary, "ncu", verb, str(self.report), *args], cwd=self.report.parent,
                             log=attempt / "stderr.log", stdout=stdout, env=env,
                             timeout=self._remaining(), resources=Resources(), gpu_device=None)
        if result["status"] != "completed":
            raise RuntimeError(f"NCU report reader query {result['status']}: {result.get('log_tail', '')[-500:]}")
        value = parse_object(stdout.read_text())
        if result.get("returncode") != 0 or value.get("error"):
            raise RuntimeError(json.dumps(value.get("error", {"message": result.get("log_tail", "query failed")})))
        if value.get("source", {}).get("kind") != "ncu" or value.get("schema") != "v1" or value.get("source", {}).get("version") != "v1":
            raise ValueError("unsupported NCU report response version")
        if not isinstance(value.get("data", {}).get("rows"), list):
            raise ValueError("NCU report reader response has no canonical data.rows list")
        self._check_inputs()
        write_json(saved, value, replace=True)
        return value

    def _load(self) -> None:
        if self.launches is not None:
            return
        if self.backend == "ncu_report":
            try:
                data = self._reader_query("launches", ["--limit", "10000"])["data"]
                if data["total_matched"] > len(data["rows"]):
                    raise ValueError("report exceeds 10000 launches; narrow the capture to the operator")
                self.launches = data["rows"]
                return
            except (OSError, ValueError, RuntimeError) as error:
                self.warnings.append(f"NCU report reader failed: {str(error)[:1200]}")
        if self.csv is None or not self.csv.is_file():
            raise ValueError("no usable NCU report reader or CSV fallback")
        self.backend = "csv"
        self.launches = read_csv(self.csv)
        self.details = {row["row_id"]: row for row in self.launches}
        self.warnings.append("CSV fallback: metric descriptions, NVIDIA rules and source/SASS attribution are unavailable.")

    def _detail(self, row_id: str) -> dict[str, Any]:
        self._load()
        if not any(row["row_id"] == row_id for row in self.launches or []):
            raise ValueError(f"unknown {row_id}; use launches to discover report row IDs")
        if row_id not in self.details:
            data = self._reader_query("inspect", ["--row-id", row_id])["data"]
            if not data["rows"] or data["rows"][0].get("type") == "not_found":
                raise ValueError(f"launch absent from report: {row_id}")
            self.details[row_id] = data["rows"][0]
        return self.details[row_id]

    def _page(self, rows: list[Any], request: ProfileQuery, *, auxiliary: Any = None,
              total: int | None = None) -> dict[str, Any]:
        limit = min(request.limit, self.settings.max_rows)
        shown = rows[request.offset:request.offset + limit]
        data = {"count": len(shown), "total_matched": len(rows) if total is None else total,
                "offset": request.offset, "rows": shown,
                "next_offset": request.offset + len(shown) if request.offset + len(shown) < len(rows) else None,
                "auxiliary": auxiliary or {}}
        return data

    def _query(self, request: ProfileQuery) -> dict[str, Any]:
        self._load()
        if request.operation == "launches":
            return self._page([{k: v for k, v in row.items() if k not in {"metrics", "rules"}}
                               for row in self.launches or []], request)
        detail = self._detail(request.row_id)
        if request.operation in {"catalog", "metrics"}:
            rows = []
            for metric in detail.get("metrics", []):
                name = metric["name"]
                description = metric.get("label") or ""
                if request.counter and not fnmatchcase(name, request.counter):
                    continue
                if any(word not in f"{name} {description}".casefold() for word in request.query.casefold().split()):
                    continue
                row = {"key": f"{request.row_id}|counter:{name}", "row_id": request.row_id,
                       "name": name, "description": description or None, "unit": metric.get("unit"),
                       "value_type": metric.get("value_type"), "metric_type": metric.get("metric_type"),
                       "rollup": metric.get("rollup")}
                if request.operation == "metrics":
                    row.update(value=metric.get("value"), status="missing" if metric.get("value") is None else "measured")
                rows.append(row)
            rows.sort(key=lambda row: row["name"])
            auxiliary = {"evidence_scope": "collected_report_only",
                "empty_result_hint": "No matching evidence in this launch. Try catalog words/pagination; "
                                     "absent metrics require a new NCU capture, not a zero-value assumption."}
            if not rows:
                base = {"operation": "catalog", "row_id": request.row_id, "limit": request.limit}
                words = list(dict.fromkeys(request.query.split()))
                if len(words) > 1:
                    auxiliary["suggested_queries"] = [{**base, "query": word} for word in words[:3]]
                    auxiliary["suggestion_basis"] = "All search words must match; try one concept at a time."
                elif request.counter and not any(char in request.counter for char in "*?["):
                    names = sorted(metric["name"] for metric in detail.get("metrics", []))
                    matches = get_close_matches(request.counter, names, n=3, cutoff=0.6)
                    if matches:
                        auxiliary["suggested_queries"] = [{**base, "counter": name} for name in matches]
                        auxiliary["suggestion_basis"] = "Name similarity only; verify descriptions before choosing a counter."
            return self._page(rows, request, auxiliary=auxiliary)
        if request.operation == "rules":
            if self.backend != "ncu_report":
                raise ValueError("NVIDIA rule findings require a readable .ncu-rep and the NCU report reader")
            return self._page(detail.get("rules", []), request,
                              auxiliary={"interpretation": "NVIDIA hypotheses; verify against measured counters and benchmark results."})
        if self.backend != "ncu_report":
            raise ValueError("source and instruction queries require a readable .ncu-rep and the NCU report reader")
        args = ["--row-id", request.row_id]
        if request.operation != "disasm":
            if request.offset >= 10000:
                raise ValueError("source query exceeds 10000 rows; narrow with file or counter")
            args += ["--by", request.by, "--limit", str(min(request.offset + request.limit, 10000))]
            if request.counter:
                args += ["--counter", request.counter]
            if request.file:
                args += ["--file", request.file]
            if request.line:
                args += ["--line", str(request.line)]
        envelope = self._reader_query(request.operation, args)
        data = envelope["data"]
        rows, auxiliary = data["rows"], dict(data.get("auxiliary") or {})
        auxiliary.pop("meta_cache_path", None)
        if request.operation == "warp-stalls" and not auxiliary.get("total_samples"):
            auxiliary["evidence_status"] = "no_samples"
            auxiliary["hint"] = "This report has no timed warp samples. This does not establish zero stalls; " \
                                "use collected PC-sampling counters or capture supported warp sampling."
        if request.operation == "disasm":
            # The reader disassembles a cubin, which can contain many unrelated
            # library kernels. Select the actual launch before paging SASS.
            auxiliary["evidence_scope"] = "cubin"
            if request.view == "ptx":
                rows = auxiliary.get("ptx_lines", [])
            elif request.view == "source":
                rows = auxiliary.get("source_index", [])
            else:
                launch = next(row for row in self.launches if row["row_id"] == request.row_id)
                symbols = {launch.get(key) for key in ("kernel_mangled", "kernel_demangled")} - {None, ""}
                selected = [row for row in rows if row.get("function_name") in symbols]
                if not selected:
                    raise ValueError("disassembly does not contain the selected launch symbol; "
                                     "unrelated cubin functions cannot be used as launch evidence")
                auxiliary.update(evidence_scope="launch_function", cubin_function_count=len(rows),
                                 selected_function_count=len(selected))
                rows = [{"kernel": row.get("function_name"), **instruction}
                        for row in selected for instruction in row.get("instructions", [])]
            auxiliary.pop("ptx_lines", None)
            auxiliary.pop("source_index", None)
            auxiliary["available_views"] = ["sass", "ptx", "source"]
            return self._page(rows, request, auxiliary=auxiliary)
        result = self._page(rows, request, auxiliary=auxiliary, total=data.get("total_matched"))
        if request.offset + result["count"] < result["total_matched"]:
            result["next_offset"] = request.offset + result["count"] if result["count"] else None
        return result

    @staticmethod
    def _select_fields(data: dict[str, Any], request: ProfileQuery) -> None:
        if not request.fields or not data["rows"]:
            return
        available = set().union(*(row.keys() for row in data["rows"]))
        unknown = set(request.fields) - available
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}; available fields: {sorted(available)}")
        # Keep metric identity, values and units alongside selected columns.
        # Source counter maps and auxiliary coverage remain interpretable.
        keep = set(request.fields) | {"key", "row_id", "launch_row_id", "name", "kernel",
            "kernel_mangled", "kernel_demangled", "file", "line", "address", "counter_name",
            "value", "unit", "status", "counters"}
        data["rows"] = [{key: value for key, value in row.items() if key in keep} for row in data["rows"]]
        data["auxiliary"]["available_fields"] = sorted(available)

    @staticmethod
    def _next_query(data: dict[str, Any], request: ProfileQuery) -> None:
        data["next_query"] = ({**request.model_dump(exclude_none=True), "offset": data["next_offset"]}
                              if data["next_offset"] is not None else None)

    def query(self, request: ProfileQuery | dict[str, Any], *, timeout: float = 60) -> dict[str, Any]:
        """Return a versioned, paged result; errors and absent evidence stay explicit."""
        self.deadline = time.monotonic() + min(timeout, self.settings.query_seconds)
        value: dict[str, Any] = {"schema_version": 1, "identity": self.identity}
        try:
            request = request if isinstance(request, ProfileQuery) else ProfileQuery.model_validate(request)
            self._check_inputs()
            data = self._query(request)
            self._select_fields(data, request)
            self._next_query(data, request)
            if "warnings" in data.get("auxiliary", {}):
                data["auxiliary"]["warnings"] = reader_diagnostic(data["auxiliary"]["warnings"])
            value.update(status="available", query=request.model_dump(exclude_none=True), data=data)
            value.update(backend=self.backend, warnings=reader_diagnostic(self.warnings[:4]), artifact_root=str(self.cache))
            # Paginate primary rows without changing their shape. Large nested metadata
            # remains structured, with explicit omission markers and the raw artifact.
            data["rows"] = [bounded(row) for row in data["rows"]]
            data["auxiliary"] = bounded(data["auxiliary"])
            while len(json.dumps(value)) > self.settings.max_result_chars and len(data["rows"]) > 1:
                data["rows"].pop()
                data["count"] = len(data["rows"])
                data["next_offset"] = request.offset + data["count"]
                self._next_query(data, request)
            if len(json.dumps(value)) > self.settings.max_result_chars:
                data["rows"] = [bounded(row, strings=250, items=3) for row in data["rows"]]
                data["auxiliary"] = bounded(data["auxiliary"], strings=250, items=3)
                value["truncated"] = True
            self._check_inputs()
        except (OSError, ValueError, RuntimeError) as error:
            value.update(status="unavailable", error={"message": reader_diagnostic(str(error))[:1600],
                         "hint": "Use launches/catalog for existing evidence. New measurements require NCU capture."})
        value.update(backend=self.backend, warnings=reader_diagnostic(self.warnings[:4]), artifact_root=str(self.cache))
        if len(json.dumps(value)) > self.settings.max_result_chars:
            # A single giant rule/counter map can defeat row pagination. Keep the
            # raw query on disk and explicitly ask for a narrower projection.
            value = {"schema_version": 1, "status": "unavailable", "backend": self.backend,
                     "error": {"message": "query result exceeds the context limit; narrow the counter, file or view",
                               "code": "result_too_large"},
                     "truncated": True, "artifact_root": str(self.cache),
                     "identity": self.identity}
        return value

    def overview(self, *, timeout: float = 60) -> dict[str, Any]:
        started = time.monotonic()
        listing = self.query({"operation": "launches", "limit": 10}, timeout=timeout)
        result = {"status": listing["status"], "identity": self.identity, "launches": listing,
                  "diagnostics": DIAGNOSTICS}
        result["available_operations"] = (["launches", "catalog", "metrics"] +
            (["rules", "source-metrics", "warp-stalls", "disasm"] if self.backend == "ncu_report" else [])
            if listing["status"] == "available" else [])
        if listing["status"] != "available" or not listing["data"]["rows"]:
            return result
        row_id = listing["data"]["rows"][0]["row_id"]
        remaining = timeout - (time.monotonic() - started)
        inventory = self.query({"operation": "catalog", "row_id": row_id, "limit": 1}, timeout=max(0, remaining))
        if inventory["status"] != "available":
            result["inventory_error"] = inventory.get("error")
            return result
        # The reader's full details remain on disk. Only a small index and
        # headline values enter the initial prompt; rule bodies are on demand.
        detail = self.details.get(row_id, {})
        families = Counter(metric["name"].split(".", 1)[0] for metric in detail.get("metrics", []))
        result.update(primary_row_id=row_id, metric_count=len(detail.get("metrics", [])),
                      family_count=len(families), metric_families=list(sorted(families))[:12],
                      families_omitted=max(0, len(families) - 12),
                      rule_count=len(detail.get("rules", [])) if self.backend == "ncu_report" else None)
        result["catalog_hint"] = "Search one concept with catalog; inspect returned names/descriptions before requesting values."
        result["rules_hint"] = "Request rules for the selected row_id when NVIDIA findings would help triage a bottleneck."
        result["headline_metrics"] = [{"name": metric["name"], "unit": metric.get("unit"),
            "value": metric.get("value"), "description": bounded(metric.get("label"), strings=250)}
            for metric in detail.get("metrics", []) if metric["name"] in {
                "gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed", "launch__registers_per_thread",
                "sm__warps_active.avg.pct_of_peak_sustained_active"}]
        return result
