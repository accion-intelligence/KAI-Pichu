"""kai-ncu-reader: query one Nsight Compute report through NVIDIA's ``ncu_report`` API.

The optimizer's profile layer asks one question per call: which launches a
report holds, every metric and NVIDIA rule of one launch, per-instruction
counters, warp-stall samples, or the disassembly. This module answers those
questions as a small command line tool that prints one JSON envelope, so the
same contract can be served by another reader if a site has one.

    kai-ncu-reader ncu launches REPORT [--limit N]
    kai-ncu-reader ncu inspect REPORT --row-id launch:0
    kai-ncu-reader ncu source-metrics REPORT --row-id launch:0 --by line|sass|file [--counter GLOB] [--file F] [--line N] [--limit N]
    kai-ncu-reader ncu warp-stalls REPORT --row-id launch:0 --by line|sass|reason [--limit N]
    kai-ncu-reader ncu disasm REPORT --row-id launch:0

Reading a report needs Nsight Compute's Python module; ``ncu_compat`` locates
it from the ``ncu`` on PATH, common install roots, or ``profile.ncu_report_dir``.
The design follows VeloQ (https://github.com/lucifer1004/veloq): narrow verbs,
a versioned envelope, rows with stable keys.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fnmatch import fnmatchcase
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ENVELOPE_SCHEMA = "v1"
# Occupancy curves and link masks are also "correlated" metrics, keyed by block
# size, register count or link index; program counters are far larger than any of those.
MIN_PROGRAM_COUNTER = 0x10000
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
SAMPLE_COUNT = "smsp__pcsamp_sample_count"
LAUNCH_METRICS = {
    "grid": ("launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z"),
    "block": ("launch__block_dim_x", "launch__block_dim_y", "launch__block_dim_z"),
}
SCALAR_LAUNCH_METRICS = {
    "duration_ns": "gpu__time_duration.sum", "registers_per_thread": "launch__registers_per_thread",
    "static_shared_bytes": "launch__shared_mem_per_block_static",
    "dynamic_shared_bytes": "launch__shared_mem_per_block_dynamic", "waves_per_sm": "launch__waves_per_multiprocessor",
}


def load_api() -> Any:
    """NVIDIA's ncu_report module, through the compatibility shim that finds and patches it."""
    from .ncu_compat import ncu_report
    return ncu_report


class Names:
    """Human names for the API's integer enums, tolerant of versions that lack some of them."""

    def __init__(self, api: Any):
        metric = api.IMetric
        self.kind = self._table(metric, "ValueKind_")
        self.metric_type = self._table(metric, "MetricType_")
        self.rollup = self._table(metric, "RollupOperation_")
        self.message = self._table(api, "MsgType_")

    @staticmethod
    def _table(owner: Any, prefix: str) -> dict[int, str]:
        return {getattr(owner, name): name[len(prefix):].lower() for name in dir(owner)
                if name.startswith(prefix) and isinstance(getattr(owner, name), int)}


class Report:
    """One loaded report: launches in capture order, each addressable as launch:N."""

    def __init__(self, path: Path):
        self.api = load_api()
        self.names = Names(self.api)
        self.path = path
        context = self.api.load_report(str(path))
        self.actions = [context.range_by_idx(r).action_by_idx(a)
                        for r in range(context.num_ranges())
                        for a in range(context.range_by_idx(r).num_actions())]

    def action(self, row_id: str) -> Any | None:
        match = re.fullmatch(r"launch:(\d+)", row_id)
        index = int(match.group(1)) if match else -1
        return self.actions[index] if 0 <= index < len(self.actions) else None

    # ---- launches and metrics -------------------------------------------------

    def launch_row(self, index: int, action: Any) -> dict[str, Any]:
        row_id = f"launch:{index}"
        row: dict[str, Any] = {
            "key": row_id, "row_id": row_id, "ncu_id": index,
            "kernel_demangled": action.name(self.api.IAction.NameBase_DEMANGLED),
            "kernel_mangled": action.name(self.api.IAction.NameBase_MANGLED),
            "kernel_function": action.name(self.api.IAction.NameBase_FUNCTION),
        }
        for label, names in LAUNCH_METRICS.items():
            dims = [self.scalar(action, name) for name in names]
            if all(dim is not None for dim in dims):
                row[label] = dims
        for label, name in SCALAR_LAUNCH_METRICS.items():
            value = self.scalar(action, name)
            if value is not None:
                row[label] = value
        return row

    def scalar(self, action: Any, name: str) -> Any:
        try:
            metric = action.metric_by_name(name)
        except Exception:  # the API raises for unknown names in some versions and returns None in others
            return None
        return None if metric is None else self.metric_value(metric)

    def metric_value(self, metric: Any) -> Any:
        """The rollup value as JSON: numbers finite, strings as is, anything else dropped."""
        try:
            value = metric.value()
        except Exception:
            return None
        return finite(value)

    def metric_row(self, metric: Any) -> dict[str, Any]:
        return {
            "name": metric.name(), "label": metric.description() or None, "unit": metric.unit() or None,
            "value": self.metric_value(metric),
            "value_type": self.names.kind.get(metric.kind()), "metric_type": self.names.metric_type.get(metric.metric_type()),
            "rollup": self.names.rollup.get(metric.rollup_operation()),
            "instances": metric.num_instances(), "correlated": bool(metric.has_correlation_ids()),
        }

    def rules(self, action: Any) -> list[dict[str, Any]]:
        rows = []
        for index, rule in enumerate(action.rule_results_as_dicts()):
            message = rule.get("rule_message") or {}
            speedup = rule.get("speedup_estimation") or rule.get("speedup") or {}
            rows.append({
                "key": f"rule:{index}", "name": rule.get("name"), "identifier": rule.get("rule_identifier"),
                "section": rule.get("section_identifier"), "title": message.get("title"),
                "message": plain_message(message.get("message")),
                "severity": self.names.message.get(message.get("type"), message.get("type")),
                "focus_metrics": [{"name": f.get("name"), "value": finite(f.get("value")), "severity": f.get("severity"),
                                   "hint": f.get("info")} for f in rule.get("focus_metrics") or []],
                "estimated_speedup_pct": finite(speedup.get("speedup")) if isinstance(speedup, dict) else None,
            })
        return rows

    def inspect_row(self, row_id: str, action: Any) -> dict[str, Any]:
        index = int(row_id.split(":")[1])
        row = self.launch_row(index, action)
        # "breakdown:" entries list the sub-metrics of a throughput; they are lists of names, not measurements.
        row["metrics"] = [self.metric_row(action.metric_by_name(name)) for name in action.metric_names()
                          if not name.startswith("breakdown:")]
        row["rules"] = self.rules(action)
        return row

    def is_per_instruction(self, action: Any, metric: Any) -> bool:
        """Correlation ids are program counters when the report can disassemble the first one."""
        if not metric.has_correlation_ids() or metric.num_instances() == 0:
            return False
        ids = metric.correlation_ids()
        kinds = self.api.IMetric
        if ids.kind() not in (kinds.ValueKind_UINT64, kinds.ValueKind_UINT32):
            return False
        first, last = int(ids.as_uint64(0)), int(ids.as_uint64(metric.num_instances() - 1))
        if min(first, last) < MIN_PROGRAM_COUNTER:
            return False
        return self.sass(action, first) is not None and self.sass(action, last) is not None

    # ---- per-instruction evidence ----------------------------------------------

    def instruction_counters(self, action: Any) -> dict[int, dict[str, float]]:
        """address -> {counter name: value} for every metric NCU attributed to program counters."""
        per_address: dict[int, dict[str, float]] = defaultdict(dict)
        kinds = self.api.IMetric
        for name in action.metric_names():
            metric = action.metric_by_name(name)
            if not self.is_per_instruction(action, metric):
                continue  # scalar, or correlated with something other than a program counter
            ids = metric.correlation_ids()
            numeric = metric.kind() in (kinds.ValueKind_DOUBLE, kinds.ValueKind_FLOAT)
            for instance in range(metric.num_instances()):
                value = metric.as_double(instance) if numeric else metric.as_uint64(instance)
                value = finite(value)
                if value is not None:
                    per_address[int(ids.as_uint64(instance))][name] = value
        return per_address

    def location(self, action: Any, address: int) -> tuple[str | None, int | None]:
        """Source file and line for a program counter, when the report carries source information."""
        try:
            info = action.source_info(address)
        except Exception:
            return None, None
        if info is None:
            return None, None
        file_name = _call_or_get(info, "file_name")
        line = _call_or_get(info, "line")
        return (str(file_name) if file_name else None), (int(line) if isinstance(line, int) and line > 0 else None)

    def sass(self, action: Any, address: int) -> str | None:
        return self._disassembly(action.sass_by_pc, address)

    def ptx(self, action: Any, address: int) -> str | None:
        return self._disassembly(action.ptx_by_pc, address)

    @staticmethod
    def _disassembly(lookup: Any, address: int) -> str | None:
        """Text for one program counter; NCU answers "N/A" for addresses it cannot place."""
        try:
            text = lookup(address)
        except Exception:
            return None
        text = str(text).strip() if text is not None else ""
        return text if text and text != "N/A" else None

    def instruction_rows(self, row_id: str, action: Any, counters: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
        rows = []
        for address in sorted(counters):
            file_name, line = self.location(action, address)
            rows.append({"key": f"{row_id}|pc:{address:#x}", "row_id": row_id, "address": f"{address:#x}",
                         "sass": self.sass(action, address), "file": file_name, "line": line,
                         "counters": dict(sorted(counters[address].items()))})
        return rows


def _call_or_get(obj: Any, attribute: str) -> Any:
    value = getattr(obj, attribute, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(attribute)
    return value() if callable(value) else value


def finite(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def plain_message(text: Any) -> str | None:
    """NCU rule text references GUI sections as @section:ID:Label@; keep the label."""
    if not text:
        return None
    return re.sub(r"@section:[^:@]*:([^@]*)@", r"\1 section", str(text)).strip()


# ---- verbs --------------------------------------------------------------------

def launches(report: Report, args: argparse.Namespace) -> dict[str, Any]:
    rows = [report.launch_row(index, action) for index, action in enumerate(report.actions)]
    return page(rows, args.limit, auxiliary={"report": str(report.path)})


def inspect(report: Report, args: argparse.Namespace) -> dict[str, Any]:
    action = report.action(args.row_id)
    if action is None:
        return page([{"type": "not_found", "row_id": args.row_id}], 1)
    return page([report.inspect_row(args.row_id, action)], 1)


def source_metrics(report: Report, args: argparse.Namespace) -> dict[str, Any]:
    action = require_action(report, args.row_id)
    counters = report.instruction_counters(action)
    names = sorted({name for values in counters.values() for name in values})
    if args.counter:
        counters = {address: {n: v for n, v in values.items() if fnmatchcase(n, args.counter)}
                    for address, values in counters.items()}
        counters = {address: values for address, values in counters.items() if values}
    rows = report.instruction_rows(args.row_id, action, counters)
    if args.file:
        rows = [row for row in rows if row["file"] and args.file in row["file"]]
    if args.line:
        rows = [row for row in rows if row["line"] == args.line]
    if args.by in ("line", "file"):
        rows = group_rows(rows, args.row_id, by=args.by)
    auxiliary = {"counters": names, "counter_filter": args.counter,
                 "instruction_attribution": bool(counters), "source_lines": any(row.get("file") for row in rows),
                 "note": ("No per-instruction counters in this report: capture with the SourceCounters section "
                          "and --import-source yes." if not counters else None)}
    return page(rows, args.limit, auxiliary={k: v for k, v in auxiliary.items() if v is not None})


def warp_stalls(report: Report, args: argparse.Namespace) -> dict[str, Any]:
    action = require_action(report, args.row_id)
    counters = report.instruction_counters(action)
    stalls: dict[int, dict[str, float]] = {}
    for address, values in counters.items():
        reasons = {name[len(STALL_PREFIX):]: value for name, value in values.items()
                   if name.startswith(STALL_PREFIX) and value}
        if reasons:
            stalls[address] = reasons
    total = sum(sum(reasons.values()) for reasons in stalls.values())
    if args.by == "reason":
        totals: dict[str, float] = defaultdict(float)
        for reasons in stalls.values():
            for reason, value in reasons.items():
                totals[reason] += value
        rows = [{"key": f"{args.row_id}|reason:{reason}", "row_id": args.row_id, "reason": reason, "samples": value,
                 "share": value / total if total else None} for reason, value in totals.items()]
        rows.sort(key=lambda row: -row["samples"])
    else:
        rows = []
        for address in sorted(stalls, key=lambda a: -sum(stalls[a].values())):
            file_name, line = report.location(action, address)
            rows.append({"key": f"{args.row_id}|pc:{address:#x}", "row_id": args.row_id, "address": f"{address:#x}",
                         "sass": report.sass(action, address), "file": file_name, "line": line,
                         "samples": sum(stalls[address].values()), "counters": dict(sorted(stalls[address].items()))})
        if args.by == "line":
            rows = group_rows(rows, args.row_id, by="line")
            for row in rows:
                row["samples"] = sum(row["counters"].values())
            rows.sort(key=lambda row: -row["samples"])
    auxiliary = {"total_samples": total, "sample_count_metric": report.scalar(action, SAMPLE_COUNT)}
    if not total:
        auxiliary["note"] = ("No warp-stall samples attributed to instructions in this report: capture with the "
                             "SourceCounters section (PC sampling).")
    return page(rows, args.limit, auxiliary=auxiliary)


def disasm(report: Report, args: argparse.Namespace) -> dict[str, Any]:
    action = require_action(report, args.row_id)
    addresses = sorted(report.instruction_counters(action))
    instructions = [{"address": f"{address:#x}", "sass": report.sass(action, address)} for address in addresses]
    instructions = [item for item in instructions if item["sass"]]
    rows = [{"key": f"{args.row_id}|function", "function_name": action.name(report.api.IAction.NameBase_MANGLED),
             "instructions": instructions}] if instructions else []
    ptx_lines = [{"address": f"{address:#x}", "ptx": report.ptx(action, address)} for address in addresses]
    ptx_lines = [item for item in ptx_lines if item["ptx"]]
    try:
        source_index = [{"file": str(name)} for name in action.source_files()]
    except Exception:
        source_index = []
    auxiliary: dict[str, Any] = {"ptx_lines": ptx_lines, "source_index": source_index, "instruction_count": len(instructions)}
    if not instructions:
        auxiliary["note"] = ("Disassembly is available only for program counters the report attributed counters to; "
                             "capture with the SourceCounters section.")
    return page(rows, len(rows) or 1, auxiliary=auxiliary)


def require_action(report: Report, row_id: str) -> Any:
    action = report.action(row_id)
    if action is None:
        raise ValueError(f"launch absent from report: {row_id}")
    return action


def group_rows(rows: list[dict[str, Any]], row_id: str, *, by: str) -> list[dict[str, Any]]:
    """Sum per-instruction counters by source line or file; unattributed instructions stay separate."""
    groups: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        group_key = (row.get("file"), row.get("line")) if by == "line" else (row.get("file"),)
        if group_key[0] is None:
            group_key = ("<unknown>", row["address"]) if by == "line" else ("<unknown>",)
        entry = groups.setdefault(group_key, {
            "key": f"{row_id}|{by}:" + ":".join(str(part) for part in group_key), "row_id": row_id,
            "file": group_key[0], **({"line": group_key[1]} if by == "line" else {}),
            "instructions": 0, "counters": defaultdict(float)})
        entry["instructions"] += 1
        for name, value in row["counters"].items():
            entry["counters"][name] += value
    result = list(groups.values())
    for entry in result:
        entry["counters"] = dict(sorted(entry["counters"].items()))
    return result


def page(rows: list[dict[str, Any]], limit: int, *, auxiliary: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"rows": rows[:limit], "count": min(len(rows), limit), "total_matched": len(rows),
            "auxiliary": auxiliary or {}}


VERBS = {"launches": launches, "inspect": inspect, "source-metrics": source_metrics,
         "warp-stalls": warp_stalls, "disasm": disasm}


def envelope(report: Path, data: dict[str, Any]) -> dict[str, Any]:
    return {"schema": ENVELOPE_SCHEMA, "source": {"kind": "ncu", "version": "v1", "reader": "kai-ncu-reader",
                                                  "report": str(report)}, "data": data}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kai-ncu-reader", description=__doc__.split("\n\n")[0])
    parser.add_argument("source", choices=["ncu"], help="Report family; only Nsight Compute reports are supported")
    parser.add_argument("verb", choices=sorted(VERBS))
    parser.add_argument("report", type=Path)
    parser.add_argument("--row-id", default="launch:0")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--by", choices=["line", "sass", "file", "reason"], default="line")
    parser.add_argument("--counter")
    parser.add_argument("--file")
    parser.add_argument("--line", type=int)
    args = parser.parse_args(argv)
    try:
        report = Report(args.report.resolve(strict=True))
        data = VERBS[args.verb](report, args)
        print(json.dumps(envelope(args.report, data), allow_nan=False))
        return 0
    except Exception as error:  # the caller reads the envelope, so every failure is reported there
        print(json.dumps({"error": {"type": type(error).__name__, "message": str(error)}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
