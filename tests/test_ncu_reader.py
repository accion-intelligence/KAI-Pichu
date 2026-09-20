"""The packaged NCU report reader, against a fake ncu_report API and through the profile layer."""
from __future__ import annotations

import json
from pathlib import Path
import stat
import sys
import types

import pytest

from kai_pichu import ncu_reader
from kai_pichu.config import ProfileConfig
from kai_pichu.profiling import ProfileReport

PC = 0xb011ab000  # program counters are large; occupancy-curve ids are small


FAKE_API = '''
class IMetric:
    ValueKind_DOUBLE, ValueKind_FLOAT, ValueKind_UINT32, ValueKind_UINT64, ValueKind_STRING = 1, 2, 3, 4, 5
    MetricType_COUNTER, MetricType_RATIO, MetricType_THROUGHPUT, MetricType_OTHER = 1, 2, 3, 4
    RollupOperation_AVG, RollupOperation_SUM, RollupOperation_NONE = 1, 2, 0

    def __init__(self, name, value=None, unit="", description="", kind=1, mtype=1, rollup=2, ids=None, values=None):
        self._name, self._value, self._unit, self._desc = name, value, unit, description
        self._kind, self._mtype, self._rollup = kind, mtype, rollup
        self._ids, self._values = ids, values or []

    def name(self): return self._name
    def description(self): return self._desc
    def unit(self): return self._unit
    def value(self): return self._value
    def kind(self): return self._kind
    def metric_type(self): return self._mtype
    def rollup_operation(self): return self._rollup
    def num_instances(self): return len(self._values)
    def has_correlation_ids(self): return self._ids is not None
    def correlation_ids(self): return self._ids
    def as_double(self, i=None): return self._values[i] if i is not None else self._value
    def as_uint64(self, i=None): return int(self._values[i]) if i is not None else int(self._value)
    def as_string(self, i=None): return str(self._value)


MsgType_OPTIMIZATION = 2


class _Ids:
    def __init__(self, values, kind): self._values, self._kind = values, kind
    def kind(self): return self._kind
    def as_uint64(self, i): return self._values[i]


class _Source:
    def __init__(self, f, l): self._f, self._l = f, l
    def file_name(self): return self._f
    def line(self): return self._l


class IAction:
    NameBase_DEMANGLED, NameBase_MANGLED, NameBase_FUNCTION = 0, 1, 2
    PCS = [0xb011ab000, 0xb011ab010, 0xb011ab020]

    def __init__(self):
        pcs = _Ids(self.PCS, IMetric.ValueKind_UINT64)
        curve = _Ids([32, 64, 128], IMetric.ValueKind_UINT64)
        self._metrics = {m.name(): m for m in [
            IMetric("gpu__time_duration.sum", 41536.0, "ns", "duration", kind=IMetric.ValueKind_DOUBLE),
            IMetric("launch__grid_dim_x", 64, "", "", kind=IMetric.ValueKind_UINT32),
            IMetric("launch__grid_dim_y", 1, kind=IMetric.ValueKind_UINT32), IMetric("launch__grid_dim_z", 1, kind=IMetric.ValueKind_UINT32),
            IMetric("launch__block_dim_x", 256, kind=IMetric.ValueKind_UINT32), IMetric("launch__block_dim_y", 1, kind=IMetric.ValueKind_UINT32),
            IMetric("launch__block_dim_z", 1, kind=IMetric.ValueKind_UINT32),
            IMetric("launch__registers_per_thread", 64, "register/thread", kind=IMetric.ValueKind_UINT32),
            IMetric("sm__throughput.avg.pct_of_peak_sustained_elapsed", float("nan"), "%", "SM throughput", kind=IMetric.ValueKind_DOUBLE),
            IMetric("breakdown:gpu__x", "a,b", kind=IMetric.ValueKind_STRING, mtype=IMetric.MetricType_OTHER),
            IMetric("derived__pct_occupancy_per_block_size", None, "%", "occupancy curve", kind=IMetric.ValueKind_DOUBLE, ids=curve, values=[50.0, 75.0, 100.0]),
            IMetric("inst_executed", 1536, "inst", "executed warp instructions", kind=IMetric.ValueKind_UINT64, ids=pcs, values=[512, 512, 512]),
            IMetric("smsp__pcsamp_warps_issue_stalled_long_scoreboard", None, "", "", kind=IMetric.ValueKind_UINT64, ids=pcs, values=[100, 0, 20]),
            IMetric("smsp__pcsamp_warps_issue_stalled_wait", None, "", "", kind=IMetric.ValueKind_UINT64, ids=pcs, values=[5, 5, 0]),
        ]}

    def name(self, base): return ["kernel(int*)", "_Z6kernelPi", "kernel"][base]
    def metric_names(self): return list(self._metrics)
    def metric_by_name(self, name): return self._metrics[name]
    def rule_results_as_dicts(self):
        return [{"rule_identifier": "SOLBottleneck", "name": "Bottleneck", "section_identifier": "SpeedOfLight",
                 "rule_message": {"title": "Small Grid", "message": "Look at @section:LaunchStats:Launch Statistics@.", "type": 2},
                 "focus_metrics": [{"name": "launch__waves_per_multiprocessor", "value": 0.33, "severity": 2, "info": "more waves"}],
                 "speedup_estimation": {"type": 1, "speedup": 49.9}}]
    def sass_by_pc(self, pc): return {0xb011ab000: "LDC R1, c[0x0][0x37c]", 0xb011ab010: "SHFL.IDX PT, R59, R22, R51, 0x1f", 0xb011ab020: "EXIT"}.get(pc, "N/A")
    def ptx_by_pc(self, pc): return "ld.param" if pc == 0xb011ab000 else "N/A"
    def source_info(self, pc): return _Source("solution.cu", 42) if pc == 0xb011ab010 else _Source("solution.cu", 7)
    def source_files(self): return ["solution.cu"]


class _Range:
    def num_actions(self): return 1
    def action_by_idx(self, i): return IAction()


class _Context:
    def num_ranges(self): return 1
    def range_by_idx(self, i): return _Range()


def load_report(path):
    return _Context()
'''


@pytest.fixture
def fake_api_dir(tmp_path):
    (tmp_path / "ncu_report.py").write_text(FAKE_API)
    return tmp_path


@pytest.fixture
def report(fake_api_dir, monkeypatch):
    module = types.ModuleType("fake_ncu_report")
    exec(FAKE_API, module.__dict__)
    monkeypatch.setattr(ncu_reader, "load_api", lambda: module)
    path = fake_api_dir / "capture.ncu-rep"
    path.write_bytes(b"report")
    return path


def run(argv):
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = ncu_reader.main(argv)
    return code, json.loads(out.getvalue())


def test_launches_and_inspect_describe_the_report(report):
    code, value = run(["ncu", "launches", str(report)])
    assert code == 0 and value["schema"] == "v1" and value["source"] == {
        "kind": "ncu", "version": "v1", "reader": "kai-ncu-reader", "report": str(report)}
    row = value["data"]["rows"][0]
    assert row["row_id"] == "launch:0" and row["kernel_mangled"] == "_Z6kernelPi" and row["grid"] == [64, 1, 1]
    assert row["duration_ns"] == 41536.0 and row["registers_per_thread"] == 64 and "metrics" not in row

    code, value = run(["ncu", "inspect", str(report), "--row-id", "launch:0"])
    detail = value["data"]["rows"][0]
    names = [m["name"] for m in detail["metrics"]]
    assert "breakdown:gpu__x" not in names and "inst_executed" in names
    nan_metric = next(m for m in detail["metrics"] if m["name"].startswith("sm__throughput"))
    assert nan_metric["value"] is None and nan_metric["unit"] == "%" and nan_metric["value_type"] == "double"
    rule = detail["rules"][0]
    assert rule["identifier"] == "SOLBottleneck" and rule["severity"] == "optimization"
    assert rule["message"] == "Look at Launch Statistics section."
    assert rule["estimated_speedup_pct"] == 49.9 and rule["focus_metrics"][0]["hint"] == "more waves"

    code, value = run(["ncu", "inspect", str(report), "--row-id", "launch:7"])
    assert value["data"]["rows"][0]["type"] == "not_found"


def test_per_instruction_evidence_ignores_occupancy_curves(report):
    code, value = run(["ncu", "source-metrics", str(report), "--row-id", "launch:0", "--by", "sass", "--limit", "10"])
    data = value["data"]
    assert data["total_matched"] == 3 and data["auxiliary"]["instruction_attribution"] is True
    assert "derived__pct_occupancy_per_block_size" not in data["auxiliary"]["counters"]
    first = data["rows"][0]
    assert first["address"] == hex(PC) and first["sass"].startswith("LDC") and first["file"] == "solution.cu"
    assert first["counters"]["inst_executed"] == 512

    code, value = run(["ncu", "source-metrics", str(report), "--row-id", "launch:0", "--by", "line", "--counter", "inst_executed"])
    lines = {(row["file"], row["line"]): row for row in value["data"]["rows"]}
    assert lines[("solution.cu", 7)]["counters"] == {"inst_executed": 1024} and lines[("solution.cu", 7)]["instructions"] == 2
    assert lines[("solution.cu", 42)]["counters"] == {"inst_executed": 512}


def test_warp_stalls_by_reason_sass_and_line(report):
    code, value = run(["ncu", "warp-stalls", str(report), "--row-id", "launch:0", "--by", "reason"])
    data = value["data"]
    assert data["auxiliary"]["total_samples"] == 130
    assert data["rows"][0] == {"key": "launch:0|reason:long_scoreboard", "row_id": "launch:0", "reason": "long_scoreboard",
                               "samples": 120, "share": pytest.approx(120 / 130)}
    code, value = run(["ncu", "warp-stalls", str(report), "--row-id", "launch:0", "--by", "sass", "--limit", "1"])
    top = value["data"]["rows"][0]
    assert top["address"] == hex(PC) and top["samples"] == 105 and top["counters"] == {"long_scoreboard": 100, "wait": 5}
    code, value = run(["ncu", "warp-stalls", str(report), "--row-id", "launch:0", "--by", "line"])
    assert value["data"]["rows"][0]["line"] == 7 and value["data"]["rows"][0]["samples"] == 125


def test_disasm_lists_the_launch_function_only_at_attributed_program_counters(report):
    code, value = run(["ncu", "disasm", str(report), "--row-id", "launch:0"])
    data = value["data"]
    function = data["rows"][0]
    assert function["function_name"] == "_Z6kernelPi" and len(function["instructions"]) == 3
    assert function["instructions"][1]["sass"].startswith("SHFL.IDX")
    assert data["auxiliary"]["ptx_lines"] == [{"address": hex(PC), "ptx": "ld.param"}]
    assert data["auxiliary"]["source_index"] == [{"file": "solution.cu"}]


def test_errors_are_reported_in_the_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(ncu_reader, "load_api", lambda: (_ for _ in ()).throw(ImportError("cannot find NVIDIA ncu_report.py")))
    path = tmp_path / "capture.ncu-rep"
    path.write_bytes(b"report")
    code, value = run(["ncu", "launches", str(path)])
    assert code == 1 and value["error"]["type"] == "ImportError" and "ncu_report" in value["error"]["message"]


def test_profile_layer_uses_the_packaged_reader_end_to_end(fake_api_dir, tmp_path):
    """ProfileReport -> kai-ncu-reader subprocess -> the ncu_compat shim -> a fake ncu_report found via profile.ncu_report_dir."""
    report = tmp_path / "capture.ncu-rep"
    report.write_bytes(b"report")
    launcher = tmp_path / "kai-ncu-reader"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} -m kai_pichu.ncu_reader \"$@\"\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    settings = ProfileConfig(report_reader=str(launcher), ncu_report_dir=str(fake_api_dir))
    view = ProfileReport(report, settings=settings)
    assert view.backend == "ncu_report"
    overview = view.overview(timeout=60)
    assert overview["status"] == "available" and overview["rule_count"] == 1
    assert "rules" in overview["available_operations"] and "warp-stalls" in overview["available_operations"]
    rules = view.query({"operation": "rules"}, timeout=60)
    assert rules["data"]["rows"][0]["title"] == "Small Grid"
    stalls = view.query({"operation": "warp-stalls", "by": "reason"}, timeout=60)
    assert stalls["status"] == "available" and stalls["data"]["rows"][0]["reason"] == "long_scoreboard"
    assert stalls["data"]["auxiliary"]["total_samples"] == 130
    metrics = view.query({"operation": "metrics", "counter": "launch__registers_per_thread"}, timeout=60)
    assert metrics["data"]["rows"][0]["value"] == 64
