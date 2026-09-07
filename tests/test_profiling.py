from __future__ import annotations

import json
import sys

import pytest

from kai_light import profiling
from kai_light.cli import main
from kai_light.config import ProfileConfig, Resources
from kai_light.process import run_process
from kai_light.profile_csv import read_csv
from kai_light.profiling import ProfileReport
from kai_light.optimizer.prompts import OPTIMIZATION_JUDGE, messages


@pytest.fixture
def reader(tmp_path, monkeypatch):
    report = tmp_path / "capture.ncu-rep"
    report.write_bytes(b"synthetic NCU report")
    binary = tmp_path / "report-reader"
    binary.write_text("synthetic executable")
    monkeypatch.setattr(profiling.shutil, "which", lambda _: str(binary))
    instance = ProfileReport(report, identity={"implementation_fingerprint": "candidate-hash", "case_id": "case-a"})
    metrics = [{"name": f"sm__counter_{i:03d}.sum", "label": f"memory transaction count {i}",
                "unit": "sector", "value": i, "value_type": "uint64", "metric_type": "counter", "rollup": "sum"}
               for i in range(65)]
    metrics.append({"name": "sm__missing.sum", "label": None, "unit": "cycle", "value": None})
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs["gpu_device"] is None
        verb = command[2]
        if verb == "launches":
            rows = [{"key": "launch:0", "row_id": "launch:0", "kernel_demangled": "operator"},
                    {"key": "launch:1", "row_id": "launch:1", "kernel_demangled": "operator_helper"}]
            auxiliary = {}
        elif verb == "inspect":
            row_id = command[command.index("--row-id") + 1]
            rows = [{"type": "launch", "key": row_id, "row_id": row_id, "metrics": metrics,
                     "rules": [{"rule_identifier": "TailEffect", "rule_message": {"message": "Check tail wave", "title": "Tail"}}]}]
            auxiliary = {}
        elif verb == "disasm":
            rows = [{"function_name": "operator", "instructions": [{"address": i * 16, "opcode": "LDG"} for i in range(8)]}]
            auxiliary = {"source_lineinfo_present": True, "warnings": ["partial PTX"],
                         "ptx_lines": [{"text": "ld.global"}], "source_index": [{"file": "operator.cu", "line": 8}]}
        else:
            rows = [{"key": "launch:0|line:operator.cu:8", "line": 8, "counters": {"sm__counter_001.sum": 3}}]
            auxiliary = {"skipped_counters": [{"name": "sm__missing.sum", "reason": "not-a-source-counter"}],
                         "unattributed_sass_counter_totals": {"sm__counter_001.sum": 9},
                         "warnings": ["missing source lines"], "total_samples": 17}
        envelope = {"schema": "v1", "source": {"kind": "ncu", "version": "v1"},
                    "data": {"count": len(rows), "total_matched": len(rows), "rows": rows, "auxiliary": auxiliary}}
        kwargs["stdout"].write_text(json.dumps(envelope))
        return {"status": "completed", "returncode": 0}

    monkeypatch.setattr(profiling, "run_process", execute)
    return instance, calls


def test_catalog_searches_descriptions_and_pages_all_metrics_beyond_24(reader):
    report, calls = reader
    names, offset = [], 0
    while True:
        result = report.query({"operation": "catalog", "query": "memory transaction", "limit": 20, "offset": offset})
        assert result["status"] == "available"
        assert result["identity"]["implementation_fingerprint"] == "candidate-hash"
        data = result["data"]
        assert data["total_matched"] == 65
        names.extend(row["name"] for row in data["rows"])
        assert data["rows"][0]["description"] and data["rows"][0]["unit"] == "sector"
        if data["next_offset"] is None:
            break
        offset = data["next_offset"]
    assert len(set(names)) == 65
    assert len(calls) == 2


def test_overview_defers_rule_details_until_requested(reader):
    report, _ = reader
    overview = report.overview()
    assert overview["status"] == "available"
    assert "rules" not in overview and "Check tail wave" not in json.dumps(overview)
    assert overview["rule_count"] == 1 and overview["metric_count"] == 66
    assert "rules" in overview["available_operations"]
    details = report.query({"operation": "rules"})
    assert details["data"]["rows"][0]["rule_message"]["message"] == "Check tail wave"


def test_field_selection_preserves_measurements_units_identity_and_full_cache(reader):
    report, calls = reader
    request = {"operation": "metrics", "counter": "sm__counter_000.sum", "fields": ["value"]}
    selected = report.query(request)
    assert selected["status"] == "available"
    row = selected["data"]["rows"][0]
    assert row["value"] == 0 and row["status"] == "measured" and row["unit"] == "sector"
    assert row["name"] == request["counter"] and row["row_id"] == "launch:0" and row["key"]
    assert "description" not in row
    assert "description" in selected["data"]["auxiliary"]["available_fields"]
    full = report.query({k: v for k, v in request.items() if k != "fields"})
    assert full["data"]["rows"][0]["description"] == "memory transaction count 0"
    assert len(calls) == 2


def test_source_field_selection_retains_attribution_and_coverage(reader):
    report, _ = reader
    result = report.query({"operation": "source-metrics", "counter": "sm__counter_001.sum", "fields": ["line"]})
    assert result["status"] == "available"
    assert result["data"]["rows"][0]["counters"] == {"sm__counter_001.sum": 3}
    assert result["data"]["auxiliary"]["unattributed_sass_counter_totals"] == {"sm__counter_001.sum": 9}
    assert result["data"]["auxiliary"]["skipped_counters"]


def test_unknown_projection_field_reports_available_fields(reader):
    report, _ = reader
    result = report.query({"operation": "catalog", "fields": ["not_a_field"]})
    assert result["status"] == "unavailable"
    assert "available fields" in result["error"]["message"]
    assert "description" in result["error"]["message"]


def test_next_query_preserves_filters_and_projection_after_size_reduction(reader):
    report, _ = reader
    report.settings = ProfileConfig(max_result_chars=4000)
    request = {"operation": "catalog", "query": "memory", "fields": ["name"], "limit": 30}
    first = report.query(request)
    assert first["status"] == "available"
    page = first["data"]
    assert 0 < page["count"] < 30
    assert page["next_query"]["offset"] == page["count"]
    assert all(page["next_query"][key] == value for key, value in request.items())
    second = report.query(page["next_query"])
    assert second["status"] == "available"
    assert not {row["key"] for row in page["rows"]} & {row["key"] for row in second["data"]["rows"]}
    assert len(json.dumps(first)) <= 4000 and len(json.dumps(second)) <= 4000


def test_empty_catalog_suggests_separate_search_words(reader):
    report, _ = reader
    result = report.query({"operation": "catalog", "query": "memory occupancy"})
    assert result["data"]["count"] == 0
    suggestions = result["data"]["auxiliary"]["suggested_queries"]
    assert [query["query"] for query in suggestions] == ["memory", "occupancy"]
    assert report.query(suggestions[0])["data"]["total_matched"] == 65


def test_missing_metric_suggests_real_names_without_inventing_values(reader):
    report, _ = reader
    result = report.query({"operation": "metrics", "counter": "sm__counter_001.su"})
    assert result["status"] == "available" and result["data"]["rows"] == []
    auxiliary = result["data"]["auxiliary"]
    assert "similarity" in auxiliary["suggestion_basis"]
    assert auxiliary["suggested_queries"]
    for query in auxiliary["suggested_queries"]:
        assert query["operation"] == "catalog"
        assert report.query(query)["data"]["count"] == 1


def test_queries_preserve_zero_missing_unknown_and_launch_identity(reader):
    report, _ = reader
    zero = report.query({"operation": "metrics", "row_id": "launch:1", "counter": "sm__counter_000.sum"})["data"]["rows"][0]
    assert zero["value"] == 0 and zero["status"] == "measured" and zero["row_id"] == "launch:1"
    missing = report.query({"operation": "metrics", "counter": "sm__missing.sum"})["data"]["rows"][0]
    assert missing["value"] is None and missing["status"] == "missing"
    absent = report.query({"operation": "metrics", "counter": "not_collected*"})
    assert absent["data"]["total_matched"] == 0 and absent["data"]["auxiliary"]["empty_result_hint"]
    invalid = report.query({"operation": "catalog", "row_id": "launch:99"})
    assert invalid["status"] == "unavailable" and "unknown launch" in invalid["error"]["message"]


def test_cache_reused_across_readers_and_modified_report_rejected(reader):
    report, calls = reader
    request = {"operation": "rules"}
    assert report.query(request)["data"]["rows"][0]["rule_identifier"] == "TailEffect"
    assert len(calls) == 2
    another = ProfileReport(report.report)
    assert another.query(request)["status"] == "available" and len(calls) == 2
    report.report.write_bytes(b"different report")
    assert report.query(request)["status"] == "unavailable"
    assert another.query(request)["status"] == "unavailable"
    assert ProfileReport(report.report).query(request)["status"] == "available"
    assert len(calls) == 4


def test_source_query_keeps_skipped_counters_warnings_and_unattributed_totals(reader):
    report, calls = reader
    result = report.query({"operation": "source-metrics", "counter": "sm__counter_001.sum", "file": "operator.cu", "line": 8})
    assert result["status"] == "available"
    auxiliary = result["data"]["auxiliary"]
    assert auxiliary["skipped_counters"][0]["reason"] == "not-a-source-counter"
    assert auxiliary["unattributed_sass_counter_totals"]["sm__counter_001.sum"] == 9
    assert auxiliary["warnings"] == ["missing source lines"]
    assert "--file" in calls[-1] and "--line" in calls[-1]


def test_absent_timed_samples_are_not_reported_as_evidence_of_zero_stalls(reader, monkeypatch):
    report, _ = reader
    original = report._reader_query
    def query(verb, args):
        if verb == "warp-stalls":
            return {"data": {"count": 0, "total_matched": 0, "rows": [], "auxiliary": {"total_samples": 0}}}
        return original(verb, args)
    monkeypatch.setattr(report, "_reader_query", query)
    result = report.query({"operation": "warp-stalls", "by": "reason"})
    assert result["data"]["auxiliary"]["evidence_status"] == "no_samples"
    assert "does not establish zero stalls" in result["data"]["auxiliary"]["hint"]


@pytest.mark.parametrize("view,field", [("sass", "opcode"), ("ptx", "text"), ("source", "file")])
def test_disassembly_has_paged_instruction_ptx_and_source_views(reader, view, field):
    report, _ = reader
    result = report.query({"operation": "disasm", "view": view, "limit": 2})
    assert field in result["data"]["rows"][0]
    assert result["data"]["auxiliary"]["warnings"] == ["partial PTX"]
    if view == "sass":
        assert result["data"]["next_offset"] == 2 and result["data"]["total_matched"] == 8


@pytest.mark.parametrize("symbol_present", [True, False])
def test_sass_selects_launch_symbol_before_paginating_shared_cubin(reader, monkeypatch, symbol_present):
    report, _ = reader
    assert report.query({"operation": "launches"})["status"] == "available"
    report.launches[0]["kernel_mangled"] = "_Z8operatorv"
    original = report._reader_query

    def query(verb, args):
        if verb != "disasm":
            return original(verb, args)
        rows = [{"function_name": "unrelated_kernel", "instructions": [{"opcode": "WRONG"}]}]
        if symbol_present:
            rows.append({"function_name": "_Z8operatorv", "instructions": [
                {"address": i * 16, "opcode": "RIGHT"} for i in range(5)]})
        return {"data": {"rows": rows, "auxiliary": {}}}

    monkeypatch.setattr(report, "_reader_query", query)
    result = report.query({"operation": "disasm", "row_id": "launch:0", "offset": 2, "limit": 2})
    if not symbol_present:
        assert result["status"] == "unavailable"
        assert "launch symbol" in result["error"]["message"]
        return
    assert result["status"] == "available"
    data = result["data"]
    assert data["total_matched"] == 5 and data["next_offset"] == 4
    assert [row["address"] for row in data["rows"]] == [32, 48]
    assert all(row["kernel"] == "_Z8operatorv" for row in data["rows"])
    assert data["auxiliary"]["evidence_scope"] == "launch_function"


@pytest.mark.parametrize("view", ["ptx", "source"])
def test_disassembly_indexes_explicitly_describe_cubin_scope(reader, view):
    report, _ = reader
    result = report.query({"operation": "disasm", "view": view})
    assert result["data"]["auxiliary"]["evidence_scope"] == "cubin"


@pytest.mark.parametrize("query_request", [
    {"operation": "shell", "command": "anything"},
    {"operation": "catalog", "report": "/another/report"},
    {"operation": "catalog", "row_id": "../report"},
    {"operation": "metrics"},
    {"operation": "metrics", "counter": "a,b"},
    {"operation": "warp-stalls", "by": "file"},
    {"operation": "source-metrics", "by": "reason", "counter": "*"},
    {"operation": "catalog", "limit": 0},
    {"operation": "catalog", "fields": ["rows[0].name"]},
])
def test_query_validation_returns_recoverable_errors_without_commands(reader, query_request):
    report, calls = reader
    assert report.query(query_request)["status"] == "unavailable"
    assert calls == []


def test_csv_fallback_has_typed_values_and_explicit_missing_capabilities(tmp_path, monkeypatch):
    csv = tmp_path / "metrics.csv"
    csv.write_text('ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n'
                   '0,op,gpu__time_duration.sum,nsecond,42\n'
                   '0,op,dram__bytes.sum,byte,NaN\n'
                   '1,op,gpu__time_duration.sum,nsecond,11\n')
    monkeypatch.setattr(profiling.shutil, "which", lambda _: None)
    report = ProfileReport(tmp_path / "capture.ncu-rep", csv_path=csv)
    metric = report.query({"operation": "metrics", "counter": "gpu__time_duration.sum", "row_id": "launch:1"})
    assert metric["backend"] == "csv" and metric["data"]["rows"][0]["value"] == 11
    assert metric["warnings"] and metric["data"]["rows"][0]["description"] is None
    assert report.query({"operation": "rules"})["status"] == "unavailable"
    assert report.query({"operation": "disasm"})["status"] == "unavailable"
    assert report.overview()["metric_count"] == 2
    csv.write_text(csv.read_text() + "changed")
    assert report.query({"operation": "catalog"})["status"] == "unavailable"


def test_csv_wide_preserves_unit_rows_and_derived_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text('==PROF== connected\n"ID","Kernel Name","CC","dram__bytes.sum","Section.derived"\n'
                    '"","","","byte","%"\n"0","op","12.0","1,024","50"\n')
    metrics = read_csv(path)[0]["metrics"]
    assert metrics[0]["value"] == 1024 and metrics[0]["unit"] == "byte"
    assert metrics[1]["name"] == "Section.derived" and metrics[1]["unit"] == "%"


def test_csv_without_ids_rejects_ambiguous_repeated_launches(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text('Kernel Name,Metric Name,Metric Value\nop,gpu__time_duration.sum,4\nop,gpu__time_duration.sum,5\n')
    with pytest.raises(ValueError, match="ambiguous"):
        read_csv(path)


def test_csv_preserves_large_integer_counts_and_rejects_partial_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("ID,Kernel Name,Metric Name,Metric Value\n0,op,dram__bytes.sum,9007199254740993\n")
    assert read_csv(path)[0]["metrics"][0]["value"] == 9007199254740993
    with path.open("a") as handle:
        handle.write("1,op,dram__bytes.sum\n")
    with pytest.raises(ValueError, match="incomplete"):
        read_csv(path)


def test_profile_cli_queries_csv_outside_optimizer(tmp_path, monkeypatch, capsys):
    path = tmp_path / "metrics.csv"
    path.write_text('ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n0,op,gpu__time_duration.sum,nsecond,42\n')
    monkeypatch.setattr(profiling.shutil, "which", lambda _: None)
    assert main(["profile", str(tmp_path / "capture.ncu-rep"), "--csv", str(path), "--request",
                 '{"operation":"metrics","counter":"gpu__time_duration.sum"}']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["data"]["rows"][0]["value"] == 42


def test_report_subprocess_separates_json_stdout_from_diagnostics(tmp_path):
    stdout = tmp_path / "stdout.json"
    result = run_process([sys.executable, "-c", 'import sys; print(\'{"data": 3}\'); print("diagnostic", file=sys.stderr)'],
        cwd=tmp_path, log=tmp_path / "stderr.log", stdout=stdout, timeout=5, resources=Resources(), gpu_device=None)
    assert result["status"] == "completed" and json.loads(stdout.read_text()) == {"data": 3}
    assert "diagnostic" in result["log_tail"] and "data" not in result["log_tail"]


def test_profile_result_budget_pages_primary_rows(reader):
    report, _ = reader
    report.settings = ProfileConfig(max_result_chars=4000)
    result = report.query({"operation": "catalog", "limit": 30})
    assert len(json.dumps(result)) <= 4000
    assert result["data"]["count"] < 30 and result["data"]["next_offset"]


def test_query_timeout_is_reported(reader, monkeypatch):
    report, _ = reader
    monkeypatch.setattr(profiling, "run_process", lambda *a, **k: {"status": "timeout"})
    result = report.query({"operation": "catalog"})
    assert result["status"] == "unavailable" and "timeout" in str(result["warnings"])


def test_unsupported_reader_schema_is_not_consumed(reader, monkeypatch):
    report, _ = reader
    def execute(*args, **kwargs):
        kwargs["stdout"].write_text('{"schema":"v2","source":{"kind":"ncu","version":"v1"},"data":{"rows":[]}}')
        return {"status": "completed", "returncode": 0}
    monkeypatch.setattr(profiling, "run_process", execute)
    result = report.query({"operation": "catalog"})
    assert result["status"] == "unavailable" and "unsupported" in str(result["warnings"])


@pytest.mark.parametrize("operation", ["catalog", "disasm"])
def test_reader_failure_uses_generic_prompt_diagnostics_but_keeps_raw_artifacts(reader, monkeypatch, operation):
    report, _ = reader
    if operation == "disasm":
        assert report.query({"operation": "catalog"})["status"] == "available"
    def execute(*args, **kwargs):
        kwargs["stdout"].write_text(json.dumps({"error": {"code": "ncu.input.missing",
            "message": "kai-ncu-reader: cache /tmp/report.ncu-rep.kai-ncu-reader/cache unavailable; set KAI_LIGHT_REPORT_READER_DIR"}}))
        return {"status": "completed", "returncode": 1}
    monkeypatch.setattr(profiling, "run_process", execute)
    result = report.query({"operation": operation})
    prompt = messages(OPTIMIZATION_JUDGE, {"hardware_feedback": result})
    assert "kai-ncu-reader" not in json.dumps(prompt).lower()
    assert result["status"] == "unavailable"
    assert "unavailable" in json.dumps(result) and "profile.ncu_report_dir" in json.dumps(result)
    assert any("kai-ncu-reader:" in path.read_text() for path in report.cache.glob("query-*/stdout.json"))


def test_reader_warning_normalization_keeps_source_identity_and_counters(reader, monkeypatch):
    report, _ = reader
    original = report._reader_query
    def query(verb, args):
        response = original(verb, args)
        if verb == "source-metrics":
            response["data"]["auxiliary"]["warnings"] = ["kai-ncu-reader: partial line information"]
        return response
    monkeypatch.setattr(report, "_reader_query", query)
    result = report.query({"operation": "source-metrics", "counter": "sm__counter_001.sum"})
    assert "kai-ncu-reader" not in json.dumps(messages(OPTIMIZATION_JUDGE, {"profile_query_results": result})).lower()
    assert "partial line information" in result["data"]["auxiliary"]["warnings"][0]
    assert result["data"]["rows"][0]["counters"] == {"sm__counter_001.sum": 3}
    assert result["identity"]["implementation_fingerprint"] == "candidate-hash"
