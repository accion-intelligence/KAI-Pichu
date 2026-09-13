from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from kai_core.benchmark import Case, Observation, Validation
from kai_core.benchmark.cli import main
from kai_core.benchmark.loading import file_inventory, load_python_file
from kai_core.benchmark.models import BenchmarkSpec, finite_number
from kai_core.benchmark.runner import BenchmarkError, Runner
from kai_core.benchmark.statistics import summarize


ADAPTER = '''
from kai_core.benchmark import Benchmark, Case, Observation, Validation, json_fingerprint, load_python_file
class Task(Benchmark):
    def cases(self, split):
        return [Case(id="one", params={"x": 3}), Case(id="two", params={"x": 7}, weight=2)]
    def prepare(self, case, seed):
        return {"x": case.params["x"] + seed, "state": 0}
    def load_implementation(self, workspace):
        return load_python_file(workspace / "solution.py")
    def reset(self, implementation, fixture):
        fixture["state"] = 0
    def run(self, implementation, fixture):
        fixture["state"] += fixture["x"]
        return Observation(fixture["state"] * implementation.FACTOR,
                           {"cost": implementation.COST, "memory": implementation.MEMORY})
    def validate(self, case, fixture, observation):
        return Validation(observation.output == fixture["x"] and fixture["state"] == fixture["x"], "wrong output/state")
    def invalid_observations(self, case, fixture, valid):
        return [Observation(valid.output + 1)]
'''


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    (root / "adapter.py").write_text(ADAPTER)
    (root / "solution.py").write_text("FACTOR = 1\nCOST = 10.0\nMEMORY = 100\n")
    spec = {
        "schema_version": 1, "name": "fixture", "description": "Synthetic metrics for SDK tests only",
        "adapter": "adapter.py:Task", "implementation": {"root": ".", "files": ["solution.py"]},
        "objective": {"metric": "cost", "unit": "test_units", "direction": "minimize", "scope": "module", "target_speedup": 1.2},
        "measurement": {"timer": "wall", "boundary": "One complete update", "cache_policy": "test",
                        "blocks": 4, "iterations": 2, "warmup": 1, "bootstrap_samples": 200},
    }
    manifest = root / "benchmark.yaml"
    manifest.write_text(yaml.safe_dump(spec))
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "solution.py").write_text("FACTOR = 1\nCOST = 5.0\nMEMORY = 100\n")
    return manifest, candidate, spec


def test_benchmark_custom_metric_and_stateful_lifecycle(bundle):
    manifest, candidate, _ = bundle
    result = Runner(manifest).run(candidate)
    assert result["status"] == "completed"
    assert result["calibration"]["passed"]
    assert result["comparison"]["overall"]["speedup"] == pytest.approx(2)
    assert result["acceptance"]["accepted"]
    assert len(result["records"]) == 4 * 4 * 2
    assert result["baseline"]["implementation_fingerprint"] != result["candidate"]["implementation_fingerprint"]
    assert len(result["input_fingerprints"]) == 2


def test_benchmark_wrong_candidate_fails_before_performance(bundle):
    manifest, candidate, _ = bundle
    (candidate / "solution.py").write_text("FACTOR = 0\nCOST = 0.1\nMEMORY = 1\n")
    with pytest.raises(BenchmarkError, match="correctness failed"):
        Runner(manifest).run(candidate)


def test_benchmark_fixture_cost_does_not_become_implementation_speedup(bundle):
    """Buffer placement can change latency even when input values are identical."""
    runner = Runner(bundle[0])
    original_prepare = runner.task.prepare
    original_run = runner.task.run
    placement_cost = {}
    preparations = 0

    def prepare(case, seed):
        nonlocal preparations
        fixture = original_prepare(case, seed)
        placement_cost[id(fixture)] = 3.0 if preparations % 2 == 0 else 1.0
        preparations += 1
        return fixture

    def run(implementation, fixture):
        result = original_run(implementation, fixture)
        result.metrics["cost"] *= placement_cost[id(fixture)]
        return result

    runner.task.prepare = prepare
    runner.task.run = run
    result = runner.run(bundle[1])
    assert result["calibration"]["passed"]
    assert result["comparison"]["overall"]["interval"] == pytest.approx([2.0, 2.0])
    for block in range(4):
        for case in runner.cases:
            for arm in ("a", "b"):
                slots = [row["fixture_slot"] for row in result["records"]
                         if row["block"] == block and row["case_id"] == case.id and row["arm"] == arm]
                assert sorted(slots) == [0, 1]


def test_benchmark_validator_must_reject_mutations(bundle):
    runner = Runner(bundle[0])
    runner.task.validate = lambda *args: Validation(True)
    with pytest.raises(BenchmarkError, match="accepted invalid"):
        runner.validate(checks_only=True)


def test_benchmark_missing_mutation_probes_fail(bundle):
    runner = Runner(bundle[0])
    runner.task.invalid_observations = lambda *args: []
    with pytest.raises(BenchmarkError, match="probe is required"):
        runner.validate(checks_only=True)


def test_benchmark_missing_reset_detected(bundle):
    runner = Runner(bundle[0])
    runner.task.reset = lambda *args: None
    with pytest.raises(BenchmarkError, match="reset"):
        runner.validate(checks_only=True)


def test_benchmark_nondeterministic_inputs_detected(bundle):
    runner = Runner(bundle[0])
    count = 0

    def prepare(case, seed):
        nonlocal count
        count += 1
        return {"x": count, "state": 0}

    runner.task.prepare = prepare
    with pytest.raises(BenchmarkError, match="different input"):
        runner.validate(checks_only=True)


def test_benchmark_cleanup_on_verification_error(bundle):
    runner = Runner(bundle[0])
    cleaned = []
    runner.task.cleanup_implementation = lambda impl: cleaned.append("implementation")
    runner.task.cleanup_fixture = lambda fixture: cleaned.append("fixture")
    runner.task.validate = lambda *args: Validation(False)
    with pytest.raises(BenchmarkError):
        runner.validate(checks_only=True)
    assert cleaned == ["fixture", "fixture", "implementation"]


def test_benchmark_checks_only_does_not_claim_readiness(bundle):
    result = Runner(bundle[0]).validate(checks_only=True)
    assert result["status"] == "checks_passed"
    assert result["ready"] is False
    assert "calibration" not in result


def test_benchmark_task_mutation_invalidates_measurement(bundle):
    runner = Runner(bundle[0])
    original = runner.task.run

    def run(*args):
        observation = original(*args)
        with (bundle[0].parent / "adapter.py").open("a") as handle:
            handle.write("\n# changed\n")
        return observation

    runner.task.run = run
    with pytest.raises(BenchmarkError, match="benchmark_fingerprint changed"):
        runner.validate(checks_only=True)


def test_benchmark_source_mutation_invalidates_measurement(bundle):
    runner = Runner(bundle[0])
    original = runner.task.run

    def run(*args):
        observation = original(*args)
        with (bundle[0].parent / "solution.py").open("a") as handle:
            handle.write("\n# changed\n")
        return observation

    runner.task.run = run
    with pytest.raises(BenchmarkError, match="implementation_fingerprint changed"):
        runner.validate(checks_only=True)


def test_benchmark_candidate_cannot_override_sdk_clock(bundle):
    runner = Runner(bundle[0])
    original = runner.task.run

    def run(*args):
        observation = original(*args)
        observation.metrics["latency_ms"] = 0.001
        return observation

    runner.task.run = run
    with pytest.raises(BenchmarkError, match="SDK-owned"):
        runner.validate(checks_only=True)


def test_benchmark_limits_inspect_raw_samples(bundle):
    manifest, candidate, spec = bundle
    spec["limits"] = [{"metric": "memory", "maximum": 50}]
    manifest.write_text(yaml.safe_dump(spec))
    result = Runner(manifest).run(candidate)
    assert result["acceptance"]["target_met"]
    assert not result["acceptance"]["accepted"]
    assert result["acceptance"]["metric_limit_failure_count"] > 0


def test_benchmark_missing_objective_metric_fails(bundle):
    manifest, _, spec = bundle
    spec["objective"]["metric"] = "nonexistent"
    manifest.write_text(yaml.safe_dump(spec))
    with pytest.raises(BenchmarkError, match="missing objective"):
        Runner(manifest).validate()


def test_benchmark_maximize_direction(bundle):
    manifest, candidate, spec = bundle
    spec["objective"]["direction"] = "maximize"
    manifest.write_text(yaml.safe_dump(spec))
    result = Runner(manifest).run(candidate)
    assert result["comparison"]["overall"]["speedup"] == pytest.approx(0.5)
    assert not result["acceptance"]["accepted"]


def test_benchmark_calibration_is_off_by_default(bundle, monkeypatch):
    manifest, candidate, _ = bundle
    runner = Runner(manifest)
    monkeypatch.setattr(runner, "_calibrate", lambda baseline: pytest.fail("A/A must not run unless enabled"))
    result = runner.run(candidate)
    assert result["status"] == "completed" and result["calibration"] == {
        "enabled": False, "passed": True, "failing_cases": [],
        "note": "A/A calibration is disabled (measurement.calibration: false)."}
    preflight = Runner(manifest).validate()
    assert preflight["status"] == "ready" and preflight["calibration"]["enabled"] is False


def test_benchmark_enabled_calibration_measures_the_baseline_against_itself(bundle):
    manifest, candidate, spec = bundle
    spec["measurement"]["calibration"] = True
    manifest.write_text(yaml.safe_dump(spec))
    result = Runner(manifest).validate()
    calibration = result["calibration"]
    assert result["status"] == "ready" and calibration["enabled"] and calibration["passed"]
    assert calibration["tolerance"] == 0.05 and len(calibration["records"]) == 4 * 4 * 2
    assert calibration["summary"]["overall"]["speedup"] == pytest.approx(1)


def test_benchmark_unstable_aa_blocks_candidate_measurement(bundle, monkeypatch):
    manifest, candidate, spec = bundle
    spec["measurement"]["calibration"] = True
    manifest.write_text(yaml.safe_dump(spec))
    runner = Runner(manifest)
    monkeypatch.setattr(runner, "_calibrate", lambda baseline: {"passed": False})
    monkeypatch.setattr(runner, "_paired", lambda *args, **kw: pytest.fail("candidate should not be measured"))
    result = runner.run(bundle[1])
    assert result["status"] == "calibration_failed"
    assert not result["acceptance"]["accepted"]


def paired_records(a=10.0, b=5.0):
    return [{"block": block, "case_id": "test", "arm": arm, "metrics": {"cost": a if arm == "a" else b}}
            for block in range(4) for arm in ("a", "b", "b", "a")]


def test_benchmark_paired_statistics_require_complete_blocks():
    records = paired_records()[:-1]
    with pytest.raises(ValueError, match="exactly two"):
        summarize(records, metric="cost", direction="minimize", weights={"test": 1},
                  confidence=0.95, bootstrap_samples=200, seed=0)


def test_benchmark_paired_statistics_remove_balanced_log_drift():
    records = paired_records()
    for index, row in enumerate(records):
        row["metrics"]["cost"] *= math.exp(index * 0.01)
    result = summarize(records, metric="cost", direction="minimize", weights={"test": 1},
                       confidence=0.95, bootstrap_samples=200, seed=0)
    assert result["overall"]["speedup"] == pytest.approx(2)
    assert result["overall"]["interval"] == pytest.approx([2, 2])


def test_benchmark_calibration_rejects_stable_bias(bundle, monkeypatch):
    runner = Runner(bundle[0])
    runner.cases = [Case(id="test")]
    # A stable 8% bias lies outside the default 5% calibration tolerance.
    monkeypatch.setattr(runner, "_paired", lambda *args, **kw: paired_records(a=10.8, b=10))
    assert not runner._calibrate(None)["passed"]


def test_benchmark_calibration_rejects_wide_interval(bundle, monkeypatch):
    runner = Runner(bundle[0])
    runner.cases = [Case(id="test")]
    records = paired_records(a=10, b=10)
    for row in records:
        if row["arm"] == "b":
            row["metrics"]["cost"] *= 0.8 if row["block"] % 2 else 1.25
    monkeypatch.setattr(runner, "_paired", lambda *args, **kw: records)
    assert not runner._calibrate(None)["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_benchmark_rejects_invalid_metric_values(value):
    with pytest.raises(ValueError):
        finite_number(value)


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"adapter": "../adapter.py:Task"},
    {"objective": {"scope": "module", "metric": "latency_ms", "unit": "seconds"}},
    {"implementation": {"root": ".", "files": ["../solution.py"]}},
])
def test_benchmark_manifest_contract_rejects_invalid_configuration(bundle, change):
    spec = copy.deepcopy(bundle[2])
    spec.update(change)
    with pytest.raises(ValueError):
        BenchmarkSpec.model_validate(spec)


def test_benchmark_manifest_unknown_fields_are_not_silently_ignored(bundle):
    spec = {**bundle[2], "measurment": {}}
    with pytest.raises(ValueError):
        BenchmarkSpec.model_validate(spec)


def test_benchmark_nested_case_mutation_invalidates_result(bundle):
    runner = Runner(bundle[0])
    original = runner.task.cleanup_implementation

    def cleanup(implementation):
        runner.cases[0].params["x"] = 999
        original(implementation)

    runner.task.cleanup_implementation = cleanup
    with pytest.raises(BenchmarkError, match="case definitions changed"):
        runner.validate(checks_only=True)


def test_benchmark_inventory_rejects_external_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass")
    (root / "link.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        file_inventory(root, ["*.py"])


def test_benchmark_leaf_modules_do_not_share_candidate_imports(tmp_path):
    path = tmp_path / "solution.py"
    path.write_text("VALUE = 1\n")
    first = load_python_file(path)
    path.write_text("VALUE = 2\n")
    second = load_python_file(path)
    assert first.VALUE == 1 and second.VALUE == 2


def test_benchmark_cli_schema_and_existing_output_preserved(tmp_path):
    path = tmp_path / "schema.json"
    assert main(["schema", "--output", str(path)]) == 0
    original = path.read_bytes()
    assert "$defs" in json.loads(original)
    assert main(["schema", "--output", str(path)]) == 2
    assert path.read_bytes() == original


def test_benchmark_ai_guide_is_shipped_and_exportable(tmp_path):
    path = tmp_path / "instructions.md"
    assert main(["guide", "--output", str(path)]) == 0
    assert "invalid_observations" in path.read_text()
    assert "kai_core benchmark validate" in path.read_text()


def test_benchmark_report_write_is_exclusive(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from kai_core.benchmark.cli import _write_new

    path = tmp_path / "report.json"

    def write(index):
        try:
            _write_new(path, {"writer": index})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, [0, 1]))
    assert sum(outcomes) == 1
    assert json.loads(path.read_text())["writer"] in (0, 1)


def test_benchmark_cli_reports_failures(bundle, tmp_path):
    runner_path = bundle[0].parent / "solution.py"
    runner_path.write_text("FACTOR = 0\nCOST = 1\nMEMORY = 1\n")
    output = tmp_path / "error.json"
    assert main(["validate", str(bundle[0]), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["status"] == "error"


def test_benchmark_cli_run_uses_structured_contract(bundle, tmp_path):
    from kai_core.cli import main as kai_main

    output = tmp_path / "report.json"
    assert kai_main(["benchmark", "run", str(bundle[0]), "--candidate", str(bundle[1]),
                     "--output", str(output)]) == 0
    assert json.loads(output.read_text())["acceptance"]["accepted"]


@pytest.mark.parametrize("template", ["stateless", "stateful", "command"])
def test_benchmark_shipped_templates_pass_conformance(template, tmp_path):
    if template == "stateless":
        pytest.importorskip("torch")
    if template == "command" and not shutil.which("c++"):
        pytest.skip("C++ compiler unavailable")
    root = tmp_path / template
    assert main(["init", str(root), "--template", template]) == 0
    result = Runner(root / "benchmark.yaml", split="acceptance").validate(checks_only=True)
    assert result["status"] == "checks_passed"
    assert len(result["checks"]) == 3
    assert main(["init", str(root), "--template", template]) == 2


def test_benchmark_entrypoint_help_does_not_import_optimization_agents():
    code = """
import sys
from kai_core.cli import main
try:
    main(['benchmark', '--help'])
except SystemExit as error:
    assert error.code == 0
assert 'kai_core.agentic_workflow' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_benchmark_wall_timer_waits_for_completion(monkeypatch):
    from kai_core.benchmark.models import Measurement
    from kai_core.benchmark.timing import Timer

    events = []
    ticks = iter([1000000, 3000000])

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr("kai_core.benchmark.timing.time.perf_counter_ns", clock)
    timer = Timer(Measurement(boundary="test", cache_policy="test"))

    def call():
        events.append("run")
        return Observation(42)

    observation, milliseconds = timer.measure(call, lambda: events.append("sync"))
    assert observation.output == 42
    assert milliseconds == 2
    assert events == ["sync", "clock", "run", "sync", "clock"]


def test_benchmark_cuda_timer_uses_configured_device_and_stream_events(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from kai_core.benchmark.models import Measurement
    from kai_core.benchmark.timing import Timer

    events = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            events.append("record")

        def synchronize(self):
            events.append("event_sync")

        def elapsed_time(self, end):
            assert isinstance(end, Event)
            return 2.5

    cuda = SimpleNamespace(
        is_available=lambda: True, get_device_name=lambda device: f"GPU{device}",
        get_device_capability=lambda device: (12, 0), device=lambda device: nullcontext(),
        synchronize=lambda device: events.append(("device_sync", device)), Event=Event,
    )
    fake = SimpleNamespace(cuda=cuda, __version__="test", version=SimpleNamespace(cuda="test"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    timer = Timer(Measurement(timer="cuda_event", device=2, boundary="test", cache_policy="test"))

    def call():
        events.append("run")
        return Observation(42)

    with timer.context():
        observation, milliseconds = timer.measure(call, lambda: events.append("adapter_sync"))
    assert observation.output == 42 and milliseconds == 2.5
    assert events == ["adapter_sync", ("device_sync", 2), "record", "run", "record", "event_sync"]


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fails", [True, False])
def test_benchmark_cuda_timing_defers_automatic_gc_and_restores_state(enabled, fails):
    import gc
    from types import SimpleNamespace

    from kai_core.benchmark.models import Measurement
    from kai_core.benchmark.timing import Timer

    class Event:
        def record(self):
            assert not gc.isenabled()

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    timer = Timer(Measurement(timer="wall", boundary="test", cache_policy="test"))
    timer.torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda device: None, Event=lambda **kwargs: Event()))
    before = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()

        def call():
            assert not gc.isenabled()
            if fails:
                raise RuntimeError("launch failed")
            return Observation(42)

        if fails:
            with pytest.raises(RuntimeError, match="launch failed"):
                timer.measure(call, lambda: None)
        else:
            assert timer.measure(call, lambda: None)[1] == 1.0
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if before else gc.disable)()


def test_benchmark_wall_timing_retains_workload_gc_policy():
    import gc

    from kai_core.benchmark.models import Measurement
    from kai_core.benchmark.timing import Timer

    timer = Timer(Measurement(timer="wall", boundary="application", cache_policy="test"))
    before = gc.isenabled()

    def call():
        assert gc.isenabled() is before
        return Observation(42)

    timer.measure(call, lambda: None)


def test_default_fixture_fingerprint_distinguishes_content_type_and_layout(tmp_path):
    from kai_core.benchmark import fixture_fingerprint
    assert fixture_fingerprint({"a": 1, "b": [1, 2]}) == fixture_fingerprint({"b": [1, 2], "a": 1})
    assert fixture_fingerprint([1, 2]) != fixture_fingerprint([2, 1])
    assert len({fixture_fingerprint(v) for v in (1, 1.0, True, "1", b"1", None)}) == 6
    assert fixture_fingerprint((1, 2)) != fixture_fingerprint([1, 2])
    data = tmp_path / "weights.bin"; data.write_bytes(b"\x00\x01")
    same = tmp_path / "copy.bin"; same.write_bytes(b"\x00\x01")
    assert fixture_fingerprint(data) == fixture_fingerprint(same)
    data.write_bytes(b"\x00\x02")
    assert fixture_fingerprint(data) != fixture_fingerprint(same)


def test_default_fixture_fingerprint_names_unsupported_objects_and_accepts_hooks():
    from kai_core.benchmark import UnsupportedFixtureValue, fixture_fingerprint

    class Opaque:
        pass

    class Described:
        def __fixture_fingerprint__(self):
            return "weights-v3"

    with pytest.raises(UnsupportedFixtureValue, match=r"fixture\['scratch'\]\[0\] holds .*Opaque"):
        fixture_fingerprint({"scratch": [Opaque()]})
    assert fixture_fingerprint({"model": Described()}) == fixture_fingerprint({"model": Described()})


def test_default_fixture_fingerprint_covers_tensor_values_shape_and_stride():
    torch = pytest.importorskip("torch")
    from kai_core.benchmark import fixture_fingerprint
    a = torch.arange(6, dtype=torch.float32)
    assert fixture_fingerprint(a) == fixture_fingerprint(a.clone())
    assert fixture_fingerprint(a) != fixture_fingerprint(a.view(2, 3))
    assert fixture_fingerprint(a.view(2, 3)) != fixture_fingerprint(a.view(2, 3).t().contiguous().t())
    assert fixture_fingerprint(a) != fixture_fingerprint(a.to(torch.float64))
    assert fixture_fingerprint(torch.zeros(3, dtype=torch.bfloat16)) != fixture_fingerprint(torch.ones(3, dtype=torch.bfloat16))
    assert fixture_fingerprint(torch.empty(0)) == fixture_fingerprint(torch.empty(0))


def test_manifest_rejects_adapter_in_agent_files():
    from pydantic import ValidationError
    from kai_core.benchmark.models import BenchmarkSpec
    base = {"name": "t", "description": "d", "adapter": "adapter.py:Task",
            "implementation": {"files": ["solution.py"]},
            "objective": {"scope": "kernel", "unit": "ms"},
            "measurement": {"timer": "wall", "boundary": "b", "cache_policy": "c"}}
    BenchmarkSpec.model_validate({**base, "agent_files": ["INTERFACE.md", "bridge.cu"]})
    for pattern in (["adapter.py"], ["*.py"], ["INTERFACE.md", "adapter.py"]):
        with pytest.raises(ValidationError, match="agent_files cannot include the adapter"):
            BenchmarkSpec.model_validate({**base, "agent_files": pattern})
