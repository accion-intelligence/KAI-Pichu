from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kai_core.benchmark.cli import main
from kai_core.benchmark.lint import lint

ADAPTER = '''
from kai_core.benchmark import Benchmark, Case, Observation, Validation
class Task(Benchmark):
    def cases(self, split):
        if split == "acceptance":
            return [Case(id="held-out")]
        return [Case(id="one")]
    def prepare(self, case, seed):
        return {"x": 1}
    def load_implementation(self, workspace):
        return workspace
    def run(self, implementation, fixture):
        return Observation(1, {"cost": 1.0})
    def validate(self, case, fixture, observation):
        return Validation(True)
    def invalid_observations(self, case, fixture, valid):
        return [Observation(-1)]
'''

SPEC = {"name": "lint_fixture", "description": "static-check fixture",
        "adapter": "adapter.py:Task", "implementation": {"root": ".", "files": ["solution.py"]},
        "objective": {"scope": "kernel", "metric": "cost", "unit": "u"},
        "measurement": {"timer": "wall", "boundary": "b", "cache_policy": "c"}}


def build(tmp_path: Path, *, spec: dict | None = None, adapter: str = ADAPTER) -> Path:
    source = tmp_path / "task"
    source.mkdir(exist_ok=True)
    (source / "adapter.py").write_text(adapter)
    (source / "solution.py").write_text("VALUE = 1\n")
    manifest = source / "benchmark.yaml"
    manifest.write_text(yaml.safe_dump({**SPEC, **(spec or {})}))
    return manifest


def codes(manifest: Path) -> set[str]:
    return {finding.code for finding in lint(manifest)}


def test_lint_passes_a_well_formed_task(tmp_path):
    assert not [f for f in lint(build(tmp_path)) if f.level == "error"]


def test_lint_never_runs_the_workload(tmp_path, monkeypatch):
    # The whole point is usability without the device the task targets.
    import kai_core.process as process
    monkeypatch.setattr(process, "run_process", lambda *a, **k: pytest.fail("subprocess started"))
    adapter = ADAPTER.replace("        return {\"x\": 1}", "        raise AssertionError('prepare called')")
    adapter = adapter.replace("        return Observation(1, {\"cost\": 1.0})", "        raise AssertionError('run called')")
    assert not [f for f in lint(build(tmp_path, adapter=adapter)) if f.level == "error"]


def test_lint_reports_an_unmatched_declared_pattern(tmp_path):
    assert "implementation.files" in codes(build(tmp_path, spec={
        "implementation": {"root": ".", "files": ["*.cuh"]}}))


def test_lint_surfaces_the_adapter_exposed_to_the_agent(tmp_path):
    # BenchmarkSpec rejects this itself; lint must report it rather than crash.
    assert codes(build(tmp_path, spec={"agent_files": ["adapter.py"]})) == {"manifest.invalid"}


def test_lint_reports_agent_files_that_are_not_text(tmp_path):
    manifest = build(tmp_path, spec={"agent_files": ["notes.bin"]})
    (manifest.parent / "notes.bin").write_bytes(b"\xff\xfe\x00binary")
    assert "agent_files.not_text" in codes(manifest)


def test_lint_reports_a_device_mismatch(tmp_path):
    assert "options.device" in codes(build(tmp_path, spec={
        "options": {"device": "cuda:1"}, "measurement": {**SPEC["measurement"], "device": 0}}))


def test_lint_reports_nondeterministic_cases(tmp_path):
    adapter = ADAPTER.replace('        return [Case(id="one")]',
                              '        import random\n        return [Case(id=f"c{random.random()}")]')
    assert "cases.nondeterministic" in codes(build(tmp_path, adapter=adapter))


def test_lint_reports_duplicate_case_ids(tmp_path):
    adapter = ADAPTER.replace('        return [Case(id="one")]', '        return [Case(id="one"), Case(id="one")]')
    assert "cases.duplicate_id" in codes(build(tmp_path, adapter=adapter))


def test_lint_warns_when_acceptance_is_not_held_out(tmp_path):
    adapter = ADAPTER.replace('        if split == "acceptance":\n            return [Case(id="held-out")]\n', "")
    findings = {f.code: f.level for f in lint(build(tmp_path, adapter=adapter))}
    assert findings.get("splits.identical") == "warning"


def test_lint_separates_a_missing_dependency_from_a_broken_task(tmp_path):
    adapter = "import a_module_that_does_not_exist\n" + ADAPTER
    findings = {f.code: f.level for f in lint(build(tmp_path, adapter=adapter))}
    assert findings.get("adapter.dependency") == "warning" and "adapter.load" not in findings


def test_lint_reports_a_broken_adapter_as_an_error(tmp_path):
    adapter = ADAPTER.replace("class Task(Benchmark):", "class Task:")
    assert "adapter.load" in codes(build(tmp_path, adapter=adapter))


def test_lint_reports_an_invalid_manifest_without_loading_the_adapter(tmp_path):
    manifest = build(tmp_path, spec={"objective": {"scope": "kernel", "metric": "latency_ms", "unit": "wrong"}})
    assert codes(manifest) == {"manifest.invalid"}


def test_lint_cli_exit_code_and_report(tmp_path, capsys):
    manifest = build(tmp_path)
    report = tmp_path / "lint.json"
    assert main(["lint", str(manifest), "--output", str(report)]) == 0
    assert "note(s)" in capsys.readouterr().out
    assert isinstance(json.loads(report.read_text()), list)
    broken = build(tmp_path, spec={"implementation": {"root": ".", "files": ["*.cuh"]}})
    assert main(["lint", str(broken)]) == 1
