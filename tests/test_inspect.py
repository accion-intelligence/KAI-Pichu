from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kai_core.cli import main
from kai_core.inspect import load, overview


# Self-contained so this file adds no coupling to the optimizer tests. The
# synthetic cost is a deterministic control-flow metric, not a benchmark.
ADAPTER = '''
from kai_core.benchmark import Benchmark, Case, Observation, Validation, json_fingerprint, load_python_file
class Task(Benchmark):
    def cases(self, split):
        return [Case(id="one", params={"x": 3 if split != "acceptance" else 17})]
    def prepare(self, case, seed):
        return {"x": case.params["x"]}
    def fingerprint(self, fixture):
        return json_fingerprint(fixture)
    def load_implementation(self, workspace):
        return load_python_file(workspace / "solution.py")
    def run(self, implementation, fixture):
        return Observation(implementation.FACTOR * fixture["x"], {"synthetic_cost": implementation.COST})
    def validate(self, case, fixture, observation):
        return Validation(observation.output == fixture["x"], "wrong output")
    def invalid_observations(self, case, fixture, valid):
        return [Observation(-99)]
'''


def reply(cost: float, factor: int = 1) -> str:
    return json.dumps({"hypothesis": f"set cost to {cost}",
                       "files": {"solution.py": f"FACTOR = {factor}\nCOST = {cost}\n"}})


@pytest.fixture
def finished_run(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter.py").write_text(ADAPTER)
    (source / "solution.py").write_text("FACTOR = 1\nCOST = 10.0\n")
    manifest = source / "benchmark.yaml"
    manifest.write_text(yaml.safe_dump({
        "name": "synthetic_control_flow_only", "description": "Test fixture, not a real speedup claim",
        "adapter": "adapter.py:Task", "implementation": {"root": ".", "files": ["solution.py"]},
        "objective": {"scope": "kernel", "metric": "synthetic_cost", "unit": "test_units", "target_speedup": 1.2},
        "measurement": {"timer": "wall", "boundary": "test only", "cache_policy": "test",
                        "blocks": 4, "iterations": 1, "warmup": 1, "bootstrap_samples": 200}}))
    config = tmp_path / "optimizer.yaml"
    config.write_text(yaml.safe_dump({
        # Round 0 produces a wrong factor, round 1 repairs it, round 2 optimizes.
        "generator": {"provider": "replay", "responses": [reply(1, factor=0), reply(6), reply(5)]},
        "judge": {"provider": "replay", "responses": [
            json.dumps({"critical_issue": "wrong factor", "why_it_matters": "output mismatch",
                        "minimal_fix_hint": "restore factor"}),
            json.dumps({"bottleneck": "synthetic test", "optimization_method": "next fixture",
                        "modification_plan": "lower synthetic cost"})]},
        "budget": {"rounds": 3, "llm_calls": 5, "acceptance_repeats": 2,
                   "seconds": 60, "evaluation_seconds": 10}}))
    output = tmp_path / "run"
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output)]) == 0
    return output


def test_inspect_names_each_round_its_phase_and_why_it_was_not_selected(finished_run, capsys):
    assert main(["inspect", str(finished_run)]) == 0
    text = capsys.readouterr().out
    assert "accepted=True" in text
    # Round 0 is the blind seed, round 1 repairs it, round 2 optimizes from the best.
    for phase in ("seed", "repair", "optimization"):
        assert phase in text
    # The failed seed is reported as not runnable, not as a silent blank row.
    seed = next(line for line in text.splitlines() if line.startswith("0 "))
    assert "no" in seed and "error" in seed
    assert "Best:    round" in text


def test_inspect_explains_a_candidate_that_ran_but_lost(finished_run):
    run = load(finished_run)
    winners = [row for row in run["rounds"] if row.selected]
    losers = [row for row in run["rounds"] if not row.selected]
    assert winners and losers
    # Every unselected round carries at least one reason drawn from the loop's rule.
    for row in losers:
        assert row.reasons, f"round {row.id} has no recorded reason"
    assert all(row.interval is None or len(row.interval) == 2 for row in run["rounds"])


def test_inspect_round_detail_recovers_the_judge_strategy(finished_run, capsys):
    assert main(["inspect", str(finished_run), "--round", "1"]) == 0
    text = capsys.readouterr().out
    assert "Round 1 (repair)" in text
    # The repair judge's decision is readable without replaying any model call.
    assert "critical_issue: wrong factor" in text
    assert "minimal_fix_hint: restore factor" in text


def test_inspect_json_output_is_machine_readable(finished_run, capsys):
    assert main(["inspect", str(finished_run), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "accepted"
    assert [row["round"] for row in payload["rounds"]] == [0, 1, 2]
    assert payload["rounds"][0]["phase"] == "seed"
    assert any(row["selected"] for row in payload["rounds"])


def test_inspect_diff_prints_the_selected_patch(finished_run, capsys):
    assert main(["inspect", str(finished_run), "--diff"]) == 0
    assert "--- a/solution.py" in capsys.readouterr().out


def test_inspect_rejects_a_directory_that_is_not_a_run(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["inspect", str(tmp_path / "empty")]) == 2
    assert "no state.json" in capsys.readouterr().err


def test_inspect_reads_a_run_with_no_completed_round(tmp_path, capsys):
    # An interrupted run is exactly when the record matters most; it must render.
    run = tmp_path / "partial"
    run.mkdir()
    (run / "state.json").write_text(json.dumps({
        "schema_version": 1, "status": "model_error", "message": "endpoint refused",
        "llm_calls": 1, "evaluations": 0, "elapsed_seconds": 2.5,
        "current": None, "best": None, "history": [], "usage": []}))
    assert main(["inspect", str(run)]) == 0
    text = capsys.readouterr().out
    assert "model_error" in text and "endpoint refused" in text
    assert "Best:    none selected" in text


def test_inspect_never_executes_a_workload(finished_run, monkeypatch):
    # Reading a record must not load an adapter or touch the GPU guard.
    import kai_core.benchmark.loading as loading
    import kai_core.process as process
    monkeypatch.setattr(loading, "load_adapter", lambda *a, **k: pytest.fail("adapter loaded"))
    monkeypatch.setattr(process, "run_process", lambda *a, **k: pytest.fail("subprocess started"))
    assert main(["inspect", str(finished_run)]) == 0
