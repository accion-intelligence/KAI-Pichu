from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from kai_pichu import dashboard
from kai_pichu.cli import main
from kai_pichu.io import write_json
from test_optimizer import task  # noqa: F401 (the replay task fixture)


def test_snapshot_reconstructs_a_finished_replay_run(task):  # noqa: F811
    manifest, config, output = task
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output)]) == 0
    snap = dashboard.snapshot(output)
    assert snap["status"] == "accepted" and snap["live"] is False and snap["activity"] is None
    assert [(r["round"], r["phase"], r["status"]) for r in snap["rounds"]] == [
        (1, "seed", "error"), (2, "repair", "completed"), (3, "optimization", "completed")]
    assert snap["best"]["round"] == 3 and snap["best"]["score"] == pytest.approx(2.0)
    assert [r["best"] for r in snap["rounds"]] == [False, False, True]
    assert [a["accepted"] for a in snap["acceptance"]] == [True, True]
    calls = [(s["actor"], s["round"]) for s in snap["steps"] if s["kind"] == "model"]
    assert calls == [("coder", 1), ("judge", 2), ("coder", 2), ("judge", 3), ("coder", 3)]
    decisions = [(s["round"], s["text"].split(" · ")[0]) for s in snap["steps"] if s["actor"] == "judge"]
    assert decisions == [(2, "decided the repair"), (3, "decided the bottleneck")]
    progress = snap["progress"]
    assert [(p["round"], p["status"], p["eligible"]) for p in progress] == [
        (1, "error", False), (2, "completed", True), (3, "completed", True)]
    assert [p["best"] for p in progress] == [None, pytest.approx(10 / 6), pytest.approx(2.0)]  # running best of eligible rounds
    assert all(a["t"] <= b["t"] and a["tokens"] <= b["tokens"] for a, b in zip(progress, progress[1:]))
    stages = [s["stage"] for s in snap["steps"] if s["kind"] == "evaluate"]
    assert stages == ["preflight", "evaluate", "evaluate", "evaluate", "acceptance", "acceptance"]
    assert snap["used"]["llm_calls"] == 5 and snap["budget"]["rounds"] == 3
    assert not list(output.glob("**/*dashboard*"))  # read-only: nothing written into the run


def _in_flight_run(root: Path) -> Path:
    write_json(root / "state.json", {"status": "running", "next_round": 1, "history": [], "llm_calls": 1,
                                     "role_calls": {"coder": 1, "judge": 0}, "usage": [], "elapsed_seconds": 5.0})
    write_json(root / "config.json", {"budget": {"rounds": 4, "llm_calls": 9, "seconds": 60},
                                      "coder": {"model": "m", "provider": "replay"}, "judge": None})
    (root / "reports").mkdir()
    (root / "reports" / "0000-preflight.log").write_text("ok\n")
    write_json(root / "reports" / "0000-preflight.process.json", {"status": "completed", "elapsed_seconds": 1.0})
    write_json(root / "reports" / "0000-preflight.json", {"status": "ready"})
    write_json(root / "llm" / "call-0000.request.json", {"role": "coder", "continuation": False, "messages": []})
    return root


def test_snapshot_reports_the_call_in_flight_while_a_process_holds_the_run(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    monkeypatch.setattr(dashboard, "is_live", lambda path: True)
    snap = dashboard.snapshot(root)
    assert snap["activity"]["stage"] == "coder" and snap["activity"]["round"] == 1
    assert snap["current_round"] == 1 and snap["current_phase"] == "seed"
    assert snap["steps"][-1]["status"] == "running"
    assert snap["live_advice"] is None


def test_the_judges_advice_shows_while_its_round_is_still_running(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    state = json.loads((root / "state.json").read_text())
    state.update(next_round=2, history=[{"id": 0, "metrics": {"status": "completed"}, "score": 1.1, "hypothesis": "h"}])
    write_json(root / "state.json", state, replace=True)
    write_json(root / "reports" / "0001-round-0000.json", {"status": "completed"})
    (root / "reports" / "0001-round-0000.log").write_text("ok\n")
    write_json(root / "reports" / "0001-round-0000.process.json", {"status": "completed", "elapsed_seconds": 0.0})
    decision = {"bottleneck": "b", "optimization_method": "m", "modification_plan": "p", "base_round": 1}
    write_json(root / "llm" / "call-0001.request.json", {"role": "judge", "continuation": False, "messages": []})
    write_json(root / "llm" / "call-0001.response.json", {"text": json.dumps(decision), "usage": {}})
    write_json(root / "llm" / "call-0002.request.json", {"role": "coder", "continuation": False, "messages": []})
    monkeypatch.setattr(dashboard, "is_live", lambda path: True)
    snap = dashboard.snapshot(root)
    assert snap["current_round"] == 2 and snap["activity"]["stage"] == "coder"
    assert snap["live_advice"] == {"round": 2, "base_round": 1, "diagnosis": {
        "bottleneck": "b", "optimization_method": "m", "modification_plan": "p"}}


def test_snapshot_describes_an_evaluation_in_flight(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    (root / "llm" / "call-0000.request.json").unlink()
    (root / "reports" / "0000-preflight.process.json").unlink()  # the baseline is still being timed
    monkeypatch.setattr(dashboard, "is_live", lambda path: True)
    snap = dashboard.snapshot(root)
    assert snap["activity"]["stage"] == "preflight"
    assert snap["steps"][-1]["text"] == "check and time the baseline"


def test_snapshot_reads_runs_recorded_with_the_generator_role(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    write_json(root / "llm" / "call-0000.request.json", {"role": "generator", "continuation": False, "messages": []}, replace=True)
    config = json.loads((root / "config.json").read_text())
    config["generator"] = config.pop("coder")
    write_json(root / "config.json", config, replace=True)
    state = json.loads((root / "state.json").read_text())
    state["role_calls"] = {"generator": 1, "judge": 0}
    write_json(root / "state.json", state, replace=True)
    monkeypatch.setattr(dashboard, "is_live", lambda path: True)
    snap = dashboard.snapshot(root)
    assert snap["activity"]["stage"] == "coder" and snap["models"]["coder"]["model"] == "m"
    assert snap["used"]["role_calls"] == {"coder": 1, "judge": 0}


def test_a_call_abandoned_by_an_interrupted_run_is_not_in_flight_after_resume(tmp_path, monkeypatch):
    import os
    root = _in_flight_run(tmp_path / "run")  # call-0000 never got a response: the run was interrupted
    write_json(root / "llm" / "call-0001.request.json", {"role": "coder", "continuation": False, "messages": []})
    first = root / "llm" / "call-0000.request.json"
    os.utime(first, (first.stat().st_mtime - 60, first.stat().st_mtime - 60))  # the resumed run's call came later
    monkeypatch.setattr(dashboard, "is_live", lambda path: True)
    snap = dashboard.snapshot(root)
    models = [s for s in snap["steps"] if s["kind"] == "model"]
    assert [s["status"] for s in models] == ["stopped", "running"]
    assert snap["activity"]["stage"] == "coder" and snap["activity"]["since"] == models[-1]["start"]


def test_snapshot_marks_a_dead_running_run_stopped(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    monkeypatch.setattr(dashboard, "is_live", lambda path: False)
    snap = dashboard.snapshot(root)
    assert snap["stale"] and snap["activity"] is None
    assert snap["steps"][-1]["status"] == "stopped"


SOL = ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
       "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")


def _capture(root: Path, tag: str, fingerprint: str, launches: list[tuple[float, float, float, float]]) -> None:
    """One NCU capture; each launch is (GPU time ns, SM %, memory %, DRAM %)."""
    folder = root / "profiles" / tag
    write_json(folder / "profile.json", {"status": "profiled", "implementation_fingerprint": fingerprint,
                                         "case_selection": {"case_id": tag.split("/")[-1]}, "process": {"elapsed_seconds": 2.0}})
    names = ["gpu__time_duration.sum", *SOL]
    lines = ['"ID","Kernel Name",' + ",".join(f'"{n}"' for n in names), '"","","ns","%","%","%"']
    lines += [f'"{i}","k{i}",' + ",".join(f'"{v}"' for v in launch) for i, launch in enumerate(launches)]
    (folder / "metrics.csv").write_text("\n".join(lines) + "\n")


def test_headroom_pairs_the_baseline_with_the_best_candidates_capture(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    monkeypatch.setattr(dashboard, "is_live", lambda path: False)
    state = json.loads((root / "state.json").read_text())
    metrics = {"status": "completed", "acceptance": {"case_speedups": {"one": {"speedup": 2.5, "interval": [2.4, 2.6]}}}}
    best = {"id": 1, "fingerprint": "best", "score": 2.5, "metrics": metrics, "hypothesis": "h"}
    state.update(best=best, latest_eligible=best, history=[{**best, "id": 0, "fingerprint": "other"}, best])
    write_json(root / "state.json", state, replace=True)
    write_json(root / "config.json", {**json.loads((root / "config.json").read_text()), "profile": {"enabled": True}}, replace=True)
    _capture(root, "baseline/one", "baseline", [(300, 70, 40, 5), (100, 30, 80, 60)])  # two launches, time-weighted
    headroom = dashboard.snapshot(root)["headroom"]
    assert headroom["note"] == "best round 2 is captured when the next round profiles it"
    _capture(root, "round-0001/one", "other", [(100, 10, 10, 10)])  # an earlier candidate, not the best
    _capture(root, "round-0002/one", "best", [(100, 95, 90, 8)])
    snap = dashboard.snapshot(root)
    [case] = snap["headroom"]["cases"]
    assert snap["headroom"]["note"] is None and snap["headroom"]["best_round"] == 2
    assert case["case"] == "one" and case["speedup"] == 2.5
    assert case["baseline"] == {"compute": pytest.approx(60), "memory": pytest.approx(50), "dram": pytest.approx(18.75), "launches": 2}
    assert case["best"]["compute"] == 95 and case["best"]["memory"] == 90
    assert any(s["text"] == "profile baseline one" and s["round"] is None for s in snap["steps"])


def test_headroom_explains_runs_without_it(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    monkeypatch.setattr(dashboard, "is_live", lambda path: False)
    assert dashboard.snapshot(root)["headroom"]["note"] == "profiling is off for this run"
    _capture(root, "round-0001/one", "x", [(100, 50, 50, 5)])
    assert dashboard.snapshot(root)["headroom"]["note"] == "this run did not capture the baseline; runs started now do"


def test_live_detection_reads_the_run_lock(tmp_path):
    import fcntl
    root = _in_flight_run(tmp_path / "run")
    with (root / ".run.lock").open("a") as handle:
        assert dashboard.is_live(root) in (False, None)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert dashboard.is_live(root) in (True, None)


def test_server_serves_only_discovered_runs(tmp_path):
    _in_flight_run(tmp_path / "runs" / "one")
    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(tmp_path / "runs"))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"<title>" in urlopen(base + "/").read()
        assert [row["name"] for row in json.loads(urlopen(base + "/api/runs").read())] == ["one"]
        assert json.loads(urlopen(base + "/api/run?name=one").read())["name"] == "one"
        for name in ("missing", "../one", "%2e%2e"):
            with pytest.raises(HTTPError) as error:
                urlopen(f"{base}/api/run?name={name}")
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_cli_rejects_a_directory_without_runs(tmp_path, capsys):
    assert main(["dashboard", str(tmp_path)]) == 2
    assert "no run directory" in capsys.readouterr().err
