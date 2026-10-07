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
    notes = [(s["round"], s["text"], s["note"]["label"], s["note"]["text"]) for s in snap["steps"] if s.get("note")]
    assert notes == [(2, "decided the repair", "critical issue", "wrong factor"),
                     (3, "decided the bottleneck", "bottleneck", "synthetic test")]
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


def test_snapshot_marks_a_dead_running_run_stopped(tmp_path, monkeypatch):
    root = _in_flight_run(tmp_path / "run")
    monkeypatch.setattr(dashboard, "is_live", lambda path: False)
    snap = dashboard.snapshot(root)
    assert snap["stale"] and snap["activity"] is None
    assert snap["steps"][-1]["status"] == "stopped"


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
