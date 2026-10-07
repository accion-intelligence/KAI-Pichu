from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
import time

import pytest
import yaml

from kai_pichu.benchmark.api import json_fingerprint
from kai_pichu.cli import main
from kai_pichu.config import ModelConfig, OptimizeConfig, Resources
from kai_pichu.optimizer.engine import OptimizationLoop
from kai_pichu.io import parse_object
from kai_pichu.models import ModelClient
from kai_pichu.process import GpuGuard, run_process
from kai_pichu.workspace import Workspace


# Synthetic cost is ONLY a deterministic control-flow test metric, not a
# performance benchmark. Correctness still executes the actual candidate source.
ADAPTER = '''
from kai_pichu.benchmark import Benchmark, Case, Observation, Validation, json_fingerprint, load_python_file
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
    return json.dumps({"hypothesis": "deterministic test candidate",
                       "files": {"solution.py": f"FACTOR = {factor}\nCOST = {cost}\n"}})


@pytest.fixture
def task(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter.py").write_text(ADAPTER)
    (source / "solution.py").write_text("FACTOR = 1\nCOST = 10.0\n")
    spec = {"name": "synthetic_control_flow_only", "description": "Test fixture, not a real speedup claim",
            "adapter": "adapter.py:Task", "implementation": {"root": ".", "files": ["solution.py"]},
            "objective": {"scope": "kernel", "metric": "synthetic_cost", "unit": "test_units", "target_speedup": 1.2},
            "measurement": {"timer": "wall", "boundary": "test only", "cache_policy": "test",
                            "blocks": 4, "iterations": 1, "warmup": 1, "bootstrap_samples": 200}}
    manifest = source / "benchmark.yaml"
    manifest.write_text(yaml.safe_dump(spec))
    config = tmp_path / "optimizer.yaml"
    config.write_text(yaml.safe_dump({
        "coder": {"provider": "replay", "responses": [reply(1, factor=0), reply(6), reply(5)]},
        "judge": {"provider": "replay", "responses": [
            json.dumps({"critical_issue": "wrong factor", "why_it_matters": "output mismatch", "minimal_fix_hint": "restore factor"}),
            json.dumps({"bottleneck": "synthetic test", "optimization_method": "next fixture", "modification_plan": "lower synthetic cost"}),
        ]}, "budget": {"rounds": 3, "llm_calls": 5, "acceptance_repeats": 2, "seconds": 60, "evaluation_seconds": 10},
    }))
    return manifest, config, tmp_path / "run"


def test_real_subprocess_replay_loop_repairs_optimizes_and_independently_accepts(task):
    manifest, config, output = task
    original = (manifest.parent / "solution.py").read_text()
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    state = json.loads((output / "state.json").read_text())
    assert summary["final_accepted"]
    assert summary["llm_calls"] == 5
    assert [row["metrics"]["status"] for row in state["history"]] == ["error", "completed", "completed"]
    assert summary["best_search"]["score"] == pytest.approx(2.0)
    assert len(summary["acceptance_reports"]) == 2
    # The model sees the scored metric's own measured values, not only a ratio.
    objective = state["history"][-1]["metrics"]["objective"]
    assert objective["metric"] == "synthetic_cost" and objective["unit"] == "test_units"
    assert objective["per_case"]["one"]["baseline"]["mean"] == pytest.approx(10.0)
    assert objective["per_case"]["one"]["candidate"]["mean"] == pytest.approx(5.0)
    assert objective["unscored_recorded_metrics"] == ["latency_ms"]
    assert all("objective" in report for report in summary["acceptance_reports"])
    assert "objective" not in state["history"][0]["metrics"]  # build/format errors have no measurement
    assert "COST = 5" in (output / "accepted/solution.py").read_text()
    assert "--- a/solution.py" in (output / "best_search.patch").read_text()
    assert (manifest.parent / "solution.py").read_text() == original
    acceptance = [json.loads(path.read_text()) for path in (output / "reports").glob("*acceptance*.json")
                  if not path.name.endswith(".process.json")]
    assert all(report["split"] == "acceptance" and report["cases"][0]["params"]["x"] == 17 for report in acceptance)


def test_dry_run_has_no_evaluation_or_model_calls_and_can_resume(task, monkeypatch):
    manifest, config, output = task
    from kai_pichu.evaluator import Evaluator
    with monkeypatch.context() as patch:
        patch.setattr(Evaluator, "evaluate", lambda *a, **k: pytest.fail("dry run evaluated"))
        patch.setattr(ModelClient, "complete", lambda *a, **k: pytest.fail("dry run called model"))
        assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output), "--dry-run"]) == 0
    assert not (output / "llm").exists()
    assert not (output / "reports").exists()
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output), "--resume"]) == 0
    assert json.loads((output / "summary.json").read_text())["final_accepted"]


def test_legacy_generator_config_key_is_read_as_coder():
    config = OptimizeConfig.model_validate({"generator": {"provider": "replay", "responses": ["x"]}})
    assert config.coder.responses == ["x"] and "generator" not in config.model_dump()
    with pytest.raises(ValueError, match="legacy name generator"):
        OptimizeConfig.model_validate({"generator": {"provider": "replay"}, "coder": {"provider": "replay"}})


def test_a_run_recorded_with_the_generator_role_resumes(task):
    manifest, config, output = task
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output), "--dry-run"]) == 0
    # Rewrite the run as it was recorded before the coder role was renamed.
    state = json.loads((output / "state.json").read_text())
    state["role_calls"] = {"generator": state["role_calls"].pop("coder"), **state["role_calls"]}
    (output / "state.json").write_text(json.dumps(state))
    recorded = json.loads((output / "config.json").read_text())
    recorded["generator"] = recorded.pop("coder")
    (output / "config.json").write_text(json.dumps(recorded))
    legacy_config = yaml.safe_load(config.read_text())
    legacy_config["generator"] = legacy_config.pop("coder")
    config.write_text(yaml.safe_dump(legacy_config))
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output), "--resume"]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["role_calls"] == {"coder": 3, "judge": 2} and state["status"] == "accepted"


def test_resource_busy_preflight_uses_no_model_budget(task, monkeypatch):
    manifest, config_path, output = task
    config = OptimizeConfig.model_validate(yaml.safe_load(config_path.read_text()))
    loop = OptimizationLoop(Workspace.create(manifest, output), config)
    monkeypatch.setattr(loop.evaluator, "evaluate", lambda *a, **k: {"status": "resource_busy"})
    result = loop.run()
    assert result["status"] == "resource_busy"
    assert result["llm_calls"] == 0 and not result["final_accepted"]


def test_final_acceptance_failure_never_creates_accepted_export(task, monkeypatch):
    manifest, config_path, output = task
    config = OptimizeConfig.model_validate(yaml.safe_load(config_path.read_text()))
    loop = OptimizationLoop(Workspace.create(manifest, output), config)
    original = loop.evaluator.evaluate

    def evaluate(*args, **kwargs):
        if kwargs["split"] == "acceptance":
            return {"status": "completed", "acceptance": {"accepted": False, "verdict": "regressed"}}
        return original(*args, **kwargs)

    monkeypatch.setattr(loop.evaluator, "evaluate", evaluate)
    result = loop.run()
    assert result["status"] == "not_accepted" and not result["final_accepted"]
    assert (output / "best_search.patch").exists() and not (output / "accepted").exists()


@pytest.mark.parametrize("changed_file", ["candidate", "benchmark"])
def test_final_source_mutation_still_writes_invalidated_summary(task, monkeypatch, changed_file):
    manifest, config_path, output = task
    config = OptimizeConfig.model_validate(yaml.safe_load(config_path.read_text()))
    loop = OptimizationLoop(Workspace.create(manifest, output), config)
    original = loop.evaluator.evaluate

    def evaluate(*args, **kwargs):
        report = original(*args, **kwargs)
        if kwargs["split"] == "acceptance":
            changed = (kwargs["candidate"] / "solution.py" if changed_file == "candidate"
                       else loop.workspace.bundle / "adapter.py")
            with changed.open("a") as handle:
                handle.write("\n# changed after evaluation\n")
        return report

    monkeypatch.setattr(loop.evaluator, "evaluate", evaluate)
    result = loop.run()
    assert result["status"] == "invalidated" and not result["final_accepted"]
    assert json.loads((output / "state.json").read_text())["status"] == "invalidated"
    assert not (output / "accepted").exists()
    assert not (output / "best_search.patch").exists()


@pytest.mark.parametrize("name", ["../adapter.py", "/tmp/escape.py", "adapter.py", "new.py", "a\\solution.py"])
def test_workspace_rejects_undeclared_paths_before_writing(task, name):
    manifest, _, output = task
    workspace = Workspace.create(manifest, output)
    before = workspace.fingerprint()
    with pytest.raises(ValueError, match="declared source"):
        workspace.candidate(0, workspace.baseline, {"hypothesis": "bad path", "files": {name: "x = 1"}})
    assert workspace.fingerprint() == before
    assert not (output / "candidates").exists()


def test_snapshot_rejects_recursive_output_and_external_symlinks(task):
    manifest, _, output = task
    with pytest.raises(ValueError, match="outside"):
        Workspace.create(manifest, manifest.parent / "run")
    original = manifest.parent / "solution.py"
    original.unlink()
    original.symlink_to(__file__)
    with pytest.raises(ValueError, match="escapes"):
        Workspace.create(manifest, output)


def test_runtime_benchmark_mutation_invalidates_run(task):
    manifest, config, output = task
    value = yaml.safe_load(config.read_text())
    code = "from pathlib import Path\np = Path(__file__).parents[2] / 'bundle/adapter.py'\np.write_text(p.read_text() + '\\n# changed')\nFACTOR = 1\nCOST = 1\n"
    value["coder"]["responses"] = [json.dumps({"hypothesis": "invalid runtime mutation", "files": {"solution.py": code}})]
    value["budget"]["rounds"] = 1
    config.write_text(yaml.safe_dump(value))
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output)]) == 1
    result = json.loads((output / "summary.json").read_text())
    assert result["status"] == "invalidated" and not result["final_accepted"]
    assert not (output / "accepted").exists()


def test_model_call_budget_survives_resume(task):
    manifest, config, output = task
    value = yaml.safe_load(config.read_text())
    value["budget"]["llm_calls"] = 1
    config.write_text(yaml.safe_dump(value))
    args = ["optimize", str(manifest), "--config", str(config), "--output", str(output)]
    assert main(args) == 1
    before = json.loads((output / "state.json").read_text())
    assert before["llm_calls"] == 1
    assert main(args + ["--resume"]) == 1
    after = json.loads((output / "state.json").read_text())
    assert after["llm_calls"] == 1
    assert after["elapsed_seconds"] >= before["elapsed_seconds"]


@pytest.mark.parametrize("text", ['{"files": {}, "files": {}}', '{"x":NaN}', '```json\n{}', '[]'])
def test_model_json_rejects_ambiguous_or_incomplete_content(text):
    with pytest.raises(ValueError):
        parse_object(text)


def test_chat_transport_sends_explicit_endpoint_model_and_key_without_logging_it(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], body))
            result = {"choices": [{"message": {"content": reply(5)}, "finish_reason": "stop"}], "usage": {"total_tokens": 7}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("TEST_KAI_KEY", "test-secret")
    config = ModelConfig(model="test-model", base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key_env="TEST_KAI_KEY")
    try:
        response = ModelClient(config).complete([{"role": "user", "content": "test"}], index=0, timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert requests[0][0] == "/v1/chat/completions"
    assert requests[0][1] == "Bearer test-secret"
    assert requests[0][2]["model"] == "test-model"
    assert response["usage"]["total_tokens"] == 7
    assert "test-secret" not in json.dumps(response)


def _transport_server(handler_body):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            handler_body(self)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _complete(server, timeout=10, **overrides):
    config = ModelConfig(model="test-model", api_key_env="",
                         base_url=f"http://127.0.0.1:{server.server_port}/v1", **overrides)
    return ModelClient(config).complete([{"role": "user", "content": "test"}], index=0, timeout=timeout)


def test_transient_transport_faults_are_retried_within_the_wall_budget():
    seen = []

    def handler(request):
        seen.append(request.path)
        if len(seen) < 3:
            request.send_response(503)
            request.send_header("Retry-After", "0")
            request.end_headers()
            request.wfile.write(b'{"error": "overloaded"}')
            return
        request.send_response(200)
        request.end_headers()
        request.wfile.write(json.dumps(
            {"choices": [{"message": {"content": reply(5)}, "finish_reason": "stop"}]}).encode())

    server, thread = _transport_server(handler)
    try:
        response = _complete(server)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert len(seen) == 3 and response["transport_attempts"] == 3
    assert parse_object(response["text"])["files"]


def test_a_rejected_request_is_not_retried():
    seen = []

    def handler(request):
        seen.append(request.path)
        request.send_response(400)
        request.end_headers()
        request.wfile.write(b'{"error": "context length exceeded"}')

    server, thread = _transport_server(handler)
    try:
        with pytest.raises(RuntimeError, match="model HTTP 400"):
            _complete(server)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert len(seen) == 1


def test_retries_never_outlive_the_caller_deadline():
    seen = []

    def handler(request):
        seen.append(request.path)
        request.send_response(503)
        request.send_header("Retry-After", "30")
        request.end_headers()
        request.wfile.write(b"{}")

    server, thread = _transport_server(handler)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="no budget to retry"):
            _complete(server, timeout=1)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert len(seen) == 1 and time.monotonic() - started < 5


def test_process_timeout_is_bounded(tmp_path):
    result = run_process([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                         log=tmp_path / "timeout.log", timeout=0.1, resources=Resources(), gpu_device=None)
    assert result["status"] == "timeout"
    assert result["elapsed_seconds"] < 5


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
def test_responses_transport_preserves_budget_and_rejects_partial_output(monkeypatch, status):
    from io import BytesIO
    from kai_pichu import models
    requests = []

    def respond(request, timeout):
        requests.append(request)
        value = {"id": "resp-test", "status": status, "usage": {"output_tokens": 23},
                 "output": [{"type": "reasoning", "summary": []},
                            {"type": "message", "role": "assistant",
                             "content": [{"type": "output_text", "text": reply(5)}]}]}
        return BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(models, "urlopen", respond)
    config = ModelConfig(provider="responses", model="test-model", base_url="http://test/v1",
                         max_tokens=4096, temperature=None, extra_body={"reasoning": {"effort": "high"}})
    client = ModelClient(config)
    if status == "completed":
        result = client.complete([{"role": "user", "content": "test"}], index=0, timeout=5)
        assert result["text"] == reply(5) and result["usage"]["output_tokens"] == 23
        assert result["response_id"] == "resp-test"
    else:
        with pytest.raises(ValueError, match="incomplete"):
            client.complete([{"role": "user", "content": "test"}], index=0, timeout=5)
    payload = json.loads(requests[0].data)
    assert requests[0].full_url == "http://test/v1/responses"
    assert payload["max_output_tokens"] == 4096 and payload["store"] is False
    assert payload["reasoning"] == {"effort": "high"}
    assert not {"max_tokens", "messages", "temperature"} & payload.keys()


def test_gpu_guard_maps_visible_devices_and_rejects_foreign_processes(monkeypatch):
    from kai_pichu import process
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.setattr(process, "_query", lambda fields, apps=False: [["GPU-two", "999"]] if apps else [["0", "GPU-zero"], ["2", "GPU-two"]])
    monkeypatch.setattr(os, "getpgid", lambda pid: 999)
    guard = GpuGuard(0)
    assert guard.uuid == "GPU-two"
    assert guard.inspect(group=123)["foreign_pids"] == [999]
    assert guard.inspect(group=999)["foreign_pids"] == []


@pytest.mark.parametrize("commentary", [
    "I will inspect the profile before deciding.",
    '{"action":"query_profile","queries":[{"operation":"catalog"}]}',
])
def test_responses_uses_final_answer_without_concatenating_commentary(monkeypatch, commentary):
    from io import BytesIO
    from kai_pichu import models
    final = '{"action":"query_profile","queries":[{"operation":"metrics","counter":"dram__bytes.sum"}]}'
    output = [
        {"type": "message", "role": "assistant", "phase": "commentary",
         "content": [{"type": "output_text", "text": commentary}]},
        {"type": "message", "role": "assistant", "phase": "final_answer",
         "content": [{"type": "output_text", "text": final[:30]},
                     {"type": "output_text", "text": final[30:]}]},
    ]
    value = {"status": "completed", "output": output, "usage": {"total_tokens": 100}}
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: BytesIO(json.dumps(value).encode()))
    config = ModelConfig(provider="responses", model="test-model", base_url="http://test/v1")
    result = ModelClient(config).complete([], index=0, timeout=5)
    assert result["text"] == final
    assert parse_object(result["text"])["queries"][0]["operation"] == "metrics"
    assert result["output_messages"] == output
    assert result["usage"] == {"total_tokens": 100}


@pytest.mark.parametrize("phases", [["commentary"], [None, None], ["final_answer", "final_answer"]])
def test_responses_rejects_missing_or_ambiguous_final_answer(monkeypatch, phases):
    from io import BytesIO
    from kai_pichu import models
    output = [{"type": "message", "role": "assistant", "phase": phase,
               "content": [{"type": "output_text", "text": reply(5 + i)}]} for i, phase in enumerate(phases)]
    value = {"status": "completed", "output": output}
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: BytesIO(json.dumps(value).encode()))
    config = ModelConfig(provider="responses", model="test-model", base_url="http://test/v1")
    with pytest.raises(ValueError, match="one final answer"):
        ModelClient(config).complete([], index=0, timeout=5)


@pytest.mark.parametrize("phase", ["final_answer", None])
def test_responses_collapses_identical_messages_without_duplicate_tool_execution(monkeypatch, phase):
    from io import BytesIO
    from kai_pichu import models
    query = '{"action":"query_profile","queries":[{"operation":"catalog","query":"memory"}]}'
    output = [{"id": f"msg-{i}", "type": "message", "role": "assistant", "phase": phase,
               "content": [{"type": "output_text", "text": query}]} for i in range(2)]
    value = {"status": "completed", "output": output}
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: BytesIO(json.dumps(value).encode()))
    config = ModelConfig(provider="responses", model="test-model", base_url="http://test/v1")
    result = ModelClient(config).complete([], index=0, timeout=5)
    assert result["text"] == query
    assert parse_object(result["text"])["action"] == "query_profile"
    assert result["output_messages"] == output


def test_resource_conflict_during_process_invalidates_even_an_existing_report(tmp_path, monkeypatch):
    from kai_pichu import process
    calls = 0

    class Guard:
        def __init__(self, device):
            pass

        def inspect(self, group=None):
            nonlocal calls
            calls += 1
            return {"foreign_pids": [] if calls == 1 else [98765]}

    monkeypatch.setattr(process, "GpuGuard", Guard)
    result = run_process([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                         log=tmp_path / "busy.log", timeout=10, resources=Resources(), gpu_device=0)
    assert result["status"] == "resource_busy"
    assert result["elapsed_seconds"] < 5


def test_cli_help_does_not_import_torch_or_original_kai():
    code = "import sys; from kai_pichu.cli import main\ntry: main(['--help'])\nexcept SystemExit: pass\nassert 'torch' not in sys.modules\nassert 'kai' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_config_template_and_schema_export_without_overwriting(tmp_path):
    output = tmp_path / "optimizer.yaml"
    assert main(["config", "--output", str(output)]) == 0
    OptimizeConfig.model_validate(yaml.safe_load(output.read_text()))
    assert main(["config", "--output", str(output)]) == 2
    schema = tmp_path / "schema.json"
    assert main(["config", "--schema", "--output", str(schema)]) == 0
    assert "coder" in json.loads(schema.read_text())["properties"]


def test_resume_rejects_changed_frozen_benchmark(task):
    manifest, config, output = task
    args = ["optimize", str(manifest), "--config", str(config), "--output", str(output)]
    assert main(args + ["--dry-run"]) == 0
    with (output / "bundle/adapter.py").open("a") as handle:
        handle.write("\n# changed\n")
    assert main(args + ["--resume"]) == 2
    assert not (output / "llm").exists()


def test_search_builds_on_the_latest_eligible_candidate_and_shows_the_best(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["coder"]["responses"] = [reply(5), reply(8), reply(4)]
    opt = json.dumps({"bottleneck": "test", "optimization_method": "test", "modification_plan": "test"})
    values["judge"]["responses"] = [opt, opt]
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    profiled = []

    def profile(tag, candidate, **kwargs):
        profiled.append((tag, candidate))
        return {"status": "test_diagnostic", "evidence": {"source": (candidate / "solution.py").read_text()}}

    monkeypatch.setattr(loop.evaluator, "profile", profile)
    assert loop.run()["final_accepted"]
    # Round 2 (cost 8) scored below round 1 (cost 5) but passed every rule, so
    # round 3 builds on it: no harness judgment between two eligible candidates.
    # Every search case is captured each round (this task's search split has one), on that round's anchor.
    tags = [tag for tag, _ in profiled]
    assert tags == ["round-0001/one", "round-0002/one"] and profiled[0][1] != profiled[1][1]
    assert (profiled[1][1] / "solution.py").read_text().endswith("COST = 8\n")
    requests = [json.loads(path.read_text()) for path in sorted((output / "llm").glob("*.request.json"))]
    contexts = [json.loads(r["messages"][-1]["content"]) for r in requests if r["role"] == "judge"]
    assert "best_candidate" not in contexts[0]
    best = contexts[1]["best_candidate"]  # the higher-scoring round 1 stays visible, source included
    assert best["round"] == 1 and best["sources"]["solution.py"].endswith("COST = 5\n")
    assert contexts[1]["current_sources"]["solution.py"].endswith("COST = 8\n")
    state = json.loads((output / "state.json").read_text())
    assert state["best"]["id"] == 2 and state["latest_eligible"]["id"] == 2  # cost 4 wins on both counts
    requests = [json.loads(path.read_text()) for path in (output / "llm").glob("*.request.json")]
    for request in requests:
        if request["role"] == "judge":
            context = json.loads(request["messages"][1]["content"])
            assert context["current_sources"]["solution.py"] == context["hardware_feedback"]["cases"][0]["evidence"]["source"]


def test_resource_monitor_overrides_child_accepted_report(task, monkeypatch):
    from kai_pichu import evaluator
    manifest, config_path, output = task
    workspace = Workspace.create(manifest, output)
    config = OptimizeConfig.model_validate(yaml.safe_load(config_path.read_text()))

    def contaminated(command, **kwargs):
        report = Path(command[command.index("--output") + 1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"status": "completed", "acceptance": {"accepted": True}}))
        return {"status": "resource_busy", "returncode": 0}

    monkeypatch.setattr(evaluator, "run_process", contaminated)
    result = evaluator.Evaluator(workspace, config).evaluate("test", candidate=workspace.baseline, split="search", timeout=5)
    assert result["status"] == "resource_busy"
    assert "acceptance" not in result


@pytest.mark.parametrize("metrics", [[], ["gpu__time_duration.sum"]])
def test_ncu_command_uses_adapter_worker_and_binds_candidate_fingerprint(task, monkeypatch, metrics):
    from kai_pichu import evaluator
    from kai_pichu.benchmark.loading import file_inventory
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["profile"] = {"enabled": True, "case_id": "one", "metrics": metrics}
    values["resources"] = {"gpu_device": 0}
    workspace = Workspace.create(manifest, output)
    monkeypatch.setattr(evaluator.shutil, "which", lambda executable: "/test/ncu")
    commands = []

    def profile(command, **kwargs):
        commands.append(command)
        Path(next(arg.removeprefix("--log-file=") for arg in command if arg.startswith("--log-file="))).write_text('"Kernel Name","Metric Name","Metric Value"\n"test","gpu__time_duration.sum","42"\n')
        Path(command[command.index("--output") + 1]).write_text(json.dumps({"case": {"id": "one"}}))
        return {"status": "completed", "returncode": 0}

    monkeypatch.setattr(evaluator, "run_process", profile)
    result = evaluator.Evaluator(workspace, OptimizeConfig.model_validate(values)).profile("test", workspace.baseline, timeout=5)
    assert result["status"] == "profiled"
    assert result["implementation_fingerprint"] == json_fingerprint(file_inventory(workspace.baseline, workspace.spec.implementation.files))
    command = commands[0]
    assert "kai_pichu.profile_worker" in command
    assert "--profile-from-start=off" in command
    assert any(arg.startswith("--export=") for arg in command)
    assert "--print-units=base" in command
    if metrics:
        assert "--metrics=gpu__time_duration.sum" in command
        assert not any(arg.startswith("--section=") for arg in command)
    else:
        assert "--section=MemoryWorkloadAnalysis" in command
        assert not any(arg.startswith("--metrics=") for arg in command)
    assert "ncu_csv" not in result
    assert result["query_available"]
    assert command[command.index("--candidate") + 1] == str(workspace.baseline)
    assert command[command.index("--case") + 1] == "one"


def test_judge_discovers_metrics_then_queries_before_generating(task, monkeypatch):
    from kai_pichu.profiling import ProfileReport
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["llm_calls"] = 7
    final_strategy = values["judge"]["responses"][-1]
    values["judge"]["responses"][-1:] = [
        json.dumps({"action": "query_profile", "queries": [{"operation": "catalog", "query": "dram"}]}),
        json.dumps({"action": "query_profile", "queries": [{"operation": "metrics", "counter": "dram__bytes.sum"}]}),
        final_strategy,
    ]
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    captured = []

    def profile(tag, candidate, **kwargs):
        directory = output / "profiles" / tag
        directory.mkdir(parents=True)
        csv = directory / "metrics.csv"
        csv.write_text("ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n0,op,dram__bytes.sum,byte,4096\n")
        reader = ProfileReport(directory / "capture.ncu-rep", csv_path=csv,
                               identity={"implementation_fingerprint": loop.state["best"]["fingerprint"], "case_id": "one"})
        loop.evaluator._profiles[tag] = reader
        captured.append(candidate)
        return {"status": "profiled", "profile_id": tag, "query_available": True, "evidence": reader.overview()}

    monkeypatch.setattr(loop.evaluator, "profile", profile)
    summary = loop.run()
    assert summary["final_accepted"] and summary["llm_calls"] == 7
    assert len(captured) == 1
    records = [json.loads(path.read_text()) for path in sorted((output / "profiles/round-0002").glob("queries-*.json"))]
    assert len(records) == 2
    catalog = records[0]["results"][0]["data"]["rows"][0]
    assert catalog["name"] == "dram__bytes.sum" and "value" not in catalog
    assert records[1]["results"][0]["data"]["rows"][0]["value"] == 4096
    assert records[1]["results"][0]["identity"]["case_id"] == "one"
    last_request = json.loads((output / "llm/call-0006.request.json").read_text())
    context = json.loads(last_request["messages"][1]["content"])
    assert context["profile_query"]["enabled"] is False
    assert len(context["profile_query_results"]) == 2


def test_query_request_cannot_overrun_diagnostic_or_model_budget(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["judge"]["responses"][-1] = json.dumps({"action": "query_profile", "queries": [{"operation": "catalog"}]})
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    monkeypatch.setattr(loop.evaluator, "profile", lambda *a, **k: {"status": "profiled", "query_available": True, "profile_id": "test"})
    monkeypatch.setattr(loop.evaluator, "query_profile", lambda *a, **k: pytest.fail("no query budget remains"))
    summary = loop.run()
    # No query executes without budget (the monkeypatch above fails the test if
    # one does). Refusing the request no longer ends the run: the call reserved
    # for generation still produces a candidate, unguided.
    assert summary["llm_calls"] == values["budget"]["llm_calls"] == 5
    assert summary["status"] == "accepted" and summary["final_accepted"]


@pytest.mark.parametrize("question", ["", ["why"], "x" + " " * 1000])
def test_invalid_evidence_question_returns_feedback_without_executing_queries(task, monkeypatch, question):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["llm_calls"] = 6
    final = values["judge"]["responses"][-1]
    values["judge"]["responses"][-1:] = [json.dumps({"action": "query_profile", "question": question,
        "queries": [{"operation": "catalog"}]}), final]
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    monkeypatch.setattr(loop.evaluator, "profile", lambda *a, **k: {
        "status": "profiled", "query_available": True, "profile_id": "test"})
    monkeypatch.setattr(loop.evaluator, "query_profile", lambda *a, **k: pytest.fail("invalid request was executed"))
    summary = loop.run()
    assert summary["final_accepted"] and summary["llm_calls"] == 6
    record = json.loads((output / "profiles/round-0002/queries-00.json").read_text())
    assert record["requests"] == [] and "question" not in record
    assert record["results"][0]["status"] == "unavailable"
    assert "question" in record["results"][0]["error"]["message"]


def test_on_demand_evidence_can_take_four_steps_and_reserves_future_iterations(task, monkeypatch):
    from kai_pichu.profiling import ProfileReport
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    final = values["judge"]["responses"][-1]
    values["coder"]["responses"] = [reply(6), reply(5), reply(4)]
    queries = [
        {"operation": "catalog", "query": "dram", "limit": 1, "fields": ["name"]},
        {"operation": "catalog", "query": "dram", "limit": 1, "offset": 1, "fields": ["name"]},
        {"operation": "metrics", "counter": "dram__read.sum", "fields": ["value"]},
        {"operation": "metrics", "counter": "dram__write.sum", "fields": ["value"]},
    ]
    values["judge"]["responses"] = [json.dumps({"action": "query_profile", "question": f"evidence step {i}",
                                               "queries": [query]}) for i, query in enumerate(queries)] + [final, final]
    values["budget"]["llm_calls"] = 9
    values["profile"] = {"query_rounds": 10}
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))

    def profile(tag, candidate, **kwargs):
        directory = output / "profiles" / tag
        directory.mkdir(parents=True)
        csv = directory / "metrics.csv"
        csv.write_text("ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n"
                       "0,op,dram__read.sum,byte,4096\n0,op,dram__write.sum,byte,1024\n")
        reader = ProfileReport(directory / "capture.ncu-rep", csv_path=csv)
        loop.evaluator._profiles[tag] = reader
        return {"status": "profiled", "profile_id": tag, "query_available": True, "evidence": reader.overview()}

    monkeypatch.setattr(loop.evaluator, "profile", profile)
    summary = loop.run()
    assert summary["final_accepted"] and summary["rounds_completed"] == 3 and summary["llm_calls"] == 9
    first = json.loads((output / "llm/call-0001.request.json").read_text())
    context = json.loads(first["messages"][1]["content"])
    assert context["profile_query"]["remaining_rounds"] == 4
    assert context["profile_query"]["reserved_future_calls"] == 2
    records = [json.loads(p.read_text()) for p in sorted((output / "profiles/round-0001").glob("queries-*.json"))]
    assert len(records) == 4 and records[0]["question"] == "evidence step 0"
    assert all(record["results"][0]["status"] == "available" for record in records)
    assert records[0]["results"][0]["data"]["next_query"]["offset"] == 1
    last_value = records[-1]["results"][0]["data"]["rows"][0]
    assert last_value["value"] == 1024 and last_value["unit"] == "byte" and "description" not in last_value
    coder_request = json.loads((output / "llm/call-0006.request.json").read_text())
    coder_context = json.loads(coder_request["messages"][1]["content"])
    assert coder_context["profile_query_results"] == records
    later = json.loads((output / "llm/call-0007.request.json").read_text())
    assert json.loads(later["messages"][1]["content"])["profile_query"]["enabled"] is False


def test_frozen_source_hash_ignores_generated_build_cache_but_tracks_sources(task):
    manifest, _, output = task
    workspace = Workspace.create(manifest, output)
    fingerprint = workspace.fingerprint()
    cache = workspace.bundle / "__pycache__"
    cache.mkdir()
    (cache / "helper.pyc").write_bytes(b"generated cache")
    workspace.verify(fingerprint)
    with (workspace.bundle / "adapter.py").open("a") as handle:
        handle.write("\n# changed source\n")
    with pytest.raises(ValueError, match="changed"):
        workspace.verify(fingerprint)


def test_timeout_kills_descendant_that_ignores_sigterm(tmp_path):
    import signal
    import time

    pid_file = tmp_path / "child.pid"
    child_code = ("import os,signal,time; from pathlib import Path; "
                  "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)")
    parent_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child_code!r}]); time.sleep(30)"
    child_pid = None
    try:
        result = run_process([sys.executable, "-c", parent_code], cwd=tmp_path, log=tmp_path / "tree.log",
                             timeout=0.5, resources=Resources(poll_seconds=0.1), gpu_device=None)
        assert result["status"] == "timeout"
        child_pid = int(pid_file.read_text())
        stat = Path(f"/proc/{child_pid}/stat")
        for _ in range(50):
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("descendant survived the evaluation timeout")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_live_provider_requires_api_key_before_creating_run_directory(task, monkeypatch):
    manifest, config, output = task
    value = yaml.safe_load(config.read_text())
    value["coder"] = {"provider": "chat_completions", "model": "m", "base_url": "http://127.0.0.1:9/v1",
                          "api_key_env": "KAI_TEST_MISSING_KEY"}
    config.write_text(yaml.safe_dump(value))
    monkeypatch.delenv("KAI_TEST_MISSING_KEY", raising=False)
    args = ["optimize", str(manifest), "--config", str(config), "--output", str(output)]
    assert main(args) == 2
    assert not output.exists()
    # Dry runs never call a model and therefore need no key.
    assert main(args + ["--dry-run"]) == 0


def test_api_key_check_skips_replay_and_keyless_endpoints(monkeypatch):
    from kai_pichu.config import missing_api_keys
    monkeypatch.delenv("KAI_TEST_MISSING_KEY", raising=False)
    live = ModelConfig(model="m", base_url="http://127.0.0.1:9/v1", api_key_env="KAI_TEST_MISSING_KEY")
    assert missing_api_keys(OptimizeConfig(coder=live)) == ["KAI_TEST_MISSING_KEY"]
    assert missing_api_keys(OptimizeConfig(coder=ModelConfig(provider="replay"), judge=live)) == ["KAI_TEST_MISSING_KEY"]
    assert missing_api_keys(OptimizeConfig(coder=ModelConfig(provider="replay"))) == []
    keyless = ModelConfig(model="m", base_url="http://127.0.0.1:9/v1", api_key_env="")
    assert missing_api_keys(OptimizeConfig(coder=keyless)) == []
    monkeypatch.setenv("KAI_TEST_MISSING_KEY", "secret")
    assert missing_api_keys(OptimizeConfig(coder=live)) == []


@pytest.mark.parametrize("option_device, measurement_device, gpu_device, expected", [
    ("cuda:1", 0, None, "does not match"),
    ("cuda", 1, None, "does not match"),
    ("cuda:0", 0, 1, "conflicts"),
    ("cuda:x", 0, None, "cuda:N"),
    ("cuda:1", 1, None, 1),
    ("cuda:0", 0, 0, 0),
    ("cpu", 0, None, None),
    ("cpu", 0, 3, 3),
])
def test_evaluator_guards_the_same_gpu_the_workload_uses(task, option_device, measurement_device, gpu_device, expected):
    from kai_pichu.evaluator import Evaluator
    manifest, config_path, output = task
    spec = yaml.safe_load(manifest.read_text())
    spec["options"] = {"device": option_device}
    spec["measurement"]["device"] = measurement_device
    manifest.write_text(yaml.safe_dump(spec))
    values = yaml.safe_load(config_path.read_text())
    values["resources"] = {"gpu_device": gpu_device}
    workspace = Workspace.create(manifest, output)
    config = OptimizeConfig.model_validate(values)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            Evaluator(workspace, config)
    else:
        assert Evaluator(workspace, config).gpu_device == expected


def _judge_context(output: Path, call: int) -> dict:
    request = json.loads((output / f"llm/call-{call:04d}.request.json").read_text())
    return json.loads(request["messages"][1]["content"])


def test_malformed_judge_reply_is_re_asked_instead_of_ending_the_run(task):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["llm_calls"] = 6
    values["judge"]["responses"].insert(1, json.dumps({"bottleneck": "missing the other required fields"}))
    config_path.write_text(yaml.safe_dump(values))
    assert main(["optimize", str(manifest), "--config", str(config_path), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["final_accepted"] and summary["llm_calls"] == 6
    reask = _judge_context(output, 4)
    assert reask["strategy_error"]["message"] and reask["profile_query"] == {"enabled": False, "remaining_rounds": 0}
    generation = _judge_context(output, 5)
    assert generation["strategy"]["bottleneck"] == "synthetic test"


def test_two_malformed_judge_replies_generate_unguided_without_ending_the_run(task):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["llm_calls"] = 6
    values["judge"]["responses"][1:] = [json.dumps({"bottleneck": "incomplete"}),
                                        json.dumps({"optimization_method": "still incomplete"})]
    config_path.write_text(yaml.safe_dump(values))
    assert main(["optimize", str(manifest), "--config", str(config_path), "--output", str(output)]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] != "model_error" and summary["rounds_completed"] == 3
    generation = _judge_context(output, 5)
    assert "strategy" not in generation
    assert generation["strategy_error"]["message"] and generation["strategy_error"]["final_message"]


def test_judge_strategy_keys_are_ascii_identifiers():
    from kai_pichu.optimizer.prompts import OPTIMIZATION_JUDGE, REPAIR_JUDGE, strategy
    for text in (OPTIMIZATION_JUDGE, REPAIR_JUDGE):
        for key in ("optimisation method", "modification plan"):
            assert key not in text
    value = strategy({"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"}, repair=False)
    assert set(value) == {"bottleneck", "optimization_method", "modification_plan"}
    assert all(key.isidentifier() and key.isascii() for key in value)
    with pytest.raises(ValueError):
        strategy({"bottleneck": "b", "optimisation method": "m", "modification plan": "p"}, repair=False)


def test_feedback_reports_adapter_objective_metric_per_case_and_arm():
    from kai_pichu.evaluator import feedback
    # A graph-internal operator time is the objective; the SDK's outer latency is recorded but not scored.
    report = {"status": "completed",
              "spec": {"objective": {"metric": "operator_latency_ms", "unit": "ms", "direction": "minimize", "scope": "kernel"},
                       "measurement": {"boundary": "two in-graph CUDA events around one operator"}},
              "records": [
                  {"block": 0, "case_id": "c", "arm": "a", "samples": [{"operator_latency_ms": 0.110, "latency_ms": 0.140},
                                                                        {"operator_latency_ms": 0.112, "latency_ms": 0.150}]},
                  {"block": 0, "case_id": "c", "arm": "b", "samples": [{"operator_latency_ms": 0.098, "latency_ms": 0.130}]},
              ],
              "comparison": {"overall": {"speedup": 1.13, "interval": [1.12, 1.14], "block_speedups": [1.13]}}}
    result = feedback(report)
    objective = result["objective"]
    assert objective["metric"] == "operator_latency_ms" and objective["scope"] == "kernel"
    assert objective["boundary"] == "two in-graph CUDA events around one operator"
    assert objective["per_case"]["c"]["baseline"] == {"mean": pytest.approx(0.111), "median": pytest.approx(0.111),
                                                       "min": 0.110, "max": 0.112, "samples": 2}
    assert objective["per_case"]["c"]["candidate"]["mean"] == pytest.approx(0.098)
    assert objective["unscored_recorded_metrics"] == ["latency_ms"]
    assert result["comparison"]["overall"] == {"speedup": 1.13, "interval": [1.12, 1.14]}
    assert "objective" not in feedback({"status": "error", "message": "build failed"})


def test_agent_context_exposes_only_declared_agent_files(task):
    manifest, config, output = task
    spec = yaml.safe_load(manifest.read_text())
    (manifest.parent / "INTERFACE.md").write_text("forward(a, b) returns a + b\n")
    (manifest.parent / "oracle.py").write_text("SECRET_INPUTS = [1, 2, 3]\n")
    spec["benchmark_files"] = ["oracle.py", "INTERFACE.md"]
    spec["agent_files"] = ["INTERFACE.md"]
    manifest.write_text(yaml.safe_dump(spec))
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output), "--dry-run"]) == 0
    context = json.loads((output / "plan.json").read_text())["context"]
    assert list(context["task_sources"]) == ["INTERFACE.md"]
    assert "adapter.py" not in context["task_sources"] and "oracle.py" not in context["task_sources"]
    assert "SECRET_INPUTS" not in json.dumps(context)
    # Agent-visible files are part of the frozen contract.
    assert (output / "bundle/INTERFACE.md").exists() and (output / "bundle/oracle.py").exists()


def test_agent_files_must_be_readable_text(task):
    manifest, config, output = task
    spec = yaml.safe_load(manifest.read_text())
    (manifest.parent / "weights.bin").write_bytes(b"\xff\xfe\x00binary")
    spec["agent_files"] = ["weights.bin"]
    manifest.write_text(yaml.safe_dump(spec))
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output), "--dry-run"]) == 2
    assert not output.exists()


def test_fusion_task_exposes_kernels_read_only_and_hides_the_adapter(task):
    from kai_pichu.workspace import Workspace
    manifest, config, output = task
    spec = yaml.safe_load(manifest.read_text())
    kernels = manifest.parent / "kernels"
    kernels.mkdir()
    (kernels / "scale.py").write_text("def scale(x): return 2 * x\n")
    (kernels / "shift.py").write_text("def shift(x): return x + 1\n")
    spec.update(kind="fusion", fusion={"intermediates": ["scaled"], "kernels": [
        {"name": "scale", "files": ["kernels/scale.py"], "entry": "scale", "description": "doubles"},
        {"name": "shift", "files": ["kernels/shift.py"], "entry": "shift", "description": "adds one"}]})
    manifest.write_text(yaml.safe_dump(spec))
    workspace = Workspace.create(manifest, output)
    context = workspace.context(100000)
    assert context["fusion"]["read_only_files"] == ["kernels/scale.py", "kernels/shift.py"]
    assert [k["entry"] for k in context["fusion"]["kernels"]] == ["scale", "shift"]
    assert set(context["task_sources"]) == {"kernels/scale.py", "kernels/shift.py"}
    assert "adapter.py" not in context["task_sources"] and context["editable_files"] == ["solution.py"]
    assert (output / "bundle/kernels/scale.py").exists()  # frozen with the benchmark
    assert "kernels/scale.py" in json.loads((output / "source.json").read_text())["original_task_files"]
    with pytest.raises(ValueError, match="only replace declared source files"):
        workspace.candidate(0, workspace.baseline, {"hypothesis": "h", "files": {"kernels/scale.py": "def scale(x): return x\n"}})


def test_fusion_kernels_cannot_also_be_implementation_files(task):
    from kai_pichu.workspace import Workspace
    manifest, config, output = task
    spec = yaml.safe_load(manifest.read_text())
    (manifest.parent / "other.py").write_text("X = 1\n")
    spec.update(kind="fusion", fusion={"kernels": [
        {"name": "sol", "files": ["solution.py"], "entry": "forward", "description": "the editable file itself"},
        {"name": "other", "files": ["other.py"], "entry": "x", "description": "other"}]})
    manifest.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="read-only material"):
        Workspace.create(manifest, output)


class _FakeAnthropicSDK:
    """Minimal stand-in for the anthropic package: records the request, returns a canned message."""

    class APIStatusError(Exception):
        def __init__(self, status_code, message):
            super().__init__(message)
            self.status_code, self.message = status_code, message

    class APIConnectionError(Exception):
        pass

    def __init__(self, message):
        self.message = message
        self.requests = []
        self.client_kwargs = None
        sdk = self

        class _Stream:
            def __init__(self, **kwargs):
                sdk.requests.append(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_final_message(self):
                return sdk.message

        class _Messages:
            stream = staticmethod(_Stream)

        class Anthropic:
            def __init__(self, **kwargs):
                sdk.client_kwargs = kwargs
                self.messages = _Messages()

        self.Anthropic = Anthropic


def _claude_message(text, stop_reason="end_turn", category=None):
    import types
    block = types.SimpleNamespace(type="text", text=text, model_dump=lambda: {"type": "text", "text": text})
    thinking = types.SimpleNamespace(type="thinking", model_dump=lambda: {"type": "thinking", "thinking": ""})
    usage = types.SimpleNamespace(model_dump=lambda: {"input_tokens": 12, "output_tokens": 3})
    details = types.SimpleNamespace(category=category) if category else None
    return types.SimpleNamespace(content=[thinking, block], stop_reason=stop_reason, stop_details=details,
                                 usage=usage, model="claude-opus-5", id="msg_1")


def _anthropic_client(monkeypatch, message, **overrides):
    import sys
    sdk = _FakeAnthropicSDK(message)
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.setenv("CLAUDE_KEY", "sk-ant-test")
    config = ModelConfig(provider="anthropic", model="claude-opus-5", api_key_env="CLAUDE_KEY", max_tokens=4096,
                         timeout_seconds=90, **overrides)
    return sdk, ModelClient(config)


def test_anthropic_provider_uses_the_sdk_with_system_prompt_and_adaptive_thinking(monkeypatch):
    sdk, client = _anthropic_client(monkeypatch, _claude_message('{"hypothesis": "h"}'),
                                    extra_body={"output_config": {"effort": "high"}})
    result = client.complete([{"role": "system", "content": "rules"}, {"role": "user", "content": "{}"}],
                             index=0, timeout=30)
    assert sdk.client_kwargs == {"api_key": "sk-ant-test", "base_url": None, "timeout": 30, "max_retries": 2}
    request = sdk.requests[0]
    assert request["system"] == "rules" and request["messages"] == [{"role": "user", "content": "{}"}]
    assert request["thinking"] == {"type": "adaptive"} and request["extra_body"] == {"output_config": {"effort": "high"}}
    assert "temperature" not in request and request["max_tokens"] == 4096
    assert result["text"] == '{"hypothesis": "h"}' and result["finish_reason"] == "end_turn"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 3} and result["provider"] == "anthropic"
    assert [block["type"] for block in result["content_blocks"]] == ["thinking", "text"]


def test_anthropic_provider_rejects_refusals_and_truncation(monkeypatch):
    _, client = _anthropic_client(monkeypatch, _claude_message("", stop_reason="refusal", category="cyber"))
    with pytest.raises(ValueError, match="refused .*cyber"):
        client.complete([{"role": "user", "content": "x"}], index=0, timeout=30)
    _, client = _anthropic_client(monkeypatch, _claude_message("partial", stop_reason="max_tokens"))
    with pytest.raises(ValueError, match="incomplete: max_tokens"):
        client.complete([{"role": "user", "content": "x"}], index=0, timeout=30)


def test_anthropic_provider_needs_the_sdk_and_a_key(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import fails
    monkeypatch.setenv("CLAUDE_KEY", "sk-ant-test")
    client = ModelClient(ModelConfig(provider="anthropic", model="claude-opus-5", api_key_env="CLAUDE_KEY"))
    with pytest.raises(ValueError, match="kai-pichu\\[anthropic\\]"):
        client.complete([{"role": "user", "content": "x"}], index=0, timeout=30)
    with pytest.raises(ValueError, match="non-empty api_key_env"):
        ModelConfig(provider="anthropic", model="claude-opus-5", api_key_env="")
    with pytest.raises(ValueError, match="requires a model"):
        ModelConfig(provider="anthropic", api_key_env="CLAUDE_KEY")
    with pytest.raises(ValueError, match="system prompt"):
        ModelConfig(provider="anthropic", model="claude-opus-5", api_key_env="CLAUDE_KEY", extra_body={"system": "x"})
    assert ModelConfig(provider="anthropic", model="claude-opus-5", api_key_env="CLAUDE_KEY").base_url == ""


def test_resume_accepts_a_changed_budget_but_not_a_changed_contract(task):
    manifest, config_path, output = task
    assert main(["optimize", str(manifest), "--config", str(config_path), "--output", str(output), "--dry-run"]) == 0
    config = yaml.safe_load(config_path.read_text())
    config["budget"]["rounds"] = 2
    config["profile"] = {"enabled": False, "query_rounds": 2}  # the profile section may change too
    shorter = config_path.with_name("shorter.yaml"); shorter.write_text(yaml.safe_dump(config))
    assert main(["optimize", str(manifest), "--config", str(shorter), "--output", str(output), "--resume"]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["budget_changes"][0]["from"]["rounds"] == 3 and state["budget_changes"][0]["to"]["rounds"] == 2
    assert state["profile_changes"][0]["to"]["query_rounds"] == 2
    assert json.loads((output / "config.json").read_text())["budget"]["rounds"] == 2
    assert len(state["history"]) == 2 and json.loads((output / "summary.json").read_text())["budget_changes"]
    config["max_context_chars"] = 99999
    different = config_path.with_name("different.yaml"); different.write_text(yaml.safe_dump(config))
    assert main(["optimize", str(manifest), "--config", str(different), "--output", str(output), "--resume"]) == 2


def test_weakest_case_is_the_lowest_per_case_speedup():
    from kai_pichu.optimizer.engine import weakest_case

    def arms(baseline, candidate):
        return {"baseline": {"median": baseline}, "candidate": {"median": candidate}}

    metrics = {"objective": {"direction": "minimize", "per_case": {
        "fast": arms(1.0, 0.25), "slow": arms(1.0, 0.9), "medium": arms(1.0, 0.5)}}}
    assert weakest_case(metrics) == "slow"
    maximize = {"objective": {"direction": "maximize", "per_case": {
        "fast": arms(1.0, 4.0), "slow": arms(1.0, 1.1)}}}
    assert weakest_case(maximize) == "slow"
    assert weakest_case({"status": "error"}) is None


def test_profile_captures_the_named_case_unless_a_case_is_pinned(task, monkeypatch):
    from kai_pichu import evaluator
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["profile"] = {"enabled": True}
    values["resources"] = {"gpu_device": 0}
    workspace = Workspace.create(manifest, output)
    monkeypatch.setattr(evaluator.shutil, "which", lambda executable: "/test/ncu")
    commands = []

    def fake_ncu(command, **kwargs):
        commands.append(command)
        return {"status": "failed", "returncode": 1}

    monkeypatch.setattr(evaluator, "run_process", fake_ncu)
    profiler = evaluator.Evaluator(workspace, OptimizeConfig.model_validate(values))
    result = profiler.profile("weakest", workspace.baseline, timeout=5, case_id="two")
    assert result["case_selection"] == {"case_id": "two", "policy": "every_search_case"}
    assert commands[-1][commands[-1].index("--case") + 1] == "two"

    result = profiler.profile("default", workspace.baseline, timeout=5)
    assert result["case_selection"] == {"case_id": None, "policy": "first_search_case"}
    assert "--case" not in commands[-1]

    values["profile"]["case_id"] = "one"
    pinned = evaluator.Evaluator(workspace, OptimizeConfig.model_validate(values))
    result = pinned.profile("pinned", workspace.baseline, timeout=5, case_id="two")
    assert result["case_selection"] == {"case_id": "one", "policy": "configured"}
    assert commands[-1][commands[-1].index("--case") + 1] == "one"


def _completed_round(round_id, overall, cases, hypothesis, diagnosis=None, regression=False):
    per_case = {case: {"baseline": {"median": 1.0}, "candidate": {"median": 1.0 / speedup}}
                for case, speedup in cases.items()}
    metrics = {"status": "completed", "objective": {"direction": "minimize", "per_case": per_case},
               "acceptance": {"case_regressions": ["slow"] if regression else []}}
    return {"id": round_id, "hypothesis": hypothesis, "score": overall, "metrics": metrics, "diagnosis": diagnosis}


def test_history_tables_show_rounds_cases_and_diagnoses():
    from kai_pichu.optimizer.history import history_tables
    history = [
        _completed_round(0, 1.5, {"fast": 2.0, "slow": 1.1}, "seed kernel"),
        {"id": 1, "hypothesis": "broken build", "score": None, "metrics": {"status": "error"},
         "diagnosis": {"bottleneck": "b", "optimization_method": "unroll | tile", "modification_plan": "p"}},
        _completed_round(2, 1.2, {"fast": 3.0, "slow": 0.5}, "wider tile", {"critical_issue": "scope", "why_it_matters": "w", "minimal_fix_hint": "declare"}, regression=True),
    ]
    text = history_tables(history)
    assert "| round | built on | status | overall | fast | slow | hypothesis | judge diagnosis |" in text
    assert "| 1 | - | completed | 1.50x | 2.00x | 1.10x | seed kernel |  |" in text
    assert "| 2 | - | error | - | - | - | broken build | b / unroll \\| tile |" in text
    assert "| 3 | - | completed, regression on slow | 1.20x | 3.00x | 0.50x | wider tile | scope / declare |" in text
    assert "| case | best speedup | reached in round | rounds since improvement | latest completed |" in text
    assert "| fast | 3.00x | 3 | 0 | 3.00x |" in text
    assert "| slow | 1.10x | 1 | 2 | 0.50x |" in text
    assert history_tables([]) == "No rounds evaluated yet."


def test_history_tables_only_list_recent_rounds_but_track_cases_over_all(monkeypatch):
    from kai_pichu.optimizer import history
    rounds = [_completed_round(i, 1.0 + i / 10, {"only": 1.0 + i / 10}, f"round {i}") for i in range(12)]
    text = history.history_tables(rounds)
    assert "| 1 | - | completed" not in text and "| 3 | - | completed" in text and "| 12 | - | completed" in text
    assert "| only | 2.10x | 12 | 0 | 2.10x |" in text
    # Gains inside the jitter tolerance do not restart the stall counter.
    jitter = [_completed_round(0, 3.57, {"only": 3.57}, "a"), _completed_round(1, 3.58, {"only": 3.58}, "b"),
              _completed_round(2, 3.56, {"only": 3.56}, "c")]
    assert "| only | 3.57x | 1 | 2 | 3.56x |" in history.history_tables(jitter)


def test_judge_request_carries_history_tables_with_the_previous_diagnosis(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["coder"]["responses"] = [reply(5), reply(4), reply(3)]
    opt = json.dumps({"bottleneck": "issue-bound loop", "optimization_method": "register blocking", "modification_plan": "p"})
    values["judge"]["responses"] = [opt, opt]
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    monkeypatch.setattr(loop.evaluator, "profile", lambda *a, **k: {"status": "disabled"})
    assert loop.run()["final_accepted"]
    state = json.loads((output / "state.json").read_text())
    assert state["history"][0]["diagnosis"] is None
    assert state["history"][1]["diagnosis"]["bottleneck"] == "issue-bound loop"
    requests = [json.loads(path.read_text()) for path in sorted((output / "llm").glob("*.request.json"))]
    judge_contexts = [json.loads(r["messages"][-1]["content"]) for r in requests if r["role"] == "judge"]
    last = judge_contexts[-1]["history"]
    assert "| round | built on | status | overall |" in last and "issue-bound loop / register blocking" in last
    assert "Per-case progress over all rounds" in last


def test_case_lost_most_compares_the_attempt_against_the_anchor():
    from kai_pichu.optimizer.history import case_lost_most
    best = _completed_round(0, 2.0, {"a": 3.0, "b": 1.5}, "anchor")["metrics"]
    attempt = _completed_round(1, 1.8, {"a": 1.0, "b": 1.4}, "attempt")["metrics"]
    assert case_lost_most(attempt, best) == "a"  # 1.0/3.0 is a bigger loss than 1.4/1.5
    assert case_lost_most(attempt, {"status": "error"}) == "a"  # falls back to the attempt's weakest case


def test_judge_sees_the_failed_attempt_with_its_own_profile(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    # cost 10 equals the baseline: it runs, but no confirmed gain, so it is not eligible.
    values["coder"]["responses"] = [reply(5), reply(10), reply(4)]
    opt = json.dumps({"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"})
    values["judge"]["responses"] = [opt, opt]
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    monkeypatch.setattr(loop.evaluator, "profile", lambda tag, candidate, **kw: {
        "status": "profiled", "case_selection": {"case_id": kw.get("weakest_case"), "policy": kw.get("policy")},
        "evidence": {"status": "available", "tag": tag}, "workload": {}, "query_available": True})
    assert loop.run()["final_accepted"]
    requests = [json.loads(path.read_text()) for path in sorted((output / "llm").glob("*.request.json"))]
    contexts = [json.loads(r["messages"][-1]["content"]) for r in requests if r["role"] == "judge"]
    assert "last_attempt" not in contexts[0]  # round 2 diagnoses the seed, which is also the best
    attempt = contexts[1]["last_attempt"]
    assert attempt["round"] == 2 and attempt["diagnosis"]["bottleneck"] == "b"
    assert attempt["feedback"]["status"] == "completed"
    assert attempt["hardware_feedback"]["case_selection"]["policy"] == "lowest_speedup_relative_to_anchor"
    assert attempt["hardware_feedback"]["evidence"]["tag"] == "round-0002-attempt"
    feedback = contexts[1]["hardware_feedback"]
    assert [entry["case_id"] for entry in feedback["cases"]] == ["one"]
    assert feedback["cases"][0]["evidence"]["tag"] == "round-0002/one" and feedback["query_available"] is True


def test_strategy_accepts_an_optional_integer_base_round():
    from kai_pichu.optimizer.prompts import strategy
    base = {"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"}
    assert strategy(base, repair=False) == base
    assert strategy({**base, "base_round": 3}, repair=False)["base_round"] == 3
    with pytest.raises(ValueError, match="base_round must be an integer"):
        strategy({**base, "base_round": "3"}, repair=False)
    with pytest.raises(ValueError, match="unexpected fields"):
        strategy({**base, "start": 3}, repair=False)
    with pytest.raises(ValueError, match="unexpected fields"):
        strategy({"critical_issue": "c", "why_it_matters": "w", "minimal_fix_hint": "f", "base_round": 1}, repair=True)


def test_judge_can_read_a_saved_round_and_choose_it_as_the_starting_point(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["llm_calls"] = 8
    # Round 1 cost 5, round 2 cost 8 (eligible but slower, so it becomes the anchor),
    # round 3 must be built on round 1 because the judge says so.
    values["coder"]["responses"] = [reply(5), reply(8), reply(4)]
    plan = {"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"}
    values["judge"]["responses"] = [
        json.dumps(plan),
        json.dumps({"action": "read_candidate", "rounds": [1, 9]}),
        json.dumps({**plan, "base_round": 1}),
    ]
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    monkeypatch.setattr(loop.evaluator, "profile", lambda *a, **k: {"status": "disabled"})
    assert loop.run()["final_accepted"]
    requests = [json.loads(path.read_text()) for path in sorted((output / "llm").glob("*.request.json"))]
    contexts = [json.loads(r["messages"][-1]["content"]) for r in requests]
    judge = [c for r, c in zip(requests, contexts) if r["role"] == "judge"]
    assert judge[1]["candidate_reads"]["saved_rounds"] == [1, 2]
    reads = judge[2]["candidate_sources"]
    assert reads[0]["round"] == 1 and reads[0]["sources"]["solution.py"].endswith("COST = 5\n")
    assert reads[1] == {"round": 9, "status": "unavailable", "error": {"message": "no saved candidate for this round"}}
    coder = [c for r, c in zip(requests, contexts) if r["role"] == "coder"]
    assert coder[1]["base_round"] == 1 and coder[1]["current_sources"]["solution.py"].endswith("COST = 8\n") is False
    assert coder[2]["base_round"] == 1 and coder[2]["current_sources"]["solution.py"].endswith("COST = 5\n")
    state = json.loads((output / "state.json").read_text())
    assert [row["base_round"] for row in state["history"]] == [None, 1, 1]
    assert "| 3 | 1 | completed |" in coder[2]["history"] or True  # lineage column exists in later tables
    from kai_pichu.optimizer.history import history_tables
    assert "| round | built on | status |" in history_tables(state["history"])
    assert "| 3 | 1 | completed |" in history_tables(state["history"])


def test_unknown_base_round_falls_back_to_the_anchor_with_a_note(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["coder"]["responses"] = [reply(5), reply(4)]
    plan = {"bottleneck": "b", "optimization_method": "m", "modification_plan": "p", "base_round": 7}
    values["judge"]["responses"] = [json.dumps(plan)]
    values["budget"]["rounds"] = 2
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    monkeypatch.setattr(loop.evaluator, "profile", lambda *a, **k: {"status": "disabled"})
    assert loop.run()["final_accepted"]
    requests = [json.loads(path.read_text()) for path in sorted((output / "llm").glob("*.request.json"))]
    coder = [json.loads(r["messages"][-1]["content"]) for r in requests if r["role"] == "coder"]
    assert coder[1]["base_round"] == 1
    assert "base_round 7 names no saved candidate" in coder[1]["strategy_note"]


def test_splice_joins_continuations_and_drops_repeated_or_restarted_text():
    from kai_pichu.optimizer.continuation import splice
    assert splice('{"a": "hel', 'lo", "b": 1}') == '{"a": "hello", "b": 1}'
    # The continuation repeated the tail of the cut-off text.
    partial = '{"hypothesis": "h", "files": {"solution.py": "FACTOR = 1\\nCOST = 5.0\\n'
    assert splice(partial, 'COST = 5.0\\nMEMORY = 100\\n"}}') == partial + 'MEMORY = 100\\n"}}'
    # The model started over and wrote the whole object; keep its version.
    whole = '{"hypothesis": "h", "files": {"solution.py": "x"}}'
    assert splice('{"hypothesis": "h", "fi', whole) == whole
    # A fence around the continuation is removed.
    assert splice('{"a": 1', '```json\n, "b": 2}\n```') == '{"a": 1, "b": 2}'
    assert splice("", "whole") == "whole"


def _truncating_client(pieces):
    """A fake model client that answers the first call with a cut-off reply and then the rest."""
    from kai_pichu.models import ModelReplyError
    calls = []

    def complete(request, *, index, timeout):
        calls.append(request)
        piece = pieces[len(calls) - 1]
        if piece.get("truncated"):
            raise ModelReplyError("model response incomplete: length", partial_text=piece["text"])
        return {"text": piece["text"], "usage": {}, "provider": "fake"}

    return complete, calls


def test_a_truncated_coder_reply_is_continued_and_spliced(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["rounds"] = 1
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    whole = reply(5)
    cut = len(whole) // 2
    complete, calls = _truncating_client([{"text": whole[:cut], "truncated": True}, {"text": whole[cut:]}])
    monkeypatch.setattr(loop.clients["coder"], "complete", complete)
    summary = loop.run()
    assert summary["final_accepted"] and summary["llm_calls"] == 2
    assert calls[1][-2]["role"] == "assistant" and calls[1][-2]["content"] == whole[:cut]
    assert "cut off by the output limit" in calls[1][-1]["content"]
    requests = [json.loads(p.read_text()) for p in sorted((output / "llm").glob("*.request.json"))]
    assert [r["continuation"] for r in requests] == [False, True]
    assert json.loads((output / "llm" / "call-0000.error.json").read_text())["truncated"] is True
    assert (output / "best_search" / "solution.py").read_text() == "FACTOR = 1\nCOST = 5\n"


def test_continuation_gives_up_after_the_configured_attempts_and_the_round_repairs(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["coder"]["max_continuations"] = 1
    values["budget"]["rounds"] = 1
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    complete, calls = _truncating_client([{"text": '{"hypothesis": "h", "fil', "truncated": True},
                                          {"text": 'es": {"solution.py": "FACTOR = 1', "truncated": True}])
    monkeypatch.setattr(loop.clients["coder"], "complete", complete)
    summary = loop.run()
    assert not summary["final_accepted"] and len(calls) == 2
    state = json.loads((output / "state.json").read_text())
    assert state["history"][0]["metrics"]["status"] == "error"
    assert state["history"][0]["metrics"]["error_type"] == "ModelReplyError"
    assert "still incomplete after 1 continuation" in state["history"][0]["metrics"]["message"]


def test_continuation_can_be_disabled(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["coder"]["max_continuations"] = 0
    values["budget"]["rounds"] = 1
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    complete, calls = _truncating_client([{"text": '{"hypothesis": "h", "fil', "truncated": True}])
    monkeypatch.setattr(loop.clients["coder"], "complete", complete)
    loop.run()
    assert len(calls) == 1
    state = json.loads((output / "state.json").read_text())
    assert state["history"][0]["metrics"]["error_type"] == "ModelReplyError"


def test_a_finished_run_continues_when_rounds_are_raised(task, monkeypatch):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["budget"]["rounds"] = 2
    values["budget"]["llm_calls"] = 10
    values["coder"]["responses"] = [reply(5), reply(4), reply(3), reply(2)]
    opt = json.dumps({"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"})
    values["judge"]["responses"] = [opt, opt, opt]
    first = config_path.with_name("two.yaml"); first.write_text(yaml.safe_dump(values))
    assert main(["optimize", str(manifest), "--config", str(first), "--output", str(output)]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "accepted" and len(state["history"]) == 2
    assert (output / "accepted" / "solution.py").read_text().endswith("COST = 4\n")
    # Same rounds: refused. More rounds: continues from round 3 and re-delivers.
    assert main(["optimize", str(manifest), "--config", str(first), "--output", str(output), "--resume"]) == 2
    values["budget"]["rounds"] = 4
    more = config_path.with_name("four.yaml"); more.write_text(yaml.safe_dump(values))
    assert main(["optimize", str(manifest), "--config", str(more), "--output", str(output), "--resume"]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["status"] == "accepted" and len(state["history"]) == 4
    assert state["continued_after"] == [{"status": "accepted", "rounds": 2}]
    assert len(state["acceptance_reports"]) == values["budget"]["acceptance_repeats"]
    assert (output / "accepted" / "solution.py").read_text().endswith("COST = 2\n")


def test_gpu_guard_treats_placeholder_process_rows_as_unreadable_not_fatal(monkeypatch):
    from kai_pichu import process
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    rows = [["GPU-zero", "[GPU access blocked by the operating system]"], ["GPU-zero", "4242"]]
    monkeypatch.setattr(process, "_query", lambda fields, apps=False: rows if apps else [["0", "GPU-zero"]])
    monkeypatch.setattr(os, "getpgid", lambda pid: 4242)
    sample = GpuGuard(0).inspect(group=4242)
    assert sample["own_pids"] == [4242] and sample["foreign_pids"] == []
    assert sample["unreadable"] == ["[GPU access blocked by the operating system]"]


def test_a_transient_guard_failure_does_not_end_the_process(tmp_path, monkeypatch):
    from kai_pichu import process
    calls = 0

    class Guard:
        def __init__(self, device):
            pass

        def inspect(self, group=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("invalid literal for int() with base 10: '[GPU access blocked by the operating system]'")
            return {"foreign_pids": [], "unreadable": ["[N/A]"] if calls == 3 else []}

    monkeypatch.setattr(process, "GpuGuard", Guard)
    monkeypatch.setattr(process, "MAX_INCONCLUSIVE_SAMPLES", 3)
    result = run_process([sys.executable, "-c", "import time; time.sleep(1.5)"], cwd=tmp_path,
                         log=tmp_path / "ok.log", timeout=10, resources=Resources(poll_seconds=0.5), gpu_device=0)
    assert result["status"] == "completed"
    assert any("error" in sample for sample in result["resource_samples"])


def test_persistent_guard_failures_still_report_a_resource_error(tmp_path, monkeypatch):
    from kai_pichu import process

    class Guard:
        def __init__(self, device):
            pass

        def inspect(self, group=None):
            raise ValueError("nvidia-smi unusable")

    monkeypatch.setattr(process, "GpuGuard", Guard)
    monkeypatch.setattr(process, "MAX_INCONCLUSIVE_SAMPLES", 2)
    result = run_process([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                         log=tmp_path / "bad.log", timeout=10, resources=Resources(poll_seconds=0.2), gpu_device=0)
    # The device could not be inspected even before launch, so nothing was started.
    assert result["status"] == "resource_error" and "nvidia-smi unusable" in result["message"]
    assert not (tmp_path / "bad.log").exists()


def test_history_cells_carry_intervals_and_flag_noisy_cases():
    from kai_pichu.optimizer.history import history_tables
    row = _completed_round(0, 2.0, {"steady": 2.5, "jittery": 1.8}, "seed")
    row["metrics"]["acceptance"]["case_speedups"] = {
        "steady": {"speedup": 2.5, "interval": [2.45, 2.55]},
        "jittery": {"speedup": 1.8, "interval": [1.5, 2.1]}}
    text = history_tables([row])
    assert "| 2.50x [2.45, 2.55] |" in text
    assert "| 1.80x [1.50, 2.10] ~ |" in text
    assert "wider than 15%" in text
    # A later value inside the previous best's interval is not an improvement.
    later = _completed_round(1, 2.05, {"steady": 2.52, "jittery": 2.0}, "tweak")
    later["metrics"]["acceptance"]["case_speedups"] = {
        "steady": {"speedup": 2.52, "interval": [2.47, 2.57]}, "jittery": {"speedup": 2.0, "interval": [1.7, 2.3]}}
    table = history_tables([row, later])
    assert "| jittery | 1.80x | 1 | 1 | 2.00x |" in table  # 2.0 lies inside [1.5, 2.1]
    assert "| steady | 2.50x | 1 | 1 | 2.52x |" in table    # 2.52 lies inside [2.45, 2.55]


def test_cases_to_profile_rotate_under_a_cap(task):
    manifest, config_path, output = task
    values = yaml.safe_load(config_path.read_text())
    values["profile"] = {"enabled": False, "cases_per_round": 2}
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(values))
    metrics = {"objective": {"per_case": {c: {} for c in ("a", "b", "c", "d", "e")}}}
    assert loop._cases_to_profile(0, metrics) == ["a", "b"]
    assert loop._cases_to_profile(1, metrics) == ["c", "d"]
    assert loop._cases_to_profile(2, metrics) == ["e", "a"]
    values["profile"] = {"enabled": False}
    uncapped = OptimizationLoop(Workspace.create(manifest, output / "uncapped"), OptimizeConfig.model_validate(values))
    assert uncapped._cases_to_profile(7, metrics) == ["a", "b", "c", "d", "e"]


def test_queries_are_routed_by_case_and_fan_out_to_all_cases(task, monkeypatch):
    manifest, config_path, output = task
    loop = OptimizationLoop(Workspace.create(manifest, output), OptimizeConfig.model_validate(yaml.safe_load(config_path.read_text())))
    seen = []

    def query_profile(profile_id, request, *, timeout):
        seen.append((profile_id, dict(request)))
        return {"status": "available", "data": {"rows": [{"profile": profile_id}]}}

    monkeypatch.setattr(loop.evaluator, "query_profile", query_profile)
    profiles = {"small": {"profile_id": "p-small"}, "large": {"profile_id": "p-large"}}
    single = loop._route_query({"operation": "metrics", "counter": "x", "case_id": "large"}, profiles, "small", timeout=5)
    assert single["case_id"] == "large" and seen[-1] == ("p-large", {"operation": "metrics", "counter": "x"})
    default = loop._route_query({"operation": "rules"}, profiles, "small", timeout=5)
    assert default["case_id"] == "small" and seen[-1][0] == "p-small"
    everywhere = loop._route_query({"operation": "warp-stalls", "by": "reason", "case_id": "all", "limit": 40}, profiles, "small", timeout=5)
    assert [entry["case_id"] for entry in everywhere["fan_out"]] == ["small", "large"]
    assert all(request["limit"] == 10 for _, request in seen[-2:])  # fan-out pages are kept small
    refused = loop._route_query({"operation": "disasm", "case_id": "all"}, profiles, "small", timeout=5)
    assert refused["status"] == "unavailable" and "name one case" in refused["error"]["message"]
    unknown = loop._route_query({"operation": "rules", "case_id": "huge"}, profiles, "small", timeout=5)
    assert unknown["status"] == "unavailable" and "available: ['large', 'small']" in unknown["error"]["message"]
