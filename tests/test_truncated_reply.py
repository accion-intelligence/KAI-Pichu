from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest
import yaml

from kai_pichu.cli import main
from kai_pichu.config import ModelConfig
from kai_pichu.models import ModelClient, ModelReplyError

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
    return json.dumps({"hypothesis": f"cost {cost}", "files": {"solution.py": f"FACTOR = {factor}\nCOST = {cost}\n"}})


def serve(handler_body):
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


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_an_unusable_reply_is_distinguishable_from_a_transport_fault(finish_reason):
    def handler(request):
        request.send_response(200)
        request.end_headers()
        request.wfile.write(json.dumps(
            {"choices": [{"message": {"content": "partial"}, "finish_reason": finish_reason}]}).encode())

    server, thread = serve(handler)
    config = ModelConfig(model="m", api_key_env="", base_url=f"http://127.0.0.1:{server.server_port}/v1")
    try:
        with pytest.raises(ModelReplyError):
            ModelClient(config).complete([{"role": "user", "content": "x"}], index=0, timeout=5)
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_an_http_fault_is_not_a_reply_error():
    def handler(request):
        request.send_response(500)
        request.end_headers()
        request.wfile.write(b"{}")

    server, thread = serve(handler)
    config = ModelConfig(model="m", api_key_env="", base_url=f"http://127.0.0.1:{server.server_port}/v1")
    try:
        with pytest.raises(RuntimeError) as caught:
            ModelClient(config).complete([{"role": "user", "content": "x"}], index=0, timeout=5)
        assert not isinstance(caught.value, ModelReplyError)
    finally:
        server.shutdown(); server.server_close(); thread.join()


@pytest.fixture
def truncating_task(tmp_path: Path):
    """A task whose endpoint truncates the first generation, then answers normally."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter.py").write_text(ADAPTER)
    (source / "solution.py").write_text("FACTOR = 1\nCOST = 10.0\n")
    manifest = source / "benchmark.yaml"
    manifest.write_text(yaml.safe_dump({
        "name": "synthetic", "description": "fixture, not a speedup claim", "adapter": "adapter.py:Task",
        "implementation": {"root": ".", "files": ["solution.py"]},
        "objective": {"scope": "kernel", "metric": "synthetic_cost", "unit": "u", "target_speedup": 1.2},
        "measurement": {"timer": "wall", "boundary": "t", "cache_policy": "t",
                        "blocks": 4, "iterations": 1, "warmup": 1, "bootstrap_samples": 200}}))
    sent = []

    def handler(request):
        sent.append(1)
        request.send_response(200)
        request.end_headers()
        # The first generation overruns max_tokens; later calls behave.
        if len(sent) == 1:
            body = {"choices": [{"message": {"content": reply(5)[:40]}, "finish_reason": "length"}]}
        else:
            body = {"choices": [{"message": {"content": reply(5)}, "finish_reason": "stop"}]}
        request.wfile.write(json.dumps(body).encode())

    server, thread = serve(handler)
    config = tmp_path / "optimizer.yaml"
    config.write_text(yaml.safe_dump({
        "generator": {"provider": "chat_completions", "model": "m", "api_key_env": "",
                      "base_url": f"http://127.0.0.1:{server.server_port}/v1"},
        # Round 1 repairs the truncated seed; round 2 optimizes. The judge's
        # contract differs between the two, so the replies must too.
        "judge": {"provider": "replay", "responses": [
            json.dumps({"critical_issue": "c", "why_it_matters": "w", "minimal_fix_hint": "h"}),
            json.dumps({"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"}),
            json.dumps({"bottleneck": "b", "optimization_method": "m", "modification_plan": "p"})]},
        "budget": {"rounds": 3, "llm_calls": 8, "acceptance_repeats": 1,
                   "seconds": 60, "evaluation_seconds": 10}}))
    yield manifest, config, tmp_path / "run", sent
    server.shutdown(); server.server_close(); thread.join()


def test_a_truncated_generation_is_continued_within_its_round(truncating_task):
    manifest, config, output, sent = truncating_task
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output)]) == 0
    state = json.loads((output / "state.json").read_text())
    # Before ModelReplyError the first truncated reply ended the run as model_error;
    # with continuation the loop asks the model to finish it and the seed round completes.
    assert state["status"] not in ("model_error", "error")
    assert len(state["history"]) == 3 and len(sent) > 1
    first = state["history"][0]["metrics"]
    assert first["status"] == "completed"
    requests = [json.loads(p.read_text()) for p in sorted((output / "llm").glob("*.request.json"))]
    assert [r["continuation"] for r in requests[:2]] == [False, True]
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": reply(5)[:40]}
    assert state["best"] is not None


def test_without_continuation_a_truncated_generation_costs_one_round_not_the_run(truncating_task):
    manifest, config, output, sent = truncating_task
    values = yaml.safe_load(config.read_text())
    values["generator"]["max_continuations"] = 0
    config.write_text(yaml.safe_dump(values))
    assert main(["optimize", str(manifest), "--config", str(config), "--output", str(output)]) == 0
    state = json.loads((output / "state.json").read_text())
    assert state["status"] not in ("model_error", "error")
    first = state["history"][0]["metrics"]
    assert first["status"] == "error" and first["error_type"] == "ModelReplyError"
    # The next round is told what actually went wrong, so it can answer shorter.
    assert "length" in first["message"]
    assert state["best"] is not None


def test_the_failed_call_is_still_charged_and_recorded(truncating_task):
    manifest, config, output, _ = truncating_task
    main(["optimize", str(manifest), "--config", str(config), "--output", str(output)])
    # Reserve-before-send is unchanged: an unusable answer is not a free call.
    record = json.loads((output / "llm/call-0000.error.json").read_text())
    assert record["error_type"] == "ModelReplyError" and "length" in record["message"] and record["truncated"] is True
    assert json.loads((output / "state.json").read_text())["llm_calls"] >= 3


@pytest.mark.parametrize("line", ["model refused the request",
                                  "model response incomplete: max_tokens",
                                  "model returned no textual candidate/strategy"])
def test_the_anthropic_provider_classifies_unusable_replies_the_same_way(line):
    # The taxonomy must not depend on which provider produced the reply.
    source = Path(__file__).parents[1] / "src/kai_pichu/models.py"
    body = source.read_text().split("_complete_anthropic", 1)[1]
    raising = next(l for l in body.splitlines() if line in l)
    assert "ModelReplyError" in raising, f"still a fatal ValueError: {raising.strip()}"


def test_the_anthropic_provider_keeps_configuration_errors_fatal():
    # A missing SDK or API key is not something a later round can repair.
    source = Path(__file__).parents[1] / "src/kai_pichu/models.py"
    body = source.read_text().split("_complete_anthropic", 1)[1]
    for line in ("needs the official SDK", "API key environment variable not set"):
        raising = next(l for l in body.splitlines() if line in l)
        assert "ModelReplyError" not in raising, f"wrongly repairable: {raising.strip()}"


def test_the_shipped_default_token_budget_fits_a_whole_file_rewrite():
    # The generator must return complete replacement text, and for a reasoning
    # provider this budget is shared with thinking tokens.
    assert ModelConfig(model="m", api_key_env="", base_url="http://x/v1").max_tokens == 262144


def test_a_reply_cut_off_by_the_output_limit_carries_its_partial_text():
    def handler(request):
        request.send_response(200)
        request.end_headers()
        request.wfile.write(json.dumps(
            {"choices": [{"message": {"content": '{"hypothesis": "h", "files": {"solution.py": "FAC'}, "finish_reason": "length"}]}).encode())

    server, thread = serve(handler)
    try:
        client = ModelClient(ModelConfig(model="m", base_url=f"http://127.0.0.1:{server.server_port}", api_key_env=""))
        with pytest.raises(ModelReplyError) as caught:
            client.complete([{"role": "user", "content": "go"}], index=0, timeout=5)
        assert caught.value.truncated and caught.value.partial_text.startswith('{"hypothesis"')
    finally:
        server.shutdown()
        thread.join()


def test_a_content_filter_reply_is_not_treated_as_truncated():
    def handler(request):
        request.send_response(200)
        request.end_headers()
        request.wfile.write(json.dumps(
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "content_filter"}]}).encode())

    server, thread = serve(handler)
    try:
        client = ModelClient(ModelConfig(model="m", base_url=f"http://127.0.0.1:{server.server_port}", api_key_env=""))
        with pytest.raises(ModelReplyError) as caught:
            client.complete([{"role": "user", "content": "go"}], index=0, timeout=5)
        assert not caught.value.truncated
    finally:
        server.shutdown()
        thread.join()


def test_responses_api_truncation_carries_the_text_written_so_far():
    def handler(request):
        request.send_response(200)
        request.end_headers()
        request.wfile.write(json.dumps({
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": '{"hypothesis": "h", '}]}],
        }).encode())

    server, thread = serve(handler)
    try:
        client = ModelClient(ModelConfig(provider="responses", model="m", base_url=f"http://127.0.0.1:{server.server_port}", api_key_env=""))
        with pytest.raises(ModelReplyError) as caught:
            client.complete([{"role": "user", "content": "go"}], index=0, timeout=5)
        assert caught.value.partial_text == '{"hypothesis": "h", '
    finally:
        server.shutdown()
        thread.join()
