"""Single-operator generation, diagnosis, repair and optimization.

Original optimizer components: MIT, Copyright (c) 2025 Zijian Zhang.
Source history and license texts are retained under licenses/."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import time
from typing import Any, Iterator

from ..benchmark.api import json_fingerprint
from ..benchmark.loading import file_inventory
from ..config import OptimizeConfig, require_api_keys
from ..evaluator import Evaluator, feedback
from ..io import parse_object, write_json
from ..models import ModelClient
from ..profiling import ProfileQuery
from ..workspace import Workspace
from .individual import KernelIndividual
from .prompts import GENERATOR, OPTIMIZATION_JUDGE, REPAIR_JUDGE, messages, strategy


class StopRun(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@contextmanager
def run_lock(root: Path) -> Iterator[None]:
    with (root / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("this run is already active") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class OptimizationLoop:
    def __init__(self, workspace: Workspace, config: OptimizeConfig):
        self.workspace = workspace
        self.config = config
        self.evaluator = Evaluator(workspace, config)
        self.clients = {"generator": ModelClient(config.generator),
                        "judge": ModelClient(config.judge or config.generator)}
        self.state: dict[str, Any] = {}
        self.started = 0.0
        self.previous_elapsed = 0.0

    def _save(self) -> None:
        self.state["elapsed_seconds"] = self.previous_elapsed + time.monotonic() - self.started
        write_json(self.workspace.root / "state.json", self.state, replace=True)

    def _remaining(self) -> float:
        remaining = self.config.budget.seconds - self.previous_elapsed - (time.monotonic() - self.started)
        if remaining <= 0:
            raise StopRun("budget_exhausted", "wall-time budget exhausted")
        return remaining

    def _verify(self) -> None:
        self.workspace.verify(self.state["frozen_fingerprint"])

    def _model(self, role: str, system: str, context: dict[str, Any]) -> dict[str, Any]:
        remaining = self._remaining()
        if self.state["llm_calls"] >= self.config.budget.llm_calls:
            raise StopRun("budget_exhausted", "model-call budget exhausted")
        count = self.state["llm_calls"]
        role_index = self.state["role_calls"][role]
        self.state["llm_calls"] += 1
        self.state["role_calls"][role] += 1
        self._save()  # Reserve before sending: an interrupted request is not free.
        request = messages(system, context)
        path = self.workspace.root / "llm" / f"call-{count:04d}"
        write_json(path.with_suffix(".request.json"), {"role": role, "messages": request})
        try:
            response = self.clients[role].complete(request, index=role_index, timeout=remaining)
        except Exception as error:
            write_json(path.with_suffix(".error.json"), {"error_type": type(error).__name__, "message": str(error)})
            raise StopRun("model_error", str(error)) from error
        write_json(path.with_suffix(".response.json"), response)
        self.state["usage"].append(response.get("usage", {}))
        self._save()
        return parse_object(response["text"])

    def _evaluate(self, label: str, candidate: Path | None, split: str) -> dict[str, Any]:
        self._verify()
        index = self.state["evaluations"]
        self.state["evaluations"] += 1
        self._save()
        report = self.evaluator.evaluate(f"{index:04d}-{label}", candidate=candidate, split=split,
                                         timeout=min(self.config.budget.evaluation_seconds, self._remaining()))
        self._verify()
        if report["status"] in ("resource_busy", "resource_error"):
            raise StopRun(report["status"], report.get("message", "GPU resource check failed"))
        if report["status"] in ("unstable", "calibration_failed"):
            raise StopRun("measurement_unstable", "A/A calibration failed; no code repair is justified by this result")
        return report

    def _path(self, individual: dict[str, Any] | None) -> Path:
        if individual is None:
            return self.workspace.baseline
        path = (self.workspace.root / individual["workspace"]).resolve(strict=True)
        if not path.is_relative_to(self.workspace.root / "candidates"):
            raise ValueError("candidate path escapes run candidates directory")
        if json_fingerprint(file_inventory(path, self.workspace.spec.implementation.files)) != individual["fingerprint"]:
            raise ValueError("saved candidate source changed")
        return path

    def _round(self, index: int, task_context: dict[str, Any]) -> None:
        current = self.state["current"]
        repair = current is not None and not current["metrics"].get("status") == "completed"
        anchor = current if repair else self.state["best"] or current
        anchor_path = self._path(anchor)
        phase = "seed" if current is None else "repair" if repair else "optimization"
        context = {**task_context, "phase": phase, "current_sources": self.workspace.sources(anchor_path),
                   "feedback": anchor["metrics"] if anchor else {},
                   "history": [{"round": row["id"], "hypothesis": row["hypothesis"], "score": row["score"],
                                "status": row["metrics"]["status"]} for row in self.state["history"][-10:]]}
        if current is not None:
            if not repair:
                profile = self.evaluator.profile(f"round-{index:04d}", anchor_path,
                                                 timeout=min(self.config.budget.evaluation_seconds, self._remaining()))
                self._verify()
                if profile.get("process", {}).get("status") in ("resource_busy", "resource_error"):
                    raise StopRun(profile["process"]["status"], "resource conflict during profiling")
                context["hardware_feedback"] = profile
            raw_strategy = (self._model("judge", REPAIR_JUDGE, context) if repair
                            else self._diagnose(index, context))
            context["strategy"] = strategy(raw_strategy, repair=repair)
        try:
            reply = self._model("generator", GENERATOR, context)
            candidate = self.workspace.candidate(index, anchor_path, reply)
        except (ValueError, json.JSONDecodeError) as error:
            # Output-format repair has the same bounded round budget as CUDA repair.
            candidate = self.workspace.root / "candidates" / f"round-{index:04d}"
            if not candidate.exists():
                shutil.copytree(anchor_path, candidate)
            reply = {"hypothesis": "invalid model response"}
            report = {"status": "error", "error_type": "CandidateFormatError", "message": str(error)}
        else:
            report = self._evaluate(f"round-{index:04d}", candidate, "search")
        score = report.get("comparison", {}).get("overall", {}).get("speedup")
        ind = KernelIndividual(index, str(candidate.relative_to(self.workspace.root)), reply["hypothesis"],
                               json_fingerprint(file_inventory(candidate, self.workspace.spec.implementation.files)),
                               feedback(report), score)
        self.state["current"] = ind.to_dict()
        self.state["history"].append(ind.to_dict())
        if ind.runnable:
            acceptance = report["acceptance"]
            constraints_ok = not acceptance["unconfirmed_case_constraints"] and not acceptance["metric_limit_failure_count"]
            lower = report["comparison"]["overall"]["interval"][0]
            previous = self.state["best"]["score"] if self.state["best"] else 1.0
            if constraints_ok and lower > 1.0 and score > previous:
                self.state["best"] = ind.to_dict()
        self._save()

    def _diagnose(self, index: int, context: dict[str, Any]) -> dict[str, Any]:
        settings = self.config.profile
        profile = context.get("hardware_feedback", {})
        query_round = 0
        while True:
            # Evidence queries must leave room for this decision/generation and
            # one decision/generation pair in every later optimization round.
            remaining_calls = self.config.budget.llm_calls - self.state["llm_calls"]
            reserved_future_calls = 2 * max(0, self.config.budget.rounds - index - 1)
            evidence_calls = max(0, remaining_calls - reserved_future_calls - 2)
            enabled = bool(profile.get("query_available")) and query_round < settings.query_rounds and evidence_calls > 0
            context["profile_query"] = {"enabled": enabled,
                "remaining_rounds": min(settings.query_rounds - query_round, evidence_calls) if enabled else 0,
                "reserved_future_calls": reserved_future_calls,
                "max_queries": settings.queries_per_round, "schema": ProfileQuery.model_json_schema()}
            response = self._model("judge", OPTIMIZATION_JUDGE, context)
            if response.get("action") != "query_profile":
                return response
            if not enabled:
                raise StopRun("model_error", "judge requested profile queries after the diagnostic budget was exhausted")
            requests = response.get("queries")
            question = response.get("question")
            valid_question = ("question" not in response or
                              isinstance(question, str) and bool(question.strip()) and len(question) <= 1000)
            if (set(response) - {"action", "queries", "question"} or not valid_question
                    or not isinstance(requests, list) or not 1 <= len(requests) <= settings.queries_per_round):
                requests = []
                results = [{"status": "unavailable", "error": {"message":
                    f"query_profile requires 1..{settings.queries_per_round} queries and an optional nonempty question (up to 1000 characters)"}}]
            else:
                results = []
                for request in requests:
                    self._verify()
                    # Validate the same source anchor before and after report queries.
                    anchor = self.state["best"] or self.state["current"]
                    self._path(anchor)
                    results.append(self.evaluator.query_profile(profile["profile_id"], request,
                        timeout=min(settings.query_seconds, self._remaining())))
                    self._path(anchor)
                    self._verify()
            record = {"round": query_round, "requests": requests, "results": results}
            if valid_question and question is not None:
                record["question"] = question
            write_json(self.workspace.root / "profiles" / f"round-{index:04d}" / f"queries-{query_round:02d}.json", record)
            context.setdefault("profile_query_results", []).append(record)
            query_round += 1

    def _deliver(self) -> None:
        best = self.state["best"]
        root = self.workspace.root
        source = None
        if best:
            try:
                self._verify()
                source = self._path(best)
            except (OSError, ValueError) as error:
                # A changed source must invalidate acceptance and still leave a
                # reviewable summary, even if mutation happened after evaluation.
                self.state.update(status="invalidated", message=str(error))
                self._save()
        if source is not None:
            delivery = root / "best_search"
            delivery.mkdir(exist_ok=True)
            for name in self.workspace.allowed:
                target = delivery / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            (root / "best_search.patch").write_text(self.workspace.patch(source))
            if self.state["status"] == "accepted":
                accepted = root / "accepted"
                shutil.copytree(source, accepted)
        summary = {"project": "KAI-light", "backend": "kai_light.optimizer",
                   "status": self.state["status"], "message": self.state.get("message", ""),
                   "final_accepted": self.state["status"] == "accepted", "best_search": best,
                   "acceptance_reports": self.state.get("acceptance_reports", []),
                   "rounds_completed": len(self.state["history"]), "llm_calls": self.state["llm_calls"],
                   "elapsed_seconds": self.state["elapsed_seconds"], "usage": self.state["usage"],
                   "frozen_fingerprint": self.state["frozen_fingerprint"],
                   "gpu_policy": self.config.resources.gpu_policy}
        write_json(root / "summary.json", summary, replace=True)

    def run(self, *, resume: bool = False, dry_run: bool = False) -> dict[str, Any]:
        root = self.workspace.root
        if not dry_run:
            require_api_keys(self.config)
        with run_lock(root):
            config_fingerprint = json_fingerprint(self.config.model_dump())
            self.started = time.monotonic()
            if resume:
                self.state = json.loads((root / "state.json").read_text())
                if self.state["config_fingerprint"] != config_fingerprint:
                    raise ValueError("resume requires the original configuration")
                if self.state["status"] in ("accepted", "not_accepted", "no_improvement"):
                    raise ValueError("completed runs cannot be resumed; use a new output directory")
                self.previous_elapsed = self.state["elapsed_seconds"]
                self._verify()
            else:
                self.state = {"schema_version": 1, "status": "created", "config_fingerprint": config_fingerprint,
                              "frozen_fingerprint": self.workspace.fingerprint(), "next_round": 0,
                              "llm_calls": 0, "role_calls": {"generator": 0, "judge": 0}, "evaluations": 0,
                              "current": None, "best": None, "history": [], "usage": [],
                              "acceptance_reports": [], "elapsed_seconds": 0.0}
                write_json(root / "config.json", self.config.model_dump())
            self._save()
            try:
                task_context = self.workspace.context(self.config.max_context_chars)
                if dry_run:
                    write_json(root / "plan.json", {"backend": "kai_light.optimizer", "context": task_context,
                                                   "first_request": messages(GENERATOR, task_context)}, replace=resume)
                    self.state["status"] = "planned"
                else:
                    self.state["status"] = "running"
                    self.state.pop("message", None)
                    baseline = self._evaluate("preflight", None, "search")
                    if baseline["status"] != "ready":
                        raise StopRun("benchmark_error", baseline.get("message", "baseline did not pass conformance/calibration"))
                    task_context["hardware"] = baseline.get("timing_environment", {})
                    for index in range(self.state["next_round"], self.config.budget.rounds):
                        self._remaining()
                        # Reserve each round before work. Resume skips an interrupted
                        # round instead of overwriting files or replaying unknown calls.
                        self.state["next_round"] = index + 1
                        self._save()
                        print(f"Round {index + 1}/{self.config.budget.rounds}", flush=True)
                        self._round(index, task_context)
                    best = self.state["best"]
                    if best is None:
                        self.state["status"] = "no_improvement"
                    else:
                        selected = self._path(best)
                        self.state["selected_fingerprint"] = best["fingerprint"]
                        self.state["acceptance_reports"] = []
                        self._save()
                        passed = True
                        for repeat in range(self.config.budget.acceptance_repeats):
                            report = self._evaluate(f"acceptance-{repeat}", selected, "acceptance")
                            self.state["acceptance_reports"].append(feedback(report))
                            passed = passed and bool(report.get("acceptance", {}).get("accepted", False))
                            self._save()
                        self.state["status"] = "accepted" if passed else "not_accepted"
            except StopRun as error:
                self.state.update(status=error.status, message=str(error))
            except KeyboardInterrupt:
                self.state.update(status="interrupted", message="Interrupted; reserved calls and rounds remain charged.")
            except Exception as error:
                self.state.update(status="error", message=f"{type(error).__name__}: {error}")
            finally:
                try:
                    self._verify()
                except Exception as error:
                    self.state.update(status="invalidated", message=str(error))
                self._save()
                self._deliver()
            return json.loads((root / "summary.json").read_text())
