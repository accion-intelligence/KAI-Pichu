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
from ..models import ModelClient, ModelReplyError
from ..profiling import ProfileQuery
from ..workspace import Workspace
from .history import case_lost_most, history_tables, weakest_case
from .individual import KernelIndividual
from .continuation import splice
from .prompts import GENERATOR, OPTIMIZATION_JUDGE, REPAIR_JUDGE, continuation_messages, messages, strategy


def _case_view(profile: dict[str, Any]) -> dict[str, Any]:
    """What the judge sees of one capture: status, identity and the evidence overview, without the process log."""
    view = {key: profile.get(key) for key in ("status", "profile_id", "case_selection", "query_available")}
    evidence = dict(profile.get("evidence") or {})
    evidence.pop("diagnostics", None)  # identical for every case; the parent carries it once
    view["evidence"] = evidence
    return view


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
        """One decision from a model: a complete reply, continued if the output limit cut it off."""
        request = messages(system, context)
        try:
            text = self._call(role, request)["text"]
        except ModelReplyError as error:
            if not error.truncated:
                raise
            text = self._continue(role, request, error)
        return parse_object(text)

    def _continue(self, role: str, request: list[dict[str, str]], cut_off: ModelReplyError) -> str:
        """Ask the model to finish a cut-off reply and splice the pieces.

        A long kernel is the normal case for a successful optimization, and a
        reply cut off by max_tokens still contains most of it. Each continuation
        is a charged model call within the same budget; when the pieces do not
        form one JSON object, the round falls back to the ordinary repair path.
        """
        attempts = self._model_config(role).max_continuations
        if attempts == 0:
            raise cut_off
        text = cut_off.partial_text
        for _ in range(attempts):
            try:
                piece = self._call(role, continuation_messages(request, text), continuation=True)["text"]
            except ModelReplyError as error:
                if not error.truncated:
                    raise
                text = splice(text, error.partial_text)
                continue
            text = splice(text, piece)
            try:
                parse_object(text)
            except (ValueError, json.JSONDecodeError):
                continue  # ask once more; the model may have stopped mid-object again
            return text
        raise ModelReplyError(f"{cut_off}; still incomplete after {attempts} continuation(s)")

    def _model_config(self, role: str) -> Any:
        return (self.config.judge or self.config.generator) if role == "judge" else self.config.generator

    def _call(self, role: str, request: list[dict[str, str]], *, continuation: bool = False) -> dict[str, Any]:
        """Send one request under the model-call budget and record it under llm/."""
        remaining = self._remaining()
        if self.state["llm_calls"] >= self.config.budget.llm_calls:
            raise StopRun("budget_exhausted", "model-call budget exhausted")
        count = self.state["llm_calls"]
        role_index = self.state["role_calls"][role]
        self.state["llm_calls"] += 1
        self.state["role_calls"][role] += 1
        self._save()  # Reserve before sending: an interrupted request is not free.
        path = self.workspace.root / "llm" / f"call-{count:04d}"
        write_json(path.with_suffix(".request.json"), {"role": role, "continuation": continuation, "messages": request})
        try:
            response = self.clients[role].complete(request, index=role_index, timeout=remaining)
        except ModelReplyError as error:
            # A call was made and answered; the answer is unusable. The caller
            # repairs this within the round budget, as it already does for a
            # malformed candidate. The call stays charged either way.
            write_json(path.with_suffix(".error.json"), {"error_type": type(error).__name__, "message": str(error),
                                                         "truncated": error.truncated})
            raise
        except Exception as error:
            write_json(path.with_suffix(".error.json"), {"error_type": type(error).__name__, "message": str(error)})
            raise StopRun("model_error", str(error)) from error
        write_json(path.with_suffix(".response.json"), response)
        self.state["usage"].append(response.get("usage", {}))
        self._save()
        return response

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
        # The search builds on the newest candidate that passed the hard rules
        # (correct, faster than the baseline overall, within every metric limit). Which of
        # two eligible candidates is "better" is left to the model, which sees
        # every round's per-case numbers; a harness gate on the aggregate score
        # would discard a real gain on one case for jitter on another.
        anchor = current if repair else self.state.get("latest_eligible") or current
        anchor_path = self._path(anchor)
        phase = "seed" if current is None else "repair" if repair else "optimization"
        context = {**task_context, "phase": phase, "current_sources": self.workspace.sources(anchor_path),
                   "feedback": anchor["metrics"] if anchor else {},
                   "history": history_tables(self.state["history"])}
        if current is not None:
            if not repair:
                context["hardware_feedback"] = self._profile_cases(index, anchor_path, anchor)
                if anchor["id"] != current["id"]:  # the last candidate completed but failed the eligibility rules
                    context["last_attempt"] = self._attempt_evidence(index, current, anchor)
                best = self.state["best"]
                if best and best["id"] != anchor["id"]:
                    context["best_candidate"] = self._best_candidate_evidence(best)
            guidance = self._strategy(index, context, repair=repair)
            if guidance is not None:
                context["strategy"] = guidance
                anchor, anchor_path = self._starting_point(context, anchor, anchor_path)
        context["base_round"] = anchor["id"] + 1 if anchor else None
        try:
            reply = self._model("generator", GENERATOR, context)
            candidate = self.workspace.candidate(index, anchor_path, reply)
        except (ValueError, json.JSONDecodeError) as error:
            # Output-format repair has the same bounded round budget as CUDA repair.
            candidate = self.workspace.root / "candidates" / f"round-{index:04d}"
            if not candidate.exists():
                shutil.copytree(anchor_path, candidate)
            reply = {"hypothesis": "invalid model response"}
            # Name the actual fault so the next round is told what to change: a
            # truncated reply needs a shorter one, not a differently formatted one.
            report = {"status": "error", "error_type": type(error).__name__, "message": str(error)}
        else:
            report = self._evaluate(f"round-{index:04d}", candidate, "search")
        score = report.get("comparison", {}).get("overall", {}).get("speedup")
        ind = KernelIndividual(index, str(candidate.relative_to(self.workspace.root)), reply["hypothesis"],
                               json_fingerprint(file_inventory(candidate, self.workspace.spec.implementation.files)),
                               feedback(report), score, diagnosis=context.get("strategy"),
                               base_round=context.get("base_round"))
        self.state["current"] = ind.to_dict()
        self.state["history"].append(ind.to_dict())
        if ind.runnable and self._eligible(report):
            self.state["latest_eligible"] = ind.to_dict()
            # The top score is bookkeeping for delivery and acceptance only; it never gates the search.
            if not self.state["best"] or score > self.state["best"]["score"]:
                self.state["best"] = ind.to_dict()
        self._save()

    @staticmethod
    def _eligible(report: dict[str, Any]) -> bool:
        """Correct, faster than the baseline with confidence overall, and within every metric limit.

        Per-case regressions are reported in the feedback and the history table
        for the model to weigh; they do not disqualify a candidate.
        """
        acceptance = report["acceptance"]
        return not acceptance["metric_limit_failure_count"] and report["comparison"]["overall"]["interval"][0] > 1.0

    def _best_candidate_evidence(self, best: dict[str, Any]) -> dict[str, Any]:
        """The highest-scoring eligible candidate, when the search has moved on from it.

        The model decides whether to carry that code forward; it can only do so
        if it can see it.
        """
        return {"round": best["id"] + 1, "score": best["score"], "hypothesis": best["hypothesis"],
                "feedback": best["metrics"], "sources": self.workspace.sources(self._path(best)),
                "note": "Highest overall speedup so far. current_sources are the latest eligible candidate; "
                        "this one is kept for acceptance unless a later candidate scores higher."}

    def _profile(self, tag: str, path: Path, case: str | None, policy: str) -> dict[str, Any]:
        profile = self.evaluator.profile(tag, path, timeout=min(self.config.budget.evaluation_seconds, self._remaining()),
                                         case_id=case, policy=policy)
        self._verify()
        if profile.get("process", {}).get("status") in ("resource_busy", "resource_error"):
            raise StopRun(profile["process"]["status"], "resource conflict during profiling")
        return profile

    def _cases_to_profile(self, index: int, metrics: dict[str, Any]) -> list[str]:
        """The search cases captured this round: all of them, or a rotating window of cases_per_round."""
        cases = sorted((metrics.get("objective") or {}).get("per_case") or {})
        cap = self.config.profile.cases_per_round
        if not cases or cap is None or cap >= len(cases):
            return cases
        start = (index * cap) % len(cases)
        return [cases[(start + offset) % len(cases)] for offset in range(cap)]

    def _profile_cases(self, index: int, path: Path, anchor: dict[str, Any]) -> dict[str, Any]:
        """One NCU capture per search case, so the judge chooses which shape's evidence to pursue.

        The harness does not rank cases by importance; that is the task's and
        the model's call. Every case's overview travels with the context, and a
        query names the case it addresses (or all of them).
        """
        cases = self._cases_to_profile(index, anchor["metrics"])
        if not cases:
            profile = self._profile(f"round-{index:04d}", path, None, "first_search_case")
            cases_out = [{"case_id": profile.get("case_selection", {}).get("case_id"), **_case_view(profile)}]
        else:
            cases_out = []
            for case in cases:
                profile = self._profile(f"round-{index:04d}/{case}", path, case, "every_search_case")
                cases_out.append({"case_id": case, **_case_view(profile)})
        weakest = weakest_case(anchor["metrics"])
        available = [c["case_id"] for c in cases_out if c.get("query_available")]
        return {
            "cases": cases_out,
            "default_case": weakest if weakest in available else (available[0] if available else None),
            "query_available": bool(available),
            "diagnostics": next((c.get("evidence", {}).get("diagnostics") for c in cases_out
                                 if c.get("evidence", {}).get("diagnostics")), None),
            "note": "One capture of the current sources per search case. Each entry holds that case's launches, "
                    "headline metrics, rule count and stall summary; a query names its case_id, or \"all\".",
        }

    def _attempt_evidence(self, index: int, attempt: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
        """Measurements and a profile of the last candidate that ran but failed the eligibility rules.

        The judge diagnoses the anchor's source, but the reason the previous
        attempt failed is only visible in the attempt itself. Profiling the
        anchor alone would show the judge the same picture every round and
        invite the same proposal; this evidence is what the attempt looked like
        on the case where it fell furthest behind the anchor.
        """
        profile = self._profile(f"round-{index:04d}-attempt", self._path(attempt),
                                case_lost_most(attempt["metrics"], anchor["metrics"]), "lowest_speedup_relative_to_anchor")
        return {"round": attempt["id"] + 1, "hypothesis": attempt["hypothesis"], "diagnosis": attempt.get("diagnosis"),
                "feedback": attempt["metrics"],
                "hardware_feedback": {key: profile.get(key) for key in ("status", "case_selection", "workload", "evidence")},
                "note": "This candidate ran but failed the eligibility rules (a metric limit, or no confirmed gain "
                        "over the baseline); profile queries address the current sources' profile, this one is an "
                        "overview only."}

    def _saved_round(self, round_number: Any) -> dict[str, Any] | None:
        """The history row of a 1-based round whose candidate is still on disk and unchanged."""
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            return None
        for row in self.state["history"]:
            if row["id"] + 1 == round_number:
                try:
                    self._path(row)
                except (OSError, ValueError):
                    return None
                return row
        return None

    def _starting_point(self, context: dict[str, Any], anchor: dict[str, Any], anchor_path: Path):
        """Honor the judge's base_round when it names a saved candidate; otherwise keep the anchor."""
        requested = context["strategy"].get("base_round")
        if requested is None or requested == anchor["id"] + 1:
            return anchor, anchor_path
        row = self._saved_round(requested)
        if row is None:
            context["strategy_note"] = (f"base_round {requested} names no saved candidate; "
                                        f"the generator builds on round {anchor['id'] + 1}")
            return anchor, anchor_path
        path = self._path(row)
        context["current_sources"] = self.workspace.sources(path)
        return row, path

    def _read_candidates(self, requests: Any) -> list[dict[str, Any]]:
        """Sources of up to three saved rounds, for a judge choosing where to start."""
        if not isinstance(requests, list) or not 1 <= len(requests) <= 3:
            return [{"status": "unavailable", "error": {"message": "read_candidate takes 1..3 round numbers"}}]
        results = []
        for number in requests:
            row = self._saved_round(number)
            if row is None:
                results.append({"round": number, "status": "unavailable",
                                "error": {"message": "no saved candidate for this round"}})
                continue
            results.append({"round": number, "status": "available", "hypothesis": row["hypothesis"],
                            "score": row["score"], "result": row["metrics"].get("status"),
                            "diagnosis": row.get("diagnosis"), "base_round": row.get("base_round"),
                            "sources": self.workspace.sources(self._path(row))})
        return results
    def _strategy(self, index: int, context: dict[str, Any], *, repair: bool) -> dict[str, str] | None:
        """Judge guidance for this round, with one corrective re-ask.

        A reply that breaks the judge's own output contract is the judge's
        mistake, not the candidate's, and a run with rounds and budget left must
        not end on it. The generator already works from feedback alone, so a
        second unusable reply degrades this round to unguided generation and
        leaves strategy_error in the recorded context.
        """
        system = REPAIR_JUDGE if repair else OPTIMIZATION_JUDGE
        try:
            raw = self._model("judge", system, context) if repair else self._diagnose(index, context)
            return strategy(raw, repair=repair)
        except (ValueError, json.JSONDecodeError) as error:
            context["strategy_error"] = {"message": str(error)[:1000], "instruction":
                "The previous reply broke the required output contract. Return only that JSON object."}
        if self.config.budget.llm_calls - self.state["llm_calls"] < 2:
            # _diagnose reserves one decision and one generation call per round.
            # A re-ask must not consume the generation: an unguided candidate is
            # worth more than guidance with nothing left to generate it.
            context["strategy_error"]["final_message"] = "no model-call budget for a re-ask"
            return None
        if not repair:
            # The re-ask decides on the evidence already gathered; no new queries.
            context["profile_query"] = {"enabled": False, "remaining_rounds": 0}
        try:
            return strategy(self._model("judge", system, context), repair=repair)
        except (ValueError, json.JSONDecodeError) as error:
            context["strategy_error"]["final_message"] = str(error)[:1000]
            return None

    FAN_OUT_OPERATIONS = {"launches", "catalog", "metrics", "rules", "warp-stalls"}

    def _route_query(self, request: Any, profiles: dict[str, dict[str, Any]], default_case: str | None,
                     *, timeout: float) -> dict[str, Any]:
        """Send one judge query to the named case's report, or fan it out to every case."""
        if not isinstance(request, dict):
            return {"status": "unavailable", "error": {"message": "each query must be an object"}}
        case = request.get("case_id") or default_case
        body = {key: value for key, value in request.items() if key != "case_id"}
        if case == "all":
            if body.get("operation") not in self.FAN_OUT_OPERATIONS:
                return {"status": "unavailable", "case_id": "all", "error": {"message":
                        f"case_id \"all\" supports {sorted(self.FAN_OUT_OPERATIONS)}; name one case for {body.get('operation')!r}"}}
            body["limit"] = min(int(body.get("limit", 20) or 20), 10)
            per_case = [{"case_id": name, **self.evaluator.query_profile(entry["profile_id"], dict(body), timeout=timeout)}
                        for name, entry in profiles.items()]
            return {"status": "available", "case_id": "all", "fan_out": per_case}
        if case not in profiles:
            return {"status": "unavailable", "case_id": case,
                    "error": {"message": f"no queryable profile for case {case!r}; available: {sorted(profiles)}"}}
        return {"case_id": case, **self.evaluator.query_profile(profiles[case]["profile_id"], body, timeout=timeout)}

    def _diagnose(self, index: int, context: dict[str, Any]) -> dict[str, Any]:
        settings = self.config.profile
        profile = context.get("hardware_feedback", {})
        profiles = {entry["case_id"]: entry for entry in profile.get("cases", []) if entry.get("query_available")}
        query_round = 0
        while True:
            # Evidence queries must leave room for this decision/generation and
            # one decision/generation pair in every later optimization round.
            remaining_calls = self.config.budget.llm_calls - self.state["llm_calls"]
            reserved_future_calls = 2 * max(0, self.config.budget.rounds - index - 1)
            evidence_calls = max(0, remaining_calls - reserved_future_calls - 2)
            evidence_ok = query_round < settings.query_rounds and evidence_calls > 0
            enabled = bool(profile.get("query_available")) and evidence_ok
            context["candidate_reads"] = {
                "enabled": evidence_ok,
                "saved_rounds": [row["id"] + 1 for row in self.state["history"]],
                "note": "read_candidate returns the saved code of past rounds; it shares the evidence budget with profile queries",
            }
            context["profile_query"] = {"enabled": enabled,
                "remaining_rounds": min(settings.query_rounds - query_round, evidence_calls) if enabled else 0,
                "reserved_future_calls": reserved_future_calls,
                "max_queries": settings.queries_per_round, "schema": ProfileQuery.model_json_schema()}
            response = self._model("judge", OPTIMIZATION_JUDGE, context)
            action = response.get("action")
            if action == "read_candidate":
                if not evidence_ok:
                    # A contract violation, not a run-ending fault: _strategy re-asks once.
                    raise ValueError("candidate reads are exhausted or unavailable; decide with the evidence already provided")
                if set(response) - {"action", "rounds"}:
                    results = [{"status": "unavailable", "error": {"message": "read_candidate takes only rounds"}}]
                else:
                    results = self._read_candidates(response.get("rounds"))
                context.setdefault("candidate_sources", []).extend(results)
                query_round += 1
                continue
            if action != "query_profile":
                return response
            if not enabled:
                # A contract violation, not a run-ending fault: _strategy re-asks once.
                raise ValueError("profile queries are exhausted or unavailable; "
                                 "decide with the evidence already provided")
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
                    anchor = self.state.get("latest_eligible") or self.state["current"]
                    self._path(anchor)
                    results.append(self._route_query(request, profiles, profile.get("default_case"),
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
                shutil.rmtree(accepted, ignore_errors=True)  # a continued run replaces its earlier delivery
                shutil.copytree(source, accepted)
        summary = {"project": "KAI Pichu", "backend": "kai_pichu.optimizer",
                   "status": self.state["status"], "message": self.state.get("message", ""),
                   "final_accepted": self.state["status"] == "accepted", "best_search": best,
                   "acceptance_reports": self.state.get("acceptance_reports", []),
                   "rounds_completed": len(self.state["history"]), "llm_calls": self.state["llm_calls"],
                   "budget_changes": self.state.get("budget_changes", []),
                   "elapsed_seconds": self.state["elapsed_seconds"], "usage": self.state["usage"],
                   "frozen_fingerprint": self.state["frozen_fingerprint"],
                   "gpu_policy": self.config.resources.gpu_policy}
        write_json(root / "summary.json", summary, replace=True)

    RESUMABLE_SECTIONS = ("budget", "profile")

    def _adopt_config_change(self, config_fingerprint: str) -> None:
        """Allow a resumed run to continue under a different budget or profiling setup.

        Rounds, model calls and wall time are limits on the experiment, and the
        NCU capture is diagnostic evidence; neither changes the task or the
        comparability of scores. Models and context settings must still match
        the recorded config.json. Every change is recorded in the checkpoint and
        config.json is replaced so the run directory describes what actually ran.
        """
        original = OptimizeConfig.model_validate(json.loads((self.workspace.root / "config.json").read_text()))
        frozen = set(self.RESUMABLE_SECTIONS)
        if original.model_dump(exclude=frozen) != self.config.model_dump(exclude=frozen):
            raise ValueError("resume requires the original configuration; only the budget and profile sections may change")
        for section in self.RESUMABLE_SECTIONS:
            before, after = getattr(original, section), getattr(self.config, section)
            if before != after:
                self.state.setdefault(f"{section}_changes", []).append({
                    "at_round": self.state["next_round"], "from": before.model_dump(), "to": after.model_dump()})
        self.state["config_fingerprint"] = config_fingerprint
        write_json(self.workspace.root / "config.json", self.config.model_dump(), replace=True)

    def run(self, *, resume: bool = False, dry_run: bool = False) -> dict[str, Any]:
        root = self.workspace.root
        if not dry_run:
            require_api_keys(self.config)
        with run_lock(root):
            config_fingerprint = json_fingerprint(self.config.model_dump())
            self.started = time.monotonic()
            if resume:
                self.state = json.loads((root / "state.json").read_text())
                if self.state["status"] in ("accepted", "not_accepted", "no_improvement"):
                    # A finished search continues only with more rounds; the final acceptance is rerun afterwards.
                    if self.config.budget.rounds <= self.state["next_round"]:
                        raise ValueError(f"this run completed {self.state['next_round']} rounds; raise budget.rounds "
                                         "above that to continue it, or use a new output directory")
                    self.state.setdefault("continued_after", []).append(
                        {"status": self.state["status"], "rounds": self.state["next_round"]})
                if self.state["config_fingerprint"] != config_fingerprint:
                    self._adopt_config_change(config_fingerprint)
                self.previous_elapsed = self.state["elapsed_seconds"]
                self._verify()
            else:
                self.state = {"schema_version": 1, "status": "created", "config_fingerprint": config_fingerprint,
                              "frozen_fingerprint": self.workspace.fingerprint(), "next_round": 0,
                              "llm_calls": 0, "role_calls": {"generator": 0, "judge": 0}, "evaluations": 0,
                              "current": None, "best": None, "latest_eligible": None, "history": [], "usage": [],
                              "acceptance_reports": [], "elapsed_seconds": 0.0}
                write_json(root / "config.json", self.config.model_dump())
            self._save()
            try:
                task_context = self.workspace.context(self.config.max_context_chars)
                if dry_run:
                    write_json(root / "plan.json", {"backend": "kai_pichu.optimizer", "context": task_context,
                                                   "first_request": messages(GENERATOR, task_context)}, replace=resume)
                    self.state["status"] = "planned"
                else:
                    self.state["status"] = "running"
                    self.state.pop("message", None)
                    baseline = self._evaluate("preflight", None, "search")
                    if baseline["status"] != "ready":
                        raise StopRun("benchmark_error", baseline.get("message", "baseline did not pass its conformance checks or timing"))
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
