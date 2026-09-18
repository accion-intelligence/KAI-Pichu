"""Deterministic validation, baseline timing, paired measurement, and acceptance."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import random
import statistics
from typing import Any

from kai_core import __version__

from .api import Benchmark, Observation, Validation, json_fingerprint
from .loading import file_inventory, load_adapter, provenance
from .models import Case, finite_number, load_spec
from .statistics import summarize
from .timing import Timer


class BenchmarkError(RuntimeError):
    pass


SDK_METRICS = ("latency_ms", "throughput")  # Computed by the runner from its own timer.


class Runner:
    """Run trusted task adapters. This is not a security/process sandbox.

    Call validate() for conformance checks plus a timing pass over the baseline,
    or run(candidate) for a paired A/B comparison against the baseline.
    """

    def __init__(self, manifest: str | Path, *, split: str = "smoke"):
        self.path = Path(manifest).expanduser().resolve(strict=True)
        self.spec = load_spec(self.path)
        self.spec_fingerprint = json_fingerprint(self.spec.model_dump())
        self.task = load_adapter(self.path, self.spec)
        self.split = split
        self.cases = list(self.task.cases(split))
        if not self.cases or any(not isinstance(case, Case) for case in self.cases):
            raise BenchmarkError("cases() must return a non-empty sequence of Case objects")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise BenchmarkError("case IDs must be unique within a split")
        descriptors = [case.model_dump() for case in self.cases]
        if descriptors != [case.model_dump() for case in self.task.cases(split)]:
            raise BenchmarkError("cases() is not deterministic")
        self.case_fingerprint = json_fingerprint(descriptors)
        self.baseline_root = (self.path.parent / self.spec.implementation.root).resolve(strict=True)
        self.timer = Timer(self.spec.measurement)
        self.input_fingerprints: dict[str, str] = {}
        self.limit_failures: list[dict[str, Any]] = []

    def _fingerprint(self, fixture: Any) -> str:
        fingerprint = self.task.fingerprint(fixture)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise BenchmarkError("fingerprint() must return a non-empty string")
        return fingerprint

    def _fixture(self, case: Case, stack: ExitStack) -> Any:
        fixture = self.task.prepare(case, self.spec.seed)
        stack.callback(self.task.cleanup_fixture, fixture)
        fingerprint = self._fingerprint(fixture)
        expected = self.input_fingerprints.setdefault(case.id, fingerprint)
        if fingerprint != expected:
            raise BenchmarkError(f"{case.id}: prepare() produced different input/state for the same seed")
        return fixture

    def _validation(self, case: Case, fixture: Any, observation: Observation) -> Validation:
        if not isinstance(observation, Observation):
            raise BenchmarkError("run() must return Observation")
        validation = self.task.validate(case, fixture, observation)
        if not isinstance(validation, Validation) or type(validation.passed) is not bool:
            raise BenchmarkError("validate() must return Validation with a boolean passed field")
        for value in validation.errors.values():
            finite_number(value)
        return validation

    def _reset(self, case: Case, implementation: Any, fixture: Any, *, check_fingerprint: bool = True) -> None:
        self.task.reset(implementation, fixture)
        self.task.synchronize(implementation)
        if check_fingerprint and self._fingerprint(fixture) != self.input_fingerprints[case.id]:
            raise BenchmarkError(f"{case.id}: reset() did not restore the declared input/state")

    def _invoke(
        self, case: Case, implementation: Any, fixture: Any, *, timed: bool,
    ) -> tuple[Observation, dict[str, float]]:
        # Hashing tensor values immediately before measurement would impose an
        # accidental cache policy. Check resets during conformance/warmup only.
        self._reset(case, implementation, fixture, check_fingerprint=not timed)
        if timed:
            observation, elapsed = self.timer.measure(
                lambda: self.task.run(implementation, fixture),
                lambda: self.task.synchronize(implementation),
            )
        else:
            observation = self.task.run(implementation, fixture)
            self.task.synchronize(implementation)
            elapsed = None
        validation = self._validation(case, fixture, observation)
        if not validation.passed:
            raise BenchmarkError(f"{case.id}: correctness failed: {validation.message}; {validation.errors}")
        metrics = {key: finite_number(value) for key, value in observation.metrics.items()}
        if any(key in metrics for key in SDK_METRICS):
            raise BenchmarkError("latency_ms and throughput are SDK-owned; use a distinct name for adapter metrics")
        metric = self.spec.objective.metric
        # An adapter-owned objective must be present on every invocation, timed or
        # not, so conformance checks catch a missing metric before any timing runs.
        if metric not in SDK_METRICS and metric not in metrics:
            raise BenchmarkError(f"{case.id}: missing objective metric {metric!r}")
        if elapsed is not None:
            metrics["latency_ms"] = finite_number(elapsed, positive=True)
            if case.work_units is not None:
                metrics["throughput"] = case.work_units * 1000 / elapsed
            if metric not in metrics:
                raise BenchmarkError(f"{case.id}: missing objective metric {metric!r}")
            finite_number(metrics[metric], positive=True)
        return observation, metrics

    def _check_cases(self, implementation: Any, *, probes: bool) -> list[dict[str, Any]]:
        checks = []
        for case in self.cases:
            with ExitStack() as stack:
                fixture = self._fixture(case, stack)
                # Independently prepare again; a single persistent object is
                # insufficient evidence that the input is reproducible.
                self._fixture(case, stack)
                observation, _ = self._invoke(case, implementation, fixture, timed=False)
                probe_count = 0
                if probes:
                    for invalid in self.task.invalid_observations(case, fixture, observation):
                        probe_count += 1
                        validation = self._validation(case, fixture, invalid)
                        if validation.passed:
                            raise BenchmarkError(f"{case.id}: validator accepted invalid observation #{probe_count}")
                    if probe_count == 0:
                        raise BenchmarkError(f"{case.id}: at least one invalid-observation probe is required")
                self._reset(case, implementation, fixture)
                checks.append({"case_id": case.id, "passed": True, "rejected_probes": probe_count})
        return checks

    def _check_limits(self, case: Case, metrics: dict[str, float]) -> None:
        for limit in self.spec.limits:
            if limit.metric not in metrics:
                raise BenchmarkError(f"missing constrained metric {limit.metric!r}")
            value = metrics[limit.metric]
            if ((limit.minimum is not None and value < limit.minimum)
                    or (limit.maximum is not None and value > limit.maximum)):
                self.limit_failures.append({"case_id": case.id, "metric": limit.metric, "value": value,
                                            "minimum": limit.minimum, "maximum": limit.maximum})

    def _paired(self, a: Any, b: Any, *, candidate_limits: bool) -> list[dict[str, Any]]:
        settings = self.spec.measurement
        records: list[dict[str, Any]] = []
        rng = random.Random(self.spec.seed)
        with ExitStack() as stack:
            fixtures = {case.id: [self._fixture(case, stack) for _ in range(2)]
                        for case in self.cases}
            implementations = {"a": a, "b": b}
            for case in self.cases:
                for _ in range(settings.warmup):
                    for fixture in fixtures[case.id]:
                        for arm in ("a", "b"):
                            self._invoke(case, implementations[arm], fixture, timed=False)
            for block in range(settings.blocks):
                cases = list(self.cases)
                rng.shuffle(cases)
                order = ("a", "b", "b", "a") if block % 2 == 0 else ("b", "a", "a", "b")
                for case in cases:
                    for position, arm in enumerate(order):
                        # Both arms use each prepared buffer once per block.
                        # Fixed arm-to-buffer assignments confound implementation
                        # speed with allocation/address/cache placement effects.
                        fixture_slot = (position // 2 + block) % 2
                        fixture = fixtures[case.id][fixture_slot]
                        samples = []
                        for _ in range(settings.iterations):
                            _, metrics = self._invoke(case, implementations[arm], fixture, timed=True)
                            if candidate_limits and arm == "b":
                                self._check_limits(case, metrics)
                            samples.append(metrics)
                        keys = set(samples[0])
                        if any(set(sample) != keys for sample in samples):
                            raise BenchmarkError("metric keys changed between invocations")
                        records.append({
                            "block": block, "position": position, "case_id": case.id, "arm": arm,
                            "fixture_slot": fixture_slot,
                            "metrics": {key: statistics.fmean(sample[key] for sample in samples) for key in keys},
                            "samples": samples,
                        })
        return records

    def _summarize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        settings = self.spec.measurement
        return summarize(
            records, metric=self.spec.objective.metric, direction=self.spec.objective.direction,
            weights={case.id: case.weight for case in self.cases}, confidence=settings.confidence,
            bootstrap_samples=settings.bootstrap_samples, seed=self.spec.seed,
        )

    def _time_baseline(self, baseline: Any) -> dict[str, Any]:
        """Time the baseline alone on every case, with the manifest's warmup and sample counts.

        Preflight uses this to prove that the baseline can be timed within the
        declared boundary and to show the task owner what it measures; it makes
        no judgment about measurement stability.
        """
        settings = self.spec.measurement
        metric = self.spec.objective.metric
        per_case: dict[str, dict[str, float | int]] = {}
        with ExitStack() as stack:
            for case in self.cases:
                fixture = self._fixture(case, stack)
                for _ in range(settings.warmup):
                    self._invoke(case, baseline, fixture, timed=False)
                values = []
                for _ in range(settings.blocks * settings.iterations):
                    _, metrics = self._invoke(case, baseline, fixture, timed=True)
                    if metric not in metrics:
                        raise BenchmarkError(f"{case.id}: baseline did not report the objective metric {metric!r}")
                    values.append(float(metrics[metric]))
                per_case[case.id] = {"mean": statistics.fmean(values), "median": statistics.median(values),
                                     "min": min(values), "max": max(values), "samples": len(values)}
        return {"metric": metric, "unit": self.spec.objective.unit, "per_case": per_case}

    def _report(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "sdk_version": __version__, "benchmark": self.spec.name,
            "created_at": datetime.now(timezone.utc).isoformat(), "split": self.split,
            "case_fingerprint": self.case_fingerprint,
            "cases": [case.model_dump() for case in self.cases],
            "input_fingerprints": self.input_fingerprints,
            "spec": self.spec.model_dump(), "timing_environment": self.timer.metadata,
            "sdk_fingerprint": json_fingerprint(file_inventory(Path(__file__).parent, ["*.py"])),
        }

    def _assert_unchanged(self, old: dict[str, Any], workspace: Path) -> None:
        if self.spec_fingerprint != json_fingerprint(self.spec.model_dump()):
            raise BenchmarkError("benchmark specification changed in memory during execution")
        if self.case_fingerprint != json_fingerprint([case.model_dump() for case in self.cases]):
            raise BenchmarkError("case definitions changed in memory during execution")
        current = provenance(self.path, self.spec, workspace)
        for key in ("benchmark_fingerprint", "implementation_fingerprint"):
            if old[key] != current[key]:
                raise BenchmarkError(f"{key} changed during execution; measurements are invalid")

    def validate(self, *, checks_only: bool = False) -> dict[str, Any]:
        report = self._report()
        report["baseline"] = provenance(self.path, self.spec, self.baseline_root)
        with self.timer.context(), ExitStack() as stack:
            baseline = self.task.load_implementation(self.baseline_root)
            stack.callback(self.task.cleanup_implementation, baseline)
            report["checks"] = self._check_cases(baseline, probes=True)
            if not checks_only:
                report["baseline_timing"] = self._time_baseline(baseline)
        self._assert_unchanged(report["baseline"], self.baseline_root)
        report["ready"] = not checks_only
        report["status"] = "ready" if report["ready"] else "checks_passed"
        return report

    def run(self, candidate: str | Path) -> dict[str, Any]:
        candidate_root = Path(candidate).expanduser().resolve(strict=True)
        self.limit_failures.clear()
        report = self._report()
        report["baseline"] = provenance(self.path, self.spec, self.baseline_root)
        report["candidate"] = provenance(self.path, self.spec, candidate_root)
        with self.timer.context(), ExitStack() as stack:
            baseline = self.task.load_implementation(self.baseline_root)
            stack.callback(self.task.cleanup_implementation, baseline)
            implementation = self.task.load_implementation(candidate_root)
            stack.callback(self.task.cleanup_implementation, implementation)
            report["checks"] = self._check_cases(baseline, probes=True)
            report["candidate_checks"] = self._check_cases(implementation, probes=False)
            records = self._paired(baseline, implementation, candidate_limits=True)
            report["records"] = records
            report["comparison"] = self._summarize(records)
            report["acceptance"] = self._acceptance(report["comparison"])
            report["status"] = "completed"
        self._assert_unchanged(report["baseline"], self.baseline_root)
        self._assert_unchanged(report["candidate"], candidate_root)
        return report

    def _acceptance(self, comparison: dict[str, Any]) -> dict[str, Any]:
        objective = self.spec.objective
        lower, upper = comparison["overall"]["interval"]
        threshold = objective.target_speedup or 1.0
        target_met = lower >= threshold if objective.target_speedup else lower > 1.0
        case_failures = []
        if objective.max_case_regression is not None:
            limit = objective.max_case_regression
            minimum = 1 / (1 + limit) if objective.direction == "minimize" else 1 - limit
            case_failures = [case for case, estimate in comparison["cases"].items()
                             if estimate["interval"][0] < minimum]
        accepted = target_met and not case_failures and not self.limit_failures
        verdict = ("accepted" if accepted else "constraint_failed" if case_failures or self.limit_failures
                   else "regressed" if upper < 1 else "improved_below_target" if lower > 1 else "inconclusive")
        return {
            "accepted": accepted, "verdict": verdict, "target_met": target_met,
            "target_speedup": objective.target_speedup,
            "unconfirmed_case_constraints": case_failures,
            "metric_limit_failure_count": len(self.limit_failures),
            "metric_limit_failures": self.limit_failures[:20],
            "note": "Acceptance uses pointwise confidence bounds. An independent frozen-candidate rerun is required for release.",
        }
