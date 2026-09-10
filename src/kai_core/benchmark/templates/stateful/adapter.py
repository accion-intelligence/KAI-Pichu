from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any, Iterable

from kai_core.benchmark import Benchmark, Case, Observation, Validation, load_python_file


class Task(Benchmark):
    def cases(self, split: str) -> Iterable[Case]:
        steps = int(self.spec.options.get("steps", 4096))
        lengths = [steps] if split == "smoke" else [1, steps, steps + 3]
        return [Case(id=f"steps{n}", params={"steps": n}, work_units=n) for n in lengths]

    def prepare(self, case: Case, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        inputs = [rng.uniform(-1, 1) for _ in range(case.params["steps"])]
        initial = rng.random()
        state = initial
        expected = []
        for value in inputs:
            state = 0.75 * state + value
            expected.append(state)
        return {"inputs": inputs, "initial": initial, "state": [initial], "expected": expected}

    def load_implementation(self, workspace: Path) -> Any:
        return load_python_file(workspace / "solution.py")

    def reset(self, implementation: Any, fixture: dict[str, Any]) -> None:
        fixture["state"][:] = [fixture["initial"]]

    def run(self, implementation: Any, fixture: dict[str, Any]) -> Observation:
        output = implementation.forward(fixture["inputs"], fixture["state"])
        return Observation({"outputs": output, "final_state": fixture["state"][0]})

    def validate(self, case: Case, fixture: dict[str, Any], observation: Observation) -> Validation:
        output = observation.output
        if not isinstance(output, dict) or not isinstance(output.get("outputs"), list):
            return Validation(False, "missing output sequence")
        got, want = output["outputs"], fixture["expected"]
        if len(got) != len(want):
            return Validation(False, "incorrect number of state updates")
        try:
            passed = all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(got, want))
            passed = passed and math.isclose(output["final_state"], want[-1], rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, KeyError):
            return Validation(False, "malformed state/output")
        return Validation(passed, "check every update and final recurrent state")

    def invalid_observations(self, case: Case, fixture: Any, valid: Observation) -> Iterable[Observation]:
        return [Observation({**valid.output, "final_state": valid.output["final_state"] + 1}),
                Observation({**valid.output, "outputs": valid.output["outputs"][:-1]})]
