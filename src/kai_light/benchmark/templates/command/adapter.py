from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any, Iterable

from kai_light.benchmark import Benchmark, Case, Observation, Validation, json_fingerprint


class Task(Benchmark):
    def cases(self, split: str) -> Iterable[Case]:
        size = int(self.spec.options.get("size", 64))
        sizes = [size] if split == "smoke" else [1, size, size + 17]
        return [Case(id=f"n{n}", params={"n": n}, work_units=n) for n in sizes]

    def prepare(self, case: Case, seed: int) -> list[int]:
        rng = random.Random(seed)
        return [rng.randrange(-100, 100) for _ in range(case.params["n"])]

    def fingerprint(self, fixture: list[int]) -> str:
        return json_fingerprint(fixture)

    def load_implementation(self, workspace: Path) -> Any:
        directory = tempfile.TemporaryDirectory(prefix="kai-native-")
        binary = Path(directory.name) / "solution"
        options = self.spec.options
        command = [str(options.get("compiler", "c++")), *options.get("compile_flags", ["-O2", "-x", "c++"]),
                   str(workspace / "solution.cu"), "-o", str(binary)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True,
                           timeout=float(options.get("timeout_s", 30)))
        except BaseException:
            directory.cleanup()
            raise
        return directory, binary

    def run(self, implementation: Any, fixture: list[int]) -> Observation:
        payload = str(len(fixture)) + "\n" + " ".join(map(str, fixture)) + "\n"
        result = subprocess.run([str(implementation[1])], input=payload, text=True,
                                capture_output=True, check=True,
                                timeout=float(self.spec.options.get("timeout_s", 30)))
        return Observation(json.loads(result.stdout)["output"])

    def validate(self, case: Case, fixture: list[int], observation: Observation) -> Validation:
        return Validation(observation.output == [x * x + 1 for x in fixture], "exact integer reference")

    def invalid_observations(self, case: Case, fixture: Any, valid: Observation) -> Iterable[Observation]:
        return [Observation(valid.output[:-1]), Observation([x + 1 for x in valid.output])]

    def cleanup_implementation(self, implementation: Any) -> None:
        implementation[0].cleanup()
