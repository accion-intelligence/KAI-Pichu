from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from kai_pichu.benchmark import Benchmark, Case, Observation, Validation, json_fingerprint, load_python_file


class Task(Benchmark):
    def cases(self, split: str) -> Iterable[Case]:
        size = int(self.spec.options.get("size", 4096))
        sizes = [size] if split == "smoke" else [size, size + 17, size * 4]
        return [Case(id=f"n{n}", params={"n": n}, work_units=n) for n in sizes]

    def prepare(self, case: Case, seed: int) -> dict[str, Any]:
        device = self.spec.options.get("device", "cpu")
        generator = torch.Generator(device=device).manual_seed(seed)
        a = torch.randn(case.params["n"], generator=generator, device=device)
        b = torch.randn(case.params["n"], generator=generator, device=device)
        return {"a": a, "b": b, "expected": a + b}

    def fingerprint(self, fixture: dict[str, Any]) -> str:
        return json_fingerprint({key: {"values": fixture[key].cpu().tolist(),
                                      "dtype": str(fixture[key].dtype), "stride": list(fixture[key].stride())}
                                 for key in ("a", "b")})

    def load_implementation(self, workspace: Path) -> Any:
        return load_python_file(workspace / "solution.py")

    def run(self, implementation: Any, fixture: dict[str, Any]) -> Observation:
        return Observation(implementation.forward(fixture["a"], fixture["b"]))

    def synchronize(self, implementation: Any) -> None:
        device = torch.device(self.spec.options.get("device", "cpu"))
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def validate(self, case: Case, fixture: dict[str, Any], observation: Observation) -> Validation:
        got, want = observation.output, fixture["expected"]
        if not isinstance(got, torch.Tensor) or got.shape != want.shape or got.dtype != want.dtype or got.device != want.device:
            return Validation(False, "output type/shape/dtype/device mismatch")
        passed = torch.allclose(got, want, atol=float(self.spec.options.get("atol", 1e-6)),
                                rtol=float(self.spec.options.get("rtol", 1e-5)))
        return Validation(bool(passed), "compare against independent torch addition")

    def invalid_observations(self, case: Case, fixture: Any, valid: Observation) -> Iterable[Observation]:
        return [Observation(valid.output + 1), Observation(valid.output[:-1])]
