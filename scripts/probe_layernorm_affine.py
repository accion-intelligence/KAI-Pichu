"""Check frozen LayerNorm candidates on nonconstant affine inputs, without timing."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

from kai_light.benchmark.loading import load_adapter
from kai_light.benchmark.models import load_spec


def probe(manifest: Path, candidate: Path, *, capture_before_update: bool = False) -> dict:
    import torch

    spec = load_spec(manifest)
    task = load_adapter(manifest, spec)
    with torch.cuda.device(spec.measurement.device), ExitStack() as cleanup:
        implementation = task.load_implementation(candidate)
        cleanup.callback(task.cleanup_implementation, implementation)
        case = list(task.cases("smoke"))[0]
        fixture = task.prepare(case, spec.seed + 2003)
        cleanup.callback(task.cleanup_fixture, fixture)
        if capture_before_update:
            task.reset(implementation, fixture)
            initial = task.run(implementation, fixture)
            task.synchronize(implementation)
            if not task.validate(case, fixture, initial).passed:
                raise ValueError("candidate failed the original affine inputs")
        with torch.random.fork_rng(devices=[spec.measurement.device]):
            torch.manual_seed(2003)
            for name in ("gamma", "beta"):
                fixture["inputs"][name].copy_(torch.randn_like(fixture["inputs"][name]))
        with torch.no_grad():
            fixture["expected"] = dict(task.definition.reference_fn(list(fixture["inputs"].items())))
        task.reset(implementation, fixture)
        observation = task.run(implementation, fixture)
        task.synchronize(implementation)
        validation = task.validate(case, fixture, observation)
        maximum_errors = {name: float((actual - fixture["expected"][name]).abs().max())
                          for name, actual in observation.output.items()}
        return {"candidate": str(candidate), "passed": validation.passed,
                "message": validation.message, "max_absolute_errors": maximum_errors,
                "capture_before_update": capture_before_update,
                "scope": "Additional correctness probe with random gamma/beta; no performance claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture-before-update", action="store_true")
    args = parser.parse_args()
    result = probe(args.manifest.resolve(), args.candidate.resolve(),
                   capture_before_update=args.capture_before_update)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
