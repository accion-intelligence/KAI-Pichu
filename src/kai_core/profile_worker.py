"""Profile the same adapter, candidate, case and scope used by the SDK."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

from .benchmark.api import json_fingerprint
from .benchmark.loading import file_inventory, load_adapter
from .benchmark.models import load_spec
from .io import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    import torch

    manifest = args.manifest.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    spec = load_spec(manifest)
    task = load_adapter(manifest, spec)
    cases = list(task.cases("search"))
    case = next((case for case in cases if case.id == args.case), None) if args.case else cases[0]
    if case is None:
        raise ValueError(f"unknown search case: {args.case}")
    fingerprint = json_fingerprint(file_inventory(candidate, spec.implementation.files))
    with torch.cuda.device(spec.measurement.device), ExitStack() as stack:
        implementation = task.load_implementation(candidate)
        stack.callback(task.cleanup_implementation, implementation)
        fixture = task.prepare(case, spec.seed)
        stack.callback(task.cleanup_fixture, fixture)
        inputs = task.fingerprint(fixture)
        for _ in range(spec.measurement.warmup):
            task.reset(implementation, fixture)
            result = task.run(implementation, fixture)
            task.synchronize(implementation)
            if not task.validate(case, fixture, result).passed:
                raise ValueError("profile workload failed correctness")
        task.reset(implementation, fixture)
        task.synchronize(implementation)
        if task.fingerprint(fixture) != inputs:
            raise ValueError("profile reset changed inputs")
        # CUDA profiler range excludes setup, compilation, oracle and warmup.
        torch.cuda.cudart().cudaProfilerStart()
        try:
            result = task.run(implementation, fixture)
            task.synchronize(implementation)
            torch.cuda.synchronize(spec.measurement.device)
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        if not task.validate(case, fixture, result).passed:
            raise ValueError("profile workload failed correctness")
    if fingerprint != json_fingerprint(file_inventory(candidate, spec.implementation.files)):
        raise ValueError("candidate changed during profiling")
    write_json(args.output, {"case": case.model_dump(), "scope": spec.objective.scope,
                            "input_fingerprint": inputs, "implementation_fingerprint": fingerprint,
                            "note": "NCU diagnostics only; profiling latency is never used as optimization score."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
