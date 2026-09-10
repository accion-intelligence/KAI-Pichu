"""Explicit source loading and provenance for benchmark bundles."""
from __future__ import annotations

import hashlib
import platform
import sys
import types
import uuid
from pathlib import Path
from typing import Any

from .api import Benchmark, json_fingerprint
from .models import BenchmarkSpec


def load_python_file(path: Path) -> types.ModuleType:
    """Load a leaf Python file under a unique module name, without stale pyc.

    This is namespace isolation for the leaf, NOT a process/security sandbox.
    For packages with colliding imports use a process-backed adapter instead.
    """
    path = path.resolve(strict=True)
    name = f"_kai_benchmark_{uuid.uuid4().hex}"
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


def load_adapter(path: Path, spec: BenchmarkSpec) -> Benchmark:
    file_name, class_name = spec.adapter.rsplit(":", 1)
    source = (path.parent / file_name).resolve(strict=True)
    if not source.is_relative_to(path.parent.resolve()):
        raise ValueError("adapter resolves outside benchmark directory")
    cls = getattr(load_python_file(source), class_name)
    if not isinstance(cls, type) or not issubclass(cls, Benchmark):
        raise TypeError("adapter class must extend kai_core.benchmark.Benchmark")
    return cls(spec)


def file_inventory(root: Path, patterns: list[str]) -> dict[str, str]:
    root = root.resolve(strict=True)
    inventory: dict[str, str] = {}
    for pattern in patterns:
        matched = False
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"fingerprinted file escapes its root: {path}")
            matched = True
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            inventory[path.relative_to(root).as_posix()] = digest.hexdigest()
        if not matched:
            raise ValueError(f"file pattern matched no files: {root} / {pattern}")
    return inventory


def provenance(path: Path, spec: BenchmarkSpec, workspace: Path) -> dict[str, Any]:
    adapter_file = spec.adapter.rsplit(":", 1)[0]
    benchmark_files = file_inventory(path.parent, [path.name, adapter_file, *spec.benchmark_files])
    implementation_files = file_inventory(workspace, spec.implementation.files)
    return {
        "benchmark_fingerprint": json_fingerprint(benchmark_files),
        "benchmark_files": benchmark_files,
        "implementation_fingerprint": json_fingerprint(implementation_files),
        "implementation_files": implementation_files,
        "workspace": str(workspace),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
