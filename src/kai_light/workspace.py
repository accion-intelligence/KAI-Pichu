from __future__ import annotations

import difflib
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

import yaml

from .benchmark.api import json_fingerprint
from .benchmark.loading import file_inventory
from .benchmark.models import BenchmarkSpec, load_spec
from .io import write_json


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.bundle = self.root / "bundle"
        self.manifest = self.bundle / "benchmark.yaml"
        self.spec = load_spec(self.manifest)
        self.baseline = self.bundle / "_baseline"
        self.allowed = sorted(file_inventory(self.baseline, self.spec.implementation.files))

    @classmethod
    def create(cls, manifest: Path, root: Path) -> Workspace:
        manifest = manifest.resolve(strict=True)
        spec = load_spec(manifest)
        baseline = (manifest.parent / spec.implementation.root).resolve(strict=True)
        if root.resolve().is_relative_to(manifest.parent) or root.resolve().is_relative_to(baseline):
            raise ValueError("run directory must be outside the task and implementation directories")
        task_files = file_inventory(manifest.parent, [
            manifest.name, spec.adapter.rsplit(":", 1)[0], *spec.benchmark_files])
        implementation_files = file_inventory(baseline, spec.implementation.files)
        if any(name == "_baseline" or name.startswith("_baseline/") for name in task_files):
            raise ValueError("_baseline is reserved for the frozen implementation")
        root.mkdir(parents=True, exist_ok=False)
        bundle = root / "bundle"
        bundle.mkdir()
        for name in task_files:
            if name == manifest.name:
                continue
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest.parent / name, target)
        for name in implementation_files:
            target = bundle / "_baseline" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(baseline / name, target)
        value = spec.model_dump()
        value["implementation"]["root"] = "_baseline"
        (bundle / "benchmark.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
        workspace = cls(root)
        write_json(root / "source.json", {
            "original_manifest": str(manifest), "original_task_files": task_files,
            "original_implementation_files": implementation_files,
            "frozen_fingerprint": workspace.fingerprint(),
        })
        return workspace

    def fingerprint(self) -> str:
        # Freeze declared sources, not bytecode/build artifacts legitimately
        # created by adapters. Owners must declare all oracle/build helpers.
        return json_fingerprint({
            "benchmark": file_inventory(self.bundle, [
                self.manifest.name, self.spec.adapter.rsplit(":", 1)[0], *self.spec.benchmark_files]),
            "implementation": file_inventory(self.baseline, self.spec.implementation.files),
        })

    def verify(self, expected: str) -> None:
        if self.fingerprint() != expected:
            raise ValueError("frozen benchmark/baseline changed; the run is invalid")

    def sources(self, directory: Path) -> dict[str, str]:
        return {name: (directory / name).read_text(encoding="utf-8") for name in self.allowed}

    def candidate(self, index: int, anchor: Path, reply: dict[str, Any]) -> Path:
        if set(reply) != {"hypothesis", "files"} or not isinstance(reply["hypothesis"], str):
            raise ValueError("candidate response requires hypothesis and files")
        files = reply["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("candidate files must be a nonempty object")
        for name, content in files.items():
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or "\\" in name
                    or name not in self.allowed or not isinstance(content, str) or not content.strip()):
                raise ValueError(f"candidate may only replace declared source files: {name}")
        target = self.root / "candidates" / f"round-{index:04d}"
        target.mkdir(parents=True, exist_ok=False)
        for name in self.allowed:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(files.get(name, (anchor / name).read_text(encoding="utf-8")), encoding="utf-8")
        return target

    def patch(self, candidate: Path) -> str:
        parts = []
        for name in self.allowed:
            parts.extend(difflib.unified_diff(
                (self.baseline / name).read_text().splitlines(keepends=True),
                (candidate / name).read_text().splitlines(keepends=True),
                fromfile=f"a/{name}", tofile=f"b/{name}"))
        return "".join(parts)

    def context(self, max_chars: int) -> dict[str, Any]:
        # Required implementation text is never silently truncated.
        baseline = self.sources(self.baseline)
        if sum(map(len, baseline.values())) > max_chars // 2:
            raise ValueError("implementation exceeds context limit; increase max_context_chars")
        task_files = {}
        remaining = max_chars // 2
        names = [self.spec.adapter.rsplit(":", 1)[0]]
        names += sorted(file_inventory(self.bundle, self.spec.benchmark_files)) if self.spec.benchmark_files else []
        for name in dict.fromkeys(names):
            data = (self.bundle / name).read_text(encoding="utf-8", errors="replace")
            task_files[name] = data[:remaining]
            if len(data) > remaining:
                task_files[name] += "\n[context limit: remaining file content omitted]"
            remaining = max(0, remaining - len(data))
        return {"benchmark": self.spec.model_dump(), "editable_files": self.allowed,
                "baseline_sources": baseline, "task_sources": task_files}
