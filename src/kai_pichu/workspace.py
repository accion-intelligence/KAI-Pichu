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
        task_files = file_inventory(manifest.parent, [manifest.name, *spec.task_file_patterns()])
        for name in file_inventory(manifest.parent, spec.agent_file_patterns()):
            try:
                (manifest.parent / name).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"agent-visible files must be UTF-8 text the model can read: {name}") from None
        if spec.fusion is not None:
            kernel_files = {(manifest.parent / name).resolve() for name in file_inventory(manifest.parent, spec.fusion.file_patterns())}
            editable = {(baseline / name).resolve() for name in file_inventory(baseline, spec.implementation.files)}
            if kernel_files & editable:
                raise ValueError("fusion kernels are read-only material and cannot also be implementation files")
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
            "benchmark": file_inventory(self.bundle, [self.manifest.name, *self.spec.task_file_patterns()]),
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
        # The agent sees the manifest, the editable sources and only the files the
        # task owner listed in agent_files (plus the kernels of a fusion task). The
        # adapter, oracle and input generation stay hidden so candidates cannot
        # specialize to the test distribution.
        task_files = {}
        remaining = max_chars // 2
        patterns = self.spec.agent_file_patterns()
        names = sorted(file_inventory(self.bundle, patterns)) if patterns else []
        for name in names:
            data = (self.bundle / name).read_text(encoding="utf-8")
            task_files[name] = data[:remaining]
            if len(data) > remaining:
                task_files[name] += "\n[context limit: remaining file content omitted]"
            remaining = max(0, remaining - len(data))
        context = {"benchmark": self.spec.model_dump(), "editable_files": self.allowed,
                   "baseline_sources": baseline, "task_sources": task_files}
        if self.spec.fusion is not None:
            context["fusion"] = {
                "goal": "Replace the baseline's sequence of kernel launches with fewer, fused launches "
                        "that compute the same result within the declared tolerance.",
                "kernels": [{"name": kernel.name, "entry": kernel.entry, "description": kernel.description,
                             "files": sorted(file_inventory(self.bundle, kernel.files))}
                            for kernel in self.spec.fusion.kernels],
                "intermediates": self.spec.fusion.intermediates,
                "read_only_files": sorted(file_inventory(self.bundle, self.spec.fusion.file_patterns())),
            }
        return context
