"""Versioned, JSON-serializable benchmark contracts."""
from __future__ import annotations

import math
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Case(Model):
    id: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0)
    work_units: float | None = Field(default=None, gt=0)


class Objective(Model):
    metric: str = Field(default="latency_ms", min_length=1)
    unit: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"] = "minimize"
    scope: Literal["kernel", "module", "end_to_end"]
    aggregation: Literal["weighted_geomean_speedup"] = "weighted_geomean_speedup"
    target_speedup: float | None = Field(default=None, gt=1)
    max_case_regression: float | None = Field(default=0.01, ge=0, lt=1)

    @model_validator(mode="after")
    def check_builtin_metric(self) -> Objective:
        if self.metric == "latency_ms" and (self.unit != "ms" or self.direction != "minimize"):
            raise ValueError("latency_ms must use unit=ms and direction=minimize")
        if self.metric == "throughput" and self.direction != "maximize":
            raise ValueError("throughput must use direction=maximize")
        return self


class MetricLimit(Model):
    metric: str = Field(min_length=1)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def check_bounds(self) -> MetricLimit:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a metric limit needs minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum exceeds maximum")
        return self


class Measurement(Model):
    timer: Literal["wall", "cuda_event"] = "wall"
    device: int = Field(default=0, ge=0)
    warmup: int = Field(default=5, ge=1)
    blocks: int = Field(default=20, ge=4)
    iterations: int = Field(default=3, ge=1)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    bootstrap_samples: int = Field(default=2000, ge=200)
    calibration: bool = Field(default=False, description=(
        "Optional baseline-vs-baseline A/A check before any comparison; a failure stops the run"))
    calibration_tolerance: float = Field(default=0.05, gt=0, lt=1)
    boundary: str = Field(min_length=1)
    cache_policy: str = Field(min_length=1)


class ImplementationSpec(Model):
    root: str = "."
    files: list[str] = Field(min_length=1)


class FusionKernel(Model):
    """One user-supplied kernel that a fusion task asks the agent to fuse."""
    name: str = Field(min_length=1)
    files: list[str] = Field(min_length=1, description="Source files of this kernel; frozen, agent-visible, read-only")
    entry: str = Field(min_length=1, description="Launcher symbol the unfused baseline calls")
    description: str = Field(min_length=1)


class FusionSpec(Model):
    kernels: list[FusionKernel] = Field(min_length=2, description="Kernels in the baseline's execution order")
    intermediates: list[str] = Field(default_factory=list, description=(
        "Buffers handed from one kernel to the next; a fused implementation may keep them on chip"))

    @model_validator(mode="after")
    def check_unique_names(self) -> FusionSpec:
        names = [kernel.name for kernel in self.kernels]
        if len(set(names)) != len(names):
            raise ValueError("fusion kernel names must be unique")
        return self

    def file_patterns(self) -> list[str]:
        return [pattern for kernel in self.kernels for pattern in kernel.files]


class BenchmarkSpec(Model):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    kind: Literal["operator", "fusion"] = Field(default="operator", description=(
        "operator: optimize one implementation; fusion: fuse the user-supplied kernels listed under `fusion`"))
    fusion: FusionSpec | None = None
    adapter: str = Field(description="Relative Python file and class, e.g. adapter.py:Task")
    implementation: ImplementationSpec
    benchmark_files: list[str] = Field(default_factory=list)
    agent_files: list[str] = Field(default_factory=list, description=(
        "Text files the optimization agent may read: interfaces, ABI headers, task descriptions. "
        "Never list input generation, oracle or adapter code; the agent must not learn the test distribution."))
    description: str = Field(min_length=1)
    objective: Objective
    measurement: Measurement
    limits: list[MetricLimit] = Field(default_factory=list)
    seed: int = Field(default=0, ge=0)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_adapter(self) -> BenchmarkSpec:
        file_name, separator, class_name = self.adapter.rpartition(":")
        if not separator or not file_name.endswith(".py") or not class_name.isidentifier():
            raise ValueError("adapter must have the form relative/path.py:ClassName")
        path = Path(file_name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("adapter must be inside the benchmark directory")
        if (self.kind == "fusion") != (self.fusion is not None):
            raise ValueError("kind: fusion requires a fusion section listing the kernels, and vice versa")
        for pattern in self.task_file_patterns() + self.implementation.files:
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError("file patterns must be relative and cannot contain '..'")
        adapter_path = PurePosixPath(file_name)
        if any(adapter_path.match(pattern) or fnmatchcase(file_name, pattern) for pattern in self.agent_file_patterns()):
            raise ValueError("agent-visible files cannot include the adapter; input synthesis and the oracle stay hidden from the agent")
        return self

    def agent_file_patterns(self) -> list[str]:
        """Text the optimization agent may read: agent_files plus, for fusion tasks, the kernels to fuse."""
        fusion_files = self.fusion.file_patterns() if self.fusion else []
        return [*self.agent_files, *fusion_files]

    def task_file_patterns(self) -> list[str]:
        """Every declared task file besides the manifest: adapter, helpers, agent-visible files and fusion kernels."""
        return [self.adapter.rsplit(":", 1)[0], *self.benchmark_files, *self.agent_file_patterns()]


def load_spec(path: Path) -> BenchmarkSpec:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BenchmarkSpec.model_validate(value)


def finite_number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a numeric metric, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("metrics must be finite; the objective must be strictly positive")
    return result
