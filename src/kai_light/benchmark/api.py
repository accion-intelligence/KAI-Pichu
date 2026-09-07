"""Task-owned semantics. No timers, ranking, or model calls belong here."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .models import BenchmarkSpec, Case


@dataclass
class Observation:
    output: Any
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class Validation:
    passed: bool
    message: str = ""
    errors: dict[str, float] = field(default_factory=dict)


def json_fingerprint(value: Any) -> str:
    """Hash JSON-compatible inputs/metadata, rejecting non-finite floats."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Benchmark(ABC):
    """One independently versioned task adapter.

    ``prepare`` and ``reset`` are outside measurement. Each invocation of
    ``run`` is the complete declared workload, with reset before EVERY call.
    Store mutable implementation state in the fixture or reset it explicitly.
    ``validate`` runs immediately after synchronization, outside measurement.
    """

    def __init__(self, spec: BenchmarkSpec):
        self.spec = spec

    @abstractmethod
    def cases(self, split: str) -> Iterable[Case]:
        """Return deterministic case descriptors for smoke/search/acceptance."""

    @abstractmethod
    def prepare(self, case: Case, seed: int) -> Any:
        """Create input data and initial state usable by either implementation.

        Paired measurement crosses both implementations over both prepared
        fixtures. Keep implementation-specific resources keyed by implementation
        identity, or in the implementation handle, and reset before every call.
        """

    @abstractmethod
    def fingerprint(self, fixture: Any) -> str:
        """Hash input values/layout and initial state (exclude scratch/output)."""

    @abstractmethod
    def load_implementation(self, workspace: Path) -> Any:
        """Build/load only the implementation at the supplied absolute root."""

    @abstractmethod
    def run(self, implementation: Any, fixture: Any) -> Observation:
        """Execute exactly one workload; materialize outputs before returning."""

    @abstractmethod
    def validate(self, case: Case, fixture: Any, observation: Observation) -> Validation:
        """Check outputs AND required side effects using a trusted oracle."""

    @abstractmethod
    def invalid_observations(
        self, case: Case, fixture: Any, valid: Observation,
    ) -> Iterable[Observation]:
        """Supply at least one task-appropriate wrong result per case.

        Used to detect vacuous validators; probes are not a proof of coverage.
        Do not mutate the live fixture/valid output while constructing probes.
        """

    def reset(self, implementation: Any, fixture: Any) -> None:
        """Restore initial state before warmup/check/each measured invocation."""

    def synchronize(self, implementation: Any) -> None:
        """Wait for async work for wall timing. No-op only for blocking/CPU calls.

        With cuda_event timing, work on auxiliary streams must join the
        measured PyTorch current stream inside ``run``.
        """

    def cleanup_fixture(self, fixture: Any) -> None:
        """Release resources created by prepare, including on failure."""

    def cleanup_implementation(self, implementation: Any) -> None:
        """Release servers, processes, modules, or other loaded resources."""
