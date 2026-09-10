"""Candidate state for the optimization loop.

Original components: MIT, Copyright (c) 2025 Zijian Zhang."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KernelIndividual:
    id: int
    workspace: str
    hypothesis: str
    fingerprint: str
    metrics: dict[str, Any]
    score: float | None = None

    @property
    def runnable(self) -> bool:
        return self.metrics.get("status") == "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
