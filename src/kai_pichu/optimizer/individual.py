"""Candidate state for the optimization loop.

Original components: MIT, Copyright (c) 2025 Zijian Zhang."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KernelIndividual:
    id: int
    workspace: str
    hypothesis: str
    fingerprint: str
    metrics: dict[str, Any]
    score: float | None = None
    # The judge strategy the coder was given for this round; None for the seed round.
    diagnosis: dict[str, Any] | None = field(default=None)
    # The 1-based round whose code the coder modified; None for the seed round.
    base_round: int | None = field(default=None)

    @property
    def runnable(self) -> bool:
        return self.metrics.get("status") == "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
