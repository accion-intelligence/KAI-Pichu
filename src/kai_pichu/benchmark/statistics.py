"""Paired-block inference. Cases in one block are resampled together."""
from __future__ import annotations

import math
import random
from typing import Any


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(
    records: list[dict[str, Any]], *, metric: str, direction: str,
    weights: dict[str, float], confidence: float, bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    blocks = sorted({row["block"] for row in records})
    if len(blocks) < 4:
        raise ValueError("at least four complete paired blocks are required")
    logs: dict[str, list[float]] = {case: [] for case in weights}
    for block in blocks:
        for case in weights:
            arms: dict[str, list[float]] = {"a": [], "b": []}
            for row in records:
                if row["block"] == block and row["case_id"] == case:
                    arms[row["arm"]].append(float(row["metrics"][metric]))
            if len(arms["a"]) != 2 or len(arms["b"]) != 2:
                raise ValueError("each block/case must have exactly two A and two B samples")
            if any(not math.isfinite(v) or v <= 0 for values in arms.values() for v in values):
                raise ValueError("objective samples must be finite and positive")
            delta = sum(math.log(v) for v in arms["a"]) / 2 - sum(math.log(v) for v in arms["b"]) / 2
            logs[case].append(delta if direction == "minimize" else -delta)
    total_weight = sum(weights.values())
    overall = [sum(weights[c] * logs[c][i] for c in weights) / total_weight for i in range(len(blocks))]
    rng = random.Random(seed)
    indices = [[rng.randrange(len(blocks)) for _ in blocks] for _ in range(bootstrap_samples)]
    alpha = (1 - confidence) / 2

    def estimate(values: list[float]) -> dict[str, Any]:
        draws = [math.exp(sum(values[i] for i in sample) / len(sample)) for sample in indices]
        return {
            "speedup": math.exp(sum(values) / len(values)),
            "interval": [_quantile(draws, alpha), _quantile(draws, 1 - alpha)],
            "block_speedups": [math.exp(v) for v in values],
        }

    return {
        "overall": estimate(overall),
        "cases": {case: estimate(values) for case, values in logs.items()},
        "confidence": confidence,
        "method": "paired_block_percentile_bootstrap_log_speedup",
        "blocks": len(blocks),
        "assumption": "Blocks are sufficiently independent; cross-run drift requires separate sessions.",
    }
