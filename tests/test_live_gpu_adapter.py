"""Explicitly opt-in checks for native GPU example measurement boundaries."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import statistics
import time

import pytest

from kai_core.benchmark.loading import load_adapter
from kai_core.benchmark.models import load_spec
from kai_core.benchmark.timing import Timer


@pytest.mark.skipif(os.environ.get("KAI_CORE_GPU_TESTS") != "1",
                    reason="requires explicit KAI_CORE_GPU_TESTS=1 and an idle CUDA GPU")
def test_graph_event_objective_excludes_host_pause_after_submission():
    import torch

    manifest = Path(__file__).parents[1] / "examples/layernorm/benchmark-graph-events.yaml"
    spec = load_spec(manifest)
    task = load_adapter(manifest, spec)
    timer = Timer(spec.measurement)
    with torch.cuda.device(0), ExitStack() as cleanup:
        implementation = task.load_implementation(manifest.parent / spec.implementation.root)
        cleanup.callback(task.cleanup_implementation, implementation)
        case = list(task.cases("smoke"))[0]
        fixture = task.prepare(case, spec.seed)
        cleanup.callback(task.cleanup_fixture, fixture)
        normal = []
        for _ in range(10):
            task.reset(implementation, fixture)
            task.synchronize(implementation)
            result = task.run(implementation, fixture)
            assert task.validate(case, fixture, result).passed
            normal.append(result.metrics["operator_latency_ms"])
        graph = fixture["graphs"][id(implementation)]

        class DelayedGraph:
            def replay(self):
                graph.replay()
                time.sleep(0.05)

        fixture["graphs"][id(implementation)] = DelayedGraph()
        for _ in range(3):
            task.reset(implementation, fixture)
            result, outer_ms = timer.measure(lambda: task.run(implementation, fixture),
                                              lambda: task.synchronize(implementation))
            assert task.validate(case, fixture, result).passed
            assert outer_ms >= 40
            assert 0 < result.metrics["operator_latency_ms"] < statistics.median(normal) * 3 + 0.01
