"""Small timing backends; importing the CPU SDK does not import PyTorch."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import time
from typing import Any, Callable

from .api import Observation
from .models import Measurement


class Timer:
    def __init__(self, settings: Measurement):
        self.settings = settings
        self.torch: Any = None
        self.metadata: dict[str, Any] = {"timer": settings.timer}
        if settings.timer == "cuda_event":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("cuda_event timing requires PyTorch with an available CUDA device")
            self.torch = torch
            device = settings.device
            self.metadata.update({
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "device": device, "device_name": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "python_gc": "automatic cyclic GC deferred during CUDA event interval",
            })

    def context(self) -> Any:
        return self.torch.cuda.device(self.settings.device) if self.torch else nullcontext()

    def measure(
        self, call: Callable[[], Observation], synchronize: Callable[[], None],
    ) -> tuple[Observation, float]:
        synchronize()
        if self.torch is None:
            start = time.perf_counter_ns()
            observation = call()
            synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
        else:
            self.torch.cuda.synchronize(self.settings.device)
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            # Host-side event/Observation allocations can trigger cyclic GC
            # between event submissions, making idle GPU time look like kernel
            # work. Defer automatic GC only for this GPU timing interval; wall
            # timing retains application GC semantics. Restore even on failure.
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                start.record()
                observation = call()
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
            finally:
                (gc.enable if gc_was_enabled else gc.disable)()
        return observation, elapsed
