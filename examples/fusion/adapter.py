"""Adapt the epilogue fusion task to the KAI Core benchmark SDK."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import torch
from torch.utils.cpp_extension import include_paths, library_paths

from kai_core.benchmark import (
    Benchmark, Case, Observation, Validation, fixture_fingerprint, load_python_file,
)

WORKSPACE_SLACK_BYTES = 16 * 1024 * 1024


@dataclass
class NativeImplementation:
    library: Any
    launch: Any
    directory: TemporaryDirectory


def workspace_bytes(rows: int, cols: int) -> int:
    """Room for the unfused baseline's two FP32 intermediates plus slack; fused code may ignore it."""
    return 2 * rows * cols * 4 + WORKSPACE_SLACK_BYTES


class Epilogue(Benchmark):
    def __init__(self, spec: Any):
        super().__init__(spec)
        self.root = Path(__file__).resolve().parent
        self.inputs = load_python_file(self.root / "inputs.py")
        self.reference = load_python_file(self.root / "reference.py")
        self.device = torch.device(spec.options["device"])
        if self.device.type != "cuda" or self.device.index != spec.measurement.device:
            raise ValueError("options.device must match measurement.device")
        if spec.options["execution"] not in ("eager", "graph_events"):
            raise ValueError("execution must be eager or graph_events")
        self.cache_buffer: torch.Tensor | None = None

    # --- task definition -------------------------------------------------------

    def cases(self, split: str) -> Iterable[Case]:
        return [Case(id=f"{profile}_{rows}x{cols}_seed{seed}",
                     params={"profile": profile, "seed_offset": seed, "rows": rows, "cols": cols},
                     work_units=float(rows * cols))
                for profile, seed, rows, cols in self.inputs.SPLITS[split]]

    def prepare(self, case: Case, seed: int) -> dict[str, Any]:
        params = case.params
        shape = (params["rows"], params["cols"])
        inputs = self.inputs.generate(params["profile"], shape, seed + params["seed_offset"], self.device)
        with torch.cuda.device(self.device), torch.no_grad():
            expected = self.reference.epilogue(inputs["x"], inputs["bias"], inputs["residual"])
        workspace = torch.empty(workspace_bytes(*shape), dtype=torch.uint8, device=self.device)
        return {"inputs": inputs, "outputs": {"out": torch.empty_like(inputs["x"])},
                "expected": {"out": expected}, "workspace": workspace, "graphs": {}, "events": {}}

    def fingerprint(self, fixture: dict[str, Any]) -> str:
        # The fixture also holds poisoned outputs, scratch and captured graphs;
        # only the inputs define the workload identity.
        return fixture_fingerprint(fixture["inputs"])

    # --- implementation lifecycle --------------------------------------------

    def load_implementation(self, workspace: Path) -> NativeImplementation:
        directory = TemporaryDirectory(prefix="kai-fusion-")
        library = Path(directory.name) / "kernel.so"
        options = self.spec.options
        # -I{workspace} resolves solution.cu; -I{root} resolves the read-only kernels/ sources.
        command = [str(options["nvcc"]), "-O3", "-std=c++17", "--use_fast_math",
                   "--expt-relaxed-constexpr", "-shared", "-Xcompiler=-fPIC",
                   f"-arch={options['cuda_arch']}", f"-I{workspace}", f"-I{self.root}"]
        command += [f"-I{path}" for path in include_paths(device_type="cuda")]
        command += [str(self.root / "bridge.cu"), "-o", str(library)]
        for path in library_paths(device_type="cuda"):
            command += [f"-L{path}", f"-Xlinker=-rpath={path}"]
        command += ["-lcudart"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True,
                           timeout=float(options["compile_timeout_seconds"]))
            loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
            launch = loaded.kai_launch
            launch.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                               + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
            launch.restype = ctypes.c_int
            return NativeImplementation(loaded, launch, directory)
        except BaseException:
            directory.cleanup()
            raise

    def cleanup_implementation(self, implementation: NativeImplementation) -> None:
        implementation.directory.cleanup()

    def cleanup_fixture(self, fixture: dict[str, Any]) -> None:
        fixture.clear()

    # --- execution -----------------------------------------------------------------

    def _launch(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> None:
        inputs = fixture["inputs"]
        rows, cols = inputs["x"].shape
        workspace = fixture["workspace"]
        status = implementation.launch(
            inputs["x"].data_ptr(), inputs["bias"].data_ptr(), inputs["residual"].data_ptr(),
            fixture["outputs"]["out"].data_ptr(), rows, cols,
            workspace.data_ptr(), workspace.numel(),
            torch.cuda.current_stream(self.device).cuda_stream)
        if status:
            raise RuntimeError(f"native epilogue launch returned status {status}")

    def reset(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> None:
        graph_mode = self.spec.options["execution"] == "graph_events"
        if graph_mode and id(implementation) not in fixture["graphs"]:
            # Capture exactly ONE pipeline invocation between two external events.
            # A fused candidate captures fewer kernel nodes; the boundary is the same.
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._launch(implementation, fixture)
            stream.synchronize()
            events = (torch.cuda.Event(enable_timing=True, external=True),
                      torch.cuda.Event(enable_timing=True, external=True))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                events[0].record()
                self._launch(implementation, fixture)
                events[1].record()
            fixture["graphs"][id(implementation)] = graph
            fixture["events"][id(implementation)] = events
        fixture["outputs"]["out"].fill_(float("nan"))
        if self.cache_buffer is None:
            self.cache_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.cache_buffer.fill_(42)

    def run(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> Observation:
        if self.spec.options["execution"] == "graph_events":
            fixture["graphs"][id(implementation)].replay()
            start, end = fixture["events"][id(implementation)]
            end.synchronize()
            return Observation({"out": fixture["outputs"]["out"]},
                               {"operator_latency_ms": start.elapsed_time(end)})
        self._launch(implementation, fixture)
        return Observation({"out": fixture["outputs"]["out"]})

    def synchronize(self, implementation: Any) -> None:
        torch.cuda.synchronize(self.device)

    # --- correctness ----------------------------------------------------------------

    def validate(self, case: Case, fixture: dict[str, Any], observation: Observation) -> Validation:
        got = observation.output
        if not isinstance(got, dict) or set(got) != {"out"}:
            return Validation(False, "required output: out")
        actual, expected = got["out"], fixture["expected"]["out"]
        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or actual.dtype != expected.dtype or actual.device != expected.device):
            return Validation(False, "out: shape/dtype/device mismatch")
        if not torch.isfinite(actual).all():
            return Validation(False, "out: non-finite values")
        atol, rtol = self.reference.ATOL, self.reference.RTOL
        error = (actual - expected).abs().max().item()
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            return Validation(False, f"out: mismatch against the PyTorch reference, max abs error {error:.3e}",
                              {"max_abs_error": error})
        return Validation(True, f"out matches the PyTorch reference at atol={atol}, rtol={rtol}",
                          {"max_abs_error": error})

    def invalid_observations(
        self, case: Case, fixture: dict[str, Any], valid: Observation,
    ) -> Iterable[Observation]:
        good = valid.output["out"]
        perturbed = good.clone()
        perturbed.view(-1)[0] += 1.0
        yield Observation({"out": perturbed})
        poisoned = good.clone()
        poisoned.view(-1)[-1] = float("nan")
        yield Observation({"out": poisoned})
        yield Observation({})
        inputs = fixture["inputs"]
        with torch.no_grad():
            # A fusion that silently dropped a stage, and one that changed a stage's math.
            yield Observation({"out": self.reference.without_bias(inputs["x"], inputs["residual"])})
            yield Observation({"out": self.reference.tanh_gelu(inputs["x"], inputs["bias"], inputs["residual"])})
