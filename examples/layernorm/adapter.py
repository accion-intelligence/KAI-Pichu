"""Adapt the LayerNorm task using public KAI-light SDK hooks."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import torch
from torch.utils.cpp_extension import include_paths, library_paths

from kai_light.benchmark import (
    Benchmark, Case, Observation, Validation, json_fingerprint, load_python_file,
)


@dataclass
class NativeImplementation:
    library: Any
    launch: Any
    directory: TemporaryDirectory


class LayerNorm(Benchmark):
    def __init__(self, spec: Any):
        super().__init__(spec)
        self.root = Path(__file__).resolve().parent
        self.definition = load_python_file(self.root / "upstream/def.py")
        self.device = torch.device(spec.options["device"])
        if self.device.type != "cuda" or self.device.index != spec.measurement.device:
            raise ValueError("options.device must match measurement.device")
        if spec.options["execution"] not in ("eager", "graph", "graph_events"):
            raise ValueError("execution must be eager, graph or graph_events")
        self.cache_buffer: torch.Tensor | None = None

    def cases(self, split: str) -> Iterable[Case]:
        seeds = [0] if split in ("smoke", "search") else [1001, 1002]
        return [Case(id=f"original_seed{seed}", params={"seed_offset": seed,
                     "rows": self.definition.ROWS, "cols": self.definition.COLS},
                     work_units=self.definition.ROWS) for seed in seeds]

    def prepare(self, case: Case, seed: int) -> dict[str, Any]:
        # Use the upstream generator and oracle verbatim, with explicit seeding.
        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed + case.params["seed_offset"])
            inputs = self.definition.get_inputs()
            with torch.no_grad():
                expected = dict(self.definition.reference_fn(inputs))
            outputs = dict(self.definition.get_outputs(inputs))
        return {"inputs": dict(inputs), "outputs": outputs, "expected": expected,
                "graphs": {}, "events": {}}

    def fingerprint(self, fixture: dict[str, Any]) -> str:
        # Hash actual device contents, not just seeds or a cached initial digest.
        digests = {}
        for name, tensor in fixture["inputs"].items():
            data = tensor.detach().cpu().contiguous().numpy()
            digests[name] = {"sha256": hashlib.sha256(memoryview(data)).hexdigest(),
                             "shape": list(tensor.shape), "stride": list(tensor.stride()),
                             "dtype": str(tensor.dtype)}
        return json_fingerprint(digests)

    def load_implementation(self, workspace: Path) -> NativeImplementation:
        directory = TemporaryDirectory(prefix="kai-layernorm-")
        library = Path(directory.name) / "kernel.so"
        command = [str(self.spec.options["nvcc"]), "-O3", "-std=c++17", "--use_fast_math",
                   "--expt-relaxed-constexpr", "-shared", "-Xcompiler=-fPIC",
                   f"-arch={self.spec.options['cuda_arch']}",
                   f"-I{workspace}", f"-I{self.root / 'upstream/layer_norm'}"]
        command += [f"-I{path}" for path in include_paths(device_type="cuda")]
        command += [str(self.root / "bridge.cu"), "-o", str(library)]
        for path in library_paths(device_type="cuda"):
            command += [f"-L{path}", f"-Xlinker=-rpath={path}"]
        command += ["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True,
                           timeout=float(self.spec.options["compile_timeout_seconds"]))
            loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
            launch = loaded.kai_launch
            launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_float, ctypes.c_void_p]
            launch.restype = ctypes.c_int
            return NativeImplementation(loaded, launch, directory)
        except BaseException:
            directory.cleanup()
            raise

    def _launch(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> None:
        inputs, outputs = fixture["inputs"], fixture["outputs"]
        pointers = [inputs[name].data_ptr() for name in ("x", "gamma", "beta")]
        pointers += [outputs[name].data_ptr() for name in ("z", "mu", "rs")]
        error = implementation.launch(*pointers, self.definition.ROWS, self.definition.COLS,
                                      self.definition.EPS,
                                      torch.cuda.current_stream(self.device).cuda_stream)
        if error:
            raise RuntimeError(f"native CUDA launch returned error {error}")

    def reset(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> None:
        if self.spec.options["execution"] in ("graph", "graph_events") and id(implementation) not in fixture["graphs"]:
            # Capture exactly ONE operation for both arms; no repeat averaging or
            # candidate-specific preprocessing. Graph creation is setup work.
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._launch(implementation, fixture)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            events = None
            if self.spec.options["execution"] == "graph_events":
                events = (torch.cuda.Event(enable_timing=True, external=True),
                          torch.cuda.Event(enable_timing=True, external=True))
                fixture["events"][id(implementation)] = events
            with torch.cuda.graph(graph, stream=stream):
                if events:
                    events[0].record()
                self._launch(implementation, fixture)
                if events:
                    events[1].record()
            fixture["graphs"][id(implementation)] = graph
        for output in fixture["outputs"].values():
            output.fill_(float("nan"))
        if self.cache_buffer is None:
            self.cache_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                                            device=self.device)
        self.cache_buffer.fill_(42)

    def run(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> Observation:
        if self.spec.options["execution"] in ("graph", "graph_events"):
            fixture["graphs"][id(implementation)].replay()
            if self.spec.options["execution"] == "graph_events":
                start, end = fixture["events"][id(implementation)]
                end.synchronize()
                return Observation(dict(fixture["outputs"]),
                                   {"operator_latency_ms": start.elapsed_time(end)})
        else:
            self._launch(implementation, fixture)
        return Observation(dict(fixture["outputs"]))

    def synchronize(self, implementation: Any) -> None:
        torch.cuda.synchronize(self.device)

    def validate(self, case: Case, fixture: dict[str, Any], observation: Observation) -> Validation:
        got = observation.output
        want = fixture["expected"]
        if not isinstance(got, dict) or set(got) != set(want):
            return Validation(False, "required outputs: z, mu, rs")
        for name, expected in want.items():
            actual = got[name]
            if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                    or actual.dtype != expected.dtype or actual.device != expected.device):
                return Validation(False, f"{name}: shape/dtype/device mismatch")
            if not torch.allclose(actual, expected, **self.definition.TOLERANCES):
                return Validation(False, f"{name}: mismatch against upstream PyTorch oracle")
        return Validation(True, "all three outputs match upstream oracle at atol=rtol=1e-4")

    def invalid_observations(
        self, case: Case, fixture: dict[str, Any], valid: Observation,
    ) -> Iterable[Observation]:
        for name in ("z", "mu", "rs"):
            changed = dict(valid.output)
            changed[name] = changed[name].clone()
            changed[name].view(-1)[0] += 1
            yield Observation(changed)
        yield Observation({"z": valid.output["z"]})
        changed = dict(valid.output)
        changed["z"] = changed["z"].clone()
        changed["z"].view(-1)[0] = float("nan")
        yield Observation(changed)

    def cleanup_fixture(self, fixture: dict[str, Any]) -> None:
        fixture.clear()

    def cleanup_implementation(self, implementation: NativeImplementation) -> None:
        implementation.directory.cleanup()
