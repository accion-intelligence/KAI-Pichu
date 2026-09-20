"""Adapt the FP16 attention forward task to the KAI Pichu benchmark SDK."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import torch
from torch.utils.cpp_extension import include_paths, library_paths

from kai_pichu.benchmark import (
    Benchmark, Case, Observation, Validation, fixture_fingerprint, load_python_file,
)

WORKSPACE_BYTES = 256 * 1024 * 1024  # Scratch offered to every implementation, contents undefined.


@dataclass
class NativeImplementation:
    library: Any
    launch: Any
    directory: TemporaryDirectory


class Attention(Benchmark):
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
        self.head_dim = int(spec.options["head_dim"])
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.cache_buffer: torch.Tensor | None = None

    # --- task definition -------------------------------------------------------

    def cases(self, split: str) -> Iterable[Case]:
        cases = []
        for profile, seed, batch, heads, seq, causal in self.inputs.SPLITS[split]:
            mask = "_causal" if causal else ""
            cases.append(Case(
                id=f"{profile}_b{batch}h{heads}s{seq}{mask}_seed{seed}",
                params={"profile": profile, "seed_offset": seed, "batch": batch, "heads": heads,
                        "seq": seq, "head_dim": self.head_dim, "causal": causal},
                work_units=float(batch * heads * seq * seq)))
        return cases

    def prepare(self, case: Case, seed: int) -> dict[str, Any]:
        params = case.params
        shape = (params["batch"], params["heads"], params["seq"], params["head_dim"])
        inputs = self.inputs.generate(params["profile"], shape, seed + params["seed_offset"], self.device)
        with torch.cuda.device(self.device), torch.no_grad():
            expected = self.reference.attention_forward(
                inputs["q"], inputs["k"], inputs["v"], scale=self.scale, causal=params["causal"])
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
        return {"inputs": inputs, "causal": params["causal"],
                "outputs": {"o": torch.empty_like(inputs["q"])}, "expected": {"o": expected},
                "workspace": workspace, "graphs": {}, "events": {}}

    def fingerprint(self, fixture: dict[str, Any]) -> str:
        # The fixture also holds poisoned outputs, scratch and captured graphs;
        # the inputs and the mask define the workload identity.
        return fixture_fingerprint({"inputs": fixture["inputs"], "causal": fixture["causal"]})

    # --- implementation lifecycle --------------------------------------------

    def load_implementation(self, workspace: Path) -> NativeImplementation:
        directory = TemporaryDirectory(prefix="kai-attention-")
        library = Path(directory.name) / "kernel.so"
        options = self.spec.options
        cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        command = [str(options["nvcc"]), "-O3", "-std=c++17", "--use_fast_math", "-lineinfo",
                   "--expt-relaxed-constexpr", "-shared", "-Xcompiler=-fPIC",
                   f"-D_GLIBCXX_USE_CXX11_ABI={cxx11_abi}",
                   f"-arch={options['cuda_arch']}", f"-I{workspace}"]
        command += [f"-I{path}" for path in include_paths(device_type="cuda")]
        command += [str(self.root / "bridge.cu"), "-o", str(library)]
        for path in library_paths(device_type="cuda"):
            command += [f"-L{path}", f"-Xlinker=-rpath={path}"]
        command += ["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda", "-lcudart", "-lcublas"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True,
                           timeout=float(options["compile_timeout_seconds"]))
            loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
            launch = loaded.kai_launch
            launch.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 4
                               + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
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
        batch, heads, seq, head_dim = inputs["q"].shape
        workspace = fixture["workspace"]
        status = implementation.launch(
            inputs["q"].data_ptr(), inputs["k"].data_ptr(), inputs["v"].data_ptr(),
            fixture["outputs"]["o"].data_ptr(),
            batch, heads, seq, head_dim, self.scale, int(fixture["causal"]),
            workspace.data_ptr(), workspace.numel(),
            torch.cuda.current_stream(self.device).cuda_stream)
        if status:
            raise RuntimeError(f"native attention launch returned status {status}")

    def reset(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> None:
        graph_mode = self.spec.options["execution"] == "graph_events"
        if graph_mode and id(implementation) not in fixture["graphs"]:
            # Capture exactly ONE operator invocation between two external events.
            # Graph creation is setup work and identical for both arms.
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
        fixture["outputs"]["o"].fill_(float("nan"))
        if self.cache_buffer is None:
            self.cache_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.cache_buffer.fill_(42)

    def run(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> Observation:
        if self.spec.options["execution"] == "graph_events":
            fixture["graphs"][id(implementation)].replay()
            start, end = fixture["events"][id(implementation)]
            end.synchronize()
            return Observation({"o": fixture["outputs"]["o"]},
                               {"operator_latency_ms": start.elapsed_time(end)})
        self._launch(implementation, fixture)
        return Observation({"o": fixture["outputs"]["o"]})

    def synchronize(self, implementation: Any) -> None:
        torch.cuda.synchronize(self.device)

    # --- correctness ----------------------------------------------------------------

    def validate(self, case: Case, fixture: dict[str, Any], observation: Observation) -> Validation:
        got = observation.output
        if not isinstance(got, dict) or set(got) != {"o"}:
            return Validation(False, "required output: o")
        actual, expected = got["o"], fixture["expected"]["o"]
        template = fixture["outputs"]["o"]
        if (not isinstance(actual, torch.Tensor) or actual.shape != template.shape
                or actual.dtype != template.dtype or actual.device != template.device):
            return Validation(False, "o: shape/dtype/device mismatch")
        if not torch.isfinite(actual).all():
            return Validation(False, "o: non-finite values")
        atol, rtol = self.reference.ATOL, self.reference.RTOL
        error = (actual.float() - expected.float()).abs()
        if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
            return Validation(False, f"o: mismatch against the FP16 reference, max abs error {error.max().item():.3e}",
                              {"max_abs_error": error.max().item()})
        return Validation(True, f"o matches the FP16 reference at atol={atol}, rtol={rtol}",
                          {"max_abs_error": error.max().item()})

    def invalid_observations(
        self, case: Case, fixture: dict[str, Any], valid: Observation,
    ) -> Iterable[Observation]:
        good = valid.output["o"]
        perturbed = good.clone()
        perturbed.view(-1)[0] += 1.0
        yield Observation({"o": perturbed})
        poisoned = good.clone()
        poisoned.view(-1)[-1] = float("nan")
        yield Observation({"o": poisoned})
        yield Observation({})
        inputs, causal = fixture["inputs"], fixture["causal"]
        # The opposite mask: the validator must tell causal from full attention.
        with torch.no_grad():
            flipped = self.reference.attention_forward(
                inputs["q"], inputs["k"], inputs["v"], scale=self.scale, causal=not causal)
        yield Observation({"o": flipped})
        # Ignoring Q and K entirely.
        yield Observation({"o": self.reference.uniform_attention(inputs["v"], causal=causal)})
