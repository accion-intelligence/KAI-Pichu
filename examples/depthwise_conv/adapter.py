"""Adapt the depthwise 7x7 convolution task to the KAI Core benchmark SDK."""
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

WORKSPACE_BYTES = 256 * 1024 * 1024  # Scratch for cuDNN algorithms or any candidate; contents undefined.


@dataclass
class NativeImplementation:
    library: Any
    launch: Any
    directory: TemporaryDirectory


def cudnn_dirs(options: dict[str, Any]) -> tuple[Path, Path, str]:
    """cuDNN include directory, library directory and library file name.

    Uses options.cudnn_home when set, otherwise the cuDNN wheel PyTorch depends
    on. That wheel ships only versioned files such as libcudnn.so.9, so the
    linker is given the exact file name instead of -lcudnn.
    """
    if options.get("cudnn_home"):
        home = Path(options["cudnn_home"])
        roots = [home]
    else:
        roots = [parent / "nvidia" / "cudnn"
                 for site in {Path(p) for p in include_paths(device_type="cuda")} for parent in site.parents]
    for root in roots:
        libraries = sorted((root / "lib").glob("libcudnn.so*")) + sorted((root / "lib64").glob("libcudnn.so*"))
        if (root / "include" / "cudnn.h").exists() and libraries:
            return root / "include", libraries[0].parent, libraries[0].name
    raise FileNotFoundError("cuDNN headers and library not found; set options.cudnn_home to a cuDNN installation")


class DepthwiseConv(Benchmark):
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
        return [Case(id=f"{profile}_n{n}c{c}_{h}x{w}_seed{seed}",
                     params={"profile": profile, "seed_offset": seed, "n": n, "c": c, "h": h, "w": w},
                     work_units=float(n * c * h * w))
                for profile, seed, n, c, h, w in self.inputs.SPLITS[split]]

    def prepare(self, case: Case, seed: int) -> dict[str, Any]:
        params = case.params
        shape = (params["n"], params["c"], params["h"], params["w"])
        inputs = self.inputs.generate(params["profile"], shape, seed + params["seed_offset"], self.device)
        with torch.cuda.device(self.device), torch.no_grad():
            expected = self.reference.depthwise_conv(inputs["x"], inputs["weight"])
        output = torch.empty_like(inputs["x"])
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)
        return {"inputs": inputs, "outputs": {"y": output}, "expected": {"y": expected},
                "workspace": workspace, "graphs": {}, "events": {}}

    def fingerprint(self, fixture: dict[str, Any]) -> str:
        # The fixture also holds poisoned outputs, scratch and captured graphs;
        # only the inputs define the workload identity.
        return fixture_fingerprint(fixture["inputs"])

    # --- implementation lifecycle --------------------------------------------

    def load_implementation(self, workspace: Path) -> NativeImplementation:
        directory = TemporaryDirectory(prefix="kai-depthwise-")
        library = Path(directory.name) / "kernel.so"
        options = self.spec.options
        cudnn_include, cudnn_lib, cudnn_library = cudnn_dirs(options)
        command = [str(options["nvcc"]), "-O3", "-std=c++17", "--use_fast_math",
                   "--expt-relaxed-constexpr", "-shared", "-Xcompiler=-fPIC",
                   f"-arch={options['cuda_arch']}", f"-I{workspace}", f"-I{cudnn_include}"]
        command += [f"-I{path}" for path in include_paths(device_type="cuda")]
        command += [str(self.root / "bridge.cu"), "-o", str(library)]
        for path in [*library_paths(device_type="cuda"), str(cudnn_lib)]:
            command += [f"-L{path}", f"-Xlinker=-rpath={path}"]
        command += [f"-l:{cudnn_library}", "-lcudart"]
        try:
            build = subprocess.run(command, capture_output=True, text=True,
                                   timeout=float(options["compile_timeout_seconds"]))
            if build.returncode != 0:
                raise RuntimeError(f"nvcc failed with status {build.returncode}:\n{build.stderr[-4000:]}")
            loaded = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
            launch = loaded.kai_launch
            launch.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
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
        x, weight = fixture["inputs"]["x"], fixture["inputs"]["weight"]
        n, c, h, w = x.shape
        workspace = fixture["workspace"]
        status = implementation.launch(
            x.data_ptr(), weight.data_ptr(), fixture["outputs"]["y"].data_ptr(), n, c, h, w,
            workspace.data_ptr(), workspace.numel(),
            torch.cuda.current_stream(self.device).cuda_stream)
        if status:
            raise RuntimeError(f"native depthwise convolution launch returned status {status}")

    def reset(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> None:
        graph_mode = self.spec.options["execution"] == "graph_events"
        if graph_mode and id(implementation) not in fixture["graphs"]:
            # Eager warm-up first: cuDNN's algorithm search must run outside capture.
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
        fixture["outputs"]["y"].fill_(float("nan"))
        if self.cache_buffer is None:
            self.cache_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.cache_buffer.fill_(42)

    def run(self, implementation: NativeImplementation, fixture: dict[str, Any]) -> Observation:
        if self.spec.options["execution"] == "graph_events":
            fixture["graphs"][id(implementation)].replay()
            start, end = fixture["events"][id(implementation)]
            end.synchronize()
            return Observation({"y": fixture["outputs"]["y"]},
                               {"operator_latency_ms": start.elapsed_time(end)})
        self._launch(implementation, fixture)
        return Observation({"y": fixture["outputs"]["y"]})

    def synchronize(self, implementation: Any) -> None:
        torch.cuda.synchronize(self.device)

    # --- correctness ----------------------------------------------------------------

    def validate(self, case: Case, fixture: dict[str, Any], observation: Observation) -> Validation:
        got = observation.output
        if not isinstance(got, dict) or set(got) != {"y"}:
            return Validation(False, "required output: y")
        actual, expected = got["y"], fixture["expected"]["y"]
        if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or actual.dtype != expected.dtype or actual.device != expected.device):
            return Validation(False, "y: shape/dtype/device mismatch")
        if not torch.isfinite(actual).all():
            return Validation(False, "y: non-finite values")
        atol, rtol = self.reference.ATOL, self.reference.RTOL
        error = (actual.float() - expected.float()).abs().max().item()
        if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
            return Validation(False, f"y: mismatch against the PyTorch reference, max abs error {error:.3e}",
                              {"max_abs_error": error})
        return Validation(True, f"y matches the PyTorch reference at atol={atol}, rtol={rtol}",
                          {"max_abs_error": error})

    def invalid_observations(
        self, case: Case, fixture: dict[str, Any], valid: Observation,
    ) -> Iterable[Observation]:
        good = valid.output["y"]
        perturbed = good.clone()
        perturbed[0, 0, 0, 0] += 1.0
        yield Observation({"y": perturbed})
        poisoned = good.clone()
        poisoned[-1, -1, -1, -1] = float("nan")
        yield Observation({"y": poisoned})
        yield Observation({})
        x, weight = fixture["inputs"]["x"], fixture["inputs"]["weight"]
        with torch.no_grad():
            # Wrong border handling and a flipped kernel are the classic depthwise mistakes.
            yield Observation({"y": self.reference.without_padding_rows(x, weight)})
            yield Observation({"y": self.reference.flipped_kernel(x, weight)})
