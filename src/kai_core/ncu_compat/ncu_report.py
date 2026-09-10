"""Expose NVIDIA's installed report API to the optional report-reader subprocess.

Older NCU versions lack timed_warp_samples; nonfinite doubles must become JSON
null. This module adapts the installed API without modifying NVIDIA's files.
"""
from __future__ import annotations

import importlib.util
import glob
import math
import os
from pathlib import Path
import shutil
import sys


def _find_report_dir() -> Path:
    configured = os.environ.get("KAI_CORE_NCU_REPORT_DIR")
    if configured:
        directory = Path(configured).expanduser().resolve()
        if not (directory / "ncu_report.py").is_file():
            raise ImportError("profile.ncu_report_dir does not contain ncu_report.py")
        return directory
    binary = shutil.which(os.environ.get("KAI_CORE_NCU", "ncu"))
    if binary:
        root = Path(binary).resolve().parent
        for parent in (root, root.parent):
            directory = parent / "extras" / "python"
            if (directory / "ncu_report.py").is_file():
                return directory
    # CUDA's bin/ncu can be a launcher script rather than a symlink to NCU.
    patterns = ["/usr/local/cuda*/nsight-compute-*/extras/python",
                "/opt/nvidia/nsight-compute/*/extras/python"]
    if binary:
        patterns.insert(0, str(Path(binary).resolve().parent.parent / "nsight-compute-*" / "extras/python"))
    for pattern in patterns:
        candidates = [Path(path) for path in glob.glob(pattern) if (Path(path) / "ncu_report.py").is_file()]
        if candidates:
            return sorted(candidates)[-1]
    raise ImportError("cannot find NVIDIA ncu_report.py; set profile.ncu_report_dir to its extras/python directory")


_directory = _find_report_dir()
sys.path.append(str(_directory))
_spec = importlib.util.spec_from_file_location("_kai_core_ncu_report", _directory / "ncu_report.py")
if _spec is None or _spec.loader is None:
    raise ImportError("cannot load NVIDIA ncu_report module")
_native = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _native
_spec.loader.exec_module(_native)
globals().update({key: value for key, value in vars(_native).items() if not key.startswith("__")})
if not hasattr(_native.IAction, "timed_warp_samples"):
    _native.IAction.timed_warp_samples = lambda self: []
_original_double = _native.IMetric.as_double


def _finite_double(self, *args):
    value = _original_double(self, *args)
    return None if isinstance(value, float) and not math.isfinite(value) else value


_native.IMetric.as_double = _finite_double
