"""Roofline points from NCU captures.

The metric names and FLOP weights follow Nsight Compute's own roofline
sections (SpeedOfLight_RooflineChart, SpeedOfLight_HierarchicalHalfRooflineChart):
an FFMA thread instruction is 2 FLOP, FADD and FMUL 1, a packed HFMA2 4 and
HADD2/HMUL2 2. Every capture collects these in addition to its sections, so a
point is the whole profiled operator: work, DRAM traffic and GPU time summed
over its kernel launches.

Tensor-core work has no FLOP counter here; when the tensor pipe is active the
point understates the operator's work and says so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .profile_csv import read_csv

ROOFLINE_METRICS = (
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained",
    "sm__cycles_elapsed.avg.per_second",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "dram__bytes.sum",
    "dram__bytes.sum.peak_sustained",
    "dram__cycles_elapsed.avg.per_second",
    "gpu__time_duration.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
)

FP32 = {"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2,
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1,
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1}
FP16 = {"smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 4,
        "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum": 2,
        "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 2}
TIME_UNITS = {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6, "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}
BYTE_UNITS = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9}


def _number(metric: dict[str, Any]) -> float | None:
    value = metric.get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def roofline_point(csv_path: Path) -> dict[str, Any] | None:
    """The operator's point and the device's roofs from one capture; None without roofline metrics."""
    try:
        launches = read_csv(csv_path)
    except (OSError, ValueError):
        return None
    flops = {"fp32": 0.0, "fp16": 0.0}
    dram_bytes = seconds = 0.0
    roofs: dict[str, float] = {}
    tensor = False
    seen = False
    for launch in launches:
        metrics = {m["name"]: m for m in launch["metrics"]}
        time, traffic = metrics.get("gpu__time_duration.sum"), metrics.get("dram__bytes.sum")
        if not time or not traffic or _number(time) is None or _number(traffic) is None:
            continue
        if not any(name in metrics for name in FP32):
            continue  # a capture taken before roofline metrics were collected
        seen = True
        seconds += _number(time) * TIME_UNITS.get((time.get("unit") or "ns").lower(), 1e-9)
        dram_bytes += _number(traffic) * BYTE_UNITS.get((traffic.get("unit") or "byte").lower(), 1)
        for kind, weights in (("fp32", FP32), ("fp16", FP16)):
            flops[kind] += sum(weight * (_number(metrics[name]) or 0.0) for name, weight in weights.items() if name in metrics)
        tensor = tensor or (_number(metrics.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", {})) or 0) > 0
        sm_hz = _number(metrics.get("sm__cycles_elapsed.avg.per_second", {}))
        dram_hz = _number(metrics.get("dram__cycles_elapsed.avg.per_second", {}))
        peaks = {"fp32": ("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained", 2, sm_hz),
                 "fp16": ("sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained", 4, sm_hz),
                 "dram": ("dram__bytes.sum.peak_sustained", 1, dram_hz)}
        for kind, (name, weight, hz) in peaks.items():
            per_cycle = _number(metrics.get(name, {}))
            if per_cycle and hz:
                roofs[kind] = max(roofs.get(kind, 0.0), per_cycle * weight * hz)
    if not seen or seconds <= 0:
        return None
    work = flops["fp32"] + flops["fp16"]
    return {"flops": work, "flops_fp32": flops["fp32"], "flops_fp16": flops["fp16"], "dram_bytes": dram_bytes,
            "seconds": seconds, "launches": len(launches), "tensor_pipe_active": tensor,
            "intensity": work / dram_bytes if dram_bytes > 0 else None,
            "flops_per_second": work / seconds, "roofs": roofs}
