from __future__ import annotations

from pathlib import Path

import pytest

from kai_pichu.roofline import ROOFLINE_METRICS, roofline_point

UNITS = {"gpu__time_duration.sum": "ns", "dram__bytes.sum": "byte", "sm__cycles_elapsed.avg.per_second": "hz",
         "dram__cycles_elapsed.avg.per_second": "hz"}


def write_capture(path: Path, launches: list[dict[str, float]]) -> Path:
    names = list(ROOFLINE_METRICS)
    lines = ['"ID","Kernel Name",' + ",".join(f'"{name}"' for name in names),
             '"","",' + ",".join(f'"{UNITS.get(name, "")}"' for name in names)]
    for index, launch in enumerate(launches):
        lines.append(f'"{index}","kernel_{index}",' + ",".join(f'"{launch.get(name, 0)}"' for name in names))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_point_sums_the_operator_launches_with_ncu_flop_weights(tmp_path):
    peaks = {"sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained": 128,
             "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained": 64,
             "sm__cycles_elapsed.avg.per_second": 2e9, "dram__bytes.sum.peak_sustained": 32,
             "dram__cycles_elapsed.avg.per_second": 1e10}
    first = {**peaks, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1000,  # 2000 FLOP
             "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 100, "dram__bytes.sum": 400, "gpu__time_duration.sum": 1000}
    second = {**peaks, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 100,  # 400 FLOP, packed half2
              "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum": 50, "dram__bytes.sum": 100, "gpu__time_duration.sum": 500,
              "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 3.5}
    point = roofline_point(write_capture(tmp_path / "metrics.csv", [first, second]))
    assert point["flops_fp32"] == 2100 and point["flops_fp16"] == 500
    assert point["dram_bytes"] == 500 and point["seconds"] == pytest.approx(1.5e-6)
    assert point["intensity"] == pytest.approx(2600 / 500)
    assert point["flops_per_second"] == pytest.approx(2600 / 1.5e-6)
    assert point["roofs"] == {"fp32": pytest.approx(128 * 2 * 2e9), "fp16": pytest.approx(64 * 4 * 2e9), "dram": pytest.approx(32 * 1e10)}
    assert point["launches"] == 2 and point["tensor_pipe_active"]


def test_captures_without_roofline_counters_have_no_point(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text('"ID","Kernel Name","gpu__time_duration.sum","dram__bytes.sum"\n"","","ns","byte"\n"0","k","10","20"\n')
    assert roofline_point(path) is None
    assert roofline_point(tmp_path / "missing.csv") is None
