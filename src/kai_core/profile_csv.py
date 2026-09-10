"""NCU CSV fallback: preserve launch identity, units and missing values."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import re
from typing import Any


def metric_value(text: str) -> Any:
    value = text.strip()
    if value.lower() in {"", "nan", "inf", "-inf", "infinity", "n/a", "no data", "none"}:
        return None
    try:
        clean = value.replace(",", "")
        if re.fullmatch(r"[+-]?\d+", clean):
            return int(clean)
        numeric = float(clean)
    except ValueError:
        return value
    return numeric if math.isfinite(numeric) else None


def read_csv(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, line in enumerate(lines) if "Kernel Name" in line and "," in line), None)
    if start is None:
        raise ValueError("NCU CSV has no Kernel Name header")
    records = list(csv.reader(io.StringIO("\n".join(lines[start:]))))
    header, rows = records[0], records[1:]
    kernel_index = header.index("Kernel Name")
    long_format = "Metric Name" in header and "Metric Value" in header
    metric_indices = (list(range(header.index("CC") + 1, len(header))) if "CC" in header
                      else [i for i, name in enumerate(header) if "__" in name])
    if not long_format and not metric_indices:
        raise ValueError("NCU CSV has no metric columns")
    units = [""] * len(header)
    if not long_format and rows and len(rows[0]) == len(header) and not rows[0][kernel_index].strip():
        units = rows.pop(0)
    launches: dict[str, dict[str, Any]] = {}
    for ordinal, cells in enumerate(rows):
        if not cells or (len(cells) == 1 and cells[0].startswith("==PROF==")):
            continue
        if len(cells) != len(header):
            raise ValueError("NCU CSV contains an incomplete row")
        if not cells[kernel_index]:
            continue
        record = dict(zip(header, cells))
        # NCU IDs identify launches; never merge distinct launches by kernel name.
        identity = record.get("ID", str(ordinal) if not long_format else cells[kernel_index])
        if identity not in launches:
            row_id = f"launch:{len(launches)}"
            launches[identity] = {"key": row_id, "row_id": row_id, "ncu_id": identity,
                                  "kernel_demangled": cells[kernel_index], "metrics": [], "rules": []}
        launch = launches[identity]
        entries = ([(record["Metric Name"], record.get("Metric Unit", ""), record["Metric Value"])]
                   if long_format else [(header[i], units[i], cells[i]) for i in metric_indices])
        for name, unit, raw in entries:
            if long_format and "ID" not in header and any(item["name"] == name for item in launch["metrics"]):
                raise ValueError("NCU long CSV without ID has ambiguous repeated launches")
            launch["metrics"].append({"name": name, "label": None, "unit": unit or None,
                                       "value": metric_value(raw)})
    if not launches:
        raise ValueError("NCU CSV has no launch data")
    return list(launches.values())
