"""Summarize every attempt from raw run reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def summarize(root: Path) -> dict[str, Any]:
    attempts = []
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for state_file in sorted(root.glob("*/state.json")):
        state = json.loads(state_file.read_text())
        run = state_file.parent
        reports = []
        for path in sorted((run / "reports").glob("*.json")):
            if path.name.endswith(".process.json"):
                continue
            report = json.loads(path.read_text())
            metric = report.get("spec", {}).get("objective", {}).get("metric")
            records = report.get("records", [])
            means = {}
            if metric:
                for arm in ("a", "b"):
                    values = [sample[metric] for row in records if row["arm"] == arm
                              for sample in row["samples"]]
                    if values:
                        means[arm] = statistics.mean(values)
            overall = report.get("comparison", {}).get("overall", {})
            reports.append({
                "path": str(path.relative_to(root)), "status": report["status"],
                "split": report.get("split"), "metric": metric,
                "speedup": overall.get("speedup"), "interval": overall.get("interval"),
                "mean_baseline_ms": means.get("a"), "mean_candidate_ms": means.get("b"),
                "accepted": report.get("acceptance", {}).get("accepted", False),
                "message": report.get("message"),
            })
        usage = {key: sum(row.get(key, 0) for row in state["usage"]) for key in totals}
        for key in totals:
            totals[key] += usage[key]
        attempts.append({"run": run.name, "status": state["status"],
                         "final_accepted": state["status"] == "accepted", "llm_calls": state["llm_calls"],
                         "rounds_completed": len(state["history"]), "elapsed_seconds": state["elapsed_seconds"],
                         "usage": usage, "reports": reports})
    return {"attempts": attempts, "optimizer_usage": totals,
            "note": "Includes every attempt; API connectivity smoke usage is separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.root), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
