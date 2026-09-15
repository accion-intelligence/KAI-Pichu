"""Standalone benchmark commands, without loading an LLM or optimization run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from .models import BenchmarkSpec
from .runner import Runner


def _write_new(path: Path, value: Any, *, text: bool = False) -> None:
    content = value if text else json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL via mode=x protects reports even when concurrent CLI processes
    # both passed the earlier existence check.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kai-core benchmark", description="Benchmark SDK v1")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a task bundle from a working example")
    init.add_argument("directory", type=Path)
    init.add_argument("--template", choices=("stateless", "stateful", "command"), default="stateless")
    schema = commands.add_parser("schema", help="Print the versioned manifest JSON Schema")
    schema.add_argument("--output", type=Path)
    guide = commands.add_parser("guide", help="Print the benchmark authoring manual")
    guide.add_argument("--output", type=Path)
    lint_command = commands.add_parser("lint", help="Static contract checks for a task; runs no workload and needs no GPU")
    lint_command.add_argument("manifest", type=Path)
    lint_command.add_argument("--output", type=Path, help="Optional JSON report path (never overwritten)")
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("manifest", type=Path)
        command.add_argument("--split", choices=("smoke", "search", "acceptance"), default="smoke")
        command.add_argument("--output", type=Path, required=True, help="New JSON report path (never overwritten)")
        if name == "validate":
            command.add_argument("--checks-only", action="store_true", help="Run conformance checks only, without timing the baseline")
        else:
            command.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            target = args.directory.expanduser().resolve()
            if target.exists():
                raise ValueError(f"destination already exists: {target}")
            source = Path(__file__).parent / "templates" / args.template
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            print(f"Created {target / 'benchmark.yaml'}")
            print("Adapt the workload and verifier, then run kai-core benchmark validate.")
            return 0
        if args.output is not None and args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        if args.command == "guide":
            instructions = (Path(__file__).parent / "MANUAL.md").read_text(encoding="utf-8")
            if args.output:
                _write_new(args.output, instructions, text=True)
            else:
                print(instructions)
            return 0
        if args.command == "lint":
            from .lint import lint, render
            findings = lint(args.manifest)
            print(render(findings))
            if args.output:
                _write_new(args.output, [{"level": f.level, "code": f.code, "message": f.message} for f in findings])
            return 1 if any(f.level == "error" for f in findings) else 0
        if args.command == "schema":
            result: dict[str, Any] = BenchmarkSpec.model_json_schema()
            if args.output:
                _write_new(args.output, result)
            else:
                print(json.dumps(result, indent=2))
            return 0
        try:
            runner = Runner(args.manifest, split=args.split)
            result = (runner.validate(checks_only=args.checks_only) if args.command == "validate"
                      else runner.run(args.candidate))
        except Exception as error:
            result = {"schema_version": 1, "status": "error", "error_type": type(error).__name__,
                      "message": str(error), "manifest": str(args.manifest.resolve())}
            for name in ("stdout", "stderr"):
                text = getattr(error, name, None)
                if text:
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="replace")
                    result[name] = str(text)[-8000:]
            _write_new(args.output, result)
            print(f"Benchmark failed: {error}\nReport: {args.output}", file=sys.stderr)
            return 2
        _write_new(args.output, result)
        print(f"Status: {result['status']}\nReport: {args.output}")
        if "comparison" in result:
            overall = result["comparison"]["overall"]
            print(f"Speedup: {overall['speedup']:.4f}x; interval={overall['interval']}")
            print(f"Acceptance: {result['acceptance']['verdict']}")
        # A completed, valid comparison is useful even if the target is unmet.
        return 0 if result["status"] in ("ready", "checks_passed", "completed") else 1
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
