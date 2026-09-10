from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import OptimizeConfig, example_config, load_config, require_api_keys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        from .benchmark.cli import main as benchmark_main
        return benchmark_main(argv[1:])
    parser = argparse.ArgumentParser(prog="kai-core", description="Single-operator GPU optimization with the Benchmark SDK and optimization loop.")
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser("optimize", help="Optimize a frozen benchmark task")
    optimize.add_argument("manifest", type=Path)
    optimize.add_argument("--config", type=Path, required=True)
    optimize.add_argument("--output", type=Path, required=True)
    optimize.add_argument("--dry-run", action="store_true", help="Freeze source and inspect prompts without GPU or model calls")
    optimize.add_argument("--resume", action="store_true", help="Resume at the next reserved round with the same config")
    commands.add_parser("benchmark", help="Benchmark SDK commands; use benchmark --help")
    config_command = commands.add_parser("config", help="Export an optimizer config template or JSON Schema")
    config_command.add_argument("--output", required=True, type=Path)
    config_command.add_argument("--schema", action="store_true")
    profile = commands.add_parser("profile", help="Query an existing NCU report without running a GPU workload")
    profile.add_argument("report", type=Path, nargs="?")
    profile.add_argument("--guide", action="store_true", help="Print the packaged AI profiling guide")
    profile.add_argument("--dependencies", action="store_true", help="Print optional report reader setup")
    profile.add_argument("--schema", action="store_true", help="Print the profile query JSON Schema")
    profile.add_argument("--csv", type=Path, help="Optional raw NCU CSV fallback")
    profile.add_argument("--request", help='Query JSON, e.g. {"operation":"catalog","query":"memory"}; default: overview')
    profile.add_argument("--report-reader", help="Optional NCU report reader executable")
    profile.add_argument("--ncu", default="ncu")
    profile.add_argument("--ncu-report-dir")
    profile.add_argument("--timeout", type=float, default=60)
    profile.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            from .config import ProfileConfig
            from .io import parse_object, write_json
            from .profiling import ProfileQuery, ProfileReport
            if args.dependencies:
                print(Path(__file__).with_name("PROFILE_DEPENDENCIES.md").read_text())
                return 0
            if args.guide:
                print(Path(__file__).with_name("PROFILE_GUIDE.md").read_text())
                return 0
            if args.schema:
                print(json.dumps(ProfileQuery.model_json_schema(), indent=2))
                return 0
            if args.report is None:
                raise ValueError("profile requires a report path, --guide or --schema")
            reader = ProfileReport(args.report, csv_path=args.csv, settings=ProfileConfig(
                report_reader=args.report_reader, ncu=args.ncu, ncu_report_dir=args.ncu_report_dir, query_seconds=args.timeout))
            result = (reader.query(parse_object(args.request), timeout=args.timeout) if args.request
                      else reader.overview(timeout=args.timeout))
            if args.output:
                write_json(args.output, result)
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "available" else 1
        if args.command == "config":
            import yaml
            from .io import write_json
            if args.schema:
                write_json(args.output, OptimizeConfig.model_json_schema())
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as handle:
                    handle.write(yaml.safe_dump(example_config(), sort_keys=False))
            print(f"Wrote {args.output}")
            return 0
        from .optimizer.engine import OptimizationLoop
        from .workspace import Workspace

        config = load_config(args.config.resolve(strict=True))
        if not args.dry_run:
            require_api_keys(config)  # Fail before the run directory and GPU preflight exist.
        root = args.output.expanduser().resolve()
        workspace = Workspace(root) if args.resume else Workspace.create(args.manifest, root)
        result = OptimizationLoop(workspace, config).run(resume=args.resume, dry_run=args.dry_run)
        print(json.dumps({"status": result["status"], "final_accepted": result["final_accepted"],
                          "summary": str(root / "summary.json")}, indent=2))
        return 0 if result["status"] in ("planned", "accepted", "no_improvement", "not_accepted") else 1
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
