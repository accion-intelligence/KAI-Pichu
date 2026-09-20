"""Give an existing NCU v1 reader the product command name without replacing it."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import shutil


def install_alias(executable: str, output: Path, *, python_dir_env: str | None = None) -> Path:
    """Install an argv-preserving launcher; optionally map the reader's env name."""
    found = shutil.which(executable)
    if found is None:
        raise ValueError("existing report reader executable was not found")
    target = Path(found).resolve()
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"alias path already exists: {output}")
    if python_dir_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", python_dir_env):
        raise ValueError("reader environment variable must be a valid identifier")
    lines = ["#!/bin/sh"]
    if python_dir_env:
        lines.extend(['if [ "${KAI_PICHU_REPORT_READER_DIR+x}" = x ]; then',
                      f'    export {python_dir_env}="$KAI_PICHU_REPORT_READER_DIR"', "fi"])
    lines.append(f'exec {shlex.quote(str(target))} "$@"')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    output.chmod(0o755)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", help="Existing NCU v1 report reader; no software is downloaded or replaced")
    parser.add_argument("--output", type=Path, default=Path.home() / ".local/bin/kai-ncu-reader")
    parser.add_argument("--python-dir-env", help="Existing reader's environment variable for its report API directory")
    args = parser.parse_args()
    try:
        print(install_alias(args.executable, args.output, python_dir_env=args.python_dir_env))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
