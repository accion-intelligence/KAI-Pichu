"""Run the CLI with one explicitly selected dotenv key and CUDA device."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex

from kai_pichu.cli import main


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--key-env", default="OPENAI_API_KEY")
    parser.add_argument("--gpu", required=True, help="Physical GPU UUID or index")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    found = False
    for line in args.env_file.read_text().splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == args.key_env:
            parts = shlex.split(value, comments=True)
            if len(parts) != 1 or not parts[0]:
                raise ValueError("selected dotenv key must have one nonempty value")
            os.environ[args.key_env] = parts[0]
            found = True
    if not found:
        raise ValueError(f"selected key {args.key_env} is absent from dotenv file")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    return main(forwarded)


if __name__ == "__main__":
    raise SystemExit(run())
