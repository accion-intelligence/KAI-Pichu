from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
        return
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def parse_object(text: str) -> dict[str, Any]:
    """Require one complete JSON object; tolerate a single markdown fence."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0] not in ("```", "```json") or lines[-1] != "```":
            raise ValueError("incomplete or unsupported JSON fence")
        text = "\n".join(lines[1:-1])

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    value = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    return value
