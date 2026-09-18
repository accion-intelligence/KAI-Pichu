"""The packaged agent skill for authoring benchmark tasks, and its installer.

Claude Code and Codex read the same skill format: a directory holding a
``SKILL.md`` with ``name`` and ``description`` front matter. The skill ships
inside the wheel so its instructions always match the installed SDK.
"""
from __future__ import annotations

from pathlib import Path
import shutil

SKILL_NAME = "kai-benchmark"
AGENT_SKILL_DIRS = {"claude": Path(".claude") / "skills", "codex": Path(".codex") / "skills"}


def skill_source() -> Path:
    return Path(__file__).parent / SKILL_NAME


def skill_destination(agent: str, *, scope: str, root: Path | None = None) -> Path:
    """Where the skill lands: the agent's skills directory under the project root or the home directory."""
    if agent not in AGENT_SKILL_DIRS:
        raise ValueError(f"unknown agent {agent!r}; choose from {sorted(AGENT_SKILL_DIRS)}")
    if scope not in ("project", "user"):
        raise ValueError("scope must be project or user")
    base = Path.home() if scope == "user" else (root or Path.cwd())
    return base.resolve() / AGENT_SKILL_DIRS[agent] / SKILL_NAME


def install_skill(agent: str, *, scope: str, root: Path | None = None, force: bool = False) -> Path:
    """Copy the packaged skill into place; an existing copy is replaced only with force."""
    destination = skill_destination(agent, scope=scope, root=root)
    if destination.exists():
        if not force:
            raise ValueError(f"skill already installed at {destination}; pass --force to replace it")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_source(), destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination
