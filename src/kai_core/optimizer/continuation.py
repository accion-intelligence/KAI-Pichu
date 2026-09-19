"""Splice a cut-off model reply with the continuation the model wrote for it.

Models asked to continue usually resume exactly, sometimes repeat the last
line or two, and occasionally start the whole reply over. The first case is a
plain concatenation; the others are recognized by overlap and handled without
help from the model.
"""
from __future__ import annotations

# Shorter matches are too likely to be coincidence (a closing brace, a newline run).
MIN_OVERLAP = 8


def splice(partial: str, continuation: str) -> str:
    """Join two reply fragments, removing text the continuation repeated."""
    continuation = _strip_fence(continuation)
    if not partial:
        return continuation
    if continuation.startswith(partial):
        return continuation  # the model started over and wrote everything
    stripped = partial.rstrip()
    overlap = _longest_overlap(stripped, continuation)
    if overlap >= MIN_OVERLAP and continuation[:overlap].strip():
        return stripped + continuation[overlap:]
    return partial + continuation


def _longest_overlap(head: str, tail: str) -> int:
    """Length of the longest suffix of head that is a prefix of tail."""
    longest = min(len(head), len(tail))
    for size in range(longest, 0, -1):
        if head.endswith(tail[:size]):
            return size
    return 0


def _strip_fence(text: str) -> str:
    """A continuation wrapped in a markdown fence despite instructions loses the fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines[1:])
    return text
