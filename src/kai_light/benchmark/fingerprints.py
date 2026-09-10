"""Content digests for prepared fixtures.

The SDK compares fixtures only through these digests: preparation must be
reproducible and reset must restore the inputs. Adapters get this behavior
without writing hashing code, as long as the fixture consists of values the
digest understands.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

FINGERPRINT_HOOK = "__fixture_fingerprint__"


class UnsupportedFixtureValue(TypeError):
    """A fixture holds an object the default digest cannot describe."""


def fixture_fingerprint(value: Any) -> str:
    """SHA-256 over the content, shape and layout of a fixture.

    Supported: None, bool, int, float, str, bytes-like, pathlib.Path (digested
    by file content), dict/list/tuple/set (recursively), NumPy arrays and
    PyTorch tensors (dtype, shape, strides and element bytes; device is not
    part of the identity). Any other object may provide ``__fixture_fingerprint__``
    returning a string. Everything else raises UnsupportedFixtureValue naming
    the offending location, so the adapter author can exclude scratch buffers
    or override ``Benchmark.fingerprint``.
    """
    digest = hashlib.sha256()
    _feed(digest, value, "fixture")
    return digest.hexdigest()


def _feed(digest: "hashlib._Hash", value: Any, location: str) -> None:
    # Every node is tagged with its type so that, for example, 1, 1.0, True and
    # "1" never collide and container boundaries stay unambiguous.
    if value is None or isinstance(value, (bool, int, float, str)):
        _tag(digest, type(value).__name__, repr(value))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        _tag(digest, "bytes", len(value)); digest.update(bytes(value))
    elif isinstance(value, Path):
        _feed_file(digest, value, location)
    elif isinstance(value, dict):
        _tag(digest, "dict", len(value))
        for key in sorted(value, key=repr):
            _feed(digest, key, f"{location} key"); _feed(digest, value[key], f"{location}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        _tag(digest, type(value).__name__, len(value))
        for index, item in enumerate(value):
            _feed(digest, item, f"{location}[{index}]")
    elif isinstance(value, (set, frozenset)):
        _tag(digest, "set", len(value))
        for item in sorted(fixture_fingerprint(item) for item in value):
            _tag(digest, "item", item)
    elif hasattr(value, FINGERPRINT_HOOK):
        _tag(digest, "custom", str(getattr(value, FINGERPRINT_HOOK)()))
    elif _module_of(value).startswith("torch"):
        _feed_tensor(digest, value)
    elif _module_of(value).startswith("numpy"):
        _feed_array(digest, value)
    else:
        raise UnsupportedFixtureValue(
            f"{location} holds {type(value).__module__}.{type(value).__qualname__}, which the default "
            "fingerprint cannot digest; keep scratch/output objects out of the fixture, give the object a "
            f"{FINGERPRINT_HOOK}() method, or override Benchmark.fingerprint")


def _tag(digest: "hashlib._Hash", kind: str, detail: Any) -> None:
    digest.update(f"{kind}:{detail}\x00".encode("utf-8", errors="surrogatepass"))


def _module_of(value: Any) -> str:
    return type(value).__module__ or ""


def _feed_file(digest: "hashlib._Hash", path: Path, location: str) -> None:
    if not path.is_file():
        raise UnsupportedFixtureValue(f"{location} refers to {path}, which is not a readable file")
    _tag(digest, "file", path.stat().st_size)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)


def _feed_tensor(digest: "hashlib._Hash", tensor: Any) -> None:
    detached = tensor.detach()
    _tag(digest, "tensor", (str(detached.dtype), tuple(detached.shape), tuple(detached.stride())))
    # Hash the raw element bytes of a contiguous host copy through a ctypes view.
    # This needs neither NumPy (absent from CPU-only torch installs) nor a
    # per-element copy, and it covers every dtype, bfloat16 included.
    host = detached.cpu().contiguous()
    nbytes = host.numel() * host.element_size()
    if nbytes:
        digest.update(memoryview((ctypes.c_ubyte * nbytes).from_address(host.data_ptr())))


def _feed_array(digest: "hashlib._Hash", array: Any) -> None:
    _tag(digest, "ndarray", (str(array.dtype), tuple(array.shape), tuple(array.strides)))
    digest.update(array.tobytes())
