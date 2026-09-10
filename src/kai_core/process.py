from __future__ import annotations

import csv
from contextlib import ExitStack
import io
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from .config import Resources


def _query(fields: str, *, apps: bool = False) -> list[list[str]]:
    flag = "--query-compute-apps=" if apps else "--query-gpu="
    result = subprocess.run(["nvidia-smi", flag + fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(result.stdout)) if row]


class GpuGuard:
    """Polling detects foreign workloads; it is not an OS-enforced GPU lease."""
    def __init__(self, logical_device: int):
        devices = _query("index,uuid")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        identifier = (visible.split(",")[logical_device].strip() if visible is not None
                      else str(logical_device))
        matches = [uuid for index, uuid in devices if identifier == index or uuid.startswith(identifier)]
        if len(matches) != 1:
            raise ValueError("cannot map logical CUDA device to a unique NVIDIA GPU UUID")
        self.uuid = matches[0]

    def inspect(self, group: int | None = None) -> dict[str, Any]:
        foreign = []
        own = []
        for uuid, pid_text in _query("gpu_uuid,pid", apps=True):
            if uuid != self.uuid:
                continue
            pid = int(pid_text)
            try:
                ours = group is not None and os.getpgid(pid) == group
            except ProcessLookupError:
                continue
            except PermissionError:
                ours = False
            (own if ours else foreign).append(pid)
        return {"gpu_uuid": self.uuid, "foreign_pids": foreign, "own_pids": own,
                "monotonic_seconds": time.monotonic()}


def _terminate(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The leader may exit while descendants ignore SIGTERM. Reap the whole
        # group, not only the process we can wait() on.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_process(
    argv: list[str], *, cwd: Path, log: Path, timeout: float,
    resources: Resources, gpu_device: int | None,
    env: dict[str, str] | None = None, stdout: Path | None = None,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    guard = None
    try:
        if gpu_device is not None and resources.gpu_policy == "exclusive":
            guard = GpuGuard(gpu_device)
            samples.append(guard.inspect())
            if samples[-1]["foreign_pids"]:
                return {"status": "resource_busy", "resource_samples": samples}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        return {"status": "resource_error", "message": str(error), "resource_samples": samples}
    started = time.monotonic()
    status = "completed"
    with ExitStack() as stack:
        output = stack.enter_context(log.open("xb"))
        standard_output = stack.enter_context(stdout.open("xb")) if stdout is not None else output
        process = subprocess.Popen(argv, cwd=cwd, stdout=standard_output, stderr=output,
                                   env=env, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    status = "timeout"
                    _terminate(process)
                    break
                if guard is not None:
                    try:
                        samples.append(guard.inspect(process.pid))
                        if samples[-1]["foreign_pids"]:
                            status = "resource_busy"
                            _terminate(process)
                            break
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        status = "resource_error"
                        samples.append({"error": str(error)})
                        _terminate(process)
                        break
                try:
                    process.wait(timeout=min(resources.poll_seconds, max(0.01, timeout - (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    pass
            # Final sampling catches a workload appearing just before exit.
            if guard is not None and status == "completed":
                try:
                    samples.append(guard.inspect(process.pid))
                    if samples[-1]["foreign_pids"]:
                        status = "resource_busy"
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    status = "resource_error"
                    samples.append({"error": str(error)})
        except BaseException:
            _terminate(process)
            raise
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 12000))
        tail = handle.read().decode("utf-8", errors="replace")
    return {"status": status, "returncode": process.returncode, "log_tail": tail,
            "resource_samples": samples, "gpu_policy": resources.gpu_policy,
            "elapsed_seconds": time.monotonic() - started}
