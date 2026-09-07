# Contributing

Use Python 3.10+, type hints and `from __future__ import annotations`. Install
with `python -m pip install -r requirements.txt` and run `python -m pytest -q` before
submitting a change. Add a failing regression for bugs involving measurements,
source boundaries, subprocesses or state transitions.

Keep workloads in benchmark adapters/examples and policy in the optimization
loop. Do not hardcode devices, source paths, model endpoints or credentials.
Tests use CPU fixtures, replayed model replies and mocked GPU resources by
default. Label live GPU/model experiments separately and retain failed runs.

Do not submit `.env`, generated candidates from private tasks, runs, model weights,
profiler captures or build products. Record source revisions when adapting code and retain
applicable license notices. Describe behavior, tested commands
and hardware/measurement scope in pull requests.
