# NCU report reader

`kai-ncu-reader` ships with this package and is the reader the profile layer
uses by default. It opens the `.ncu-rep` file Nsight Compute wrote and answers
one question per call: the launches in the report, every metric and NVIDIA rule
of one launch, per-instruction counters, warp-stall samples, or the SASS/PTX of
the launch. Its query design follows [VeloQ](https://github.com/lucifer1004/veloq).

```bash
kai-ncu-reader ncu launches REPORT.ncu-rep
kai-ncu-reader ncu inspect REPORT.ncu-rep --row-id launch:0
kai-ncu-reader ncu warp-stalls REPORT.ncu-rep --row-id launch:0 --by reason
```

## Requirements

Reading a report needs the `ncu_report` Python module that Nsight Compute
installs under `extras/python`. The reader finds it from the `ncu` on PATH,
under `/usr/local/cuda*/nsight-compute-*` or `/opt/nvidia/nsight-compute/*`, or
from `profile.ncu_report_dir` (CLI `--ncu-report-dir`) when the installation is
elsewhere. Nothing is installed automatically.

Per-instruction evidence exists only if the capture collected it: the default
capture adds the `SourceCounters` section and `--import-source yes`, and source
lines appear when the kernel was compiled with `-lineinfo`. Without those the
reader still serves launches, metrics and rules, and says explicitly that
instruction attribution is absent rather than reporting zeros.

## Without a readable report

If the reader or the `ncu_report` module is unavailable, the profile layer falls
back to the CSV Nsight Compute wrote alongside the report. The CSV carries the
metric values but no descriptions, no NVIDIA rules and no instruction-level
attribution; every query result names its backend (`ncu_report` or `csv`).

## Another reader

`profile.report_reader` may name a different executable that speaks the same
contract: `<reader> ncu <verb> <report> [options]` printing one JSON envelope
with `schema: v1`, `source.kind: ncu`, `source.version: v1` and `data.rows`.
