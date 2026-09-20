# NCU report reader setup

KAI Pichu calls an existing external NCU v1 report reader. The report parser,
source attribution and full binary disassembly remain in that external program.
KAI Pichu does not contain a replacement implementation.

The default command name is `kai-ncu-reader`. Set `profile.report_reader` or CLI
`--report-reader` to select an existing compatible reader command or launcher.
A launcher can give your existing installation this command name without changing
its implementation. From this source distribution, run:

```bash
python scripts/install_report_reader_alias.py /path/to/existing-reader \
  --python-dir-env EXISTING_READER_REPORT_DIRECTORY_VARIABLE
```

Replace the executable path and environment-variable placeholder with those of
your existing reader installation. The launcher forwards all arguments unchanged.
It maps `KAI_PICHU_REPORT_READER_DIR` to the reader's existing environment variable,
so the packaged report API compatibility adapter continues to work. If the reader
already accepts `KAI_PICHU_REPORT_READER_DIR`, omit `--python-dir-env`.
The script creates `~/.local/bin/kai-ncu-reader`; add that directory to PATH.
It does not overwrite an existing command. The external reader is not bundled.

For a fresh report, the reader needs NVIDIA's installed `ncu_report` Python API.
Set `profile.ncu_report_dir` or CLI `--ncu-report-dir` to its `extras/python`
directory if automatic discovery fails. Source/SASS/PTX queries also depend on
captured binary/source data and CUDA's disassembly tools.

The benchmark SDK and optimization loop require no report reader. With profiling
enabled, Nsight Compute collects the report and CSV. The CSV fallback works
without another executable. Public query results identify the backend as
`ncu_report` or `csv`. No component automatically installs software or invokes sudo.
