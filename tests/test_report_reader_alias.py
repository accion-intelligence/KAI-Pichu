from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def alias_installer():
    path = Path(__file__).resolve().parents[1] / "scripts/install_report_reader_alias.py"
    spec = importlib.util.spec_from_file_location("report_reader_alias", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.install_alias


def test_alias_preserves_existing_reader_argv_environment_and_exit_code(tmp_path, alias_installer):
    reader = tmp_path / "existing reader"
    reader.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                      "print(json.dumps({'argv':sys.argv[1:],'directory':os.environ['READER_API_DIR']}))\n"
                      "sys.exit(7)\n")
    reader.chmod(0o755)
    alias = alias_installer(str(reader), tmp_path / "kai-ncu-reader", python_dir_env="READER_API_DIR")
    args = ["ncu", "disasm", "capture with spaces.ncu-rep", "--row-id", "launch:0", "literal;$HOME"]
    result = subprocess.run([str(alias), *args], capture_output=True, text=True,
                            env={**os.environ, "KAI_PICHU_REPORT_READER_DIR": "/api path/with spaces"})
    assert result.returncode == 7
    assert json.loads(result.stdout) == {"argv": args, "directory": "/api path/with spaces"}


def test_alias_does_not_overwrite_an_existing_command(tmp_path, alias_installer):
    path = tmp_path / "kai-ncu-reader"
    path.write_text("existing command")
    with pytest.raises(ValueError, match="already exists"):
        alias_installer(sys.executable, path)
    assert path.read_text() == "existing command"


def test_alias_rejects_invalid_environment_variable(tmp_path, alias_installer):
    path = tmp_path / "kai-ncu-reader"
    with pytest.raises(ValueError, match="valid identifier"):
        alias_installer(sys.executable, path, python_dir_env="READER;exit 0")
    assert not path.exists()
