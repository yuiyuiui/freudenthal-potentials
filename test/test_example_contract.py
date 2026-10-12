"""Existing output directories must survive failed repeat invocations."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["run_example.py", "plot_results.py"])
def test_example_scripts_refuse_overwriting(tmp_path, script):
    output = tmp_path / "previous-run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve this run")
    args = [sys.executable, str(ROOT / "numerical_case" / script), "--output", str(output)]
    if script == "plot_results.py":
        args += ["--input", str(tmp_path / "missing"), "--reference", str(tmp_path / "missing")]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert sentinel.read_text() == "preserve this run"
    assert sorted(path.name for path in output.iterdir()) == ["keep.txt"]
