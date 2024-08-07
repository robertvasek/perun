"""Basic tests for checking the correctness of the VCS modules"""

from __future__ import annotations

# Standard Imports
import os

# Third-Party Imports
from click.testing import CliRunner

# Perun Imports
from perun import cli


def test_imports(pcs_with_svs):
    pool_path = os.path.join(os.path.split(__file__)[0], "sources", "imports")

    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 0
    runner = CliRunner()
    result = runner.invoke(
        cli.cli,
        ["import", "-c", "ls", "-w", ".", "perf", "record", os.path.join(pool_path, "import.data")],
    )
    assert result.exit_code == 0
    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 2

    result = runner.invoke(
        cli.cli,
        [
            "import",
            "-c",
            "ls",
            "-w",
            "al",
            "perf",
            "script",
            os.path.join(pool_path, "import.script"),
        ],
    )
    assert result.exit_code == 0
    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 3

    result = runner.invoke(
        cli.cli,
        [
            "import",
            "--machine-info",
            os.path.join(pool_path, "machine_info.json"),
            "perf",
            "stack",
            os.path.join(pool_path, "import.stack"),
        ],
    )
    assert result.exit_code == 0
    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 4

    result = runner.invoke(
        cli.cli,
        [
            "import",
            "--save-to-index",
            "-c",
            "ls",
            "-w",
            "..",
            "perf",
            "stack",
            os.path.join(pool_path, "import.stack"),
            os.path.join(pool_path, "import.stack.gz"),
        ],
    )
    assert result.exit_code == 0
    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 4

    result = runner.invoke(
        cli.cli,
        [
            "import",
            "-c",
            "ls",
            "-w",
            "..",
            "-d",
            pool_path,
            "perf",
            "stack",
            "import.csv",
        ],
    )
    assert result.exit_code == 0
    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 5

    result = runner.invoke(
        cli.cli,
        [
            "import",
            "-c",
            "ls",
            "-w",
            ".",
            "perf",
            "record",
            os.path.join(pool_path, "import.stack"),
        ],
    )
    assert result.exit_code == 1
    assert len(os.listdir(os.path.join(".perun", "jobs"))) == 5
