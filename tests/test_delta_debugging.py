from __future__ import annotations

# Standard Imports
from pathlib import Path
# Third-Party Imports
import pytest
from click.testing import CliRunner

# Perun Imports
from perun import cli
from perun.testing import asserts

@pytest.mark.usefixtures("cleandir")
def test_delta_debugging_correct():
    """Runs basic tests for delta debugging CLI"""
    runner = CliRunner()
    examples = Path(__file__).parent / "sources" / "delta_debugging_examples"

    num_workload = examples / "samples" / "txt" / "simple.txt"

    delta_debugging_test = examples / "dd-minimal-test" / "dd-minimal"
    result = runner.invoke(
        cli.deltadebugging,
        [
            str(delta_debugging_test),
            str(num_workload),
            str(examples),
            "-t", "2",
        ],
    )

    asserts.predicate_from_cli(result, result.exit_code == 0)
    asserts.predicate_from_cli(result, "---" in result.output)
