"""Collection of global methods for delta debugging.
https://www.debuggingbook.org/html/DeltaDebugger.html
"""

from __future__ import annotations

import subprocess
from perun.utils.external import commands as external_commands
from perun.utils import log
from typing import Any, TYPE_CHECKING
from pathlib import Path
from perun.utils.common import common_kit

if TYPE_CHECKING:
    from perun.utils.structs import Executable


def run_delta_debugging_for_command(
    executable: Executable,
    input_sample: str,
    **kwargs: Any,
) -> None:

    timeout: float = kwargs.get("timeout", 1.0)
    output_dir = Path(kwargs["output_dir"]).resolve()
    read_input(input_sample)
    inp = read_input(input_sample)
    n = 2  # granularity

    while len(inp) >= 2:
        start = 0
        subset_length = int(len(inp) / n)
        program_fails = False

        while start < len(inp):
            complement = inp[: int(start)] + inp[int(start + subset_length) :]
            try:
                full_cmd = f"{executable} {complement}"
                external_commands.run_safely_external_command(full_cmd, True, True, timeout)

            except subprocess.TimeoutExpired:
                inp = complement
                n = max(n - 1, 2)
                program_fails = True
                break
            start += subset_length

        if not program_fails:
            if n == len(inp):
                break
            n = min(n * 2, len(inp))

    log.minor_info("shortest failing input = " + inp)
    return create_debugging_file(output_dir, input_sample, inp)


def read_input(input_file: str) -> str:
    input_path = Path(input_file)
    if input_path.is_file():
        with open(input_path, "r") as file:
            input_value = file.read()
    else:
        input_value = input_file

    return input_value


def create_debugging_file(output_dir: Path, file_name: str, input_data: str) -> None:
    output_dir = output_dir.resolve()
    dir_name = "delta_debugging"
    full_dir_path = output_dir / dir_name
    file_path = Path(file_name)
    common_kit.touch_dir(str(full_dir_path))
    file_path = full_dir_path / file_path.name

    with open(file_path, "w") as file:
        file.write(input_data)
