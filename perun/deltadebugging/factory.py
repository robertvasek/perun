"""Collection of global methods for delta debugging.
https://www.debuggingbook.org/html/DeltaDebugger.html
"""
from __future__ import annotations

import os

import subprocess
from perun.utils.external import commands as external_commands
from perun.utils import log
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from perun.utils.structs import Executable

def run_delta_debugging_for_command(
    executable: Executable,
    input_sample: list[str],
    **kwargs: Any,
) -> None:

    timeout: float = kwargs.get("timeout", 1.0)
    output_dir: str = os.path.abspath(kwargs["output_dir"])
    readInput(input_sample)
    inp = readInput(input_sample)
    n = 2  # granularity

    while len(inp) >= 2:
        start = 0
        subset_length = int(len(inp) / n)
        program_fails = False

        while start < len(inp):
            complement = (inp[:int(start)] + inp[int(start + subset_length):])
            try:
                full_cmd = f"{executable} {complement}"
                external_commands.run_safely_external_command(full_cmd, True, True, timeout)

            except subprocess.TimeoutExpired:
                inp = complement
                n = max(n - 1, 2)
                program_fails = True
                print(complement)
                break
            start += subset_length

        if not program_fails:
            if n == len(inp):
                break
            n = min(n * 2, len(inp))

    log.minor_info('shortest failing input = ' + inp)
    return create_debugging_file(output_dir, input_sample, inp)


def readInput(input_file):
    if os.path.isfile(input_file):
        with open(input_file, 'r') as file:
            input_value = file.read()
    else:
        input_value = input_file

    return input_value

def create_debugging_file(output_dir, file_name, input_data):
    dir_name = 'delta_debugging'
    full_dir_path = os.path.join(output_dir, dir_name)

    file_dir = os.path.dirname(file_name)
    if file_dir:
        full_dir_path = os.path.join(full_dir_path, file_dir)

    os.makedirs(full_dir_path, exist_ok=True)
    file_path = os.path.join(full_dir_path, os.path.basename(file_name))

    with open(file_path, 'w') as file:
        file.write(input_data)