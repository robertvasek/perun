"""Helper functions for working with commands.

This contains functions for getting outputs from commands or running commands or external executables.
"""

from __future__ import annotations

# Standard Imports
from typing import Optional, IO, Any
from collections import defaultdict
import os
import shlex
import subprocess

# Third-Party Imports

# Perun Imports
from perun.utils import log
from perun.utils.common import common_kit
from perun.logic import pcs, config

log_file_cache: dict[str, int] = defaultdict(int)


def save_output_of_command(command: str, content: bytes, extension: str = "out") -> None:
    """Saves output of command to log

    :param command: command for which we are storing the logs
    :param content: content we are storing
    :param extension: extension of the saved log (to differentiated between stderr and stdout)
    """
    if log.LOGGING:
        log_directory = config.lookup_key_recursively("path.logs", default=pcs.get_log_directory())
        log_file = common_kit.sanitize_filepart(command.split()[0])
        log_file_cache[f"{log_file}.{extension}"] += 1
        log_no = log_file_cache[f"{log_file}.{extension}"]
        target_file = os.path.join(log_directory, f"{log_file}.{log_no}.{extension}")
        with open(target_file, "w") as target_handle:
            target_handle.write(f"# cmd: {command}\n")
            target_handle.write(content.decode("utf-8"))
        log.minor_status(
            f"Saved {extension} of {log.cmd_style(command)}", log.path_style(target_file)
        )


def get_stdout_from_external_command(command: list[str], stdin: Optional[IO[bytes]] = None) -> str:
    """Runs external command with parameters, checks its output and provides its output.

    :param list command: list of arguments for command
    :param handle stdin: the command input as a file handle
    :return: string representation of output of command
    """
    output = subprocess.check_output(
        [c for c in command if c != ""], stderr=subprocess.STDOUT, stdin=stdin
    )
    save_output_of_command(";".join(command), output)
    return output.decode("utf-8")


def run_safely_external_command(
    cmd: str,
    check_results: bool = True,
    quiet: bool = True,
    timeout: Optional[float | int] = None,
    **kwargs: Any,
) -> tuple[bytes, bytes]:
    """Safely runs the piped command, without executing of the shell

    Courtesy of: https://blog.avinetworks.com/tech/python-best-practices

    :param str cmd: string with command that we are executing
    :param bool check_results: check correct command exit code and raise exception in case of fail
    :param bool quiet: if set to False, then it will print the output of the command
    :param int timeout: timeout of the command
    :param dict kwargs: additional args to subprocess call
    :return: returned standard output and error
    :raises subprocess.CalledProcessError: when any of the piped commands fails
    """
    # Split
    unpiped_commands = list(map(str.strip, cmd.split(" | ")))
    cmd_no = len(unpiped_commands)

    # Run the command through pipes
    objects: list[subprocess.Popen[bytes]] = []
    for i in range(cmd_no):
        executed_command = shlex.split(unpiped_commands[i])

        # set streams
        stdin = None if i == 0 else objects[i - 1].stdout
        stderr = subprocess.STDOUT if i < (cmd_no - 1) else subprocess.PIPE

        # run the piped command and close the previous one
        piped_command = subprocess.Popen(
            executed_command,
            shell=False,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
            **kwargs,
        )
        if i != 0:
            # Fixme: we ignore this, as it is tricky to handle
            objects[i - 1].stdout.close()  # type: ignore
        objects.append(piped_command)

    try:
        # communicate with the last piped object
        cmdout, cmderr = objects[-1].communicate(timeout=timeout)

        for i in range(len(objects) - 1):
            objects[i].wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        for p in objects:
            p.terminate()
        raise

    # collect the return codes
    if check_results:
        for i in range(cmd_no):
            if objects[i].returncode:
                if not quiet and (cmdout or cmderr):
                    log.cprintln(f"captured stdout: {cmdout.decode('utf-8')}", "red")
                    log.cprintln(f"captured stderr: {cmderr.decode('utf-8')}", "red")
                save_output_of_command(cmd, cmdout, "stdout")
                save_output_of_command(cmd, cmderr, "stderr")
                raise subprocess.CalledProcessError(objects[i].returncode, unpiped_commands[i])

    save_output_of_command(cmd, cmdout, "stdout")
    save_output_of_command(cmd, cmderr, "stderr")
    return cmdout, cmderr


def run_safely_list_of_commands(cmd_list: list[str]) -> None:
    """Runs safely list of commands

    :param list cmd_list: list of external commands
    :raise subprocess.CalledProcessError: when there is an error in any of the commands
    """
    for cmd in cmd_list:
        log.write(">", cmd)
        out, err = run_safely_external_command(cmd)
        if out:
            log.write(out.decode("utf-8"), end="")
        if err:
            log.cprint(err.decode("utf-8"), "red")


def run_external_command(cmd_args: list[str], **subprocess_kwargs: Any) -> int:
    """Runs external command with parameters.

    :param list cmd_args: list of external command and its arguments to be run
    :param subprocess_kwargs: additional parameters to the subprocess object
    :return: return value of the external command that was run
    """
    process = subprocess.Popen(cmd_args, **subprocess_kwargs)
    process.wait()
    return process.returncode


def is_executable(command: str) -> bool:
    """Tests if command is executable

    :return: true if the command is executable
    """
    try:
        run_safely_external_command(command)
        return True
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        return False
