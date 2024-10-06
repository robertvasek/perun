"""Module for providing coverage-based testing.

Collects functions for preparing the workspace before testing (so it removes remaining files),
collecting source files, initial and common testing and gathering coverage information by
executing gcov tool and parsing its output.
"""

from __future__ import annotations

# Standard Imports
from typing import Any, TYPE_CHECKING
import os
import statistics
import subprocess

# Third-Party Imports

# Perun Imports
from perun.utils import log
from perun.utils.exceptions import SuppressedExceptions
from perun.utils.external import commands

if TYPE_CHECKING:
    from perun.fuzz.structs import (
        FuzzingConfiguration,
        Mutation,
        FuzzingProgress,
        CoverageConfiguration,
    )
    from perun.utils.structs.common_structs import Executable


def prepare_workspace(source_path: str) -> None:
    """Prepares workspace for yielding coverage information using gcov.

    The .gcda file is generated when a program containing object files built with the GCC
    -fprofile-arcs() option is executed. We use --coverage which is used to compile and link code
    instrumented for coverage analysis. The option is a synonym for -fprofile-arcs -ftest-coverage
    (when compiling) and -lgcov (when linking).
    A separate .gcda file is created for each object file compiled with this option. It contains arc
    transition counts, and some summary information. Files with coverage data created using gcov
    utility have extension .gcov.
    Before meaningful testing, residual .gcda and .gcov files have to be removed.

    :param source_path: path to dir with source files, where coverage info files are stored
    """
    for file in os.listdir(source_path):
        if file.endswith(".gcda") or file.endswith(".gcov") or file.endswith(".gcov.json.gz"):
            os.remove(os.path.join(source_path, file))


def get_src_files(source_path: str) -> list[str]:
    """Gathers all C/C++ source files in the directory specified by `source_path`

    C/C++ files are identified with extensions: .c, .cc, .cpp. All these types of files located in
    source directory are collected together.

    :param source_path: path to directory with source files
    :return: list of source files (their absolute paths) located in `source_path`
    """
    sources = []

    for root, _, files in os.walk(source_path):
        if files:
            sources.extend(
                [
                    os.path.join(os.path.abspath(root), filename)
                    for filename in files
                    if os.path.splitext(filename)[-1] in [".c", ".cpp", ".cc", ".h"]
                ]
            )

    return sources


def baseline_testing(
    executable: Executable, workloads: list[Mutation], config: FuzzingConfiguration
) -> int:
    """Coverage based testing initialization. Wrapper over function `get_initial_coverage`.

    :param executable: called command with arguments
    :param workloads: workloads for initial testing
    :param config: configuration of the fuzzing
    :return: median of measured coverages
    """
    # get source files (.c, .cc, .cpp, .h)
    config.coverage.source_files = get_src_files(config.coverage.source_path)
    log.minor_status(
        f"{log.highlight('gcov')} version", status=f"version {config.coverage.gcov_version}"
    )

    return get_initial_coverage(executable, workloads, config.hang_timeout, config)


def get_initial_coverage(
    executable: Executable,
    seeds: list[Mutation],
    timeout: int | float,
    fuzzing_config: FuzzingConfiguration,
) -> int:
    """Provides initial testing with initial samples given by user.

    :param timeout: specified timeout for run of target application
    :param executable: called command with arguments
    :param seeds: initial sample files
    :param fuzzing_config: configuration of the fuzzing
    :return: median of measured coverages
    """
    coverages = []

    # run program with each seed
    log.minor_info("Running program with seeds")
    log.increase_indent()
    for seed in log.progress(seeds, description="Running Seeds"):
        prepare_workspace(fuzzing_config.coverage.gcno_path)

        command = " ".join([os.path.abspath(executable.cmd), seed.path])

        try:
            commands.run_safely_external_command(command, timeout=timeout)
            log.minor_success(f"{log.cmd_style(command)}")
        except subprocess.CalledProcessError as serr:
            log.minor_fail(f"{log.cmd_style(command)}")
            log.error("Initial testing with file " + seed.path + " caused " + str(serr))
        seed.cov = get_coverage_from_dir(os.getcwd(), fuzzing_config.coverage)

        coverages.append(seed.cov)
    log.decrease_indent()

    return int(statistics.median(coverages))


def target_testing(
    executable: Executable,
    workload: Mutation,
    config: FuzzingConfiguration,
    parent: Mutation,
    fuzzing_progress: FuzzingProgress,
    **__: Any,
) -> bool:
    """
    Testing function for coverage based fuzzing. Before testing, it prepares the workspace
    using `prepare_workspace` func, executes given command and `get_coverage_info` to
    obtain coverage information.

    :param executable: called command with arguments
    :param workload: testing workload
    :param parent: parent we are mutating
    :param config: config of the fuzzing
    :param fuzzing_progress: progress of the fuzzing process
    :param __: additional information containing base result and path to .gcno files
    :return: true if the base coverage has just increased
    """
    prepare_workspace(config.coverage.gcno_path)
    command = " ".join([executable.cmd, workload.path])

    try:
        commands.run_safely_external_command(command, timeout=config.hang_timeout)
    except subprocess.CalledProcessError as err:
        log.error_msg("Testing with file " + workload.path + " caused an error: " + str(err))
        raise err

    workload.cov = get_coverage_from_dir(os.getcwd(), config.coverage)
    return check_if_coverage_increased(
        fuzzing_progress.base_cov, workload.cov, parent.cov, config.cov_rate
    )


def get_gcov_files(directory: str) -> list[str]:
    """Searches for gcov files in `directory`.
    :param directory: path of a directory, where searching will be provided
    :return: absolute paths of found gcov files
    """
    gcov_files = []
    for file in os.listdir(directory):
        if os.path.isfile(file) and file.endswith("gcov"):
            gcov_file = os.path.abspath(os.path.join(directory, file))
            gcov_files.append(gcov_file)
    return gcov_files


def parse_coverage_from_line(line: str, coverage_config: CoverageConfiguration) -> int:
    """Parses coverage information out of the line according to the version of gcov

    :param line: one line in coverage info
    :param coverage_config: configuration of the coverage testing
    :return: coverage info from one line
    """
    with SuppressedExceptions(ValueError):
        # intermediate text format
        if coverage_config.has_intermediate_format() and "lcount" in line:
            return int(line.split(",")[1])
        # standard gcov file format
        elif coverage_config.has_common_format():
            return int(line.split(":")[0])
    return 0


def get_coverage_from_dir(cwd: str, config: CoverageConfiguration) -> int:
    """Executes gcov utility with source files, and gathers all output .gcov files.

    First of all, it changes current working directory to directory specified by
    `source_path` and then executes utility gcov over all source files.
    By execution, .gcov files was created as output in intermediate text format("-i") if possible.
    Otherwise, standard gcov output file format  will be parsed.
    Current working directory is now changed back.

    :param cwd: current working directory for changing back
    :param config: configuration for coverage
    :return: absolute paths of generated .gcov files
    """
    os.chdir(config.gcno_path)

    cmd = ["gcov", "-i", "-o", "."] if config.has_intermediate_format() else ["gcov", "-o", "."]
    cmd.extend(config.source_files)

    with SuppressedExceptions(subprocess.CalledProcessError):
        commands.run_safely_external_command(" ".join(cmd))

    # searching for gcov files, if they are not already known
    if not config.gcov_files:
        config.gcov_files = get_gcov_files(".")

    execs = 0
    for gcov_file in config.gcov_files:
        with open(gcov_file, "r") as gcov_fp:
            for line in gcov_fp:
                execs += parse_coverage_from_line(line, config)
    os.chdir(cwd)
    return execs


def check_if_coverage_increased(
    base_cov: int, cov: int, parent_cov: int, increase_ratio: float = 1.5
) -> bool:
    """Condition for adding mutated input to set of candidates(parents).

    :param base_cov: base coverage
    :param cov: measured coverage of the current mutation
    :param parent_cov: coverage of mutation parent
    :param increase_ratio: desired coverage increase ration between `base_cov` and `cov`
    :return: True if `cov` is greater than `base_cov` * `deg_ratio`, False otherwise
    """
    return cov > int(base_cov * increase_ratio) and cov > parent_cov
