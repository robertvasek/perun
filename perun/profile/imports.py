"""Functions for importing Profile from different formats"""

from __future__ import annotations

# Standard Imports
from typing import Any, Optional
import json
import os
import subprocess

# Third-Party Imports

# Perun Imports
from perun.collect.kperf import parser
from perun.profile import helpers as profile_helpers
from perun.logic import commands, index, pcs
from perun.utils import log, streams
from perun.utils.common import script_kit
from perun.utils.external import commands as external_commands, environment
from perun.utils.structs import MinorVersion
from perun.profile.factory import Profile


def get_machine_info(machine_info: Optional[str] = None) -> dict[str, Any]:
    """Returns machine info either from input file or constructs it from environment

    :param machine info: file in json format, which contains machine specification
    :return: parsed dictionary format of machine specification
    """
    if machine_info is not None:
        return json.load(machine_info)
    else:
        return environment.get_machine_specification()


def import_from_string(
    out: str,
    minor_version: MinorVersion,
    machine_info: Optional[str] = None,
    with_sudo: bool = False,
    save_to_index: bool = False,
    **kwargs: Any,
):
    resources = parser.parse_events(out.split("\n"))
    prof = Profile(
        {
            "global": {
                "time": "???",
                "resources": resources,
            }
        }
    )
    prof.update({"origin": minor_version.checksum})
    prof.update({"machine": get_machine_info(machine_info)})
    prof.update(
        {
            "header": {
                "type": "time",
                "cmd": kwargs.get("cmd", "?"),
                "workload": kwargs.get("cmd", "?"),
                "units": {"time": "sample"},
            }
        }
    )
    prof.update(
        {
            "collector_info": {
                "name": "kperf",
                "params": {
                    "with_sudo": with_sudo,
                    "warmup": kwargs.get("warmup", "?"),
                    "repeat": kwargs.get("repeat", 0),
                },
            }
        }
    )
    prof.update({"postprocessors": []})

    full_profile_name = profile_helpers.generate_profile_name(prof)
    profile_directory = pcs.get_job_directory()
    full_profile_path = os.path.join(profile_directory, full_profile_name)

    streams.store_json(prof.serialize(), full_profile_path)
    log.minor_status(
        "stored generated profile ",
        status=f"{log.path_style(os.path.relpath(full_profile_path))}",
    )
    if save_to_index:
        commands.add([full_profile_path], minor_version, keep_profile=False)
    else:
        # Else we register the profile in pending index
        index.register_in_pending_index(full_profile_path, prof)


def import_perf_from_record(
    imported: list[str],
    machine_info: Optional[str],
    minor_version_list: list[MinorVersion],
    with_sudo: bool = False,
    save_to_index: bool = False,
    **_: Any,
) -> None:
    """Imports profile collected by `perf record`"""
    assert (
        len(minor_version_list) == 1
    ), f"One can import profile for single version only (got {len(minor_version_list)} instead)"

    parse_script = script_kit.get_script("stackcollapse-perf.pl")
    out = ""

    for imported_file in imported:
        if with_sudo:
            perf_script_command = f"sudo perf script -i {imported_file} | {parse_script}"
        else:
            perf_script_command = f"perf script -i {imported_file} | {parse_script}"
        try:
            out, _ = external_commands.run_safely_external_command(perf_script_command)
            log.minor_success(f"Raw data from {log.path_style(imported_file)}", "collected")
        except subprocess.CalledProcessError:
            log.minor_fail(f"Raw data from {log.path_style(imported_file)}", "not collected")
        import_from_string(
            out.decode("utf-8"),
            minor_version_list[0],
            machine_info,
            with_sudo=with_sudo,
            save_to_index=save_to_index,
        )
        log.minor_success(log.path_style(imported_file), "imported")


def import_perf_from_script(
    imported: list[str],
    machine_info: Optional[str],
    minor_version_list: list[MinorVersion],
    save_to_index: bool = False,
    **_: Any,
) -> None:
    """Imports profile collected by `perf record; perf script`"""
    assert (
        len(minor_version_list) == 1
    ), f"One can import profile for single version only (got {len(minor_version_list)} instead)"

    parse_script = script_kit.get_script("stackcollapse-perf.pl")
    out = ""

    for imported_file in imported:
        perf_script_command = f"cat {imported_file} | {parse_script}"
        out, _ = external_commands.run_safely_external_command(perf_script_command)
        log.minor_success(f"Raw data from {log.path_style(imported_file)}", "collected")
        import_from_string(
            out.decode("utf-8"),
            minor_version_list[0],
            machine_info,
            save_to_index=save_to_index,
        )
        log.minor_success(log.path_style(imported_file), "imported")


def import_perf_from_stack(
    imported: list[str],
    machine_info: Optional[str],
    minor_version_list: list[MinorVersion],
    save_to_index: bool = False,
    **_: Any,
) -> None:
    """Imports profile collected by `perf record; perf script | stackcollapse-perf.pl`"""
    assert (
        len(minor_version_list) == 1
    ), f"One can import profile for single version only (got {len(minor_version_list)} instead)"

    for imported_file in imported:
        with open(imported_file, "r", encoding="utf-8") as imported_handle:
            out = imported_handle.read()
        import_from_string(
            out,
            minor_version_list[0],
            machine_info,
            save_to_index=save_to_index,
        )
        log.minor_success(log.path_style(imported_file), "imported")
