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
from perun.vcs import vcs_kit


def get_machine_info(machine_info: Optional[str] = None) -> dict[str, Any]:
    """Returns machine info either from input file or constructs it from environment

    :param machine info: file in json format, which contains machine specification
    :return: parsed dictionary format of machine specification
    """
    if machine_info is not None:
        with open(machine_info, "r") as machine_handle:
            return json.load(machine_handle)
    else:
        return environment.get_machine_specification()


def import_from_string(
    out: str,
    minor_version: MinorVersion,
    machine_info: Optional[str] = None,
    with_sudo: bool = False,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
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
                "cmd": kwargs.get("cmd", ""),
                "exitcode": kwargs.get("exitcode", "?"),
                "workload": kwargs.get("workload", ""),
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
                    "warmup": kwargs.get("warmup", 0),
                    "repeat": kwargs.get("repeat", 1),
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
        commands.add([full_profile_path], minor_version.checksum, keep_profile=False)
    else:
        # Else we register the profile in pending index
        index.register_in_pending_index(full_profile_path, prof)


@vcs_kit.lookup_minor_version
def import_perf_from_record(
    imported: list[str],
    machine_info: Optional[str],
    minor_version: str,
    with_sudo: bool = False,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record`"""
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)

    parse_script = script_kit.get_script("stackcollapse-perf.pl")
    out = b""

    for imported_file in imported:
        perf_script_command = (
            f"{'sudo ' if with_sudo else ''}perf script -i {imported_file} | {parse_script}"
        )
        try:
            out, _ = external_commands.run_safely_external_command(perf_script_command)
            log.minor_success(f"Raw data from {log.path_style(imported_file)}", "collected")
        except subprocess.CalledProcessError as err:
            log.minor_fail(f"Raw data from {log.path_style(imported_file)}", "not collected")
            log.error(f"Cannot load data due to: {err}")
        import_from_string(
            out.decode("utf-8"),
            minor_version_info,
            machine_info,
            with_sudo=with_sudo,
            save_to_index=save_to_index,
            **kwargs,
        )
        log.minor_success(log.path_style(imported_file), "imported")


@vcs_kit.lookup_minor_version
def import_perf_from_script(
    imported: list[str],
    machine_info: Optional[str],
    minor_version: str,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record; perf script`"""
    parse_script = script_kit.get_script("stackcollapse-perf.pl")
    out = b""
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)

    for imported_file in imported:
        perf_script_command = f"cat {imported_file} | {parse_script}"
        out, _ = external_commands.run_safely_external_command(perf_script_command)
        log.minor_success(f"Raw data from {log.path_style(imported_file)}", "collected")
        import_from_string(
            out.decode("utf-8"),
            minor_version_info,
            machine_info,
            save_to_index=save_to_index,
            **kwargs,
        )
        log.minor_success(log.path_style(imported_file), "imported")


@vcs_kit.lookup_minor_version
def import_perf_from_stack(
    imported: list[str],
    machine_info: Optional[str],
    minor_version: str,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record; perf script | stackcollapse-perf.pl`"""
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)

    for imported_file in imported:
        with open(imported_file, "r", encoding="utf-8") as imported_handle:
            out = imported_handle.read()
        import_from_string(
            out,
            minor_version_info,
            machine_info,
            save_to_index=save_to_index,
            **kwargs,
        )
        log.minor_success(log.path_style(imported_file), "imported")
