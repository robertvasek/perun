"""Functions for importing Profile from different formats"""

from __future__ import annotations

# Standard Imports
from typing import Any, Optional, Iterator, Callable
from pathlib import Path
import json
import csv
import os
import subprocess
import statistics
from dataclasses import dataclass, field, asdict

# Third-Party Imports
import gzip

# Perun Imports
from perun.collect.kperf import parser
from perun.profile import helpers as p_helpers
from perun.logic import commands, index, pcs
from perun.utils import log, streams
from perun.utils.common import script_kit
from perun.utils.external import commands as external_commands, environment
from perun.utils.structs import MinorVersion
from perun.profile.factory import Profile
from perun.vcs import vcs_kit


# TODO: add documentation
# TODO: fix stats in other types of diffviews
# TODO: refactor the perf import type commands: there is a lot of code duplication


@dataclass
class ImportProfileSpec:
    path: Path
    exit_code: int = 0
    values: list[float] = field(default_factory=list)


class ImportedProfiles:
    __slots__ = "import_dir", "stats", "profiles"

    def __init__(self, targets: list[str], import_dir: str | None, stats_info: str | None) -> None:
        self.import_dir: Path = Path(import_dir) if import_dir is not None else Path.cwd()
        # Parse the CLI stats if available
        self.stats: list[p_helpers.ProfileStat] = []
        self.profiles: list[ImportProfileSpec] = []

        if stats_info is not None:
            self.stats = [
                p_helpers.ProfileStat.from_string(*stat.split("|"))
                for stat in stats_info.split(",")
            ]

        for target in targets:
            if target.lower().endswith(".csv"):
                # The input is a csv file
                self._parse_import_csv(target)
            else:
                # The input is a file path
                self._add_imported_profile(target.split(","))

    def __iter__(self) -> Iterator[ImportProfileSpec]:
        return iter(self.profiles)

    def __len__(self) -> int:
        return len(self.profiles)

    def get_exit_codes(self) -> str:
        return ", ".join(str(p.exit_code) for p in self.profiles)

    def aggregate_stats(
        self, agg: Callable[[list[float | int]], float]
    ) -> Iterator[p_helpers.ProfileStat]:
        stat_value_lists: list[list[float | int]] = [[] for _ in range(len(self.stats))]
        for profile in self.profiles:
            value_list: list[float | int]
            stat_value: float | int
            for value_list, stat_value in zip(stat_value_lists, profile.values):
                value_list.append(stat_value)
        for value_list, stat_obj in zip(stat_value_lists, self.stats):
            stat_obj.value = agg(value_list)
            yield stat_obj

    def _parse_import_csv(self, target: str) -> None:
        with open(self.import_dir / target, "r") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            header: list[str] = next(csv_reader)
            stats: list[p_helpers.ProfileStat] = [
                p_helpers.ProfileStat.from_string(*stat_definition.split("|"))
                for stat_definition in header[2:]
            ]
            # Parse the CSV stat definition and check that they are not in conflict with the CLI
            # stat definitions, if any
            for idx, stat in enumerate(stats):
                if idx >= len(self.stats):
                    self.stats.append(stat)
                elif stat != self.stats[idx]:
                    log.warn(
                        f"Mismatching profile stat definition from CLI and CSV: "
                        f"cli.{self.stats[idx].name} != csv.{stat.name}. "
                        f"Using the CLI stat definition."
                    )
            # Parse the remaining rows that should represent profile specifications
            for row in csv_reader:
                self._add_imported_profile(row)

    def _add_imported_profile(self, target: list[str]) -> None:
        if len(target) == 0:
            # Empty profile specification, warn
            log.warn("Empty import profile specification. Skipping.")
        else:
            self.profiles.append(
                ImportProfileSpec(
                    self.import_dir / target[0],
                    int(target[1]) if len(target) >= 2 else ImportProfileSpec.exit_code,
                    list(map(float, target[2:])),
                )
            )


def load_file(filepath: Path) -> str:
    if filepath.suffix.lower() == ".gz":
        with open(filepath, "rb") as f:
            header = f.read(2)
            f.seek(0)
            assert header == b"\x1f\x8b"
            with gzip.GzipFile(fileobj=f) as gz:
                return gz.read().decode("utf-8")
    with open(filepath, "r", encoding="utf-8") as imported_handle:
        return imported_handle.read()


def get_machine_info(machine_info: Optional[str] = None) -> dict[str, Any]:
    """Returns machine info either from input file or constructs it from environment

    :param machine_info: file in json format, which contains machine specification
    :return: parsed dictionary format of machine specification
    """
    if machine_info is not None:
        with open(machine_info, "r") as machine_handle:
            return json.load(machine_handle)
    else:
        return environment.get_machine_specification()


def import_profile(
    profiles: ImportedProfiles,
    resources: list[dict[str, Any]],
    minor_version: MinorVersion,
    machine_info: Optional[str] = None,
    with_sudo: bool = False,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
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
    prof.update({"stats": [asdict(stat) for stat in profiles.aggregate_stats(statistics.median)]}),
    prof.update(
        {
            "header": {
                "type": "time",
                "cmd": kwargs.get("cmd", ""),
                "exitcode": profiles.get_exit_codes(),
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
                    "repeat": len(profiles),
                },
            }
        }
    )
    prof.update({"postprocessors": []})

    full_profile_name = p_helpers.generate_profile_name(prof)
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
    import_dir: str | None,
    stats_info: str | None,
    minor_version: str,
    with_sudo: bool = False,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record`"""
    parse_script = script_kit.get_script("stackcollapse-perf.pl")
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)

    profiles = ImportedProfiles(imported, import_dir, stats_info)

    resources = []
    for imported_file in profiles:
        perf_script_command = (
            f"{'sudo ' if with_sudo else ''}perf script -i {imported_file.path} | {parse_script}"
        )
        try:
            out, _ = external_commands.run_safely_external_command(perf_script_command)
            log.minor_success(
                f"Raw data from {log.path_style(str(imported_file.path))}", "collected"
            )
        except subprocess.CalledProcessError as err:
            log.minor_fail(
                f"Raw data from {log.path_style(str(imported_file.path))}", "not collected"
            )
            log.error(f"Cannot load data due to: {err}")
        resources.extend(parser.parse_events(out.decode("utf-8").split("\n")))
        log.minor_success(log.path_style(str(imported_file.path)), "imported")
    import_profile(profiles, resources, minor_version_info, with_sudo=with_sudo, **kwargs)


@vcs_kit.lookup_minor_version
def import_perf_from_script(
    imported: list[str],
    import_dir: str | None,
    stats_info: str | None,
    minor_version: str,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record; perf script`"""
    parse_script = script_kit.get_script("stackcollapse-perf.pl")
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)

    profiles = ImportedProfiles(imported, import_dir, stats_info)

    resources = []
    for imported_file in profiles:
        perf_script_command = f"cat {imported_file.path} | {parse_script}"
        out, _ = external_commands.run_safely_external_command(perf_script_command)
        log.minor_success(f"Raw data from {log.path_style(str(imported_file.path))}", "collected")
        resources.extend(parser.parse_events(out.decode("utf-8").split("\n")))
        log.minor_success(log.path_style(str(imported_file.path)), "imported")
    import_profile(profiles, resources, minor_version_info, **kwargs)


@vcs_kit.lookup_minor_version
def import_perf_from_stack(
    imported: list[str],
    import_dir: str | None,
    stats_info: str | None,
    minor_version: str,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record; perf script | stackcollapse-perf.pl`"""
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)
    profiles = ImportedProfiles(imported, import_dir, stats_info)

    resources = []

    for imported_profile in profiles:
        out = load_file(imported_profile.path)
        resources.extend(parser.parse_events(out.split("\n")))
        log.minor_success(log.path_style(str(imported_profile.path)), "imported")
    import_profile(profiles, resources, minor_version_info, **kwargs)
