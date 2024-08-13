"""Functions for importing Profile from different formats"""

from __future__ import annotations

# Standard Imports
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Iterator, Callable
import csv
import json
import os
import statistics
import subprocess

# Third-Party Imports
import gzip

# Perun Imports
from perun.collect.kperf import parser
from perun.profile import helpers as profile_helpers
from perun.logic import commands, index, pcs
from perun.utils import log, streams
from perun.utils.common import script_kit, common_kit
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
    """
    Note: I would reconsider this class or refactor it, removing the logical elements, it obfuscates the logic a little
      and makes the functions less readable (there are not streams/pipes as is most of the logic/perun); I for one am
      rather "fan" of generic functions that takes structures and returns structure than classes with methods/logic.
    TODO: the import-dir could be removed by extracting this functionality to command-line callback and massage
    the paths during the CLI parsing; hence assuming that the paths are correct when importing. I think the parameter
    only complicates the code.
    """

    __slots__ = "import_dir", "stats", "profiles"

    def __init__(self, targets: list[str], import_dir: str | None, stats_info: str | None) -> None:
        self.import_dir: Path = Path(import_dir) if import_dir is not None else Path.cwd()
        # Parse the CLI stats if available
        self.stats: list[profile_helpers.ProfileStat] = []
        self.profiles: list[ImportProfileSpec] = []

        if stats_info is not None:
            self.stats = [
                profile_helpers.ProfileStat.from_string(*stat.split("|"))
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

    def get_exit_codes(self) -> list[int]:
        return [p.exit_code for p in self.profiles]

    def aggregate_stats(
        self, agg: Callable[[list[float | int]], float]
    ) -> Iterator[profile_helpers.ProfileStat]:
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
            stats: list[profile_helpers.ProfileStat] = [
                profile_helpers.ProfileStat.from_string(*stat_definition.split("|"))
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
            profile_info = ImportProfileSpec(
                self.import_dir / target[0],
                int(target[1]) if len(target) >= 2 else ImportProfileSpec.exit_code,
                list(map(float, target[2:])),
            )
            if profile_info.exit_code != 0:
                log.warn("Importing a profile with non-zero exit code.")
            self.profiles.append(profile_info)


def load_file(filepath: Path) -> str:
    """Tests if the file is packed by gzip and unpacks it, otherwise reads it as a text file

    :param filepath: path with source file
    :return: the content of the file
    """
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


def import_perf_profile(
    profiles: ImportedProfiles,
    resources: list[dict[str, Any]],
    minor_version: MinorVersion,
    machine_info: Optional[str] = None,
    with_sudo: bool = False,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
    """Constructs the profile for perf-collected data and saves them to jobs or index

    :param profiles: list of to-be-imported profiles
    :param resources: list of parsed resources
    :param minor_version: minor version corresponding to the imported profiles
    :param machine_info: additional dictionary with machine specification
    :param with_sudo: indication whether the data were collected with sudo
    :param save_to_index: indication whether we should save the imported profiles to index
    :param kwargs: rest of the paramters
    """
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

    save_imported_profile(prof, save_to_index, minor_version)


def save_imported_profile(prof: Profile, save_to_index: bool, minor_version: MinorVersion) -> None:
    """Saves the imported profile either to index or to pending jobs

    :param prof: imported profile
    :param minor_version: minor version corresponding to the imported profiles
    :param save_to_index: indication whether we should save the imported profiles to index
    """
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
    import_dir: str | None,
    stats_info: str | None,
    minor_version: str,
    with_sudo: bool = False,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record`

    It does some black magic in ImportedProfiles probably, then for each filename it runs the
    perf script + parser script to generate the profile.

    :param imported: list of files with imported data
    :param import_dir: different directory for importing the profiles
    :param stats_info: additional statistics collected for the profile (i.e. non-resource types)
    :param minor_version: minor version corresponding to the imported profiles
    :param with_sudo: indication whether the data were collected with sudo
    :param kwargs: rest of the paramters
    """
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
    import_perf_profile(profiles, resources, minor_version_info, with_sudo=with_sudo, **kwargs)


@vcs_kit.lookup_minor_version
def import_perf_from_script(
    imported: list[str],
    import_dir: str | None,
    stats_info: str | None,
    minor_version: str,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record | perf script`

    It does some black magic in ImportedProfiles probably, then for each filename it runs the
    parser script to generate the profile.

    :param imported: list of files with imported data
    :param import_dir: different directory for importing the profiles
    :param stats_info: additional statistics collected for the profile (i.e. non-resource types)
    :param minor_version: minor version corresponding to the imported profiles
    :param kwargs: rest of the paramters
    """
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
    import_perf_profile(profiles, resources, minor_version_info, **kwargs)


@vcs_kit.lookup_minor_version
def import_perf_from_stack(
    imported: list[str],
    import_dir: str | None,
    stats_info: str | None,
    minor_version: str,
    **kwargs: Any,
) -> None:
    """Imports profile collected by `perf record | perf script`

    It does some black magic in ImportedProfiles probably, then for each filename parses the files.

    :param imported: list of files with imported data
    :param import_dir: different directory for importing the profiles
    :param stats_info: additional statistics collected for the profile (i.e. non-resource types)
    :param minor_version: minor version corresponding to the imported profiles
    :param kwargs: rest of the paramters
    """
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)
    profiles = ImportedProfiles(imported, import_dir, stats_info)

    resources = []

    for imported_profile in profiles:
        out = load_file(imported_profile.path)
        resources.extend(parser.parse_events(out.split("\n")))
        log.minor_success(log.path_style(str(imported_profile.path)), "imported")
    import_perf_profile(profiles, resources, minor_version_info, **kwargs)


def extract_machine_info_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extracts the parts of the profile, that corresponds to machine info

    Note that not many is collected from the ELK formats and it can vary greatly,
    hence, most of the machine specification and environment should be in metadata instead.

    :param metadata: metadata extracted from the ELK profiles
    :return: machine info extracted from the profiles
    """
    machine_info = {
        "architecture": metadata.get("machine.arch", "?"),
        "system": metadata.get("machine.os", "?").capitalize(),
        "release": metadata.get("extra.machine.platform", "?"),
        "host": metadata.get("machine.hostname", "?"),
        "cpu": {
            "physical": "?",
            "total": metadata.get("machine.cpu-cores", "?"),
            "frequency": "?",
        },
        "memory": {
            "total_ram": metadata.get("machine.ram", "?"),
            "swap": "?",
        },
    }

    machine_info["boot_info"] = "?"
    machine_info["mem_details"] = {}
    machine_info["cpu_details"] = []
    return machine_info


def import_elk_profile(
    resources: list[dict[str, Any]],
    metadata: dict[str, Any],
    minor_version: MinorVersion,
    save_to_index: bool = False,
    **kwargs: Any,
) -> None:
    """Constructs the profile for elk-stored data and saves them to jobs or index

    :param resources: list of parsed resources
    :param metadata: parts of the profiles that will be stored as metadata in the profile
    :param minor_version: minor version corresponding to the imported profiles
    :param save_to_index: indication whether we should save the imported profiles to index
    :param kwargs: rest of the paramters
    """
    prof = Profile(
        {
            "global": {
                "time": "???",
                "resources": resources,
            }
        }
    )
    prof.update({"origin": minor_version.checksum})
    prof.update({"metadata": metadata})
    prof.update({"machine": extract_machine_info_from_metadata(metadata)})
    prof.update(
        {
            "header": {
                "type": "time",
                "cmd": kwargs.get("cmd", ""),
                "exitcode": "?",
                "workload": kwargs.get("workload", ""),
                "units": {"time": "sample"},
            }
        }
    )
    prof.update(
        {
            "collector_info": {
                "name": "???",
                "params": {},
            }
        }
    )
    prof.update({"postprocessors": []})

    save_imported_profile(prof, save_to_index, minor_version)


def extract_from_elk(
    elk_query: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """For the given elk query, extracts resources and metadata.

    For metadata we consider any key that has only single value through the profile,
    and is not linked to keywords `metric` or `benchmarking`.
    For resources we consider anything that is not identified as metadata

    :param elk_query: query from the elk in form of list of resource
    :return: list of resources and metadata
    """
    res_counter = defaultdict(set)
    for res in elk_query:
        for key, val in res.items():
            res_counter[key].add(val)
    metadata_keys = {
        k
        for (k, v) in res_counter.items()
        if not k.startswith("metric") and not k.startswith("benchmarking") and len(v) == 1
    }

    metadata = {k: res_counter[k].pop() for k in metadata_keys}
    resources = [
        {
            k: common_kit.try_convert(v, [int, float, str])
            for k, v in res.items()
            if k not in metadata_keys
        }
        for res in elk_query
    ]
    # We register uid
    for res in resources:
        res["uid"] = res["metric.name"]
        res["benchmarking.time"] = res["benchmarking.end-ts"] - res["benchmarking.start-ts"]
        res.pop("benchmarking.end-ts")
        res.pop("benchmarking.start-ts")
    return resources, metadata


@vcs_kit.lookup_minor_version
def import_elk_from_json(
    imported: list[str],
    minor_version: str,
    **kwargs: Any,
) -> None:
    """Imports the ELK stored data from the json data.

    The loading expects the json files to be in form of `{'queries': []}`.

    :param imported: list of filenames with elk data.
    :param minor_version: minor version corresponding to the imported profiles
    :param kwargs: rest of the paramters
    """
    minor_version_info = pcs.vcs().get_minor_version_info(minor_version)

    resources: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for imported_file in imported:
        with open(imported_file, "r") as imported_handle:
            imported_json = json.load(imported_handle)
            assert (
                "queries" in imported_json.keys()
            ), "expected the JSON to contain list of dictionaries in 'queries' key"
            r, m = extract_from_elk(imported_json["queries"])
        resources.extend(r)
        metadata.update(m)
        log.minor_success(log.path_style(str(imported_file)), "imported")
    import_elk_profile(resources, metadata, minor_version_info, **kwargs)
