"""``perun.profile.factory`` specifies collective interface for basic
manipulation with profiles.

The format of profiles is w.r.t. :ref:`profile-spec`. This module contains
helper functions for loading and storing of the profiles either in the
persistent memory or in filesystem (in this case, the profile is in
uncompressed format).

.. _Python JSON library: https://docs.python.org/3.7/library/json.html

For further manipulations refer either to :ref:`profile-conversion-api`
(implemented in ``perun.profile.convert`` module) or :ref:`profile-query-api`
(implemented in ``perun.profile.query module``). For full specification how to
handle the JSON objects in Python refer to `Python JSON library`_.
"""

from __future__ import annotations

# Standard Imports
import dataclasses
import json
import operator
import os
import re
import time
from typing import Any, TYPE_CHECKING, Union

# Third-Party Imports

# Perun Imports
from perun.logic import config, index, pcs, store
from perun import profile as profiles
from perun.utils import decorators, log as perun_log
from perun.utils.common import common_kit
from perun.utils.external import environment
from perun.utils.exceptions import (
    InvalidParameterException,
    TagOutOfRangeException,
)
from perun.utils.structs.common_structs import Unit, Executable, Job, SortOrder
from perun.vcs import vcs_kit

if TYPE_CHECKING:
    import types

# A tuple representation of the ProfileHeaderEntry object
ProfileHeaderTuple = tuple[str, Union[str, float], str, dict[str, Union[str, float]]]


PROFILE_COUNTER: int = 0
DEFAULT_SORT_KEY: str = "time"


def lookup_value(container: dict[str, str] | profiles.Profile, key: str, missing: str) -> str:
    """Helper function for getting the key from the container. If it is not present in the container,
    or it is empty string or empty object, the function should return the missing constant.

    :param container: dictionary container
    :param key: string representation of the key
    :param missing: string constant that is returned if key is not present in container,
        or is set to empty string or None.
    :return:
    """
    return str(container.get(key, missing)) or missing


def lookup_param(profile: profiles.Profile, unit: str, param: str) -> str:
    """Helper function for looking up the unit in the profile (can be either collector or
    postprocessor) and finds the value of the param in it

    :param profile: dictionary with profile information w.r.t profile specification
    :param unit: unit in which the parameter is located
    :param param: parameter we will use in the resulting profile
    :return: collector or postprocess unit name that is incorporated in the filename
    """
    unit_param_map = {post["name"]: post["params"] for post in profile.get("postprocessors", [])}
    used_collector = profile["collector_info"]
    unit_param_map.update({used_collector.get("name", "?"): used_collector.get("params", {})})

    # Lookup the unit params
    unit_params = unit_param_map.get(unit)
    if unit_params:
        return (
            common_kit.sanitize_filepart(list(profiles.all_key_values_of(unit_params, param))[0])
            or "_"
        )
    else:
        return "_"


def generate_profile_name(profile: profiles.Profile) -> str:
    """Constructs the profile name with the extension .perf from the job.

    The profile is identified by its binary, collector, workload and the time
    it was run.

    Valid tags:
        `%collector%`:
            Name of the collector
        `%postprocessors%`:
            Joined list of postprocessing phases
        `%<unit>.<param>%`:
            Parameter of the collector given by concrete name
        `%cmd%`:
            Command of the job
        `%workload%`:
            Workload of the job
        `%label%`:
            A custom string label associated with the profile
        `%type%`:
            Type of the generated profile
        `%kernel%`:
            Underlying kernel
        `%date%`:
            Current date
        `%origin%`:
            Origin of the profile
        `%counter%`:
            Increasing argument

    :param profile: generate the corresponding profile for given name
    :return: string for the given profile that will be stored
    """
    global PROFILE_COUNTER
    # TODO: This might be broken in new versions?
    fmt_parser = re.Scanner(  # type: ignore
        [
            (
                r"%collector%",
                lambda scanner, token: lookup_value(profile["collector_info"], "name", "_"),
            ),
            (
                r"%postprocessors%",
                lambda scanner, token: (
                    ("after-" + "-and-".join(map(lambda p: p["name"], profile["postprocessors"])))
                    if profile["postprocessors"]
                    else "_"
                ),
            ),
            (
                r"%[^.]+\.[^%]+%",
                lambda scanner, token: lookup_param(profile, *token[1:-1].split(".", maxsplit=1)),
            ),
            (
                r"%cmd%",
                lambda scanner, token: "["
                + common_kit.sanitize_filepart(
                    os.path.split(lookup_value(profile["header"], "cmd", "_"))[-1]
                )
                + "]",
            ),
            (
                r"%workload%",
                lambda scanner, token: "["
                + common_kit.sanitize_filepart(
                    os.path.split(lookup_value(profile["header"], "workload", "_"))[-1]
                )
                + "]",
            ),
            (
                r"%label%",
                lambda scanner, token: lookup_value(profile["header"], "label", "_"),
            ),
            (r"%kernel%", lambda scanner, token: environment.get_kernel()),
            (
                r"%type%",
                lambda scanner, token: lookup_value(profile["header"], "type", "_"),
            ),
            (
                r"%date%",
                lambda scanner, token: time.strftime(
                    "%Y-%m-%d-%H-%M-%S", config.runtime().safe_get("current_time", time.gmtime())
                ),
            ),
            (r"%origin%", lambda scanner, token: lookup_value(profile, "origin", "_")),
            (r"%counter%", lambda scanner, token: str(PROFILE_COUNTER)),
            (r"%%", lambda scanner, token: token),
            ("[^%]+", lambda scanner, token: token),
        ]
    )
    PROFILE_COUNTER += 1

    # Obtain the formatting template from the configuration
    template = config.lookup_key_recursively("format.output_profile_template")
    tokens, rest = fmt_parser.scan(template)
    if rest:
        perun_log.error(
            f"formatting string '{template}' could not be parsed\n\n"
            + "Run perun config to modify the formatting pattern. "
            "Refer to documentation for more information about formatting patterns"
        )
    return "".join(tokens) + ".perf"


def load_list_for_minor_version(minor_version: str) -> list["ProfileInfo"]:
    """Returns profiles assigned to the given minor version.

    :param minor_version: identification of the commit (preferably sha1)
    :return: list of ProfileInfo parsed from index of the given minor_version
    """
    # Compute the
    profile_list = index.get_profile_list_for_minor(pcs.get_object_directory(), minor_version)
    profile_info_list = []
    for index_entry in profile_list:
        inside_info = {
            "header": {
                "type": index_entry.type,
                "cmd": index_entry.cmd,
                "workload": index_entry.workload,
                "label": index_entry.label,
            },
            "collector_info": {"name": index_entry.collector},
            "postprocessors": [{"name": p} for p in index_entry.postprocessors],
        }
        _, profile_name = store.split_object_name(pcs.get_object_directory(), index_entry.checksum)
        profile_info = ProfileInfo(index_entry.path, profile_name, index_entry.time, inside_info)
        profile_info_list.append(profile_info)

    return profile_info_list


@vcs_kit.lookup_minor_version
def get_nth_profile_of(position: int, minor_version: str) -> str:
    """Returns the profile at nth position in the index

    :param position: position of the profile we are obtaining
    :param minor_version: looked up minor version for the wrapped vcs

    :return: path of the profile at nth position in the index
    """
    registered_profiles = load_list_for_minor_version(minor_version)
    sort_profiles(registered_profiles)
    if 0 <= position < len(registered_profiles):
        return registered_profiles[position].realpath
    else:
        raise TagOutOfRangeException(position, len(registered_profiles) - 1, "i")


@vcs_kit.lookup_minor_version
def find_profile_entry(profile: str, minor_version: str) -> index.BasicIndexEntry:
    """Finds the profile entry within the index file of the minor version.

    :param profile: the profile identification, can be given as tag, sha value,
                        sha-path (path to tracked profile in obj) or source-name
    :param minor_version: the minor version representation or None for HEAD

    :return: the profile entry from the index file
    """

    minor_index = index.find_minor_index(minor_version)

    # If profile is given as tag, obtain the sha-path of the file
    tag_match = store.INDEX_TAG_REGEX.match(profile)
    if tag_match:
        profile = get_nth_profile_of(int(tag_match.group(1)), minor_version)
    # Transform the sha-path (obtained or given) to the sha value
    if not store.is_sha1(profile) and not profile.endswith(".perf"):
        profile = store.version_path_to_sha(profile) or ""

    # Search the minor index for the requested profile
    with open(minor_index, "rb") as index_handle:
        # The profile can be only sha value or source path now
        if store.is_sha1(profile):
            return index.lookup_entry_within_index(
                index_handle, lambda x: x.checksum == profile, profile
            )
        else:
            return index.lookup_entry_within_index(
                index_handle, lambda x: x.path == profile, profile
            )


def generate_units(collector: types.ModuleType) -> dict[str, str]:
    """Generate information about units used by the collector.

    Note that this is mostly placeholder for future extension, how the units will be handled.

    :param collector: collector module that collected the data
    :return: dictionary with map of resources to units
    """
    return collector.COLLECTOR_DEFAULT_UNITS


def generate_header_for_profile(job: Job) -> dict[str, Any]:
    """
    :param job: job with information about the computed profile
    :return: dictionary in form of {'header': {}} corresponding to the perun specification
    """
    # At this point, the collector module should be valid
    collector = common_kit.get_module(".".join(["perun.collect", job.collector.name]))

    return {
        "type": collector.COLLECTOR_TYPE,
        "cmd": job.executable.cmd,
        "workload": job.executable.workload,
        "units": generate_units(collector),
        "exitcode": config.runtime().safe_get("exitcode", "?"),
    }


def generate_collector_info(job: Job) -> dict[str, Any]:
    """
    :param job: job with information about the computed profile
    :return: dictionary in form of {'collector_info': {}} corresponding to the perun
        specification
    """
    return {"name": job.collector.name, "params": job.collector.params}


def generate_postprocessor_info(job: Job) -> list[dict[str, Any]]:
    """
    :param job: job with information about the computed profile
    :return: dictionary in form of {'postprocess_info': []} corresponding to the perun spec
    """
    return [
        {"name": postprocessor.name, "params": postprocessor.params}
        for postprocessor in job.postprocessors
    ]


def finalize_profile_for_job(profile: profiles.Profile, job: Job) -> profiles.Profile:
    """
    :param profile: collected profile through some collector
    :param job: job with information about the computed profile
    :return: valid profile JSON file
    """
    profile.update({"origin": pcs.vcs().get_minor_head()})
    profile.update({"header": generate_header_for_profile(job)})
    profile.update({"machine": environment.get_machine_specification()})
    profile.update({"collector_info": generate_collector_info(job)})
    profile.update({"postprocessors": generate_postprocessor_info(job)})
    return profile


def to_string(profile: profiles.Profile) -> str:
    """Converts profile from dictionary to string

    :param profile: profile we are converting
    :return: string representation of profile
    """
    return json.dumps(profile.serialize())


def to_config_tuple(profile: profiles.Profile) -> tuple[str, str, str, str]:
    """Converts the profile to the tuple representing its configuration

    :param profile: profile we are converting to configuration tuple
    :returns: (collector.name, cmd, args, workload, postprocessors joined by ', ')
    """
    profile_header = profile["header"]
    return (
        profile["collector_info"]["name"],
        profile_header.get("cmd", ""),
        profile_header.get("workload", ""),
        ", ".join([postprocessor["name"] for postprocessor in profile["postprocessors"]]),
    )


def config_tuple_to_cmdstr(config_tuple: tuple[str, str, str, str]) -> str:
    """Converts tuple to command string

    :param config_tuple: tuple of (collector, cmd, workload, postprocessors)
    :return: string representing the executed command
    """
    return " ".join(filter(lambda x: x, config_tuple[1:3]))


def extract_job_from_profile(profile: profiles.Profile) -> Job:
    """Extracts information from profile about job, that was done to generate the profile.

    Fixme: Add assert that profile is profile

    :param profile: dictionary with valid profile
    :return: job according to the profile information
    """
    collector_record = profile["collector_info"]
    collector = Unit(collector_record["name"], collector_record["params"])

    posts = []
    for postprocessor in profile["postprocessors"]:
        posts.append(Unit(postprocessor["name"], postprocessor["params"]))

    cmd = profile["header"]["cmd"]
    workload = profile["header"]["workload"]
    executable = Executable(cmd, workload)

    return Job(collector, posts, executable)


def is_key_aggregatable_by(profile: profiles.Profile, func: str, key: str, keyname: str) -> bool:
    """Check if the key can be aggregated by the function.

    Everything is countable and hence 'count' and 'nunique' (number of unique values) are
    valid aggregation functions for everything. Otherwise, (e.g. sum, mean), we need numerical
    values.

    :param profile: profile that will be used against in the validation
    :param func: function used for aggregation of the data
    :param key: key that will be aggregated in the graph
    :param keyname: name of the validated key
    :return: true if the key is aggregatable by the function
    :raises InvalidParameterException: if the of_key does not support the given function
    """
    # Everything is countable ;)
    if func in ("count", "nunique"):
        return True

    # Get all valid numeric keys and validate
    valid_keys = set(profiles.all_numerical_resource_fields_of(profile))
    if key not in valid_keys:
        choices = "(choose either count/nunique as aggregation function;"
        choices += f" or from the following keys: {', '.join(map(str, valid_keys))})"
        raise InvalidParameterException(keyname, key, choices)
    return True


def sort_profiles(profile_list: list["ProfileInfo"]) -> None:
    """Sorts the profiles according to the key and ordering set in configuration.

    The key can either be specified in temporary configuration, or in any of the local or global
    configs as the key :ckey:`format.sort_profiles_by` attributes. Be default, profiles are sorted
    by time of creation, resp. modification. In case of any errors (invalid sort key or missing key)
    the profiles will be sorted by the default key instead.

    The profiles may further be sorted either in ascending or descending order w.r.t. the sort key.
    By default, the profiles are sorted in ascending order so the profiles are shown in the
    order from least recent to most recent (with the default sort key being time). This ensures
    that, by default, the numeric profile tags do not change when new profiles are created.
    The sort order can again be specified in the temporary, local or global configs using the key
    :ckey:`format.sort_profiles_order` attribute.

    :param profile_list: list of ProfileInfo object
    """
    sort_key = config.safely_lookup_key_recursively(
        "format.sort_profiles_by", ProfileInfo.valid_attributes, DEFAULT_SORT_KEY
    )
    sort_order = config.safely_lookup_key_recursively(
        "format.sort_profiles_order", SortOrder.supported(), SortOrder.default()
    )
    profile_list.sort(
        key=operator.attrgetter(sort_key), reverse=SortOrder(sort_order).as_sort_flag()
    )


def merge_resources_of(
    lhs: profiles.Profile | dict[str, Any], rhs: profiles.Profile | dict[str, Any]
) -> profiles.Profile:
    """Merges the resources of lhs and rhs profiles

    :param lhs: left operator of the profile merge
    :param rhs: right operator of the profile merge
    :return: profile with merged resources
    """
    # Not Good: Temporary solution:
    lhs = common_kit.ensure_type(lhs, profiles.Profile)
    rhs = common_kit.ensure_type(rhs, profiles.Profile)

    # Return lhs/rhs if rhs/lhs is empty
    if rhs.resources_size() == 0:
        return lhs
    elif lhs.resources_size() == 0:
        return rhs

    lhs_res = [res[1] for res in lhs.all_resources()] if lhs else []
    rhs_res = [res[1] for res in rhs.all_resources()] if rhs else []
    lhs_res.extend(rhs_res)
    lhs.update_resources(lhs_res, clear_existing_resources=True)

    return lhs


def _get_default_variable(profile: profiles.Profile, supported_variables: list[str]) -> str:
    """Helper function that determines default variable for profile based on list of supported
    variables.

    Note that this returns the first suitable candidate, so it is expected that supported_variables
    are sorted by their priority.

    :param profile: input profile
    :param supported_variables: list of supported fields
    :return: default key picked from the list of supported fields (either for dependent or
        independent variables)
    """
    resource_fields = list(profile.all_resource_fields())
    candidates = [var for var in supported_variables if var in set(resource_fields)]
    if candidates:
        # Return first suitable candidate, according to the given order
        return candidates[0]
    else:
        perun_log.error(
            f"Profile does not contain (in)dependent variable. Has to be one of: ({', '.join(supported_variables)})"
        )


def get_default_independent_variable(profile: profiles.Profile) -> str:
    """Returns default independent variable for the given profile

    :param profile: input profile
    :return: default independent variable
    """
    return _get_default_variable(profile, profiles.Profile.independent)


def get_default_dependent_variable(profile: profiles.Profile) -> str:
    """Returns default dependent variable for the given profile

    :param profile: input profile
    :return: default dependent variable
    """
    return _get_default_variable(profile, profiles.Profile.dependent)


class ProfileInfo:
    """Structure for storing information about profiles.

    This is mainly used for formatted output of the profile list using
    the command line interface
    """

    __slots__ = [
        "_is_raw_profile",
        "source",
        "realpath",
        "time",
        "type",
        "cmd",
        "workload",
        "label",
        "collector",
        "postprocessors",
        "checksum",
        "config_tuple",
    ]

    def __init__(
        self,
        path: str,
        real_path: str,
        mtime: str,
        profile_info: profiles.Profile | dict[str, Any],
        is_raw_profile: bool = False,
    ) -> None:
        """
        :param path: contains the name of the file, which identifies it in the index
        :param real_path: real path to the profile, i.e. how can it really be accessed
            this is either in jobs, in objects or somewhere else
        :param mtime: time of the modification of the profile
        :param is_raw_profile: true if the stored profile is raw, i.e. in json and not
            compressed
        """
        self._is_raw_profile = is_raw_profile
        self.source = path
        self.realpath = os.path.relpath(real_path, os.getcwd())
        self.time = mtime
        self.type = profile_info["header"]["type"]
        self.cmd = profile_info["header"]["cmd"]
        self.workload = profile_info["header"]["workload"]
        self.label = profile_info["header"].get("label", "")
        self.collector = profile_info["collector_info"]["name"]
        self.postprocessors = [
            postprocessor["name"] for postprocessor in profile_info["postprocessors"]
        ]
        self.checksum = None
        self.config_tuple = (
            self.collector,
            self.cmd,
            self.workload,
            ",".join(self.postprocessors),
        )

    def load(self) -> profiles.Profile:
        """Loads the profile from given file

        This is basically a wrapper that loads the profile, whether it is raw (i.e. in pending)
        or not raw and stored in index

        :return: loaded profile in dictionary format, w.r.t :ref:`profile-spec`
        """
        return store.load_profile_from_file(self.realpath, self._is_raw_profile)

    def is_compatible_with_profile(self, profile: profiles.Profile) -> bool:
        """Tests if the profile info is compatible with other profile

        :param profile: profile, that is compared with this profile
        :return: true if this profile is compatible with the other profile
        """
        profile_postprocessors = {
            postprocessor["name"] for postprocessor in profile["postprocessors"]
        }
        return (
            self.type == profile["header"]["type"]
            and self.cmd == profile["header"]["cmd"]
            and self.workload == profile["header"]["workload"]
            and self.collector == profile["collector_info"]["name"]
            and all(post in profile_postprocessors for post in self.postprocessors)
        )

    valid_attributes: list[str] = [
        "realpath",
        "type",
        "time",
        "cmd",
        "workload",
        "label",
        "collector",
        "checksum",
        "source",
    ]


@dataclasses.dataclass
class ProfileHeaderEntry:
    """A representation of a single profile header entry.

    :ivar name: the name (key) of the header entry
    :ivar value: the value of the header entry
    :ivar description: detailed description of the header entry
    :ivar details: nested key: value data
    """

    name: str
    value: str | float
    description: str = ""
    details: dict[str, str | float] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_string(cls, header: str) -> ProfileHeaderEntry:
        """Constructs a `ProfileHeaderRecord` object from a string representation.

        :param header: the string representation of a header entry

        :return: the constructed ProfileHeaderRecord object
        """
        split = header.split("|")
        name = split[0]
        value = common_kit.try_convert(split[1] if len(split) > 1 else "[empty]", [float, str])
        desc = split[2] if len(split) > 2 else ProfileHeaderEntry.description
        details: dict[str, str | float] = {}
        for detail in split[4:]:
            detail_key, detail_value = detail.split(maxsplit=1)
            details[detail_key] = common_kit.try_convert(detail_value, [float, str])
        return cls(name, value, desc, details)

    @classmethod
    def from_profile(cls, header: dict[str, Any]) -> ProfileHeaderEntry:
        """Constructs a ProfileHeaderEntry object from a dictionary representation used in Profile.

        :param header: the dictionary representation of a header entry

        :return: the constructed ProfileHeaderEntry object
        """
        return cls(**header)

    def as_tuple(self) -> ProfileHeaderTuple:
        """Converts the header object into a tuple.

        :return: the tuple representation of a header entry
        """
        return self.name, self.value, self.description, self.details
