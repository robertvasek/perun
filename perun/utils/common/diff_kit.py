"""Set of helper functions for working with perun.diff_view"""

from __future__ import annotations


# Standard Imports
from typing import Any, Optional, Iterable, Literal
import difflib
import os

# Third-Party Imports

# Perun Imports
from perun.profile import helpers
from perun.profile.factory import Profile
from perun.utils import log


def save_diff_view(
    output_file: Optional[str],
    content: str,
    output_type: str,
    lhs_profile: Profile,
    rhs_profile: Profile,
) -> str:
    """Saves the content to the output file; if no output file is stated, then it is automatically
    generated.

    :param output_file: file, where the content will be stored
    :param content: content of the output file
    :param output_type: type of the output
    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :return: name of the output file
    """
    if output_file is None:
        lhs_name = os.path.splitext(helpers.generate_profile_name(lhs_profile))[0]
        rhs_name = os.path.splitext(helpers.generate_profile_name(rhs_profile))[0]
        output_file = f"{output_type}-diff-of-{lhs_name}-and-{rhs_name}" + ".html"

    if not output_file.endswith("html"):
        output_file += ".html"

    with open(output_file, "w", encoding="utf-8") as template_out:
        template_out.write(content)

    return output_file


def get_candidate_keys(candidate_keys: Iterable[str]) -> list[str]:
    """Returns list of candidate keys

    :param candidate_keys: list of candidate keys
    :return: list of supported keys
    """
    allowed_keys = [
        "Total Exclusive T [ms]",
        "Total Inclusive T [ms]",
        "amount",
        "ncalls",
        # "E Min",
        # "E Max",
        # "I Min",
        # "I Max",
        # "Callees Mean [#]",
        # "Callees [#]",
        # "Total Inclusive T [%]",
        # "Total Exclusive T [%]",
        # "I Mean",
        # "E Mean",
    ]
    return sorted([candidate for candidate in candidate_keys if candidate in allowed_keys])


def generate_header(profile: Profile) -> list[tuple[str, Any, str]]:
    """Generates header for given profile

    :param profile: profile for which we are generating the header
    :return: list of tuples (key and value)
    """
    command = " ".join([profile["header"]["cmd"], profile["header"]["workload"]]).strip()
    exitcode = profile["header"].get("exitcode", "?")
    machine_info = profile.get("machine", {})
    return [
        (
            "origin",
            profile.get("origin", "?"),
            "The version control version, for which the profile was measured.",
        ),
        ("command", command, "The workload / command, for which the profile was measured."),
        ("exitcode", exitcode, "The maximal exit code that was returned by underlying command."),
        (
            "collector command",
            log.collector_to_command(profile.get("collector_info", {})),
            "The collector / profiler, which collected the data.",
        ),
        (
            "kernel",
            machine_info.get("release", "?"),
            "The underlying kernel version, where the results were measured.",
        ),
        (
            "boot info",
            machine_info.get("boot_info", "?"),
            "The contents of `/proc/cmdline` containing boot information about kernel",
        ),
        ("host", machine_info["host"], "The hostname, where the results were measured."),
        (
            "cpu (total)",
            machine_info.get("cpu", {"total": "?"}).get("total", "?"),
            "The total number (physical and virtual) of CPUs available on the host.",
        ),
        (
            "memory (total)",
            machine_info.get("memory", {"total_ram": "?"}).get("total_ram", "?"),
            "The total number of RAM available on the host.",
        ),
    ]


def diff_to_html(diff: list[str], start_tag: Literal["+", "-"]) -> str:
    """Create a html tag with differences for either + or - changes

    :param diff: diff computed by difflib.ndiff
    :param start_tag: starting point of the tag
    """
    tag_to_color = {
        "+": "green",
        "-": "red",
    }
    result = []
    for chunk in diff:
        if chunk.startswith("  "):
            result.append(chunk[2:])
        if chunk.startswith(start_tag):
            result.append(
                f'<span style="color: {tag_to_color.get(start_tag, "grey")}; '
                f'font-weight: bold">{chunk[2:]}</span>'
            )
    return " ".join(result)


def _color_stat_value_diff(
    lhs_stat: helpers.ProfileStat, rhs_stat: helpers.ProfileStat
) -> tuple[str, str]:
    """Color the stat values on the LHS and RHS according to their difference.

    The color is determined by the stat ordering and the result of the stat values comparison.

    :param lhs_stat: a stat from the baseline
    :param rhs_stat: a stat from the target
    :return: colored LHS and RHS stat values
    """
    # Map the colors based on the value ordering
    color_map: dict[bool, str] = {
        lhs_stat.ordering: "red",
        not lhs_stat.ordering: "green",
    }
    lhs_value, rhs_value = str(lhs_stat.value), str(rhs_stat.value)
    if lhs_stat.ordering != rhs_stat.ordering:
        # Conflicting ordering in baseline and target, do not compare
        log.warn(
            f"Profile stats '{lhs_stat.name}' have conflicting ordering in baseline and target."
            f" The stats will not be compared."
        )
    elif lhs_value != rhs_value:
        # Different stat values, color them
        is_lhs_lower = lhs_stat.value < rhs_stat.value
        lhs_value = (
            f'<span style="color: {color_map[is_lhs_lower]}; '
            f'font-weight: bold">{lhs_value}</span>'
        )
        rhs_value = (
            f'<span style="color: {color_map[not is_lhs_lower]}; '
            f'font-weight: bold">{rhs_value}</span>'
        )
    return lhs_value, rhs_value


def generate_diff_of_stats(
    lhs_stats: list[helpers.ProfileStat], rhs_stats: list[helpers.ProfileStat]
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Re-generate the stats with CSS diff styles suitable for an output.

    :param lhs_stats: stats from the baseline
    :param rhs_stats: stats from the target
    :return: collection of LHS and RHS stats (stat-name [stat-unit], stat-value, stat-tooltip)
             with CSS styles that reflect the stat diffs.
    """

    # Get all the stats that occur in either lhs or rhs and match those that exist in both
    stats_map: dict[str, dict[str, helpers.ProfileStat]] = {}
    for stat_source, stat_list in [("lhs", lhs_stats), ("rhs", rhs_stats)]:
        for stat in stat_list:
            stats_map.setdefault(stat.name, {})[stat_source] = stat
    # Iterate the stats and format them according to their diffs
    lhs_diff, rhs_diff = [], []
    for stat_key in sorted(stats_map.keys()):
        lhs_stat: helpers.ProfileStat | None = stats_map[stat_key].get("lhs", None)
        rhs_stat: helpers.ProfileStat | None = stats_map[stat_key].get("rhs", None)
        lhs_tooltip = lhs_stat.get_normalized_tooltip() if lhs_stat is not None else ""
        rhs_tooltip = rhs_stat.get_normalized_tooltip() if rhs_stat is not None else ""
        if rhs_stat and lhs_stat is None:
            # There is no matching stat on the LHS
            lhs_diff.append((f"{rhs_stat.name} [{rhs_stat.unit}]", "-", rhs_tooltip))
            rhs_diff.append(
                (f"{rhs_stat.name} [{rhs_stat.unit}]", str(rhs_stat.value), rhs_tooltip)
            )
        elif lhs_stat and rhs_stat is None:
            # There is no matching stat on the RHS
            lhs_diff.append(
                (f"{lhs_stat.name} [{lhs_stat.unit}]", str(lhs_stat.value), lhs_tooltip)
            )
            rhs_diff.append((f"{lhs_stat.name} [{lhs_stat.unit}]", "-", lhs_tooltip))
        elif lhs_stat and rhs_stat:
            # The stat is present on both LHS and RHS
            lhs_value, rhs_value = _color_stat_value_diff(lhs_stat, rhs_stat)
            lhs_diff.append((f"{lhs_stat.name} [{lhs_stat.unit}]", lhs_value, lhs_tooltip))
            rhs_diff.append((f"{rhs_stat.name} [{rhs_stat.unit}]", rhs_value, rhs_tooltip))
    return lhs_diff, rhs_diff


def generate_diff_of_headers(
    lhs_header: list[tuple[str, Any, str]], rhs_header: list[tuple[str, Any, str]]
) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]]]:
    """Regenerates header with differences between individual parts of the info

    :param lhs_header: header for baseline
    :param rhs_header: header for target
    """
    lhs_diff, rhs_diff = [], []
    for (lhs_key, lhs_value, lhs_info), (rhs_key, rhs_value, _) in zip(lhs_header, rhs_header):
        assert (
            lhs_key == rhs_key
            and f"Configuration keys in headers are wrongly ordered (expected {lhs_key}; "
            f"got {rhs_key})"
        )
        if lhs_value != rhs_value:
            diff = list(difflib.ndiff(str(lhs_value).split(), str(rhs_value).split()))
            key = f'<span style="color: red; font-weight: bold">{lhs_key}</span>'
            lhs_diff.append((key, diff_to_html(diff, "-"), lhs_info))
            rhs_diff.append((key, diff_to_html(diff, "+"), lhs_info))
        else:
            lhs_diff.append((lhs_key, lhs_value, lhs_info))
            rhs_diff.append((rhs_key, rhs_value, lhs_info))
    return lhs_diff, rhs_diff


def generate_headers(
    lhs_profile: Profile, rhs_profile: Profile
) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]]]:
    """Generates headers for lhs and rhs profile

    :param lhs_profile: profile for baseline
    :param rhs_profile: profile for target
    :return: pair of headers for lhs (baseline) and rhs (target)
    """
    lhs_header = generate_header(lhs_profile)
    rhs_header = generate_header(rhs_profile)
    return generate_diff_of_headers(lhs_header, rhs_header)
