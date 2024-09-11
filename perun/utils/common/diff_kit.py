"""Set of helper functions for working with perun.diff_view"""

from __future__ import annotations

# Standard Imports
import dataclasses
import difflib
import os
from typing import Any, Optional, Iterable, Literal, cast

# Third-Party Imports

# Perun Imports
from perun.profile import helpers
from perun.profile.factory import Profile
from perun.profile import stats as pstats
from perun.utils import log
from perun.utils.common.common_kit import ColorChoiceType


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
    exitcode = _format_exit_codes(profile["header"].get("exitcode", "?"))
    machine_info = profile.get("machine", {})
    return [
        (
            "origin",
            profile.get("origin", "?"),
            "The version control version, for which the profile was measured.",
        ),
        ("command", command, "The workload / command, for which the profile was measured."),
        ("exitcode", exitcode, "The exit code that was returned by the underlying command."),
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
    tag_to_color: dict[str, ColorChoiceType] = {
        "+": "green",
        "-": "red",
    }
    result = []
    for chunk in diff:
        if chunk.startswith("  "):
            result.append(chunk[2:])
        if chunk.startswith(start_tag):
            result.append(_emphasize(tag_to_color.get(start_tag, "grey"), chunk[2:]))
    return " ".join(result)


def generate_diff_of_stats(
    lhs_stats: list[pstats.ProfileStat], rhs_stats: list[pstats.ProfileStat]
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Re-generate the stats with CSS diff styles suitable for an output.

    :param lhs_stats: stats from the baseline
    :param rhs_stats: stats from the target
    :return: collection of LHS and RHS stats (stat-name [stat-unit], stat-value, stat-tooltip)
             with CSS styles that reflect the stat diffs.
    """
    # Get all the stats that occur in either lhs or rhs and match those that exist in both
    stats_map: dict[str, dict[str, pstats.ProfileStat]] = {}
    for stat_source, stat_list in [("lhs", lhs_stats), ("rhs", rhs_stats)]:
        for stat in stat_list:
            stats_map.setdefault(stat.name, {})[stat_source] = stat
    # Iterate the stats and format them according to their diffs
    lhs_diff, rhs_diff = [], []
    for stat_key in sorted(stats_map.keys()):
        lhs_stat: pstats.ProfileStat | None = stats_map[stat_key].get("lhs", None)
        rhs_stat: pstats.ProfileStat | None = stats_map[stat_key].get("rhs", None)
        lhs_diff.append(_generate_stat_diff_record(lhs_stat, rhs_stat))
        rhs_diff.append(_generate_stat_diff_record(rhs_stat, lhs_stat))
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
            key = _emphasize("red", lhs_key)
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


def generate_metadata(
    lhs_profile: Profile, rhs_profile: Profile
) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]]]:
    """Generates metadata for lhs and rhs profile

    :param lhs_profile: profile for baseline
    :param rhs_profile: profile for target
    :return: pair of metadata for lhs (baseline) and rhs (target)
    """
    data: dict[str, Any]

    lhs_metadata = sorted(
        [
            helpers.ProfileMetadata.from_profile(data).as_tuple()
            for data in lhs_profile.get("metadata", [])
        ],
        key=lambda x: x[0],
    )
    rhs_metadata = sorted(
        [
            helpers.ProfileMetadata.from_profile(data).as_tuple()
            for data in rhs_profile.get("metadata", [])
        ],
        key=lambda x: x[0],
    )
    return generate_diff_of_headers(lhs_metadata, rhs_metadata)


def normalize_stat_tooltip(tooltip: str, ordering: pstats.ProfileStatOrdering) -> str:
    ordering_str: str = f'[{ordering.value.replace("_", " ")}]'
    return f"{tooltip} {ordering_str}"


def _emphasize(color: ColorChoiceType, value: str) -> str:
    return f'<span style="color: {color}; font-weight: bold">{value}</span>'


def _format_exit_codes(exit_code: str | list[str] | list[int]) -> str:
    # Unify the exit code types
    exit_codes: list[str] = []
    if isinstance(exit_code, str):
        exit_codes.append(exit_code)
    else:
        exit_codes = list(map(str, exit_code))
    # Color exit codes that are not zero
    return ", ".join(code if code == "0" else _emphasize("red", code) for code in exit_codes)


def _generate_stat_diff_record(
    stat: pstats.ProfileStat | None, other_stat: pstats.ProfileStat | None
) -> tuple[str, str, str]:
    if stat is None:
        # The stat is missing, use some info from the other stat
        assert other_stat is not None
        return f"{other_stat.name} [{other_stat.unit}]", "-", "missing stat info"
    else:
        stat_agg = pstats.aggregate_stats(stat)
        tooltip = normalize_stat_tooltip(stat.tooltip, stat_agg.infer_auto_ordering(stat.ordering))
        return (
            f"{stat.name} [{stat.unit}] ({stat_agg.normalize_aggregate_key(stat.aggregate_by)})",
            str(stat_agg.as_table()[0]),
            tooltip,
        )


def _color_stat_value_diff(
    lhs_stat_agg: pstats.ProfileStatAggregation,
    rhs_stat_agg: pstats.ProfileStatAggregation,
    aggregate_key: str,
    ordering: pstats.ProfileStatOrdering,
) -> tuple[str, str]:
    """Color the stat values on the LHS and RHS according to their difference.

    The color is determined by the stat ordering and the result of the stat values comparison.

    :param lhs_stat: a stat from the baseline
    :param rhs_stat: a stat from the target
    :return: colored LHS and RHS stat values
    """
    comparison_result = pstats.compare_stats(lhs_stat_agg, rhs_stat_agg, aggregate_key, ordering)
    color_map: dict[pstats.StatComparisonResult, tuple[ColorChoiceType, ColorChoiceType]] = {
        pstats.StatComparisonResult.INVALID: ("red", "red"),
        pstats.StatComparisonResult.UNEQUAL: ("red", "red"),
        pstats.StatComparisonResult.EQUAL: ("black", "black"),
        pstats.StatComparisonResult.BASELINE_BETTER: ("green", "red"),
        pstats.StatComparisonResult.TARGET_BETTER: ("red", "green"),
    }

    baseline_color, target_color = color_map[comparison_result]
    if comparison_result == pstats.StatComparisonResult.INVALID:
        baseline_value, target_value = "<invalid comparison>", "<invalid comparison>"
    else:
        baseline_value = str(lhs_stat_agg.as_table()[0])
        target_value = str(rhs_stat_agg.as_table()[0])
    return _emphasize(baseline_color, baseline_value), _emphasize(target_color, target_value)
