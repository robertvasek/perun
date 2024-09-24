"""Set of helper functions for working with perun.diff_view"""

from __future__ import annotations

# Standard Imports
import dataclasses
import difflib
import enum
import os
from typing import Any, Optional, Iterable, Literal

# Third-Party Imports

# Perun Imports
from perun.profile import helpers
from perun.profile.factory import Profile
from perun.profile import stats as pstats
from perun.utils import log
from perun.utils.common import common_kit
from perun.utils.common.common_kit import ColorChoiceType


class MetadataDisplayStyle(enum.Enum):
    """Supported styles of displaying metadata."""

    FULL = "full"
    DIFF = "diff"

    @staticmethod
    def supported() -> list[str]:
        """Obtain the collection of supported metadata display styles.

        :return: the collection of valid display styles
        """
        return [style.value for style in MetadataDisplayStyle]

    @staticmethod
    def default() -> str:
        """Provide the default metadata display style.

        :return: the default display style
        """
        return MetadataDisplayStyle.FULL.value


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


def generate_diff_of_headers(
    lhs_header: list[tuple[str, Any, str]], rhs_header: list[tuple[str, Any, str]]
) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]]]:
    """Regenerates header entries with HTML diff styles suitable for an output.

    :param lhs_header: header for baseline
    :param rhs_header: header for target

    :return: pair of header entries for lhs (baseline) and rhs (target)
    """
    lhs_diff, rhs_diff = [], []
    for lhs_record, rhs_record in zip(lhs_header, rhs_header):
        _, lhs_diff_record, rhs_diff_record = generate_diff_of_header_record(lhs_record, rhs_record)
        lhs_diff.append(lhs_diff_record)
        rhs_diff.append(rhs_diff_record)
    return lhs_diff, rhs_diff


def generate_diff_of_metadata(
    lhs_metadata: Iterable[helpers.ProfileMetadata],
    rhs_metadata: Iterable[helpers.ProfileMetadata],
    display_style: MetadataDisplayStyle,
) -> tuple[list[tuple[str, Any, str]], list[tuple[str, Any, str]]]:
    """Generates metadata entries with HTML diff styles suitable for an output.

    :param lhs_metadata: a collection of metadata entries for baseline
    :param rhs_metadata: a collection of metadata entries for target
    :param display_style: the metadata display style; DIFF produces only entries that have diffs

    :return: pair of metadata entries for lhs (baseline) and rhs (target)
    """
    metadata_map = common_kit.match_dicts_by_keys(
        {data.name: data for data in lhs_metadata}, {data.name: data for data in rhs_metadata}
    )
    lhs_list, rhs_list = [], []
    for metadata_key in sorted(metadata_map.keys()):
        lhs_data: helpers.ProfileMetadata | None
        rhs_data: helpers.ProfileMetadata | None
        lhs_data, rhs_data = metadata_map[metadata_key]
        lhs_tuple = (
            lhs_data.as_tuple()
            if lhs_data is not None
            else (metadata_key, "-", "missing metadata info")
        )
        rhs_tuple = (
            rhs_data.as_tuple()
            if rhs_data is not None
            else (metadata_key, "-", "missing metadata info")
        )
        if lhs_data is not None and rhs_data is not None:
            is_diff, lhs_tuple, rhs_tuple = generate_diff_of_header_record(lhs_tuple, rhs_tuple)
            if display_style == MetadataDisplayStyle.DIFF and not is_diff:
                # We wish to display only differing metadata, skip this one
                continue
        lhs_list.append(lhs_tuple)
        rhs_list.append(rhs_tuple)
    return lhs_list, rhs_list


def generate_diff_of_stats(
    lhs_stats: Iterable[pstats.ProfileStat], rhs_stats: Iterable[pstats.ProfileStat]
) -> tuple[list[tuple[str, str, str, dict[str, Any]]], list[tuple[str, str, str, dict[str, Any]]]]:
    """Generates the profile stats with HTML diff styles suitable for an output.

    :param lhs_stats: stats from the baseline
    :param rhs_stats: stats from the target
    :return: collection of LHS and RHS stats (stat-description, stat-value, stat-description)
             with HTML styles that reflect the stat diffs.
    """
    # Get all the stats that occur in either lhs or rhs and match those that exist in both
    stats_map = common_kit.match_dicts_by_keys(
        {stats.name: stats for stats in lhs_stats}, {stats.name: stats for stats in rhs_stats}
    )
    # Iterate the stats and format them according to their diffs
    lhs_diff, rhs_diff = [], []
    for stat_key in sorted(stats_map.keys()):
        lhs_stat: pstats.ProfileStat | None
        rhs_stat: pstats.ProfileStat | None
        lhs_stat, rhs_stat = stats_map[stat_key]
        lhs_info, rhs_info = generate_diff_of_stats_record(lhs_stat, rhs_stat)
        lhs_diff.append(lhs_info)
        rhs_diff.append(rhs_info)
    return lhs_diff, rhs_diff


def generate_diff_of_header_record(
    lhs_record: tuple[str, Any, str], rhs_record: tuple[str, Any, str]
) -> tuple[bool, tuple[str, Any, str], tuple[str, Any, str]]:
    """Generates a single diffed LHS and RHS header entry.

    :param lhs_record: the LHS (baseline) entry
    :param rhs_record: the RHS (target) entry

    :return: pair of a single diffed LHS (baseline) and RHS (target) header entry
    """
    (lhs_key, lhs_value, lhs_info), (rhs_key, rhs_value, _) = lhs_record, rhs_record
    assert (
        lhs_key == rhs_key
        and f"Configuration keys in headers are wrongly ordered (expected {lhs_key}; got {rhs_key})"
    )
    if lhs_value != rhs_value:
        diff = list(difflib.ndiff(str(lhs_value).split(), str(rhs_value).split()))
        key = _emphasize(lhs_key, "red")
        return (
            True,
            (key, diff_to_html(diff, "-"), lhs_info),
            (key, diff_to_html(diff, "+"), lhs_info),
        )
    else:
        return False, (lhs_key, lhs_value, lhs_info), (rhs_key, rhs_value, lhs_info)


def generate_diff_of_stats_record(
    lhs_stat: pstats.ProfileStat | None, rhs_stat: pstats.ProfileStat | None
) -> tuple[tuple[str, str, str, dict[str, Any]], tuple[str, str, str, dict[str, Any]]]:
    """Generates a single diffed LHS and RHS profile stats entry.

    :param lhs_stat: the LHS (baseline) stats entry
    :param rhs_stat: the RHS (target) stats entry

    :return: pair of a single diffed LHS (baseline) and RHS (target) stats entry
    """
    lhs_diff = _StatsDiffRecord.from_stat(lhs_stat, rhs_stat)
    rhs_diff = _StatsDiffRecord.from_stat(rhs_stat, lhs_stat)
    if lhs_diff.stat_agg is not None and rhs_diff.stat_agg is not None:
        assert lhs_stat is not None
        lhs_diff.value, rhs_diff.value = _color_stat_record_diff(
            lhs_diff.stat_agg,
            rhs_diff.stat_agg,
            lhs_stat.aggregate_by,
            lhs_diff.stat_agg.infer_auto_comparison(lhs_stat.cmp),
        )
    return lhs_diff.to_tuple(), rhs_diff.to_tuple()


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
            result.append(_emphasize(chunk[2:], tag_to_color.get(start_tag, "grey")))
    return " ".join(result)


def stat_description_to_tooltip(description: str, comparison: pstats.ProfileStatComparison) -> str:
    """Transform a stat description into a tooltip by including the comparison type as well.

    :param description: the original stat description
    :param comparison: the comparison type of the stat

    :return: the generated tooltip
    """
    comparison_str: str = f'[{comparison.value.replace("_", " ")}]'
    return f"{description} {comparison_str}"


def _emphasize(value: str, color: ColorChoiceType) -> str:
    """Emphasize a string with a HTML color.

    :param value: the string to emphasize
    :param color: the color to use

    :return: the emphasized string
    """
    return f'<span style="color: {color}; font-weight: bold">{value}</span>'


def _format_exit_codes(exit_code: str | list[str] | list[int]) -> str:
    """Format (a collection of) exit code(s) for HTML output.

    Exit codes that are non-zero will be emphasized with a color.

    :param exit_code: the exit code(s)

    :return: the formatted exit codes
    """
    # Unify the exit code types
    exit_codes: list[str] = []
    if isinstance(exit_code, (str, int)):
        exit_codes.append(str(exit_code))
    else:
        exit_codes = list(map(str, exit_code))
    # Color exit codes that are not zero
    return ", ".join(code if code == "0" else _emphasize(code, "red") for code in exit_codes)


def _color_stat_record_diff(
    lhs_stat_agg: pstats.ProfileStatAggregation,
    rhs_stat_agg: pstats.ProfileStatAggregation,
    compare_key: str,
    comparison: pstats.ProfileStatComparison,
) -> tuple[str, str]:
    """Color the stats values on the LHS and RHS according to their difference.

    The color is determined by the stat comparison type and the comparison result.

    :param lhs_stat_agg: a baseline stat aggregation
    :param rhs_stat_agg: a target stat aggregation
    :param compare_key: the key by which to compare the stats
    :param comparison: the comparison type of the stat

    :return: colored LHS and RHS stat values
    """
    # Build a color map for different comparison results
    color_map: dict[pstats.StatComparisonResult, tuple[ColorChoiceType, ColorChoiceType]] = {
        pstats.StatComparisonResult.INVALID: ("red", "red"),
        pstats.StatComparisonResult.UNEQUAL: ("red", "red"),
        pstats.StatComparisonResult.EQUAL: ("black", "black"),
        pstats.StatComparisonResult.BASELINE_BETTER: ("green", "red"),
        pstats.StatComparisonResult.TARGET_BETTER: ("red", "green"),
    }
    # Compare and color the stat entry
    comparison_result = pstats.compare_stats(lhs_stat_agg, rhs_stat_agg, compare_key, comparison)
    baseline_color, target_color = color_map[comparison_result]
    if comparison_result == pstats.StatComparisonResult.INVALID:
        baseline_value, target_value = "invalid comparison", "invalid comparison"
    else:
        baseline_value = _format_stat_value(lhs_stat_agg.as_table(compare_key)[0])
        target_value = _format_stat_value(rhs_stat_agg.as_table(compare_key)[0])
    return _emphasize(baseline_value, baseline_color), _emphasize(target_value, target_color)


def _format_stat_value(value: str | float | tuple[str, int]) -> str:
    """Formats float stat values to have a fixed number of decimal digits.

    Non-float stat values are kept as is.

    :param value: the value to format.

    :return: the formatted value.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


@dataclasses.dataclass
class _StatsDiffRecord:
    """A helper struct for storing a difference entry of baseline/target stats.

    :ivar name: name of the stats entry
    :ivar value: the value of the stats entry
    :ivar tooltip: the tooltip of the stats entry
    :ivar stat_agg: the stats aggregation
    """

    name: str = ""
    value: str = "-"
    tooltip: str = "missing stat info"
    stat_agg: pstats.ProfileStatAggregation | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_stat(
        cls, stat: pstats.ProfileStat | None, other_stat: pstats.ProfileStat | None
    ) -> _StatsDiffRecord:
        """Construct a difference record from a baseline/target profile stat.

        If the baseline/target profile stat is not available, we use the other
        (i.e., target/baseline) stat to construct a stub entry to indicate a missing stat.

        :param stat: the baseline/target profile stat to construct from, if available
        :param other_stat: the other profile stat to use for construction as a fallback

        :return: the constructed difference record
        """
        if stat is None:
            # Fallback construction from the other stat
            assert other_stat is not None
            unit = f" [{other_stat.unit}]" if other_stat.unit else ""
            return cls(f"{other_stat.name}{unit}")
        # The standard construction
        stat_agg = pstats.aggregate_stats(stat)
        unit = f" [{stat.unit}]" if stat.unit else ""
        agg_key = stat_agg.normalize_aggregate_key(stat.aggregate_by)
        name = f"{stat.name}{unit} " f"({agg_key})"
        value, details = stat_agg.as_table(agg_key)
        tooltip = stat_description_to_tooltip(
            stat.description, stat_agg.infer_auto_comparison(stat.cmp)
        )
        return cls(name, str(value), tooltip, stat_agg, details)

    def to_tuple(self) -> tuple[str, str, str, dict[str, Any]]:
        """Convert the difference record to a tuple.

        :return: the tuple representation of the difference record
        """
        return self.name, self.value, self.tooltip, self.details
