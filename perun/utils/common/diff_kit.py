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
    """Saves the content to the output file; if no output file is stated, then it is automatically generated

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
    machine_info = profile.get("machine", {})
    return [
        (
            "origin",
            profile.get("origin", "?"),
            "The version control version, for which the profile was measured.",
        ),
        ("command", command, "The workload / command, for which the profile was measured."),
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
                f'<span style="color: {tag_to_color.get(start_tag, "grey")}; font-weight: bold">{chunk[2:]}</span>'
            )
    return " ".join(result)


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
            and f"Configuration keys in headers are wrongly order (expected {lhs_key}; got {rhs_key})"
        )
        if lhs_value != rhs_value:
            diff = list(difflib.ndiff(lhs_value.split(), rhs_value.split()))
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
