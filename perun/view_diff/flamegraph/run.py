"""Flamegraph difference of the profile"""
from __future__ import annotations

# Standard Imports
from typing import Any
import re

# Third-Party Imports
import click
import jinja2

# Perun Imports
from perun.utils import log
from perun.utils.common import diff_kit
from perun.profile.factory import Profile
from perun.profile import convert
from perun.view.flamegraph import flamegraph as flamegraph_factory
from perun.view_diff.table import run as table_run


DEFAULT_HEIGHT: int = 14
DEFAULT_WIDTH: int = 600
TAGS_TO_INDEX: list[str] = []


def escape_content(tag: str, content: str) -> str:
    """Escapes content, so there are no clashes in the files

    :param tag: tag used to prefix all the functions and ids
    :param content: generated svg content
    :return: escaped content
    """
    if tag not in TAGS_TO_INDEX:
        TAGS_TO_INDEX.append(tag)
    functions = [
        r"(?<!\w)(c)\(",
        r"(?<!\w)(find_child)\(",
        r"(?<!\w)(g_to_func)\(",
        r"(?<!\w)(g_to_text)\(",
        r"(?<!\w)(init)\(",
        r"(?<!\w)(orig_load)\(",
        r"(?<!\w)(orig_save)\(",
        r"(?<!\w)(reset_search)\(",
        r"(?<!\w)(s)\(",
        r"(?<!\w)(search)\(",
        r"(?<!\w)(search_prompt)\(",
        r"(?<!\w)(searchout)\(",
        r"(?<!\w)(searchover)\(",
        r"(?<!\w)(unzoom)\(",
        r"(?<!\w)(update_text)\(",
        r"(?<!\w)(zoom)\(",
        r"(?<!\w)(zoom_child)\(",
        r"(?<!\w)(zoom_parent)\(",
        r"(?<!\w)(zoom_reset)\(",
    ]
    other = [
        (r"func_g", f"{tag}_func_g"),
        (r"\"unzoom\"", f'"{tag}_unzoom"'),
        (r"\"search\"", f'"{tag}_search"'),
        (r"\"matched\"", f'"{tag}_matched"'),
        (r"details", f"{tag}_details"),
        (r"searchbtn", f"{tag}_searchbtn"),
        (r"searching", f"{tag}_searching"),
        (r"matchedtxt", f"{tag}_matchedtxt"),
        (r"svg\.", f"{tag}_svg."),
        (r"svg =", f"{tag}_svg ="),
        (r"svg;", f"{tag}_svg;"),
        (r"\[0\]", f"[{TAGS_TO_INDEX.index(tag)}]"),
        (r"document.", f"{tag}_svg."),
        (f"({tag}_(svg|details|searchbtn|matchedtxt)) = {tag}_svg.", "\\1 = document."),
        # Huge thanks to following article:
        # https://chartio.com/resources/tutorials/how-to-resize-an-svg-when-the-window-is-resized-in-d3-js/
        # Which helped to solve the issue with non-resizable flamegraphs
        (
            '<svg version="1.1" width="[0-9]+" height="[0-9]+"',
            '<svg version="1.1" preserveAspectRatio="xMinYMin meet" class="svg-content"',
        ),
    ]
    for func in functions:
        content = re.sub(func, f"{tag}_\\1(", content)
    for unit, sub in other:
        content = re.sub(unit, sub, content)
    return content


def get_uids(profile: Profile) -> set[str]:
    """For given profile return set of uids

    :param profile: profile
    :return: set of unique uids in profile
    """
    df = convert.resources_to_pandas_dataframe(profile)
    return set(df["uid"].unique())


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


def generate_flamegraph_difference(
    lhs_profile: Profile, rhs_profile: Profile, **kwargs: Any
) -> None:
    """Generates differences of two profiles as two side-by-side flamegraphs

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :param kwargs: additional arguments
    """
    lhs_types = list(lhs_profile.all_resource_fields())
    rhs_types = list(rhs_profile.all_resource_fields())
    data_types = diff_kit.get_candidate_keys(set(lhs_types).union(set(rhs_types)))
    data_type = list(data_types)[0]

    log.major_info("Generating Flamegraph Difference")
    flamegraphs = []
    for i, data_type in enumerate(data_types):
        lhs_graph = flamegraph_factory.draw_flame_graph(
            lhs_profile,
            kwargs.get("height", DEFAULT_HEIGHT),
            kwargs.get("width", DEFAULT_WIDTH),
            title="Baseline Flamegraph",
            profile_key=data_type,
        )
        escaped_lhs = escape_content(f"lhs_{i}", lhs_graph)
        log.minor_success(f"Baseline flamegraph ({data_type})", "generated")

        rhs_graph = flamegraph_factory.draw_flame_graph(
            rhs_profile,
            kwargs.get("height", DEFAULT_HEIGHT),
            kwargs.get("width", DEFAULT_WIDTH),
            title="Target Flamegraph",
            profile_key=data_type,
        )
        escaped_rhs = escape_content(f"rhs_{i}", rhs_graph)
        log.minor_success(f"Target flamegraph ({data_type})", "generated")

        diff_graph = flamegraph_factory.draw_flame_graph_difference(
            lhs_profile,
            rhs_profile,
            kwargs.get("height", DEFAULT_HEIGHT),
            kwargs.get("width", DEFAULT_WIDTH),
            title="Difference Flamegraph",
            profile_key=data_type,
        )
        escaped_diff = escape_content(f"diff_{i}", diff_graph)
        log.minor_success(f"Diff flamegraph ({data_type})", "generated")
        flamegraphs.append((escaped_lhs, escaped_rhs, escaped_diff))

    env = jinja2.Environment(loader=jinja2.PackageLoader("perun", "templates"))
    template = env.get_template("diff_view_flamegraph.html.jinja2")
    content = template.render(
        flamegraphs=flamegraphs,
        lhs_header=generate_header(lhs_profile),
        lhs_tag="Baseline (base)",
        lhs_top=table_run.get_top_n_records(lhs_profile, top_n=10, aggregated_key=data_type),
        lhs_uids=get_uids(lhs_profile),
        rhs_header=generate_header(rhs_profile),
        rhs_tag="Target (tgt)",
        rhs_top=table_run.get_top_n_records(rhs_profile, top_n=10, aggregated_key=data_type),
        rhs_uids=get_uids(rhs_profile),
        title="Differences of profiles (with flamegraphs)",
        data_types=data_types,
    )
    log.minor_success("Difference report", "generated")
    output_file = diff_kit.save_diff_view(
        kwargs.get("output_file"), content, "flamegraph", lhs_profile, rhs_profile
    )
    log.minor_status("Output saved", log.path_style(output_file))


@click.command()
@click.pass_context
@click.option(
    "-w",
    "--width",
    type=click.INT,
    default=DEFAULT_WIDTH,
    help="Sets the width of the flamegraph (default=600px).",
)
@click.option(
    "-h",
    "--height",
    type=click.INT,
    default=DEFAULT_HEIGHT,
    help="Sets the height of the flamegraph (default=14).",
)
@click.option("-o", "--output-file", help="Sets the output file (default=automatically generated).")
def flamegraph(ctx: click.Context, *_: Any, **kwargs: Any) -> None:
    """ """
    assert ctx.parent is not None and f"impossible happened: {ctx} has no parent"
    profile_list = ctx.parent.params["profile_list"]
    generate_flamegraph_difference(profile_list[0], profile_list[1], **kwargs)
