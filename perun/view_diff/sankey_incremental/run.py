"""Sankey difference of the profiles

The difference is in form of:

cmd       | cmd
workload  | workload
collector | kernel

| ---\          /======|
     |-----|====
|---/          \------|

"""
from __future__ import annotations

# Standard Imports
from typing import Any, Literal

# Third-Party Imports
from collections import defaultdict
from dataclasses import dataclass
import click
import jinja2
import progressbar

# Perun Imports
from perun.profile import convert
from perun.profile.factory import Profile
from perun.utils import log
from perun.utils.common import diff_kit, common_kit
from perun.view_diff.flamegraph import run as flamegraph_run


class ColorPalette:
    """ """

    Baseline: str = "rgba(49, 48, 77, 0.4)"
    Target: str = "rgba(255, 201, 74, 0.4)"
    NotInBaseline: str = "rgba(255, 0, 0, 0.4)"
    NotInTarget: str = "rgba(0, 255, 0, 0.4)"
    InBoth: str = "rgba(0, 0, 255, 0.4)"
    Highlight: str = "rgba(0, 0, 0, 0.7)"
    NoHighlight: str = "rgba(0, 0, 0, 0.2)"


def generate_sankey_difference(lhs_profile: Profile, rhs_profile: Profile, **kwargs: Any) -> None:
    """Generates differences of two profiles as sankey diagram

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :param kwargs: additional arguments
    """
    # We automatically set the value of True for kperf, which samples
    if lhs_profile.get("collector_info", {}).get("name") == "kperf":
        kwargs["trace_is_inclusive"] = True

    log.major_info("Generating Sankey Graph Difference")

    selection_table = []

    env = jinja2.Environment(loader=jinja2.PackageLoader("perun", "templates"))
    template = env.get_template("diff_view_sankey_incremental.html.jinja2")
    content = template.render(
        title="Differences of profiles (with sankey)",
        lhs_tag="Baseline (base)",
        lhs_header=flamegraph_run.generate_header(lhs_profile),
        rhs_tag="Target (tgt)",
        rhs_header=flamegraph_run.generate_header(rhs_profile),
        palette=ColorPalette,
    )
    log.minor_success("Difference sankey", "generated")
    output_file = diff_kit.save_diff_view(
        kwargs.get("output_file"), content, "sankey", lhs_profile, rhs_profile
    )
    log.minor_status("Output saved", log.path_style(output_file))


@click.command()
@click.option("-o", "--output-file", help="Sets the output file (default=automatically generated).")
@click.pass_context
def sankey_incr(ctx: click.Context, *_: Any, **kwargs: Any) -> None:
    """Creates sankey graphs representing the differences between two profiles"""
    assert ctx.parent is not None and f"impossible happened: {ctx} has no parent"
    profile_list = ctx.parent.params["profile_list"]
    generate_sankey_difference(profile_list[0], profile_list[1], **kwargs)
