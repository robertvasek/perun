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
from typing import Any

# Third-Party Imports
import click
import jinja2

# Perun Imports
from perun.profile.factory import Profile
from perun.utils import log
from perun.utils.common import diff_kit
from perun.view_diff.flamegraph import run as flamegraph_run


def generate_sankey_difference(lhs_profile: Profile, rhs_profile: Profile, **kwargs: Any) -> None:
    """Generates differences of two profiles as sankey diagram

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :param kwargs: additional arguments
    """

    log.major_info("Generating Flamegraph Difference")

    env = jinja2.Environment(loader=jinja2.PackageLoader("perun", "templates"))
    template = env.get_template("diff_view_sankey.html.jinja2")
    content = template.render(
        title="Differences of profiles (with sankey)",
        lhs_tag="Baseline (base)",
        lhs_header=flamegraph_run.generate_header(lhs_profile),
        rhs_tag="Target (tgt)",
        rhs_header=flamegraph_run.generate_header(rhs_profile),
    )
    log.minor_success("Difference sankey", "generated")
    output_file = diff_kit.save_diff_view(
        kwargs.get("output_file"), content, "sankey", lhs_profile, rhs_profile
    )
    log.minor_status("Output saved", log.path_style(output_file))


@click.command()
@click.option("-o", "--output-file", help="Sets the output file (default=automatically generated).")
@click.pass_context
def sankey(ctx: click.Context, *_: Any, **kwargs: Any) -> None:
    """ """
    assert ctx.parent is not None and f"impossible happened: {ctx} has no parent"
    profile_list = ctx.parent.params["profile_list"]
    generate_sankey_difference(profile_list[0], profile_list[1], **kwargs)
