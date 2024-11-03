"""Flame graph visualization of the profiles."""

from __future__ import annotations

# Standard Imports
from typing import Any

# Third-Party Imports
import click

# Perun Imports
from perun.profile import convert
import perun.view.flamegraph.flamegraph as flame
import perun.profile.factory as profile_factory


def save_flamegraph(profile: profile_factory.Profile, filename: str) -> None:
    """Draws and saves flamegraph to file

    :param profile: profile for which we are saving flamegraph
    :param filename: name of the file where the flamegraph will be saved
    """
    flamegraph_content = flame.draw_flame_graph(
        convert.to_flame_graph_format(profile),
        flame.generate_title(profile["header"]),
        flame.get_units(profile["header"]["type"]),
    )
    with open(filename, "w") as file_handle:
        file_handle.write(flamegraph_content)


@click.command()
@click.option(
    "--filename",
    "-f",
    default="flame.svg",
    help="Sets the output file of the resulting flame graph.",
)
@profile_factory.pass_profile
def flamegraph(profile: profile_factory.Profile, filename: str, **_: Any) -> None:
    """Flame graph interprets the relative and inclusive presence of the
    resources according to the stack depth of the origin of resources.

    \b
      * **Limitations**: `memory` profiles generated by
        :ref:`collectors-memory`.
      * **Interpretation style**: graphical
      * **Visualization backend**: HTML

    Flame graph intends to quickly identify hotspots, that are the source of
    the resource consumption complexity. On X axis, a relative consumption of
    the data is depicted, while on Y axis a stack depth is displayed. The wider
    the bars are on the X axis are, the more the function consumed resources
    relative to others.

    **Acknowledgements**: Big thanks to Brendan Gregg for creating the original
    perl script for creating flame graphs w.r.t simple format. If you like this
    visualization technique, please check out this guy's site
    (https://brendangregg.com) for more information about performance, profiling
    and useful talks and visualization techniques!

    The example output of the flamegraph is more or less as follows::

        \b
                            `
                            -                         .
                            `                         |
                            -              ..         |     .
                            `              ||         |     |
                            -              ||        ||    ||
                            `            |%%|       |--|  |!|
                            -     |## g() ##|     |#g()#|***|
                            ` |&&&& f() &&&&|===== h() =====|
                            +````||````||````||````||````||````

    Refer to :ref:`views-flame-graph` for more thorough description and
    examples of the interpretation technique. Refer to
    :func:`perun.profile.convert.to_flame_graph_format` for more details how
    the profiles are converted to the flame graph format.
    """
    save_flamegraph(profile, filename)
