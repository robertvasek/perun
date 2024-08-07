"""Group of CLI commands used for importing profiles"""

from __future__ import annotations

# Standard Imports
from typing import Any

# Third-Party Imports
import click

# Perun Imports
from perun.logic import commands
from perun.profile import imports


@click.group("import")
@click.option(
    "--machine-info",
    "-i",
    type=click.Path(resolve_path=True, readable=True),
    help="Imports machine info from file in JSON format (by default, machine info is loaded from "
    "the current host). You can use `utils/generate_machine_info.sh` script to generate the "
    "machine info file.",
)
@click.option(
    "--import-dir",
    "-d",
    type=click.Path(resolve_path=True, readable=True),
    help="Specifies the directory to import profiles from.",
)
@click.option(
    "--minor-version",
    "-m",
    nargs=1,
    default=None,
    is_eager=True,
    help="Specifies the head minor version, for which the profiles will be imported.",
)
@click.option(
    "--stats-info",
    "-t",
    nargs=1,
    default=None,
    metavar="<stat1-description,...>",
    help="Describes the stats associated with the imported profiles. Please see the import "
    "documentation for details regarding the stat description format.",
)
@click.option(
    "--cmd",
    "-c",
    nargs=1,
    default="",
    help=(
        "Command that was being profiled. Either corresponds to some"
        " script, binary or command, e.g. ``./mybin`` or ``perun``."
    ),
)
@click.option(
    "--workload",
    "-w",
    nargs=1,
    default="",
    help="Inputs for <cmd>. E.g. ``./subdir`` is possible workload for ``ls`` command.",
)
@click.option(
    "--save-to-index",
    "-s",
    is_flag=True,
    help="Saves the imported profile to index.",
    default=False,
)
@click.pass_context
def import_group(ctx: click.Context, **kwargs: Any) -> None:
    """Imports Perun profiles from different formats"""
    commands.try_init()
    ctx.obj = kwargs


@import_group.group("perf")
@click.option(
    "--warmup",
    "-w",
    default=0,
    help="Sets [INT] warm up iterations of ith profiled command.",
)
@click.pass_context
def perf_group(ctx: click.Context, **kwargs: Any) -> None:
    """Imports Perun profiles from perf results

    This supports either profiles collected in:

      1. Binary format: e.g., `collected.data` files, that are results of `perf record`
      2. Text format: result of `perf script` that parses the binary into user-friendly and
         parsing-friendly text format
    """
    ctx.obj.update(kwargs)


@perf_group.command("record")
@click.argument("imported", nargs=-1, required=True)
@click.pass_context
@click.option(
    "--with-sudo",
    "-s",
    is_flag=True,
    help="Runs the conversion of the data in sudo mode.",
    default=False,
)
def from_binary(ctx: click.Context, imported: list[str], **kwargs: Any) -> None:
    """Imports Perun profiles from binary generated by `perf record` command"""
    kwargs.update(ctx.obj)
    imports.import_perf_from_record(imported, **kwargs)


@perf_group.command("script")
@click.argument("imported", type=str, nargs=-1, required=True)
@click.pass_context
def from_text(ctx: click.Context, imported: list[str], **kwargs: Any) -> None:
    """Import Perun profiles from output generated by `perf script` command"""
    kwargs.update(ctx.obj)
    imports.import_perf_from_script(imported, **kwargs)


@perf_group.command("stack")
@click.argument("imported", type=str, nargs=-1, required=True)
@click.pass_context
def from_stacks(ctx: click.Context, imported: list[str], **kwargs: Any) -> None:
    """Import Perun profiles from output generated by `perf script | stackcollapse-perf.pl`
    command
    """
    kwargs.update(ctx.obj)
    imports.import_perf_from_stack(imported, **kwargs)
