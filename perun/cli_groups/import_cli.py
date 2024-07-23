"""Group of CLI commands used for importing profiles"""

from __future__ import annotations

# Standard Imports
from typing import Any

# Third-Party Imports
import click

# Perun Imports
from perun.logic import commands
from perun.profile import imports
from perun.utils.common import cli_kit


@click.group("import")
@click.option(
    "--machine-info",
    "-m",
    type=click.Path(resolve_path=True, readable=True),
    help="Imports machine info from file in JSON format (by default, machine info is loaded from the current host)."
    "You can use `utils/generate_machine_info.sh` script to generate the machine info file.",
)
@click.option(
    "--minor-version",
    "-m",
    "minor_version_list",
    nargs=1,
    multiple=True,
    callback=cli_kit.minor_version_list_callback,
    default=["HEAD"],
    help="Specifies the head minor version, for which the profiles will be imported.",
)
@click.option(
    "--exitcode",
    "-e",
    nargs=1,
    required=False,
    multiple=True,
    default=["?"],
    help=("Exit code of the command."),
)
@click.option(
    "--cmd",
    "-c",
    nargs=1,
    required=False,
    multiple=True,
    default=[""],
    help=(
        "Command that was being profiled. Either corresponds to some"
        " script, binary or command, e.g. ``./mybin`` or ``perun``."
    ),
)
@click.option(
    "--workload",
    "-w",
    nargs=1,
    required=False,
    multiple=True,
    default=[""],
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
    multiple=True,
    default=[0],
    help="Sets [INT] warm up iterations of ith profiled command.",
)
@click.option(
    "--repeat",
    "-r",
    multiple=True,
    default=[1],
    help="Sets [INT] samplings of the ith profiled command.",
)
@click.pass_context
def perf_group(ctx: click.Context, **kwargs: Any) -> None:
    """Imports Perun profiles from perf results

    This supports either profiles collected in:

      1. Binary format: e.g., `collected.data` files, that are results of `perf record`
      2. Text format: result of `perf script` that parses the binary into user-friendly and parsing-friendly text format
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
@click.argument("imported", type=click.Path(resolve_path=True), nargs=-1, required=True)
@click.pass_context
def from_text(ctx: click.Context, imported: list[str], **kwargs: Any) -> None:
    """Import Perun profiles from output generated by `perf script` command"""
    kwargs.update(ctx.obj)
    imports.import_perf_from_script(imported, **kwargs)


@perf_group.command("stack")
@click.argument("imported", type=click.Path(resolve_path=True), nargs=-1, required=True)
@click.pass_context
def from_stacks(ctx: click.Context, imported: list[str], **kwargs: Any) -> None:
    """Import Perun profiles from output generated by `perf script | stackcollapse-perf.pl` command"""
    kwargs.update(ctx.obj)
    imports.import_perf_from_stack(imported, **kwargs)
