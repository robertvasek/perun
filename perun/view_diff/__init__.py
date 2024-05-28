"""
Base package for difference of profiles
"""

from __future__ import annotations

from typing import Callable, Any


def lazy_get_cli_commands() -> list[Callable[..., Any]]:
    """
    Lazily imports CLI commands
    """
    import perun.view_diff.flamegraph.run as flamegraph_run
    import perun.view_diff.datatables.run as datatables_run
    import perun.view_diff.sankey.run as sankey_run
    import perun.view_diff.report.run as report_run
    import perun.view_diff.short.run as short_run

    return [
        flamegraph_run.flamegraph,
        short_run.short,
        datatables_run.datatables,
        sankey_run.sankey,
        report_run.report,
    ]
