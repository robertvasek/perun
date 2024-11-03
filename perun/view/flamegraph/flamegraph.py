"""This module provides wrapper for the Flame graph visualization"""

from __future__ import annotations

# Standard Imports
from dataclasses import dataclass, field
import os
import tempfile
from typing import Any

# Third-Party Imports

# Perun Imports
from perun.utils import mapping
from perun.utils.common import script_kit
from perun.utils.external import commands


def draw_flame_graph_difference(
    lhs_flame_data: list[str],
    rhs_flame_data: list[str],
    title: str,
    units: str = "samples",
    img_width: int = 1200,
    fg_flags: str = "",
    fg_min_width: str = "1",
    fg_max_trace: int = 0,
    fg_max_resource: float = 0.0,
) -> str:
    """Draws difference of two flame graphs from two profiles

    :param lhs_flame_data: baseline flame graph data
    :param rhs_flame_data: target flame graph data
    :param title: title of the flame graph
    :param units: the units of the flame graph data
    :param img_width: width of the graph
    :param fg_flags: additional flags to pass to the flame graph generator script
    :param fg_min_width: minimum width of the flame graph rectangles that will be drawn
    :param fg_max_trace: maximum length of traces that are being drawn
    :param fg_max_resource: maximum number of samples collected

    :return: the difference flame graph
    """
    with open("lhs.flame", "w") as lhs_handle:
        lhs_handle.write("".join(lhs_flame_data))

    with open("rhs.flame", "w") as rhs_handle:
        rhs_handle.write("".join(rhs_flame_data))

    diff_script = script_kit.get_script("difffolded.pl")
    flame_script = script_kit.get_script("flamegraph.pl")
    difference_script = (
        f"{diff_script} -n lhs.flame rhs.flame "
        f"| {flame_script} --title '{title}' --countname {units} --reverse "
        f"--width {img_width} --minwidth {fg_min_width} --maxtrace {fg_max_trace}"
    )
    if fg_max_resource > 0.0:
        difference_script += f' --total {fg_max_resource} --rootnode "Maximum (Baseline, Target)"'
    if fg_flags:
        difference_script += f" {fg_flags}"
    out, _ = commands.run_safely_external_command(difference_script)
    os.remove("lhs.flame")
    os.remove("rhs.flame")

    return out.decode("utf-8")


def draw_flame_graph(
    flame_data: list[str],
    title: str,
    units: str = "samples",
    img_width: int = 1200,
    fg_min_width: str = "1",
    fg_max_trace: int = 0,
    fg_max_resource: float = 0.0,
) -> str:
    """Draw Flame graph from flame data.

        To create Flame graphs we use perl script created by Brendan Gregg.
        https://github.com/brendangregg/FlameGraph/blob/master/flamegraph.pl

    :param flame_data: the data to generate the flame graph from
    :param title: title of the flame graph
    :param units: the units of the flame graph data
    :param img_width: width of the graph
    :param fg_min_width: minimum width of the flame graph rectangles that will be drawn
    :param fg_max_trace: maximum length of traces that are being drawn
    :param fg_max_resource: maximum number of samples collected
    """
    # converting profile format to format suitable to Flame graph visualization
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write("".join(flame_data).encode("utf-8"))
        tmp.close()
        cmd = " ".join(
            [
                script_kit.get_script("flamegraph.pl"),
                tmp.name,
                "--cp",
                "--title",
                f'"{title}"',
                "--countname",
                f"{units}",
                "--reverse",
                "--width",
                f"{img_width}",
                "--maxtrace",
                f"{fg_max_trace}",
                "--minwidth",
                f"{fg_min_width}",
            ]
        )
        if fg_max_resource > 0.0:
            cmd += f' --total {fg_max_resource} --rootnode "Maximum (Baseline, Target)"'
        out, _ = commands.run_safely_external_command(cmd)
        os.remove(tmp.name)
    return out.decode("utf-8")


def generate_title(profile_header: dict[str, Any]) -> str:
    """Generate a title for flame graph based on the profile header.

    :param profile_header: the profile header

    :return: the title of the flame graph
    """
    profile_type = profile_header["type"]
    cmd, workload = (profile_header["cmd"], profile_header["workload"])
    return f"{profile_type} consumption of {cmd} {workload}"


def get_units(profile_key: str) -> str:
    """Obtain the units of the flame graph based on the profile key.

    :param profile_key: the profile key

    :return: the units of the flame graph
    """
    return mapping.get_unit(mapping.get_readable_key(profile_key))


def compute_max_traces(
    flame_data: list[str],
    img_width: float,
    min_width: str = "1",
) -> tuple[int, int, int]:
    """Recreate Brendan Gregg's max trace depth computation for correct flamegraph height.

    This function provides the maximum length of traces and filtered maximum that takes into
    account the filtering of subtraces based on their sample count, as well as the total number of
    samples collected.

    :param flame_data: the flame graph data
    :param img_width: the width of the graph image
    :param min_width: the minimum width of the flame graph rectangles that will be drawn

    :return: the maximum trace length, the maximum length of traces that are being drawn, the total
             number of samples collected
    """
    max_unfiltered_trace = 0
    flame_stacks: _PerfStackRecord = _PerfStackRecord()
    # Process each flame data record
    for stack_trace in flame_data:
        # Update the perf stack traces representation with this record
        stack_str, samples = stack_trace.rsplit(maxsplit=1)
        stack = stack_str.split(";")
        max_unfiltered_trace = max(len(stack), max_unfiltered_trace)
        flame_stacks.update_stack(stack, int(float(samples)))
    min_width_f = _compute_minwidth_samples(flame_stacks.inclusive_samples, img_width, min_width)
    return max_unfiltered_trace, flame_stacks.filter(min_width_f), flame_stacks.inclusive_samples


def _compute_minwidth_samples(max_resource: float, img_width: float, min_width: str) -> float:
    """Computes the minimum width threshold for flamegraph blocks to be displayed.

    Reconstructed from the flamegraph.pl script.

    :param max_resource: the total number of samples collected
    :param img_width: the width of the graph image
    :param min_width: the minimum width of the flame graph rectangles that will be drawn
    """
    try:
        if min_width.endswith("%"):
            return max_resource * float(min_width[:-1]) / 100
        else:
            x_padding: int = 10
            width_per_time: float = (img_width - 2 * x_padding) / max_resource
            return float(min_width) / width_per_time
    except ZeroDivisionError:
        # Unknown or invalid max_resource, we set the threshold so that it does not filter anything
        return 0.0


@dataclass
class _PerfStackRecord:
    """Representation of a single perf stack frame record from flame data.

    :ivar inclusive_samples: the number of samples in which a given stack frame record occurred
    :ivar nested: callee stack frame records
    """

    inclusive_samples: int = 0
    nested: dict[str, _PerfStackRecord] = field(default_factory=dict)

    def update_stack(self, stack: list[str], samples: int) -> None:
        """Update the stack traces with new flame data record.

        :param stack: the stack trace
        :param samples: the number of samples that contain this stack trace
        """
        self.inclusive_samples += samples
        if stack:
            func = stack.pop()
            self.nested.setdefault(func, _PerfStackRecord()).update_stack(stack, samples)

    def filter(self, min_width: float) -> int:
        """Filter the stack subtraces (recursively) that do not meet the minimum sample threshold.

        :param min_width: the samples threshold

        :return: the maximum stack length after the filtering
        """
        depth: int = 0
        delete_list: list[str] = []
        for nested_func, nested_record in self.nested.items():
            if nested_record.inclusive_samples < min_width:
                # This nested subtrace does not meet the samples threshold, we cut the entire
                # subtrace off including its subtraces.
                delete_list.append(nested_func)
            else:
                # This subtrace meets the threshold, check recursively
                depth = max(depth, nested_record.filter(min_width) + 1)
        for del_key in delete_list:
            del self.nested[del_key]
        return depth
