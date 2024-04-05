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
from perun.utils.common import diff_kit
from perun.view_diff.flamegraph import run as flamegraph_run


NOT_IN_BASE: str = "rgba(255, 0, 0, 0.4)"
NOT_IN_TARGET: str = "rgba(0, 255, 0, 0.4)"
IN_BOTH: str = "rgba(0, 0, 255, 0.4)"


@dataclass
class SuccessorStats:
    """Represents statistic about successors between sankey nodes

    :ivar baseline: stats for baseline profile
    :ivar target: stats for target profile
    """

    __slots__ = ["baseline", "target"]

    baseline: int
    target: int


@dataclass
class SankeyNode:
    """Single node in the sankey graph

    :ivar uid: string representation of the uid/node
    :ivar tag: uid joined with order in the traces
    :ivar id: integer representation of the node in the graph
    :ivar succs: list of successor sankey nodes together with the stats
    """

    __slots__ = ["uid", "id", "succs", "tag"]

    uid: str
    tag: str
    id: int
    succs: dict[int, SuccessorStats]


@dataclass
class SankeyGraph:
    """Representation of sankey graph data consisting of links between nodes

    :ivar uid: function for which they sankey graph is corresponding
    :ivar label: list of labes for each node in the graph
    :ivar source: sources of the edges
    :ivar target: targets of the edges
    :ivar value: values of edges
    :ivar color: colours of edges
    :ivar width: the size of the longest trace
    :ivar height: the maximal number of paralel traces
    """

    __slots__ = ["label", "source", "target", "value", "color", "uid", "width", "height"]
    uid: str
    label: list[str]
    source: list[int]
    target: list[int]
    value: list[int]
    color: list[str]
    width: int
    height: int


def get_sankey_point(sankey_points: dict[str, SankeyNode], key: str) -> SankeyNode:
    """Retrieves the sankey node from the list of points.

    If the point does not exist, then it is created

    :param sankey_points: map of keys to sankey nodes
    :param key: key identifying sankey node
    :return: sankey node corresponding to given key
    """
    if key not in sankey_points:
        sankey_points[key] = SankeyNode(key.split("#")[0], key, len(sankey_points), {})
    return sankey_points[key]


def process_edge(
    sankey_map: dict[str, dict[str, SankeyNode]],
    trace: list[str],
    profile_type: Literal["baseline", "target"],
    src: str,
    tgt: str,
    amount: int,
) -> None:
    """Creates and edge in each of the sankey graph along the trace

    Increases the valuation of appropriate statistic

    :param sankey_map: list of sankey points corresponding to each uid
    :param trace: list of uid representing the trace
    :param profile_type: identification of either baseline or target
    :param src: source id of the edge
    :param tgt: target id of the edge
    :param amount: valuation of the edge
    """
    for t in trace:
        sankey_points = sankey_map[t]
        src_point = get_sankey_point(sankey_points, src)
        tgt_point = get_sankey_point(sankey_points, tgt)
        if tgt_point.id not in src_point.succs:
            src_point.succs[tgt_point.id] = SuccessorStats(0, 0)
        if profile_type == "baseline":
            src_point.succs[tgt_point.id].baseline += amount
        else:
            assert profile_type == "target"
            src_point.succs[tgt_point.id].target += amount


def process_traces(
    profile: Profile,
    sankey_map: dict[str, dict[str, SankeyNode]],
    profile_type: Literal["baseline", "target"],
) -> None:
    """Processes all traces in the profile

    Iterates through all traces and creates edges for each pair
    of source and target.

    :param profile: input profile
    :param sankey_map: output sankey map with sankey nodes
    :param profile_type: type of the profile
    """
    for _, resource in progressbar.progressbar(profile.all_resources()):
        trace_len = len(resource["trace"])
        full_trace = [convert.to_uid(t) for t in resource["trace"]] + [
            convert.to_uid(resource["uid"])
        ]
        if trace_len > 0:
            amount = int(resource["amount"])
            for i in range(0, trace_len - 1):
                src = f"{full_trace[i]}#{i}"
                tgt = f"{full_trace[i+1]}#{i+1}"
                process_edge(sankey_map, full_trace, profile_type, src, tgt, amount)
            src = f"{full_trace[-2]}#{trace_len-1}"
            tgt = f"{full_trace[-1]}"
            process_edge(sankey_map, full_trace, profile_type, src, tgt, amount)


def create_edge(graph: SankeyGraph, src: int, tgt: int, value: int, color: str) -> None:
    """Creates single edge in the sankey graph

    :param graph: output sankey graph
    :param src: source of the edge
    :param tgt: target of the edge
    :param value: valuation of the edge
    :param color: color of the edge
    """
    graph.source.append(src)
    graph.target.append(tgt)
    graph.value.append(value)
    graph.color.append(color)


def extract_graphs_from_sankey_map(
    sankey_map: dict[str, dict[str, SankeyNode]]
) -> list[SankeyGraph]:
    """For computed maps of sankey edges computes the list of actual sankey graphs

    :param sankey_map: mapping of function ids to their sankey points
    :return: list of sankey graphs
    """
    sankey_graphs = []

    for uid, sankey_points in progressbar.progressbar(sankey_map.items()):
        sankey_graph = SankeyGraph(uid, [], [], [], [], [], 0, 0)
        positions = []

        for sankey_point in sankey_points.values():
            if "#" in sankey_point.tag:
                positions.append(int(sankey_point.tag.split("#")[-1]))
            sankey_graph.label.append(sankey_point.uid)
            for successor, value in sankey_point.succs.items():
                value_diff = abs(value.baseline - value.target)
                if value.baseline == value.target:
                    create_edge(sankey_graph, sankey_point.id, successor, value.baseline, IN_BOTH)
                elif value.baseline > value.target:
                    create_edge(sankey_graph, sankey_point.id, successor, value.target, IN_BOTH)
                    create_edge(sankey_graph, sankey_point.id, successor, value_diff, NOT_IN_TARGET)
                else:
                    create_edge(sankey_graph, sankey_point.id, successor, value.baseline, IN_BOTH)
                    create_edge(sankey_graph, sankey_point.id, successor, value_diff, NOT_IN_BASE)

        sankey_graph.width = max(positions)
        sankey_graph.height = max(positions.count(pos) for pos in set(positions))
        sankey_graphs.append(sankey_graph)

    return sankey_graphs


def to_sankey_graphs(lhs_profile: Profile, rhs_profile: Profile) -> list[SankeyGraph]:
    """Converts difference of two profiles to sankey format

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    """
    sankey_map: dict[str, dict[str, SankeyNode]] = defaultdict(dict)

    process_traces(lhs_profile, sankey_map, "baseline")
    process_traces(rhs_profile, sankey_map, "target")

    return extract_graphs_from_sankey_map(sankey_map)


def generate_sankey_difference(lhs_profile: Profile, rhs_profile: Profile, **kwargs: Any) -> None:
    """Generates differences of two profiles as sankey diagram

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :param kwargs: additional arguments
    """

    log.major_info("Generating Sankey Graph Difference")

    sankey_graphs = to_sankey_graphs(lhs_profile, rhs_profile)
    sankey_graphs = sorted(sankey_graphs, key=lambda x: x.uid)
    sankey_keys = [graph.uid for graph in sankey_graphs]

    env = jinja2.Environment(loader=jinja2.PackageLoader("perun", "templates"))
    template = env.get_template("diff_view_sankey.html.jinja2")
    content = template.render(
        title="Differences of profiles (with sankey)",
        lhs_tag="Baseline (base)",
        lhs_header=flamegraph_run.generate_header(lhs_profile),
        rhs_tag="Target (tgt)",
        rhs_header=flamegraph_run.generate_header(rhs_profile),
        sankey_graphs=sankey_graphs,
        uids=sankey_keys,
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
    """Creates sankey graphs representing the differences between two profiles"""
    assert ctx.parent is not None and f"impossible happened: {ctx} has no parent"
    profile_list = ctx.parent.params["profile_list"]
    generate_sankey_difference(profile_list[0], profile_list[1], **kwargs)
