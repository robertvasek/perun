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
from operator import itemgetter
from typing import Any, Literal, Type, Callable
from collections import defaultdict
from dataclasses import dataclass

# Third-Party Imports
import click
import jinja2
import progressbar

# Perun Imports
from perun.profile import convert
from perun.profile.factory import Profile
from perun.utils import log
from perun.utils.common import diff_kit, common_kit
from perun.view_diff.flamegraph import run as flamegraph_run


def singleton_class(cls: Type) -> Callable[[], Config]:
    """Helper class for creating singleton objects"""
    instances = {}

    def getinstance() -> Config:
        """Singleton instance"""
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton_class
class Config:
    """Singleton config for generation of sankey graphs

    :ivar trace_is_inclusive: if set to true, then the amounts are distributed among the whole traces
    """

    def __init__(self):
        """Initializes the config

        By default we consider, that the traces are not inclusive
        """
        self.trace_is_inclusive: bool = False


class ColorPalette:
    """Colour palette is a static object for grouping colours used in visualizations"""

    Baseline: str = "rgba(49, 48, 77, 0.4)"
    Target: str = "rgba(255, 201, 74, 0.4)"
    NotInBaseline: str = "rgba(255, 0, 0, 0.4)"
    NotInTarget: str = "rgba(0, 255, 0, 0.4)"
    InBoth: str = "rgba(0, 0, 255, 0.4)"
    Highlight: str = "rgba(0, 0, 0, 0.7)"
    NoHighlight: str = "rgba(0, 0, 0, 0.2)"


@dataclass
class TraceStat:
    """Statistics for single trace

    TODO: Add per partes stats

    :ivar trace: list of called UID
    :ivar baseline_cost: overall cost of the traces
    :ivar target_cost: overall cost of the traces
    :ivar baseline_inverse_exclusive: overall cost minus last call
    :ivar target_inverse_exclusive: overall cost minus last call
    """

    __slots__ = [
        "trace",
        "baseline_cost",
        "target_cost",
        "baseline_inverse_exclusive",
        "target_inverse_exclusive",
    ]
    trace: list[str]
    baseline_cost: dict[str, float]
    target_cost: dict[str, float]
    baseline_inverse_exclusive: dict[str, float]
    target_inverse_exclusive: dict[str, float]


@dataclass
class SelectionRow:
    """Helper dataclass for displaying selection of data

    :ivar uid: uid of the selected graph
    :ivar index: index in the sorted list of data
    :ivar abs_amount: absolute change in the units
    :ivar fresh: the state of the uid (whether it is new (added in target), removed (removed in target), or
        possibly unchanged
    :ivar main_stat: type of the
    :ivar rel_amount: relative change in the units
    """

    __slots__ = [
        "uid",
        "index",
        "fresh",
        "main_stat",
        "abs_amount",
        "rel_amount",
        "stats",
        "trace_stats",
    ]
    uid: str
    index: int
    fresh: Literal["new", "removed", "(un)changed"]
    main_stat: str
    abs_amount: float
    rel_amount: float
    stats: list[list[str, float, float]]
    trace_stats: list[list[str, str, float, float, str]]


class Skeleton:
    """Represents single fully computed sankey graph: a skeleton

    :ivar stat: stat for which the skeleton is generated
    :ivar label: list of node labels
    :ivar color: list of node colors
    :ivar source: list of sources of edges
    :ivar target: list of target of edges
    :ivar value: list of values of edges
    """

    __slots__ = ["stat", "label", "color", "source", "target", "value", "link_color", "height"]

    def __init__(self, stat: str):
        """Initializes the skeleton"""
        self.stat: str = stat
        self.label: list[str] = []
        self.color: list[str] = []
        self.source: list[int] = []
        self.target: list[int] = []
        self.link_color: list[str] = []
        self.value: list[float] = []
        self.height: int = 0


class Graph:
    """Represents single sankey graph

    :ivar nodes: mapping of uids#pos to concrete sankey nodes
    :ivar uid_to_nodes: mapping of uid to list of its uid#pos, i.e. uid in different contexts
    :ivar uid_to_id: mapping of uid to unique identifiers
    :ivar stats_to_id: mapping of stats to unique identifiers
    """

    __slots__ = ["nodes", "uid_to_nodes", "uid_to_id", "stats_to_id", "uid_to_traces"]

    nodes: dict[str, Node]
    uid_to_traces: dict[str, list[list[str]]]
    uid_to_nodes: dict[str, list[Node]]
    uid_to_id: dict[str, int]
    stats_to_id: dict[str, int]

    def __init__(self):
        """Initializes empty graph"""
        self.nodes = {}
        self.uid_to_nodes = defaultdict(list)
        self.uid_to_traces = defaultdict(list)
        self.stats_to_id = {}
        self.uid_to_id = {}

    def get_node(self, node: str) -> Node:
        """For give uid#pos returns its corresponding node

        If the uid is not yet assigned unique id, we assign it.
        If the node has corresponding node created we return it.

        :param node: node in form of uid#pos
        :return: corresponding node in sankey graph
        """
        uid = node.split("#")[0]
        if uid not in self.uid_to_id:
            self.uid_to_id[uid] = len(self.uid_to_id)
        if node not in self.nodes:
            self.nodes[node] = Node(node)
            self.uid_to_nodes[uid].append(self.nodes[node])
        return self.nodes[node]

    def translate_stats(self, stats: str) -> int:
        """Translates string representation of stats to unique id

        :param stats: stats represented as string
        :return: unique id for the string
        """
        if stats not in self.stats_to_id:
            self.stats_to_id[stats] = len(self.stats_to_id)
        return self.stats_to_id[stats]

    def translate_node(self, node) -> str:
        """Translates the node to its unique id

        :param node: node which we are translating to id
        :return: unique identifier for the node uid
        """
        if "#" in node:
            return f"{self.uid_to_id[node.split('#')[0]]}"
        return str(self.uid_to_id[node])

    def get_callee_stats(self, src: str, tgt: str) -> Stats:
        """Returns stats for callees of the src

        :param src: source node
        :param tgt: target callee node
        :return: stats
        """
        # Create node if it does not already exist
        src_node = self.get_node(src)

        # Get link if it does not already exist
        if tgt not in src_node.callees:
            tgt_node = self.get_node(tgt)
            src_node.callees[tgt] = Link(tgt_node, Stats())

        return src_node.callees[tgt].stats

    def get_caller_stats(self, src: str, tgt: str) -> Stats:
        """Returns stats for callers of the src

        :param src: source node
        :param tgt: target caller node
        :return: stats
        """
        # Create node if it does not already exist
        src_node = self.get_node(src)

        # Get link if it does not already exist
        if tgt not in src_node.callers:
            tgt_node = self.get_node(tgt)
            src_node.callers[tgt] = Link(tgt_node, Stats())

        return src_node.callers[tgt].stats

    def to_jinja_string(self, link_type: Literal["callers", "callees"] = "callers") -> str:
        """Since jinja seems to be awfully slow with this, we render the result ourselves

        1. Target nodes of "uid#pos" are simplified to "uid", since you can infer pos to be pos+1 of source
        2. Stats are merged together: first half is for baseline, second half is for target

        TODO: switch callees to callers and callers to callees

        :param link_type: either callers for callers or callee for callees
        :return string representation of the caller or callee relation
        """

        def comma_control(commas: list[bool], pos: int) -> str:
            """Helper function for comma control

            :param pos: position in the nesting
            :param commas: list of boolean flags for comma control (true = we should output)
            """
            if commas[pos]:
                return ","
            commas[pos] = True
            return ""

        output = "{"
        commas = [False, False, False]
        for uid, nodes in progressbar.progressbar(self.uid_to_nodes.items()):
            output += comma_control(commas, 0) + f"{self.translate_node(uid)}:" + "{"
            commas[1] = False
            for node in nodes:
                output += comma_control(commas, 1) + f"{node.get_order()}:" + "{"
                commas[2] = False
                for link in node.get_links(link_type).values():
                    assert link_type == "callees" or int(node.get_order()) + 1 == int(
                        link.target.get_order()
                    )
                    assert (
                        link_type == "callers"
                        or int(node.get_order()) == int(link.target.get_order()) + 1
                    )
                    output += comma_control(commas, 2) + f"{self.translate_node(link.target.uid)}:"
                    stats = f"[{','.join(link.stats.to_array('baseline') + link.stats.to_array('target'))}]"
                    output += str(self.translate_stats(stats))
                output += "}"
            output += "}"
        output += "}"
        return output


@dataclass
class Node:
    """Single node in sankey graph

    :ivar uid: unique identifier of the node (the label)
    :ivar callers: mapp of positions to edge relation for callers
    :ivar callees: mapp of positions to edge relation for callees
    """

    __slots__ = ["uid", "callers", "callees"]

    uid: str
    callers: dict[str, Link]
    callees: dict[str, Link]

    def __init__(self, uid: str):
        """Initializes the node"""
        self.uid = uid
        self.callers = {}
        self.callees = {}

    def get_links(self, link_type: Literal["callers", "callees"]) -> dict[str, Link]:
        """Returns linkage based on given type

        :param link_type: either callers or callees
        :return: linkage of the given ty pe
        """
        if link_type == "callers":
            return self.callers
        assert link_type == "callees"
        return self.callees

    def get_order(self) -> int:
        """Gets position/order in the call traces

        :return: order/position in the call traces
        """
        return int(self.uid.split("#")[1])


@dataclass
class Link:
    """Helper dataclass for linking two nodes

    :ivar target: target of the edge
    :ivar stats: stats of the edge
    """

    __slots__ = ["stats", "target"]
    target: Node
    stats: Stats


class Stats:
    """Statistics for a given edge

    :ivar baseline: baseline stats
    :ivar target: target stats
    """

    __slots__ = ["baseline", "target"]
    KnownStats: set[str] = set()

    def __init__(self):
        """Initializes the stat"""
        self.baseline: dict[str, float] = defaultdict(float)
        self.target: dict[str, float] = defaultdict(float)

    def add_stat(
        self, stat_type: Literal["baseline", "target"], stat_key: str, stat_val: int | float
    ) -> None:
        """Adds stat of given type

        :ivar stat_type: type of the stat (either baseline or target)
        :ivar stat_key: type of the metric
        :ivar stat_val: value of the metric
        """
        Stats.KnownStats.add(stat_key)
        if stat_type == "baseline":
            self.baseline[stat_key] += stat_val
        else:
            self.target[stat_key] += stat_val

    def to_array(self, stat_type: Literal["baseline", "target"]) -> list[str]:
        """Converts stats to single compact array"""
        # TODO: Add different type for int/float
        stats = self.baseline if stat_type == "baseline" else self.target
        return [
            common_kit.compact_convert_num_to_str(stats.get(stat, 0), 2)
            for stat in Stats.KnownStats
        ]


def process_edge(
    graph: Graph,
    profile_type: Literal["baseline", "target"],
    resource: dict[str, Any],
    src: str,
    tgt: str,
) -> None:
    """Processes single edge with given resources

    :param graph: sankey graph
    :param profile_type: type of the profile
    :param resource: consumed resources
    :param src: caller
    :param tgt: callee
    """
    src_stats = graph.get_caller_stats(src, tgt)
    tgt_stats = graph.get_callee_stats(tgt, src)
    for key in resource:
        amount = common_kit.try_convert(resource[key], [float])
        if amount is None or key in ("time",):
            continue
        src_stats.add_stat(profile_type, key, amount)
        tgt_stats.add_stat(profile_type, key, amount)


def process_traces(
    profile: Profile, profile_type: Literal["baseline", "target"], graph: Graph
) -> None:
    """Processes all traces in the profile

    Iterates through all traces and creates edges for each pair of source and target.

    :param profile: input profile
    :param profile_type: type of the profile
    :param graph: sankey graph
    """
    for _, resource in progressbar.progressbar(profile.all_resources()):
        full_trace = [convert.to_uid(t) for t in resource["trace"]] + [
            convert.to_uid(resource["uid"])
        ]
        trace_len = len(full_trace)
        if trace_len > 1:
            if Config().trace_is_inclusive:
                for i in range(0, trace_len - 1):
                    src = f"{full_trace[i]}#{i}"
                    tgt = f"{full_trace[i+1]}#{i+1}"
                    process_edge(graph, profile_type, resource, src, tgt)
            else:
                src = f"{full_trace[-2]}#{trace_len-2}"
                tgt = f"{full_trace[-1]}#{trace_len-1}"
                process_edge(graph, profile_type, resource, src, tgt)
            for uid in full_trace:
                graph.uid_to_traces[uid].append(full_trace)


def generate_skeletons(graph: Graph, traces: dict[str, list[TraceStat]]) -> list[Skeleton]:
    """Generates skeletons of graphs for each amount"""
    stat_to_traces = defaultdict(list)
    processed = set()
    for val in traces.values():
        for trace in val:
            trace_key = ",".join(trace.trace)
            if trace_key in processed:
                continue
            processed.add(trace_key)
            for stat in Stats.KnownStats:
                abs_amount = trace.target_cost[stat] - trace.baseline_cost[stat]
                rel_amount = (trace.target_cost[stat] - trace.baseline_cost[stat]) / max(
                    trace.target_cost[stat], trace.baseline_cost[stat]
                )
                if 0 < abs(rel_amount) < 1.0:
                    stat_to_traces[stat].append([trace.trace, abs_amount])

    skeletons: list[Skeleton] = []
    processed = set()
    for stat, stat_traces in stat_to_traces.items():
        skeleton = Skeleton(stat)
        sorted_traces = sorted(
            sorted(stat_traces, key=lambda t: t[1])[-10:], key=lambda key: ",".join(key[0])
        )
        node_map = []
        for trace, rel in sorted_traces:
            trace_len = len(trace)
            skeleton.height = max(skeleton.height, trace_len)
            for i in range(0, trace_len - 1):
                src, tgt = f"{trace[i]}#{i}", f"{trace[i+1]}#{i+1}"
                if f"{src},{tgt}" in processed:
                    continue
                processed.add(f"{src},{tgt}")
                stats = graph.get_node(src).callers[tgt].stats
                if src not in node_map:
                    node_map.append(src)
                    skeleton.label.append(src.split("#")[0])
                    skeleton.color.append(ColorPalette.NoHighlight)
                if tgt not in skeleton.label:
                    node_map.append(tgt)
                    skeleton.label.append(tgt.split("#")[0])
                    skeleton.color.append(ColorPalette.NoHighlight)
                src_i, tgt_i = node_map.index(src), node_map.index(tgt)
                src_s, tgt_s = stats.baseline[stat], stats.target[stat]
                skeleton.source.append(src_i)
                skeleton.target.append(tgt_i)
                skeleton.value.append(abs(tgt_s - src_s))
                skeleton.link_color.append(
                    ColorPalette.NotInTarget if tgt_s > src_s else ColorPalette.NotInBaseline
                )
        skeletons.append(skeleton)
    return skeletons


def generate_trace_stats(graph: Graph) -> dict[str, list[TraceStat]]:
    """Generates trace stats

    :param graph: sankey graph
    :return: trace stats
    """
    trace_stats = defaultdict(list)
    for uid, traces in graph.uid_to_traces.items():
        processed = set()
        for trace in traces:
            key = ",".join(trace)
            if key not in processed:
                baseline_excl = defaultdict(float)
                target_excl = defaultdict(float)
                trace_len = len(trace)
                for i in range(0, trace_len - 2):
                    stats = graph.get_node(f"{trace[i]}#{i}").callers[f"{trace[i+1]}#{i+1}"].stats
                    for stat in Stats.KnownStats:
                        if Config().trace_is_inclusive:
                            baseline_excl[stat] = (
                                stats.baseline[stat]
                                if baseline_excl[stat] == 0
                                else min(baseline_excl[stat], stats.baseline[stat])
                            )
                            target_excl[stat] = (
                                stats.target[stat]
                                if target_excl[stat] == 0
                                else min(target_excl[stat], stats.target[stat])
                            )
                        else:
                            baseline_excl[stat] += stats.baseline[stat]
                            target_excl[stat] += stats.target[stat]
                last_stats = (
                    graph.get_node(f"{trace[trace_len-2]}#{trace_len-2}")
                    .callers[f"{trace[trace_len-1]}#{trace_len-1}"]
                    .stats
                )
                baseline_overall = {
                    stat: (last_stats.baseline[stat] + baseline_excl[stat])
                    if Config().trace_is_inclusive
                    else min(last_stats.baseline[stat], baseline_excl[stat])
                    for stat in Stats.KnownStats
                }
                target_overall = {
                    stat: (last_stats.target[stat] + target_excl[stat])
                    if Config().trace_is_inclusive
                    else min(last_stats.target[stat], target_excl[stat])
                    for stat in Stats.KnownStats
                }

                processed.add(key)
                trace_stats[uid].append(
                    TraceStat(trace, baseline_overall, target_overall, baseline_excl, target_excl)
                )
    return trace_stats


def generate_selection(graph: Graph, trace_stats: dict[str, list[TraceStat]]) -> list[SelectionRow]:
    """Generates selection table

    :param graph: sankey graph
    :param trace_stats: stats for traces for each uid
    :return: list of selection rows for table
    """
    selection = []
    for uid, nodes in graph.uid_to_nodes.items():
        baseline_overall = defaultdict(float)
        target_overall = defaultdict(float)
        stats = []
        for node in nodes:
            for known_stat in Stats.KnownStats:
                stat = " ".join(["callee", known_stat])
                for link in node.callees.values():
                    baseline_overall[stat] += link.stats.baseline[known_stat]
                    target_overall[stat] += link.stats.target[known_stat]
                stat = " ".join(["caller", known_stat])
                for link in node.callers.values():
                    baseline_overall[stat] += link.stats.baseline[known_stat]
                    target_overall[stat] += link.stats.target[known_stat]
        for stat in baseline_overall:
            baseline, target = baseline_overall[stat], target_overall[stat]
            if baseline != 0 or target != 0:
                abs_diff = target - baseline
                rel_diff = round(100 * abs_diff / max(baseline, target), 2)
                stats.append([stat, abs_diff, rel_diff])
        stats = sorted(stats, key=itemgetter(2))
        if all(val == 0 for val in baseline_overall.values()):
            state = "new"
        elif all(val == 0 for val in target_overall.values()):
            state = "removed"
        else:
            state = "(un)changed"

        # Prepare trace stats
        uid_stats = trace_stats[uid]
        uid_trace_stats = []
        for trace in uid_stats:
            # Trace is in form of [short_trace, stat_type, abs, rel, long_trace]
            for stat in Stats.KnownStats:
                abs_amount = trace.target_cost[stat] - trace.baseline_cost[stat]
                rel_amount = (trace.target_cost[stat] - trace.baseline_cost[stat]) / max(
                    trace.target_cost[stat], trace.baseline_cost[stat]
                )

                short_id = ";".join(
                    [
                        str(graph.uid_to_id[trace.trace[0]]),
                        str(graph.uid_to_id[trace.trace[-1]]),
                    ]
                )
                uid_trace_stats.append(
                    [
                        short_id,
                        stat,
                        abs_amount,
                        rel_amount,
                        [graph.uid_to_id[t] for t in trace.trace],
                    ]
                )
        uid_trace_stats = sorted(uid_trace_stats, key=itemgetter(3))
        selection.append(
            SelectionRow(
                uid,
                graph.uid_to_id[uid],
                state,
                stats[0][0],
                stats[0][1],
                stats[0][2],
                stats,
                uid_trace_stats[:3],
            )
        )
    return selection


def generate_sankey_difference(lhs_profile: Profile, rhs_profile: Profile, **kwargs: Any) -> None:
    """Generates differences of two profiles as sankey diagram

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :param kwargs: additional arguments
    """
    # We automatically set the value of True for kperf, which samples
    if lhs_profile.get("collector_info", {}).get("name") == "kperf":
        Config().trace_is_inclusive = True

    log.major_info("Generating Sankey Graph Difference")

    graph = Graph()

    process_traces(lhs_profile, "baseline", graph)
    process_traces(rhs_profile, "target", graph)

    trace_stats = generate_trace_stats(graph)
    selection_table = generate_selection(graph, trace_stats)
    skeletons = generate_skeletons(graph, trace_stats)
    log.minor_success("Sankey graphs", "generated")

    env = jinja2.Environment(loader=jinja2.PackageLoader("perun", "templates"))
    template = env.get_template("diff_view_sankey_incremental.html.jinja2")
    content = template.render(
        title="Differences of profiles (with sankey)",
        lhs_tag="Baseline (base)",
        lhs_header=flamegraph_run.generate_header(lhs_profile),
        rhs_tag="Target (tgt)",
        rhs_header=flamegraph_run.generate_header(rhs_profile),
        palette=ColorPalette,
        caller_graph=graph.to_jinja_string("callers"),
        callee_graph=graph.to_jinja_string("callees"),
        stat_list=list(Stats.KnownStats),
        stats="["
        + ",".join(
            list(map(itemgetter(0), sorted(list(graph.stats_to_id.items()), key=itemgetter(1))))
        )
        + "]",
        nodes=list(map(itemgetter(0), sorted(list(graph.uid_to_id.items()), key=itemgetter(1)))),
        node_map=[
            sorted([node.get_order() for node in nodes])
            for nodes in map(
                itemgetter(1),
                sorted(list(graph.uid_to_nodes.items()), key=lambda x: graph.uid_to_id[x[0]]),
            )
        ],
        skeletons=skeletons,
        selection_table=selection_table,
    )
    log.minor_success("HTML template", "rendered")
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
