"""HTML report difference of the profiles"""
from __future__ import annotations

# Standard Imports
from dataclasses import dataclass
from typing import Any

# Third-Party Imports
import click
import jinja2
import progressbar

# Perun Imports
from perun.utils import log
from perun.utils.common import diff_kit, traces_kit
from perun.profile.factory import Profile
from perun.profile import convert
from perun.view_diff.flamegraph import run as flamegraph_run
from perun.view_diff.table import run as table_run


PRECISION: int = 2


@dataclass
class TableRecord:
    """Represents single record on top of the consumption

    :ivar uid: uid of the records
    :ivar trace: trace of the record
    :ivar abs_amount: absolute value of the uid
    :ivar rel_amount: relative value of the uid
    """

    __slots__ = [
        "uid",
        "trace",
        "cluster",
        "short_trace",
        "trace_list",
        "abs_amount",
        "rel_amount",
        "type",
    ]

    uid: str
    trace: str
    cluster: traces_kit.TraceCluster
    short_trace: str
    trace_list: [str]
    abs_amount: float
    rel_amount: float
    type: str


def to_short_trace(trace: str) -> str:
    """Converts longer traces to short representation

    :param trace: trace, delimited by ','
    :return: shorter representation of trace
    """
    split_trace = trace.split(",")
    if len(split_trace) <= 3:
        return trace
    return " -> ".join([split_trace[0], "...", split_trace[-1]])


@dataclass
class TraceInfo:
    """Trace info is a helper structure that encapsulates selected characteristics of traces.

    :ivar long_str: long representation of trace (calls delimited by `,`)
    :ivar short_str: shorter string ideal for display
    :ivar cluster: classified cluster
    """

    __slots__ = ["long_str", "short_str", "cluster"]
    long_str: str
    short_str: str
    cluster: traces_kit.TraceCluster

    def __eq__(self, other):
        """Two trace infos are equal if their long strings are equals

        :param other: other traceinfo
        :return: whether two objects are equal
        """
        if isinstance(other, TraceInfo):
            return self.long_str == other.long_str
        return False

    def __hash__(self):
        """Hash of the trace info is hash of its long string

        :return: hash of the trace info
        """
        return hash(self.long_str)

    def __lt__(self, other):
        """Trace infos are sorted by their long strings

        :return: comparison result
        """
        if isinstance(other, TraceInfo):
            return self.long_str < other.long_str
        raise TypeError("Cannot compare TraceInfo with other types.")


def profile_to_data(
    profile: Profile, classifier: traces_kit.TraceClassifier
) -> tuple[list[TableRecord], list[str]]:
    """Converts profile to list of columns and list of list of values

    :param profile: converted profile
    :param classifier: classifier used for classification of traces
    :return: list of columns and list of rows and list of selectable types
    """
    df = convert.resources_to_pandas_dataframe(profile)

    # Convert traces to some trace objects
    trace_info_map = {}
    for trace in progressbar.progressbar(df["trace"].unique()):
        trace_as_list = trace.split(",")
        long_trace = ",".join(
            traces_kit.fold_recursive_calls_in_trace(trace_as_list, generalize=True)
        )
        trace_info_map[trace] = TraceInfo(
            long_trace, to_short_trace(long_trace), classifier.classify_trace(long_trace.split(","))
        )

    def process_traces(value: str) -> TraceInfo:
        """Converts single string to trace info

        :param value: single string
        """
        return trace_info_map[value]

    df["trace"] = df["trace"].apply(process_traces)

    # TODO: This could be more effective
    data = []
    aggregation_keys = diff_kit.get_candidate_keys(df.columns)
    for aggregation_key in aggregation_keys:
        grouped_df = df.groupby(["uid", "trace"]).agg({aggregation_key: "sum"}).reset_index()
        sorted_df = grouped_df.sort_values(by=aggregation_key, ascending=False)
        amount_sum = df[aggregation_key].sum()
        for _, row in progressbar.progressbar(sorted_df.iterrows()):
            data.append(
                TableRecord(
                    row["uid"],
                    row["trace"].long_str,
                    row["trace"].cluster,
                    row["trace"].short_str,
                    table_run.generate_trace_list(row["trace"].long_str, row["uid"]),
                    row[aggregation_key],
                    round(100 * row[aggregation_key] / amount_sum, PRECISION),
                    aggregation_key,
                )
            )
    return data, aggregation_keys


def generate_html_report(lhs_profile: Profile, rhs_profile: Profile, **kwargs: Any) -> None:
    """Generates HTML report of differences

    :param lhs_profile: baseline profile
    :param rhs_profile: target profile
    :param kwargs: other parameters
    """
    classifier = traces_kit.TraceClassifier()
    log.major_info("Generating HTML Report", no_title=True)
    lhs_data, lhs_types = profile_to_data(lhs_profile, classifier)
    log.minor_success("Baseline data", "generated")
    rhs_data, rhs_types = profile_to_data(rhs_profile, classifier)
    log.minor_success("Target data", "generated")
    # TODO: Remake obtaining of the unit somehow
    data_types = diff_kit.get_candidate_keys(set(lhs_types).union(set(rhs_types)))
    columns = [
        ("uid", "The measured symbol (click [+] for full trace)."),
        ("[unit]", "The absolute measured value."),
        ("[%]", "The relative measured value (in percents overall)."),
    ]

    env = jinja2.Environment(loader=jinja2.PackageLoader("perun", "templates"))
    template = env.get_template("diff_view_report.html.jinja2")
    content = template.render(
        lhs_tag="Baseline (base)",
        lhs_columns=columns,
        lhs_data=lhs_data,
        lhs_header=flamegraph_run.generate_header(lhs_profile),
        rhs_tag="Target (tgt)",
        rhs_columns=columns,
        rhs_data=rhs_data,
        rhs_header=flamegraph_run.generate_header(rhs_profile),
        data_types=data_types,
        cluster_types=sorted(traces_kit.TraceCluster.cluster_dict.keys()),
        title="Difference of profiles (with tables)",
    )
    log.minor_success("HTML report ", "generated")
    output_file = diff_kit.save_diff_view(
        kwargs.get("output_file"), content, "report", lhs_profile, rhs_profile
    )
    log.minor_status("Output saved", log.path_style(output_file))


@click.command()
@click.option("-o", "--output-file", help="Sets the output file (default=automatically generated).")
@click.pass_context
def report(ctx: click.Context, *_: Any, **kwargs: Any) -> None:
    assert ctx.parent is not None and f"impossible happened: {ctx} has no parent"
    profile_list = ctx.parent.params["profile_list"]
    generate_html_report(profile_list[0], profile_list[1], **kwargs)
