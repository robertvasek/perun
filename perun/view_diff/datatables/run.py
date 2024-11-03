"""HTML report difference of the profiles"""

from __future__ import annotations

# Standard Imports
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Third-Party Imports
import click

# Perun Imports
import perun
from perun.templates import factory as templates
from perun.utils import log
from perun.utils.common import diff_kit, traces_kit
from perun.profile.factory import Profile
from perun.profile import convert
from perun.view_diff.short import run as table_run


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
    trace_list: list[str]
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
        return isinstance(other, TraceInfo) and self.long_str == other.long_str

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


def construct_filters(kwargs: dict[str, Any]) -> list[tuple[str, str]]:
    """Constructs list of filters for filtering the dataframe

    :param kwargs: arguments from command line
    :return: list of callable functions on pandas series
    """
    return [
        ("ncalls", f"ncalls > {kwargs.get('filter_by_calls', 1)}"),
        (
            "Total Inclusive T [ms]",
            f"`Total Inclusive T [ms]` > {kwargs.get('filter_by_time', 0.01)}",
        ),
        (
            "Total Exclusive T [ms]",
            f"`Total Exclusive T [ms]` > {kwargs.get('filter_by_time', 0.01)}",
        ),
    ]


def profile_to_data(
    profile: Profile,
    classifier: traces_kit.TraceClassifier,
    filters: list[tuple[str, str]],
) -> tuple[list[TableRecord], list[str]]:
    """Converts profile to list of columns and list of list of values

    :param profile: converted profile
    :param classifier: classifier used for classification of traces
    :param filters: list of filters for filtering the data
    :return: list of columns and list of rows and list of selectable types
    """
    df = convert.resources_to_pandas_dataframe(profile)
    applicable_filters = " & ".join(f[1] for f in filters if f[0] in df.columns)
    if applicable_filters:
        log.minor_status("Applying filters", f"{log.highlight(applicable_filters)}")
        df = df.query(applicable_filters)

    # Convert traces to some trace objects
    trace_info_map = {}
    for trace in log.progress(df["trace"].unique(), description="Converting Traces"):
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

    df["trace"] = df["trace"].apply(process_traces)  # type: ignore

    # TODO: This could be more effective
    data = []
    aggregation_keys = diff_kit.get_candidate_keys(df.columns)
    for aggregation_key in aggregation_keys:
        grouped_df = df.groupby(["uid", "trace"]).agg({aggregation_key: "sum"}).reset_index()
        sorted_df = grouped_df.sort_values(by=aggregation_key, ascending=False)
        amount_sum = df[aggregation_key].sum()
        for _, row in log.progress(sorted_df.iterrows(), description="Processing Traces"):
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
    filters = construct_filters(kwargs)
    classifier = traces_kit.TraceClassifier(
        strategy=traces_kit.ClassificationStrategy(kwargs.get("classify_traces_by", "identity")),
        threshold=0.5,
    )
    log.major_info("Generating HTML Report", no_title=True)
    lhs_data, lhs_types = profile_to_data(lhs_profile, classifier, filters)
    log.minor_success("Baseline data", "generated")
    rhs_data, rhs_types = profile_to_data(rhs_profile, classifier, filters)
    log.minor_success("Target data", "generated")
    # TODO: Remake obtaining of the unit somehow
    data_types = diff_kit.get_candidate_keys(set(lhs_types).union(set(rhs_types)))
    columns = [
        ("uid", "The measured symbol (click [+] for full trace)."),
        ("[unit]", "The absolute measured value."),
        ("[%]", "The relative measured value (in percents overall)."),
    ]

    lhs_header, rhs_header = diff_kit.generate_diff_of_headers(
        diff_kit.generate_specification(lhs_profile), diff_kit.generate_specification(rhs_profile)
    )
    template = templates.get_template("diff_view_datatables.html.jinja2")
    content = template.render(
        perun_version=perun.__version__,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC",
        lhs_tag="Baseline (base)",
        lhs_columns=columns,
        lhs_data=lhs_data,
        lhs_header=lhs_header,
        rhs_tag="Target (tgt)",
        rhs_columns=columns,
        rhs_data=rhs_data,
        rhs_header=rhs_header,
        data_types=data_types,
        cluster_types=sorted(list(traces_kit.TraceCluster.id_set)),
        title="Difference of profiles (with tables)",
    )
    log.minor_success("HTML report ", "generated")
    output_file = diff_kit.save_diff_view(
        kwargs.get("output_file"), content, "datatables", lhs_profile, rhs_profile
    )
    log.minor_status("Output saved", log.path_style(output_file))


@click.command()
@click.option("-o", "--output-file", help="Sets the output file (default=automatically generated).")
@click.option(
    "-c",
    "--classify-traces-by",
    help="Classifies the traces by selected classifier. "
    "One of: 1) identity (each uid is classified to its own cluster); "
    "2) best-fit (each uid is classified to best cluster); "
    "3) first-fit (each uid is classified to first suitable cluster). "
    "(default=identity)",
    type=click.Choice(["identity", "best-fit", "first-fit"]),
    default="identity",
)
@click.option(
    "-fc",
    "--filter-by-calls",
    nargs=1,
    help="Filters records that have less or equal calls than [INT] (default=1)",
    type=click.INT,
    default=1,
)
@click.option(
    "-ft",
    "--filter-by-time",
    nargs=1,
    help="Filters records based on the 'Total {Inclusive,Exclusive} Time T [ms]' column. "
    "It filters values that are lesser or equal than [FLOAT] (default=0.1).",
    type=click.FLOAT,
    default=0.1,
)
@click.pass_context
def datatables(ctx: click.Context, *_: Any, **kwargs: Any) -> None:
    assert ctx.parent is not None, f"impossible happened: {ctx} has no parent"
    profile_list = ctx.parent.params["profile_list"]
    generate_html_report(profile_list[0], profile_list[1], **kwargs)
