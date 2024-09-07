from __future__ import annotations

# Standard Imports
from collections import Counter
import dataclasses
import enum
import statistics
from typing import Any, Protocol, Iterable, ClassVar, Union, cast

# Third-Party Imports

# Perun Imports
from perun.utils import log as perun_log


class ProfileStatOrdering(str, enum.Enum):
    # The class is derived from str so that ProfileStat serialization using asdict() works properly
    AUTO = "auto"
    HIGHER = "higher_is_better"
    LOWER = "lower_is_better"
    EQUALITY = "equality"

    @staticmethod
    def supported() -> set[str]:
        return {ordering.value for ordering in ProfileStatOrdering}

    @staticmethod
    def default() -> ProfileStatOrdering:
        return ProfileStatOrdering.AUTO


class StatComparisonResult(enum.Enum):
    EQUAL = 1
    UNEQUAL = 2
    BASELINE_BETTER = 3
    TARGET_BETTER = 4
    INVALID = 5


@dataclasses.dataclass
class ProfileStat:
    name: str
    ordering: ProfileStatOrdering = ProfileStatOrdering.default()
    unit: str = "#"
    aggregate_by: str = ""
    tooltip: str = ""
    value: object = ""

    @classmethod
    def from_string(
        cls,
        name: str = "[empty]",
        ordering: str = "",
        unit: str = "#",
        aggregate_by: str = "",
        tooltip: str = "",
        *_: Any,
    ) -> ProfileStat:
        if name == "[empty]":
            # Invalid stat specification, warn
            perun_log.warn("Empty profile stat specification. Creating a dummy '[empty]' stat.")
        ordering_enum = cls._convert_ordering(ordering)
        return cls(name, ordering_enum, unit, aggregate_by, tooltip)

    @classmethod
    def from_profile(cls, stat: dict[str, Any]) -> ProfileStat:
        stat["ordering"] = cls._convert_ordering(stat.get("ordering", ""))
        return cls(**stat)

    @staticmethod
    def _convert_ordering(ordering: str) -> ProfileStatOrdering:
        if not ordering:
            return ProfileStatOrdering.default()
        try:
            return ProfileStatOrdering(ordering.strip())
        except ValueError:
            # Invalid stat ordering, warn
            perun_log.warn(
                f"Unknown stat ordering: {ordering}. Using the default stat ordering value instead."
                f" Please choose one of ({', '.join(ProfileStatOrdering.supported())})."
            )
            return ProfileStatOrdering.default()


class ProfileStatAggregation(Protocol):
    _SUPPORTED_KEYS: ClassVar[set[str]] = set()
    _DEFAULT_KEY: ClassVar[str] = ""

    def normalize_aggregate_key(self, key: str = _DEFAULT_KEY) -> str:
        if key not in self._SUPPORTED_KEYS:
            if key:
                # A key was provided, but it is an invalid one
                perun_log.warn(
                    f"{self.__class__.__name__}: Invalid aggregate key '{key}'. "
                    f"Using the default key '{self._DEFAULT_KEY}' instead."
                )
            key = self._DEFAULT_KEY
        return key

    def get_value(self, key: str = _DEFAULT_KEY) -> Any:
        return getattr(self, self.normalize_aggregate_key(key))

    def infer_auto_ordering(self, ordering: ProfileStatOrdering) -> ProfileStatOrdering: ...

    # header, table
    def as_table(
        self, key: str = _DEFAULT_KEY
    ) -> tuple[str | float | tuple[str, int], dict[str, Any]]: ...


@dataclasses.dataclass
class SingleValue(ProfileStatAggregation):
    _SUPPORTED_KEYS: ClassVar[set[str]] = {"value"}
    _DEFAULT_KEY = "value"

    value: str | float = "[missing]"

    def infer_auto_ordering(self, ordering: ProfileStatOrdering) -> ProfileStatOrdering:
        if ordering != ProfileStatOrdering.AUTO:
            return ordering
        if isinstance(self.value, str):
            return ProfileStatOrdering.EQUALITY
        return ProfileStatOrdering.HIGHER

    def as_table(self, _: str = "") -> tuple[str | float, dict[str, str | float]]:
        # There are no details of a single value to generate into a table
        return self.value, {}


@dataclasses.dataclass
class StatisticalSummary(ProfileStatAggregation):
    _SUPPORTED_KEYS: ClassVar[set[str]] = {
        "min",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "max",
        "mean",
    }
    _DEFAULT_KEY: ClassVar[str] = "median"

    min: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    median: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    max: float = 0.0
    mean: float = 0.0

    @classmethod
    def from_values(cls, values: Iterable[float]) -> StatisticalSummary:
        # We assume there aren't too many values so that multiple passes of the list don't matter
        # too much. If this becomes a bottleneck, we can use pandas describe() instead.
        values = list(values)
        quantiles = statistics.quantiles(values, n=20, method="inclusive")
        return cls(
            float(min(values)),
            quantiles[2],  # p10
            quantiles[5],  # p25
            quantiles[10],  # p50
            quantiles[15],  # p75
            quantiles[18],  # p90
            float(max(values)),
            statistics.mean(values),
        )

    def infer_auto_ordering(self, ordering: ProfileStatOrdering) -> ProfileStatOrdering:
        if ordering != ProfileStatOrdering.AUTO:
            return ordering
        return ProfileStatOrdering.HIGHER

    def as_table(self, key: str = _DEFAULT_KEY) -> tuple[float, dict[str, float]]:
        return self.get_value(key), dataclasses.asdict(self)


@dataclasses.dataclass
class StringCollection(ProfileStatAggregation):
    _SUPPORTED_KEYS: ClassVar[set[str]] = {
        "total",
        "unique",
        "min_count",
        "max_count",
        "counts",
        "sequence",
    }
    _DEFAULT_KEY: ClassVar[str] = "unique"

    sequence: list[str] = dataclasses.field(default_factory=list)
    counts: Counter[str] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.counts = Counter(self.sequence)

    @property
    def unique(self) -> int:
        return len(self.counts)

    @property
    def total(self) -> int:
        return len(self.sequence)

    @property
    def min_count(self) -> tuple[str, int]:
        return self.counts.most_common()[-1]

    @property
    def max_count(self) -> tuple[str, int]:
        return self.counts.most_common()[0]

    def infer_auto_ordering(self, ordering: ProfileStatOrdering) -> ProfileStatOrdering:
        if ordering != ProfileStatOrdering.AUTO:
            return ordering
        return ProfileStatOrdering.EQUALITY

    def as_table(
        self, key: str = _DEFAULT_KEY
    ) -> tuple[int | str | tuple[str, int], dict[str, int] | dict[str, str]]:
        header: str | int | tuple[str, int]
        if key in ("counts", "sequence"):
            # The Counter and list objects are not suitable for direct printing in a table.
            header = f"[{key}]"
        else:
            # A little type hinting help. The list and Counter types have already been covered.
            header = cast(Union[str, int, tuple[str, int]], self.get_value(key))
        if key == "sequence":
            # The 'sequence' key table format is a bit different from the rest.
            return header, {f"{idx}.": value for idx, value in enumerate(self.sequence)}
        return header, self.counts


def aggregate_stats(stat: ProfileStat) -> ProfileStatAggregation:
    if isinstance(stat.value, (str, float, int)):
        return SingleValue(stat.value)
    if isinstance(stat.value, Iterable):
        # Iterable types are converted to a list
        values = list(stat.value)
        if len(values) == 0:
            perun_log.warn(f"ProfileStat aggregation: Missing value of stat '{stat.name}'")
            return SingleValue()
        elif len(values) == 1:
            return SingleValue(values[0])
        elif all(isinstance(value, (int, float)) for value in values):
            # All values are integers or floats
            return StatisticalSummary.from_values(map(float, values))
        else:
            # Even heterogeneous lists will be aggregated as lists of strings
            return StringCollection(list(map(str, values)))
    perun_log.warn(
        f"ProfileStat aggregation: Unknown type '{type(stat.value)}' of stat '{stat.name}'"
    )
    return SingleValue()


def compare_stats(
    stat: ProfileStatAggregation,
    other_stat: ProfileStatAggregation,
    key: str,
    ordering: ProfileStatOrdering,
) -> StatComparisonResult:
    value, other_value = stat.get_value(key), other_stat.get_value(key)
    # Handle auto ordering according to the aggregation type
    ordering = stat.infer_auto_ordering(ordering)
    if type(stat) is not type(other_stat):
        # Invalid comparison attempt
        perun_log.warn(
            f"Invalid comparison of {stat.__class__.__name__} and {other_stat.__class__.__name__}."
        )
        return StatComparisonResult.INVALID
    if value == other_value:
        # The values are the same, the result is the same regardless of the ordering used
        return StatComparisonResult.EQUAL
    if ordering == ProfileStatOrdering.EQUALITY:
        # The values are different and we compare for equality
        return StatComparisonResult.UNEQUAL
    elif value > other_value:
        return (
            StatComparisonResult.BASELINE_BETTER
            if ordering == ProfileStatOrdering.HIGHER
            else StatComparisonResult.TARGET_BETTER
        )
    elif value < other_value:
        return (
            StatComparisonResult.BASELINE_BETTER
            if ordering == ProfileStatOrdering.LOWER
            else StatComparisonResult.TARGET_BETTER
        )
    return StatComparisonResult.UNEQUAL
