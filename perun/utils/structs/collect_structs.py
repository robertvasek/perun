# Standard Imports
from enum import Enum

# Third-Party Imports
# Perun Imports


class Optimizations(Enum):
    """Enumeration of the implemented methods and their CLI name."""

    BASELINE_STATIC = "baseline-static"
    BASELINE_DYNAMIC = "baseline-dynamic"
    CALL_GRAPH_SHAPING = "cg-shaping"
    DYNAMIC_SAMPLING = "dynamic-sampling"
    DIFF_TRACING = "diff-tracing"
    DYNAMIC_PROBING = "dynamic-probing"
    TIMED_SAMPLING = "timed-sampling"

    @staticmethod
    def supported():
        """List the currently supported optimization methods.

        :return: CLI names of the supported optimizations
        """
        return [optimization.value for optimization in Optimizations]


class Pipeline(Enum):
    """Enumeration of the implemented pipelines and their CLI name.
    Custom represents a defualt pipeline that has no pre-configured methods or parameters
    """

    CUSTOM = "custom"
    BASIC = "basic"
    ADVANCED = "advanced"
    FULL = "full"

    @staticmethod
    def supported():
        """List the currently supported optimization pipelines.

        :return: CLI names of the supported pipelines
        """
        return [pipeline.value for pipeline in Pipeline]

    @staticmethod
    def default():
        """Name of the default pipeline.

        :return: the CLI name of the default pipeline
        """
        return Pipeline.CUSTOM.value

    def map_to_optimizations(self):
        """Map the selected optimization pipeline to the set of employed optimization methods.

        :return: list of the Optimizations enumeration objects
        """
        if self == Pipeline.BASIC:
            return [Optimizations.CALL_GRAPH_SHAPING, Optimizations.BASELINE_DYNAMIC]
        if self == Pipeline.ADVANCED:
            return [
                Optimizations.DIFF_TRACING,
                Optimizations.CALL_GRAPH_SHAPING,
                Optimizations.BASELINE_DYNAMIC,
                Optimizations.DYNAMIC_SAMPLING,
            ]
        if self == Pipeline.FULL:
            return [
                Optimizations.DIFF_TRACING,
                Optimizations.CALL_GRAPH_SHAPING,
                Optimizations.BASELINE_STATIC,
                Optimizations.BASELINE_DYNAMIC,
                Optimizations.DYNAMIC_SAMPLING,
                Optimizations.DYNAMIC_PROBING,
            ]
        return []


class CallGraphTypes(Enum):
    """Enumeration of the implemented call graph types and their CLI names."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    MIXED = "mixed"

    @staticmethod
    def supported():
        """List the currently supported call graph types.

        :return: CLI names of the supported cg types
        """
        return [cg.value for cg in CallGraphTypes]

    @staticmethod
    def default():
        """Name of the default cg type.

        :return: the CLI name of the default cg type
        """
        return CallGraphTypes.STATIC.value


class Parameters(Enum):
    """Enumeration of the currently supported CLI options for optimization methods and pipelines."""

    DIFF_VERSION = "diff-version"
    DIFF_KEEP_LEAF = "diff-keep-leaf"
    DIFF_INSPECT_ALL = "diff-inspect-all"
    DIFF_CG_MODE = "diff-cfg-mode"
    SOURCE_FILES = "source-files"
    SOURCE_DIRS = "source-dirs"
    STATIC_COMPLEXITY = "static-complexity"
    STATIC_KEEP_TOP = "static-keep-top"
    CG_SHAPING_MODE = "cg-mode"
    CG_PROJ_LEVELS = "cg-proj-levels"
    CG_PROJ_KEEP_LEAF = "cg-proj-keep-leaf"
    DYNSAMPLE_STEP = "dyn-sample-step"
    DYNSAMPLE_THRESHOLD = "dyn-sample-threshold"
    PROBING_THRESHOLD = "probing-threshold"
    PROBING_REATTACH = "probing-reattach"
    TIMEDSAMPLE_FREQ = "timed-sample-freq"
    DYNBASE_SOFT_THRESHOLD = "dyn-base-soft-threshold"
    DYNBASE_HARD_THRESHOLD = "dyn-base-hard-threshold"
    THRESHOLD_MODE = "threshold-mode"

    @staticmethod
    def supported():
        """List the currently supported optimization parameters.

        :return: CLI names of the supported parameters
        """
        return [parameter.value for parameter in Parameters]
