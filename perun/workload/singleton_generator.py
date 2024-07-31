"""Singleton Generator generates only one single value. This generator corresponds to the default
behaviour of Perun, i.e. when each specified workload in :munit:`workloads` was passed to profiled
program as string.

Currently, be default any string specified in :munit:`workloads`, that does not correspond to some
generator specified in :ckey:`generators.workload`, is converted to Singleton Generator.

The Singleton Generator can be configured by following options:

  * ``value``: singleton value that is passed as workload.

"""

from __future__ import annotations

# Standard Imports
from typing import Any, Iterable

# Third-Party Imports

# Perun Imports
from perun.utils.structs.common_structs import Job
from perun.workload.generator import WorkloadGenerator


class SingletonGenerator(WorkloadGenerator):
    """Generator of singleton values

    :ivar value: singleton value used as workload
    """

    __slots__ = ["value"]

    def __init__(self, job: Job, value: Any, **kwargs: Any) -> None:
        """Initializes the generator of singleton workload

        :param job: job for which we are generating the workloads
        :param value: singleton value that is used as workload
        """
        super().__init__(job, **kwargs)

        self.value: Any = value

    def _generate_next_workload(self) -> Iterable[tuple[Any, dict[str, Any]]]:
        """Generates the next integer as the workload

        :return: single value
        """
        yield self.value, {}
