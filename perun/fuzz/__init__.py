"""Collective package for fuzz performance testing

Contains actual fuzzing rules for different types of workloads,
the fuzzing loop and strategies/heuristics for enqueueing newly discovered
workloads for further fuzzing.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
