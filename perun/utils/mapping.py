"""Helper module for mapping various aspects of perun

Mainly, this currently holds a mapping of keys in profiles to human readable equivalents
"""
from __future__ import annotations

# Standard Imports

# Third-Party Imports

# Perun Imports
from perun.utils import decorators
from perun.logic import config


@decorators.singleton_with_args
def get_readable_key(key: str) -> str:
    """For given key returns a human-readable key

    :param key: transformed key
    :return: human readable key
    """
    profiles = config.runtime().get("context.profiles")
    if key == "amount":
        if all(p.get("collector_info", {}).get("name") == "kperf" for p in profiles):
            return "Inclusive Samples"
    return key
