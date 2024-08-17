"""Helper module for mapping various aspects of perun

Mainly, this currently holds a mapping of keys in profiles to human readable equivalents
"""

from __future__ import annotations

# Standard Imports
import re

# Third-Party Imports

# Perun Imports
from perun.logic import config
from perun.utils import log


def get_readable_key(key: str) -> str:
    """For given key returns a human-readable key

    :param key: transformed key
    :return: human readable key
    """
    profiles = config.runtime().safe_get("context.profiles", default=[])
    if key == "amount":
        if all(
            p.get("collector_info", {}).get("name") == "kperf"
            for p in profiles
            if "collector_info" in p
        ):
            return "Inclusive Samples [#]"
        if all(
            p.get("collector_info", {}).get("name") == "memory"
            for p in profiles
            if "collector_info" in p
        ):
            return "Allocated Memory [B]"
    if key == "ncalls":
        return "Number of Calls [#]"
    if key == "benchmarking.time":
        return "Benchmarking Time [ms]"
    if key in ("I Mean", "I Max", "I Min"):
        return key.replace("I ", "Inclusive ") + " [ms]"
    if key in ("E Mean", "E Max", "E Min"):
        return key.replace("E ", "Exclusive ") + " [ms]"
    return key


def from_readable_key(key: str) -> str:
    """For given key returns a human-readable key

    :param key: transformed key
    :return: human readable key
    """
    if key == "Inclusive Samples [#]":
        return "amount"
    if key == "Allocated Memory [B]":
        return "amount"
    if key == "Number of Calls [#]":
        return "ncalls"
    if key == "Benchmarking Time [ms]":
        return "benchmarking.time"
    if key in ("Inclusive Mean [ms]", "Inclusive Max [ms]", "Inclusive Min [ms]"):
        return key.replace("Inclusive ", "I ").replace(" [ms]", "")
    if key in ("Exclusive Mean [ms]", "Exclusive Max [ms]", "Exclusive Min [ms]"):
        return key.replace("Exclusive ", "E ").replace(" [ms]", "")

    return key


def get_unit(key: str) -> str:
    """Returns unit forgiven any key"""
    if "[#]" in key:
        return "samples"
    if m := re.search(r"\[(?P<unit>[^]]+)\]", key):
        return m.group("unit")
    else:
        log.warn(f"Unregistered unit for '{key}'")
        return "?"
