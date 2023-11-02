"""Module for automatic recognizing file type and choosing appropriate fuzzing rules."""
from __future__ import annotations

import mimetypes
import re

from typing import Callable, Optional

import perun.fuzz.randomizer as randomizer

import perun.fuzz.methods.binary as binary
import perun.fuzz.methods.xml as xml
import perun.fuzz.methods.textfile as textfile
from perun.fuzz.structs import RuleSet


def custom_rules(
    regex_rules: dict[str, str],
    fuzzing_methods: list[tuple[Callable[[list[str]], None], str]],
) -> None:
    """Adds custom rules specified by regexps, and read from the file in YAML format.
    Format:
        del: add
        ([0-9]{6}),([0-9]{2}): \\1.\\2
        (\\w+)=(\\w+): \\2=\\1

    :param dict regex_rules: dict of custom regular expression rules
    :param list fuzzing_methods: list of functions, fuzzing (mutation) strategies
    """
    for key, value in regex_rules.items():

        def custom_rule(lines: list[str]) -> None:
            comp_regexp = re.compile(key, flags=re.IGNORECASE)
            index = randomizer.rand_index(len(lines))
            lines[index] = comp_regexp.sub(value, lines[index])

        fuzzing_methods.append(
            (custom_rule, 'User rule: "' + key + '" -> "' + value + '"')
        )


def get_filetype(file: str) -> tuple[bool, Optional[str]]:
    """Recognizes whether file is binary or text and its type using `mimetypes`

    Fixme: this might need refactoring to handle some edge cases

    :param str file: file name
    :return tuple: is_file_binary, file_type
    """
    try:
        guessed_type = (mimetypes.guess_type(file))[0]
        subfiletype = guessed_type.split("/")[-1] if guessed_type is not None else None
        filetype = guessed_type.split("/")[0] if guessed_type is not None else None
        return (
            subfiletype is None or (filetype != "text" and subfiletype != "xml"),
            subfiletype,
        )
    except AttributeError:
        return True, None


def choose_ruleset(file: str, regex_rules: dict[str, str]) -> RuleSet:
    """Automatically collects appropriate fuzz methods according to file type.

    :param str file: path to file
    :param dict regex_rules: dict of custom regex rules
    :return list: list of tuples fuzz_method_function, description
    """
    fuzzing_methods: list[tuple[Callable[[list[str]], None], str]] = []
    if regex_rules:
        custom_rules(regex_rules, fuzzing_methods)

    is_binary, filetype = get_filetype(file)
    if is_binary:
        fuzzing_methods.extend(binary.FUZZING_METHODS)
    else:
        if filetype in ["xml", "html", "svg", "xhtml", "xul"]:
            fuzzing_methods.extend(xml.FUZZING_METHODS)
        fuzzing_methods.extend(textfile.FUZZING_METHODS)

    # last element is for total num of cov increases or perf degradations
    return RuleSet(fuzzing_methods, [0] * (len(fuzzing_methods) + 1))
