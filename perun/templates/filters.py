"""
Filters contain custom filters for working with jinja2
"""
from __future__ import annotations

# Standard Imports

# Third-Party Imports

# Perun Imports

def sanitize_variable_name(value: str) -> str:
    """Sanitiezes the string so it is applicable as variable name

    :param value: string to be sanitized
    :return: sanitized value
    """
    invalid_characters = r"[]#%{}\<>*?/$!'\":@"
    sanitized_str = "".join(
        "" if c in invalid_characters else ("_" if c == " " else c) for c in str(value)
    )
    sanitized_str = sanitized_str.replace("%", "pct")
    sanitized_str = sanitized_str.replace(" ", "_")
    return sanitized_str
