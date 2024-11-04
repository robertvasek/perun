from __future__ import annotations

# Standard Imports
import enum

# Third-Party Imports

# Perun Imports


class HeaderDisplayStyle(enum.Enum):
    """Supported styles of displaying profile specification and metadata."""

    FULL = "full"
    DIFF = "diff"

    @staticmethod
    def supported() -> list[str]:
        """Obtain the collection of supported display styles.

        :return: the collection of valid display styles
        """
        return [style.value for style in HeaderDisplayStyle]

    @staticmethod
    def default() -> str:
        """Provide the default display style.

        :return: the default display style
        """
        return HeaderDisplayStyle.FULL.value
