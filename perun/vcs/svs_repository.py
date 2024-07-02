"""SVS is a dummy Single Version System, which tracks the results as a single version, nothing more

Contains concrete implementation of the functions needed by Perun to work without any version control system,
like GIT, SVN, or other.
"""

from __future__ import annotations

# Standard Imports
from typing import Any, Iterator
import getpass
import os
import socket
import sys
import time


# Third-Party Imports

# Perun Imports
from perun.utils.structs import MajorVersion, MinorVersion
from perun.vcs.abstract_repository import AbstractRepository

SINGLE_VERSION_BRANCH: str = "master"
SINGLE_VERSION_TAG: str = "HEAD"


def get_time_of_creation(path: str) -> str:
    return time.ctime(
        os.path.getctime(path) if sys.platform.startswith("win") else os.stat(path).st_ctime
    )


class SvsRepository(AbstractRepository):
    def __init__(self, vcs_path: str) -> None:
        super().__init__()
        self.vcs_path: str = vcs_path

    def get_minor_head(self) -> str:
        """Returns always single version tag --- the same version"""
        return SINGLE_VERSION_TAG

    def init(self, _: dict[str, Any]) -> bool:
        """Svs is always initialized"""
        return True

    def walk_minor_versions(self, _: str) -> Iterator[MinorVersion]:
        """Yield single version"""
        yield self.get_minor_version_info(SINGLE_VERSION_TAG)

    def walk_major_versions(self) -> Iterator[MajorVersion]:
        """Yields single branch with single version"""
        yield MajorVersion(SINGLE_VERSION_BRANCH, SINGLE_VERSION_TAG)

    def get_minor_version_info(self, minor_version: str) -> MinorVersion:
        """Returns valid minor version info only for a SINGLE_VERSION TAG"""
        assert minor_version == SINGLE_VERSION_TAG, f"unknown version `{minor_version}`"
        return MinorVersion(
            get_time_of_creation(self.vcs_path),
            getpass.getuser(),
            f"{getpass.getuser()}@{socket.gethostname()}",
            SINGLE_VERSION_TAG,
            "Singleton version",
            [],
        )

    def minor_versions_diff(self, baseline_minor_version: str, target_minor_version: str) -> str:
        """There is no diff, as there is single version"""
        assert (
            baseline_minor_version == target_minor_version
        ), f"{baseline_minor_version} or {target_minor_version} is unsupported"
        assert (
            baseline_minor_version == SINGLE_VERSION_TAG
        ), f"{baseline_minor_version} is unsupported"
        return ""

    def get_head_major_version(self) -> str:
        """Returns single branch"""
        return SINGLE_VERSION_BRANCH

    def check_minor_version_validity(self, minor_version: str) -> None:
        """Only SINGLE_VERSION_TAG is valid minor version in SVS"""
        assert minor_version == SINGLE_VERSION_TAG, f"invalid minor version {minor_version}"

    def massage_parameter(self, parameter: str, _: str | None = None) -> str:
        """No massaging in SVS"""
        return parameter

    def is_dirty(self) -> bool:
        """Nothing is really tracked, so it is never dirty"""
        return False

    def save_state(self) -> tuple[bool, str]:
        """Nothing was stashed, and we are still on the same version"""
        return False, SINGLE_VERSION_TAG

    def restore_state(self, _: bool, __: str) -> None:
        """Nothing to restore"""
        pass

    def checkout(self, _: str) -> None:
        """There is single version, so nothing is checked out"""
        pass
