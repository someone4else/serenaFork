# SPDX-License-Identifier: GPL-3.0-or-later
"""
The origin of an installed distribution, as recorded by the installer, which yields the commit a build was created
from also when the installation contains no repository of its own (e.g. a cache entry created from a git URL).
"""

import json
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class InstallationOrigin:
    """
    The origin of an installed distribution.
    """

    is_editable: bool
    """whether the installation refers to a working tree rather than to a copy of the code"""
    vcs_commit_hash: str | None
    """the commit the distribution was installed from, provided that the installer recorded a version control origin"""


def get_installation_origin(distribution_name: str) -> InstallationOrigin | None:
    """
    Determines the origin an installer recorded for a distribution.

    Installers record the origin in the distribution's ``direct_url.json``.

    :param distribution_name: the name of the installed distribution, e.g. "serena-agent"
    :return: the origin, or None if the distribution is not installed or records no origin
    """
    try:
        direct_url_text = metadata.distribution(distribution_name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None

    if not direct_url_text:
        return None

    # parse the recorded origin
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None

    # an installation from a version control system records the commit, an editable one refers to a working tree
    dir_info = direct_url.get("dir_info")
    is_editable = isinstance(dir_info, dict) and bool(dir_info.get("editable"))

    vcs_commit_hash = None
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, dict):
        commit_id = vcs_info.get("commit_id")
        if isinstance(commit_id, str):
            vcs_commit_hash = commit_id

    return InstallationOrigin(is_editable=is_editable, vcs_commit_hash=vcs_commit_hash)
