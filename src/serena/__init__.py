# SPDX-License-Identifier: GPL-3.0-or-later

__version__ = "2.1.0"

import logging

log = logging.getLogger(__name__)

_DISTRIBUTION_NAME = "serena-agent"
"""the name of the distribution installed from this package, under which installers record the installation's origin"""


def serena_version() -> str:
    """
    :return: the version of the package, including the commit the build was created from and a marker for uncommitted
        changes where these can be determined.
    """
    from serena.util.git import get_git_status
    from serena.util.installation import get_installation_origin

    # marks the build of the fork someone4else/serenaFork (__version__ itself is kept for version comparisons)
    version = __version__ + "+fork"

    # an installation from a version control system is an immutable copy of one commit, which the installer records
    origin = get_installation_origin(_DISTRIBUTION_NAME)
    if origin is not None and origin.vcs_commit_hash is not None and not origin.is_editable:
        return version + f"-{origin.vcs_commit_hash[:8]}"

    # otherwise the working tree is authoritative, and it may contain changes which are not committed yet
    try:
        git_status = get_git_status()
    except:
        git_status = None
    if git_status is None:
        return version
    version += f"-{git_status.commit[:8]}"
    if not git_status.is_clean:
        version += "-dirty"
    return version


def _init_log_configuration() -> None:
    from sensai.util import logging

    def _configure() -> None:
        logging.getLogger("PIL").setLevel(logging.WARNING)

    logging.set_configure_callback(_configure)


_init_log_configuration()
