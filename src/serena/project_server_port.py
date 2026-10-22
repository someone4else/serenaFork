# SPDX-License-Identifier: GPL-3.0-or-later
"""
The default port of the project server in the Serena fork, which an environment variable can override (e.g. for a trial
server whose clients must not reach the regular one).
"""

import os

from serena.constants import SerenaPorts


class ProjectServerPort:
    """
    The default port of the project server, used by the server and by its clients.
    """

    ENVIRONMENT_VARIABLE = "SERENA_PROJECT_SERVER_PORT"

    @classmethod
    def default(cls) -> int:
        """
        :return: the port given by the environment variable `ENVIRONMENT_VARIABLE`, if set, otherwise Serena's standard port
        :raises ValueError: if the environment variable is set to a value that is not a port number
        """
        value = os.environ.get(cls.ENVIRONMENT_VARIABLE, "").strip()
        if not value:
            return SerenaPorts.PROJECT_SERVER_PORT
        if not value.isdigit() or not 0 < int(value) < 65536:
            raise ValueError(f"Invalid project server port in {cls.ENVIRONMENT_VARIABLE}: '{value}'")
        return int(value)
