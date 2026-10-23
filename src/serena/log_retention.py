# SPDX-License-Identifier: GPL-3.0-or-later
"""
Retention of the date-stamped log directories in Serena's user home, one of which every session that writes a log
file creates, so that their number would otherwise be unbounded.
"""

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_DATE_DIRECTORY_NAME = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class LogRetention:
    """
    The retention policy for the date-stamped log directories, each of which holds the logs written during one day.
    """

    days: int = 14
    """the number of days whose log directories are kept; directories of earlier days are removed"""

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError(f"The log retention period must be at least one day, got {self.days}")

    def prune(self, logs_root: str, today: date | None = None) -> list[str]:
        """
        Removes the log directories that are older than the retention period.

        Only directories whose name is a date are considered, so that pruning never removes anything which the
        creation of log files did not produce itself. Failures to remove a directory are logged rather than raised,
        because the retention is a maintenance task and must not prevent a session from starting.

        :param logs_root: the directory that contains the date-stamped log directories
        :param today: the day against which the retention period is measured; defaults to the current day
        :return: the names of the removed directories, in ascending order
        """
        today = today or datetime.now().date()
        cutoff = today - timedelta(days=self.days)

        # determine which of the entries are log directories of days which are no longer retained
        try:
            entries = sorted(Path(logs_root).iterdir())
        except OSError as e:
            log.debug("Could not read the log directory %s: %s", logs_root, e)
            return []
        expired_names = []
        for entry in entries:
            if not entry.is_dir() or not _DATE_DIRECTORY_NAME.fullmatch(entry.name):
                continue
            try:
                day = date.fromisoformat(entry.name)
            except ValueError:
                continue
            if day < cutoff:
                expired_names.append(entry.name)

        # remove them, tolerating a directory which another session has already removed
        removed_names = []
        for name in expired_names:
            directory = Path(logs_root, name)
            try:
                shutil.rmtree(directory)
            except FileNotFoundError:
                continue
            except OSError as e:
                log.warning("Could not remove the expired log directory %s: %s", directory, e)
                continue
            removed_names.append(name)
        return removed_names
