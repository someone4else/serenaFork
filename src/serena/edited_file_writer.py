# SPDX-License-Identifier: GPL-3.0-or-later
"""
Writing of edited text files by the Serena fork, which can keep each existing file's own line endings.
"""

import os
from typing import TYPE_CHECKING

from serena.util.file_system import write_file_atomic

if TYPE_CHECKING:
    from serena.project import Project


class EditedFileWriter:
    """
    Writes the contents of edited text files atomically, applying either the configured line ending convention or,
    for an existing file with line breaks, the line endings that prevail in the file.
    """

    def __init__(self, encoding: str, newline: str | None, preserve_line_endings: bool) -> None:
        """
        :param encoding: the encoding with which files are written (and read for the detection of their line endings)
        :param newline: the configured line ending convention, as a ``newline`` value of :func:`open` (None for the
            platform's convention)
        :param preserve_line_endings: whether an existing file with line breaks keeps the line endings that prevail
            in it; if False, every file is written with the configured convention
        """
        self._encoding = encoding
        self._newline = newline
        self._preserve_line_endings = preserve_line_endings

    @classmethod
    def for_project(cls, project: "Project") -> "EditedFileWriter":
        """
        Creates the writer for the files of a project.

        :param project: the project
        :return: a writer applying the project's encoding, its line ending convention and the global setting
            `SerenaConfig.preserve_line_endings`
        """
        return cls(project.project_config.encoding, project.line_ending.newline_str, project.serena_config.preserve_line_endings)

    def write(self, path: str, contents: str) -> None:
        """
        Writes a file, replacing its previous contents.

        :param path: the path of the file
        :param contents: the new contents, whose line breaks are written with the applicable line ending
        """
        # determine the line ending, normalising CRLF line breaks in the contents (which would otherwise be written
        # as CR CR LF)
        newline = self._newline
        if self._preserve_line_endings:
            contents = contents.replace("\r\n", "\n")
            prevailing_newline = self._read_prevailing_newline(path)
            if prevailing_newline is not None:
                newline = prevailing_newline

        write_file_atomic(path, contents, encoding=self._encoding, newline=newline)

    def _read_prevailing_newline(self, path: str) -> str | None:
        """
        Determines the line ending that prevails in an existing file.

        :param path: the path of a file
        :return: the line ending that prevails in the file (CRLF in case of a tie), or None if the file does not exist
            or contains no line breaks
        """
        if not os.path.isfile(path):
            return None

        # count the line endings
        with open(path, encoding=self._encoding, errors="replace", newline="") as f:
            text = f.read()
        num_crlf = text.count("\r\n")
        num_lf = text.count("\n") - num_crlf

        if num_crlf == 0 and num_lf == 0:
            return None
        return "\r\n" if num_crlf >= num_lf else "\n"
