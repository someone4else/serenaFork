# SPDX-License-Identifier: MIT
"""
The paths under which files reported by a language server are addressed in location data.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from solidlsp.ls_utils import PathUtils


@dataclass(frozen=True)
class LocationFilePath:
    """
    The path under which a file reported by a language server is addressed in the location data of
    symbols and references.

    A file within a workspace folder is addressed by its path relative to the repository root, such that
    locations stay project-relative. A file outside all workspace folders - for example a source that the
    language server generated from package metadata in a temporary directory - has no such path and is
    therefore addressed by its absolute path.
    """

    path: str
    """the path under which the file is addressed"""
    is_in_workspace_folder: bool
    """whether the file lies within one of the language server's workspace folders"""

    @classmethod
    def from_abs_path(cls, abs_path: str, repository_root_path: str, workspace_folders: Sequence[str]) -> Self:
        """
        :param abs_path: the absolute path of the file, as reported by the language server
        :param repository_root_path: the root path of the repository analysed by the language server
        :param workspace_folders: the absolute paths of the language server's workspace folders
        :return: the path under which the file is addressed, which is relative to the repository root
            if the file lies within a workspace folder and the absolute path otherwise
        """
        relative_path = PathUtils.get_relative_path(abs_path, repository_root_path)

        # a file inside the repository root keeps the path as reported, without resolving symbolic links,
        # because the language server may address it through aliases within the project
        if relative_path is not None and not PathUtils.traverses_outside_root(relative_path):
            return cls(path=relative_path, is_in_workspace_folder=True)

        # a file in another workspace folder is addressed through the relative path that traverses to it
        if relative_path is not None and cls._is_within_any_folder(abs_path, workspace_folders):
            return cls(path=relative_path, is_in_workspace_folder=True)

        # any other file lies outside the workspaces and can only be addressed by its absolute path
        return cls(path=abs_path, is_in_workspace_folder=False)

    @staticmethod
    def _is_within_any_folder(abs_path: str, workspace_folders: Sequence[str]) -> bool:
        """
        :param abs_path: the absolute path of the file
        :param workspace_folders: the absolute paths of the language server's workspace folders
        :return: whether the file lies within one of the folders, symbolic links being resolved
        """
        resolved_path = Path(abs_path).resolve()
        return any(resolved_path.is_relative_to(Path(folder).resolve()) for folder in workspace_folders)
