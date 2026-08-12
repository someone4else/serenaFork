import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Self

from serena.jetbrains import jetbrains_types as jb

if TYPE_CHECKING:
    from serena.project import Project

log = logging.getLogger(__name__)


class FileProxy(ABC):
    @abstractmethod
    def get_contents(self) -> str:
        """:return: the contents of the file as a string."""

    @abstractmethod
    def get_relative_path(self) -> str:
        """:return: the relative path reported by Serena (actual relative path or encoded external path)"""

    @abstractmethod
    def is_glob_supported(self):
        """
        :return: whether the proxy supports glob filtering based on its relative path
        """

    def is_searchable(self) -> bool:
        """
        :return: whether it is worth searching this file's contents. Proxies that can cheaply
            establish that a file cannot usefully be searched (binary content, excessive size)
            return False, so that bulk consumers such as the project-wide text search can skip
            it instead of reading it into memory. The default implementation returns True.
        """
        return True

    @staticmethod
    def is_external_path(relative_path: str) -> bool:
        """
        :return: whether the given relative path is an encoded external path (not a local project file)
        """
        # This is intended to be extended once we also support external paths in other backends
        return jb.is_external_path(relative_path)

    @classmethod
    def from_project_relative_path(cls, project: "Project", relative_path: str) -> "FileProxy":
        if cls.is_external_path(relative_path):
            if project.language_backend.is_jetbrains():
                return JetBrainsFileProxy(relative_path, project)
        return LocalProjectFileProxy(relative_path, project)


class LocalProjectFileProxy(FileProxy):
    MAX_SEARCHABLE_FILE_SIZE = 20 * 1024 * 1024
    """
    Upper bound on the size of a file that will be read in order to search it. Files larger than
    this are almost never source code we want to grep (data dumps, bundled/minified assets, ...),
    and loading them would waste memory and CPU. The limit is deliberately generous.
    """

    def __init__(self, relative_path: str, project: "Project"):
        self._relative_path = relative_path
        self._project = project

    def _get_abs_path(self) -> str:
        return os.path.join(self._project.project_root, self._relative_path)

    def get_contents(self) -> str:
        with open(self._get_abs_path(), encoding=self._project.project_config.encoding) as f:
            return f.read()

    def get_relative_path(self) -> str:
        return self._relative_path

    def is_glob_supported(self):
        return True

    def is_searchable(self) -> bool:
        # local import to avoid a circular import (solidlsp imports back into serena.util)
        from solidlsp.ls_utils import FileUtils

        # Both checks are cheap (a stat and a small header sniff), so they pay for themselves by
        # avoiding a full read of e.g. a compiled assembly or a multi-megabyte data blob.
        abs_path = self._get_abs_path()
        try:
            if os.path.getsize(abs_path) > self.MAX_SEARCHABLE_FILE_SIZE:
                log.debug(f"Skipping {self._relative_path}: exceeds the maximum searchable file size")
                return False
        except OSError:
            # The size could not be determined (e.g. the path does not exist); fall through and let
            # the file reader surface the error, preserving the previous behavior.
            return True
        if FileUtils.is_binary_file(abs_path):
            log.debug(f"Skipping {self._relative_path}: binary file")
            return False
        return True


class JetBrainsFileProxy(FileProxy):
    """
    Retrieves the contents of a file from the JetBrains plugin via the plugin client, given its relative path,
    which may be an external path (e.g., "<ext:FileUtil.class|472e0a13>")
    """

    def __init__(self, relative_path: str, project: "Project"):
        self._relative_path = relative_path
        self._project = project

    def get_contents(self) -> str:
        from serena.jetbrains.jetbrains_plugin_client import JetBrainsPluginClient

        client = JetBrainsPluginClient.from_project(self._project)
        return client.read_file(self._relative_path)

    def get_relative_path(self) -> str:
        return self._relative_path

    def is_glob_supported(self):
        return False


class FileCollection:
    def __init__(self, file_proxies: list[FileProxy]):
        self._file_proxies = file_proxies

    def __len__(self) -> int:
        return len(self._file_proxies)

    def __iter__(self) -> Iterator[FileProxy]:
        return iter(self._file_proxies)

    @classmethod
    def from_local_project_paths(cls, relative_paths: list[str], project: "Project") -> Self:
        return cls([LocalProjectFileProxy(path, project) for path in relative_paths])

    def filter_glob(self, paths_include_glob: str | None = None, paths_exclude_glob: str | None = None) -> "FileCollection":
        """
        Filters the collection based on the given patterns.
        Note: Filtering is applied only to local project files. Other files are always retained.

        :param paths_include_glob: optional glob pattern to include files from the list
        :param paths_exclude_glob: optional glob pattern to exclude files from the list
        :return: the filtered collection
        """
        from serena.util.text_utils import GlobMatcher

        if paths_include_glob is None and paths_exclude_glob is None:
            return self

        include_glob_matcher = GlobMatcher(paths_include_glob) if paths_include_glob else None
        exclude_glob_matcher = GlobMatcher(paths_exclude_glob) if paths_exclude_glob else None

        filtered_files = []
        for f in self._file_proxies:
            if f.is_glob_supported():
                path = f.get_relative_path()
                if include_glob_matcher:
                    if not include_glob_matcher.matches(path):
                        log.debug(f"Skipping {path}: does not match include pattern {paths_include_glob}")
                        continue
                if exclude_glob_matcher:
                    if exclude_glob_matcher.matches(path):
                        log.debug(f"Skipping {path}: matches exclude pattern {paths_exclude_glob}")
                        continue
            filtered_files.append(f)

        return FileCollection(filtered_files)
