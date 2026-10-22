# SPDX-License-Identifier: GPL-3.0-or-later
"""
Preparation of git worktrees as Serena projects in the Serena fork: a worktree receives the project configuration and
the language servers' raw symbol caches of the repository's main directory, is registered, and is announced to the
running project server, through which sessions query and edit it; the release reverses this.
"""

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
import requests

from serena.config.serena_config import ProjectConfig, RegisteredProject, SerenaConfig
from serena.project_server import ProjectServer
from serena.project_server_ext import ProjectServerRegistryFile
from serena.util.yaml import load_yaml
from solidlsp.ls import SolidLanguageServer

log = logging.getLogger(__name__)


class WorktreeProjectError(Exception):
    """
    Raised when a worktree cannot be prepared or released as a project.
    """


class ProjectServerAdminClient:
    """
    A client for the routes of the running project server that register, warm up and unload projects.
    """

    TIMEOUT_SECONDS = 60.0

    def __init__(self, port: int, auth_secret: str) -> None:
        """
        :param port: the server's port (on the local host)
        :param auth_secret: the shared authentication secret
        """
        self._base_url = f"http://127.0.0.1:{port}"
        self._headers = {"Authorization": f"Bearer {auth_secret}"}

    @classmethod
    def find_running(cls, serena_config: SerenaConfig) -> "ProjectServerAdminClient | None":
        """
        :param serena_config: Serena's configuration, providing the authentication secret
        :return: a client for the project server on the default port, provided that its registry file names a running
            process and the server answers its heartbeat; None otherwise
        """
        try:
            with open(ProjectServerRegistryFile.path_for(ProjectServer.PORT), encoding="utf-8") as f:
                registry = json.load(f)
        except (OSError, ValueError):
            return None
        pid, port = registry.get("pid"), registry.get("port")
        if not isinstance(pid, int) or not isinstance(port, int) or not psutil.pid_exists(pid):
            return None
        client = cls(port, serena_config.auth_secret)
        try:
            response = requests.get(client._base_url + "/heartbeat", headers=client._headers, timeout=5)
        except requests.RequestException:
            return None
        return client if response.status_code == 200 else None

    def post(self, route: str, body: dict[str, Any]) -> requests.Response:
        """
        :param route: the route, e.g. "/register"
        :param body: the JSON body
        :return: the server's response
        """
        return requests.post(self._base_url + route, json=body, headers=self._headers, timeout=self.TIMEOUT_SECONDS)


class ProjectServerState(str, Enum):
    """
    The state of a worktree project at the project server after its preparation or release.
    """

    ABSENT = "absent"
    """no project server is running"""
    ANNOUNCED = "announced"
    """the project server registered the project (and warms it up, unless this was disabled)"""
    UNLOADED = "unloaded"
    """the project server released the project"""
    REJECTED = "rejected"
    """the project server rejected the project (see the message)"""


@dataclass
class WorktreeProjectReport:
    """
    The outcome of the preparation or release of a worktree project.
    """

    name: str | None
    root: str
    copied: bool
    """whether the main directory's project configuration and symbol caches were copied (preparation only)"""
    registered: bool
    """whether the project is registered in Serena's configuration afterwards"""
    project_server: ProjectServerState
    message: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {**asdict(self), "project_server": self.project_server.value}

    def describe(self) -> str:
        """
        :return: a human-readable description of the outcome
        """
        registration = "registered" if self.registered else "not registered"
        description = f"Project '{self.name}' ({self.root}): {registration}, project server: {self.project_server.value}"
        return description if self.message is None else f"{description} ({self.message})"


class WorktreeProjects:
    """
    Prepares and releases git worktrees of a repository as Serena projects, based on the project of the repository's
    main directory, whose configuration is only read.
    """

    UNLOAD_RETRY_INTERVAL_SECONDS = 2.0

    def __init__(self, serena_config: SerenaConfig, project_server: ProjectServerAdminClient | None) -> None:
        """
        :param serena_config: Serena's configuration, in which projects are registered
        :param project_server: the client for the running project server; None if none is running
        """
        self._serena_config = serena_config
        self._project_server = project_server

    def prepare(self, worktree_root: str, main_root: str, name: str | None = None, warm_up: bool = True) -> WorktreeProjectReport:
        """
        Prepares a worktree as a project, unless it is prepared already, and registers it with Serena and the project
        server. A first preparation writes the main directory's project configuration with the project's name (the
        languages and ignored paths being kept) and copies the main directory's raw symbol caches, which, unlike the
        processed ones, contain no absolute paths.

        :param worktree_root: the root of the worktree
        :param main_root: the root of the repository's main directory, which must be a Serena project
        :param name: the project's name; if None, the name of an already prepared worktree is kept, and a new one is
            named after the main project and the worktree's last two path components
        :param warm_up: whether the project server shall warm up the project
        :return: the report
        :raises WorktreeProjectError: if a root is invalid, the worktree is prepared under another name, or the name is
            used by another existing project
        """
        worktree = Path(worktree_root).resolve()
        main = Path(main_root).resolve()
        if not worktree.is_dir():
            raise WorktreeProjectError(f"Worktree root '{worktree}' is not a directory")
        if worktree == main:
            raise WorktreeProjectError("The worktree root must differ from the main directory's root")
        main_yml_path = self._serena_config.get_project_yml_location(main)
        if not os.path.isfile(main_yml_path):
            raise WorktreeProjectError(f"The main directory '{main}' is not a Serena project (no {main_yml_path})")
        worktree_yml_path = self._serena_config.get_project_yml_location(worktree)

        # determine the project's name, keeping the name of an already prepared worktree
        is_prepared = os.path.isfile(worktree_yml_path)
        if is_prepared:
            prepared_name = ProjectConfig.load(worktree, serena_config=self._serena_config).project_name
            if name is not None and name != prepared_name:
                raise WorktreeProjectError(f"Worktree '{worktree}' is already prepared as project '{prepared_name}'")
            name = prepared_name
        elif name is None:
            main_name = load_yaml(main_yml_path).get("project_name") or main.name
            name = f"{main_name}@{worktree.parent.name}-{worktree.name}"
        self._release_name(name, worktree)

        # write the configuration and copy the caches of a new worktree project
        if not is_prepared:
            os.makedirs(os.path.dirname(worktree_yml_path), exist_ok=True)
            shutil.copyfile(main_yml_path, worktree_yml_path)
            project_config = ProjectConfig.load(worktree, serena_config=self._serena_config)
            project_config.project_name = name
            project_config.save(worktree_yml_path, save_project_local_yml=False)
            self._copy_raw_symbol_caches(main, worktree)

        # register the project
        if self._find_registered_project(worktree) is None:
            self._serena_config.add_registered_project(RegisteredProject.from_project_root(worktree, serena_config=self._serena_config))
            log.info("Registered worktree project '%s' (%s)", name, worktree)

        state, message = self._announce(worktree, warm_up)
        return WorktreeProjectReport(name, str(worktree), not is_prepared, True, state, message)

    def release(self, worktree_root: str, delete_data: bool = False, timeout: float = 300.0) -> WorktreeProjectReport:
        """
        Releases a worktree project: the project server unloads and forgets it (once no request uses it any more),
        and the project is removed from Serena's configuration.

        :param worktree_root: the root of the worktree (which may no longer exist)
        :param delete_data: whether to also delete the worktree's Serena data folder (configuration and caches)
        :param timeout: the maximum time in seconds to wait for the project server to unload the project
        :return: the report
        :raises WorktreeProjectError: if the project server does not unload the project
        """
        worktree = Path(worktree_root).resolve()

        # unload the project at the project server, retrying while the server is busy with it
        state, message = ProjectServerState.ABSENT, None
        if self._project_server is not None:
            deadline = time.monotonic() + timeout
            while True:
                response = self._project_server.post("/unload", {"project": str(worktree), "forget": True})
                if response.status_code != 409 or time.monotonic() >= deadline:
                    break
                time.sleep(self.UNLOAD_RETRY_INTERVAL_SECONDS)
            if response.status_code == 200:
                state = ProjectServerState.UNLOADED
            elif response.status_code == 404:
                state, message = ProjectServerState.UNLOADED, "The project server did not know the project"
            else:
                raise WorktreeProjectError(f"The project server did not unload the project ({response.status_code}): {response.text}")

        # unregister the project and delete its data
        registered_project = self._find_registered_project(worktree)
        name = None if registered_project is None else registered_project.project_name
        if registered_project is not None:
            self._serena_config.remove_registered_project(registered_project)
        if delete_data:
            data_folder = Path(self._serena_config.get_project_serena_folder(worktree))
            if data_folder.is_dir():
                shutil.rmtree(data_folder)
        return WorktreeProjectReport(name, str(worktree), False, False, state, message)

    def _find_registered_project(self, root: Path) -> RegisteredProject | None:
        """
        :param root: a project root (which may no longer exist)
        :return: the registered project with this root, if any
        """
        for registered_project in self._serena_config.projects:
            if registered_project.matches_root_path(root) or os.path.normcase(str(registered_project.project_root)) == os.path.normcase(
                str(root)
            ):
                return registered_project
        return None

    def _release_name(self, name: str, root: Path) -> None:
        """
        Ensures that a project name can be used for the project with the given root: registrations of the name whose
        root no longer exists (e.g. of removed worktrees) are removed.

        :param name: the project name
        :param root: the root of the project to be named
        :raises WorktreeProjectError: if another project with an existing root has the name
        """
        own_registration = self._find_registered_project(root)
        for registered_project in list(self._serena_config.projects):
            if registered_project.project_name != name or registered_project is own_registration:
                continue
            if registered_project.project_root.is_dir():
                raise WorktreeProjectError(f"Project name '{name}' is already used by '{registered_project.project_root}'")
            self._serena_config.remove_registered_project(registered_project)

    def _copy_raw_symbol_caches(self, main: Path, worktree: Path) -> None:
        """
        Copies the raw symbol caches of all language servers of the main directory to the worktree.

        :param main: the root of the main directory
        :param worktree: the root of the worktree
        """
        file_name = SolidLanguageServer.RAW_DOCUMENT_SYMBOL_CACHE_FILENAME
        main_cache_dir = Path(self._serena_config.get_project_serena_folder(main)) / "cache"
        worktree_cache_dir = Path(self._serena_config.get_project_serena_folder(worktree)) / "cache"
        if not main_cache_dir.is_dir():
            return
        for language_cache_dir in main_cache_dir.iterdir():
            source = language_cache_dir / file_name
            target = worktree_cache_dir / language_cache_dir.name / file_name
            if not source.is_file() or target.exists():
                continue

            # the copy is written under a temporary name first, such that a language server never reads a partial file
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_target = target.with_name(f"{file_name}.{os.getpid()}.copy.tmp")
            shutil.copyfile(source, temporary_target)
            os.replace(temporary_target, target)

    def _announce(self, worktree: Path, warm_up: bool) -> tuple[ProjectServerState, str | None]:
        """
        Registers the worktree project with the running project server and, optionally, starts its warm-up.

        :return: the project's state at the project server and the server's message, if any
        """
        if self._project_server is None:
            return ProjectServerState.ABSENT, None
        response = self._project_server.post("/register", {"project": str(worktree)})
        if response.status_code != 200:
            return ProjectServerState.REJECTED, f"{response.status_code}: {response.text}"
        if warm_up:
            response = self._project_server.post("/warmup", {"project": str(worktree)})
            if response.status_code != 202:
                return ProjectServerState.REJECTED, f"{response.status_code}: {response.text}"
        return ProjectServerState.ANNOUNCED, None
