# SPDX-License-Identifier: GPL-3.0-or-later
"""
Extensions of the Serena fork for a long-running project server: projects are warmed up at startup and their readiness
is reported by the heartbeat, a registry file lets clients find the server, an allowlist restricts the tools the server
accepts, projects within writable roots (e.g. git worktrees) can be registered and unloaded at runtime, and a shutdown
route stops the server gracefully (saving the language servers' caches).
"""

import _thread
import gc
import json
import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import requests as requests_lib
from flask import Response, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from serena import serena_version
from serena.config.serena_config import RegisteredProject, SerenaConfig, SerenaPaths
from serena.constants import SerenaPorts
from serena.edited_file_writer import WritableRoots, WriteConfinement
from serena.project_server import CallFacadeMethodRequest, ProjectServer, ProjectServerClient, QueryProjectRequest
from serena.tools.tools_base import ToolCallError
from serena.util.file_system import write_file_atomic

if TYPE_CHECKING:
    from serena.project import Project
    from serena.repl.facade import FacadeMethod
    from solidlsp.ls import SolidLanguageServer

log = logging.getLogger(__name__)


class ProjectServerSettingsError(ValueError):
    """
    Raised when the project server is created with an inadmissible combination of settings.
    """


class ToolNotAllowedError(Exception):
    """
    Raised when a request asks the project server for a tool or facade method outside of its allowlist.
    """


class ProjectRequestRejectedError(Exception):
    """
    Raised when the project server rejects a request concerning a project; the message is the reason reported to the
    client, along with the HTTP status `status`.
    """

    status = 400


class ProjectNotFoundError(ProjectRequestRejectedError):
    """
    Raised when a request refers to a project that is not registered or has no project configuration.
    """

    status = 404


class ProjectNotAllowedError(ProjectRequestRejectedError):
    """
    Raised when a request refers to a project outside of the project server's writable roots.
    """

    status = 403


class ProjectConflictError(ProjectRequestRejectedError):
    """
    Raised when a request conflicts with the state of a project (e.g. a project being unloaded); unless stated
    otherwise in the message, the request can be repeated later.
    """

    status = 409


class ProjectRequestFailedError(ProjectRequestRejectedError):
    """
    Raised when a request concerning a project fails for a reason other than the request (e.g. the project cannot be
    loaded).
    """

    status = 500


class ToolFailedError(ProjectRequestRejectedError):
    """
    Raised when a tool executed for a request fails (e.g. because a symbol is not found).
    """

    status = 422


class EditOutcomeUnknownError(ProjectRequestRejectedError):
    """
    Raised when an edit did not complete within the tool timeout; it may still be applied afterwards.
    """

    status = 504


class ProjectUsageGate:
    """
    Admission of requests to the projects of the project server: a request uses a project (identified by its root
    path) while it is processed, and a project that is being unloaded admits no requests, its unloading waiting for
    the requests that use it.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._num_users_by_root: dict[str, int] = {}
        self._unloading_roots: set[str] = set()

    @contextmanager
    def use(self, project_root: str) -> Iterator[None]:
        """
        Context for a request using a project.

        :param project_root: the project's root path
        :raises ProjectConflictError: if the project is being unloaded
        """
        with self._condition:
            if project_root in self._unloading_roots:
                raise ProjectConflictError(f"Project '{project_root}' is being unloaded")
            self._num_users_by_root[project_root] = self._num_users_by_root.get(project_root, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                num_users = self._num_users_by_root[project_root] - 1
                if num_users > 0:
                    self._num_users_by_root[project_root] = num_users
                else:
                    del self._num_users_by_root[project_root]
                self._condition.notify_all()

    @contextmanager
    def unloading(self, project_root: str, timeout: float) -> Iterator[None]:
        """
        Context for the unloading of a project, which is entered once no request uses the project, new requests for
        the project being rejected from the start of the wait until the end of the context.

        :param project_root: the project's root path
        :param timeout: the maximum time in seconds to wait for the requests using the project
        :raises ProjectConflictError: if the project is already being unloaded or still in use after the timeout
        """
        with self._condition:
            if project_root in self._unloading_roots:
                raise ProjectConflictError(f"Project '{project_root}' is already being unloaded")
            self._unloading_roots.add(project_root)
            if not self._condition.wait_for(lambda: project_root not in self._num_users_by_root, timeout=timeout):
                self._unloading_roots.discard(project_root)
                raise ProjectConflictError(f"Project '{project_root}' is still in use after {timeout} seconds")
        try:
            yield
        finally:
            with self._condition:
                self._unloading_roots.discard(project_root)


class ToolAllowlist:
    """
    The tools a project server accepts.

    An empty allowlist imposes no restriction (the upstream behaviour). A non-empty one restricts ``/query_project``
    to the listed tools and ``/call_facade_method`` to the non-editing facade methods whose corresponding tool is
    listed; facade methods without a corresponding tool are rejected.
    """

    def __init__(self, tool_names: Sequence[str] = ()) -> None:
        """
        :param tool_names: the names of the accepted tools; empty for no restriction
        """
        self._tool_names = frozenset(tool_names)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tool_names)

    def check_tool(self, tool_name: str) -> None:
        """
        :param tool_name: the name of a requested tool
        :raises ToolNotAllowedError: if the tool is not accepted
        """
        if self._tool_names and tool_name not in self._tool_names:
            raise ToolNotAllowedError(f"Tool '{tool_name}' is not allowed by this project server (--allowed-tool)")

    def check_facade_method(self, method: "FacadeMethod") -> None:
        """
        :param method: a requested facade method
        :raises ToolNotAllowedError: if the method is not accepted
        """
        if not self._tool_names:
            return
        tool_name = method.info.get_corresponding_tool_name()
        if method.info.can_edit or tool_name is None or tool_name not in self._tool_names:
            raise ToolNotAllowedError(f"Facade method '{method.qualified_name}' is not allowed by this project server (--allowed-tool)")


class ProjectState(str, Enum):
    """
    The state of a project that is warmed up by the project server.
    """

    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ProjectReadiness:
    """
    The readiness of a project that is warmed up by the project server.
    """

    name: str
    state: ProjectState
    started_at: float
    """the monotonic time at which the warm-up started"""
    seconds: float | None = None
    """the duration of the warm-up, once it is finished"""
    error: str | None = None
    """the error that made the warm-up fail, if it failed"""

    def to_json_dict(self) -> dict[str, Any]:
        """
        :return: the representation reported by the heartbeat and the registry file
        """
        return {"name": self.name, "state": self.state.value, "seconds": self.seconds, "error": self.error}


class ProjectReadinessTracker:
    """
    Thread-safe record of the readiness of the projects warmed up by the project server, which reports every change of
    the record (with the record's new snapshot) to a listener.
    """

    def __init__(self, on_change: Callable[[dict[str, dict[str, Any]]], None]) -> None:
        """
        :param on_change: the listener, which is called (while the record is locked, such that snapshots are reported
            in order) with the snapshot after each change
        """
        self._on_change = on_change
        self._lock = threading.Lock()
        self._readiness_by_root: dict[str, ProjectReadiness] = {}

    def begin(self, project_root: str, project_name: str) -> bool:
        """
        Records the start of the warm-up of a project, unless the project is already loading or ready.

        :param project_root: the project's root path
        :param project_name: the project's name
        :return: whether the warm-up shall be started
        """
        with self._lock:
            readiness = self._readiness_by_root.get(project_root)
            if readiness is not None and readiness.state != ProjectState.FAILED:
                return False
            self._readiness_by_root[project_root] = ProjectReadiness(project_name, ProjectState.LOADING, time.monotonic())
            self._on_change(self._snapshot())
            return True

    def finish(self, project_root: str, error: str | None = None) -> None:
        """
        Records the end of the warm-up of a project.

        :param project_root: the project's root path
        :param error: the error that made the warm-up fail; None if the project is ready
        """
        with self._lock:
            readiness = self._readiness_by_root[project_root]
            readiness.state = ProjectState.READY if error is None else ProjectState.FAILED
            readiness.seconds = round(time.monotonic() - readiness.started_at, 1)
            readiness.error = error
            self._on_change(self._snapshot())

    def forget(self, project_root: str) -> None:
        """
        Removes the readiness record of a project, if any.

        :param project_root: the project's root path
        """
        with self._lock:
            if self._readiness_by_root.pop(project_root, None) is not None:
                self._on_change(self._snapshot())

    def is_loading(self, project_root: str) -> bool:
        """
        :param project_root: the project's root path
        :return: whether the project's warm-up is in progress
        """
        with self._lock:
            readiness = self._readiness_by_root.get(project_root)
            return readiness is not None and readiness.state == ProjectState.LOADING

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """
        :return: the JSON representations of the readiness records (see `ProjectReadiness.to_json_dict`) by project root
        """
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        return {root: readiness.to_json_dict() for root, readiness in self._readiness_by_root.items()}


class ProjectWarmer:
    """
    Loads projects in background threads and probes each of their language servers once, such that the first request
    finds the project loaded and its language servers responsive.

    The probe talks to the language servers directly rather than through a tool, as tools are executed under the
    project server's process-wide lock: a cold language server (e.g. C# loading a large solution) would otherwise
    delay the requests for all other projects.
    """

    def __init__(self, load_project: Callable[[str], "Project"], tracker: ProjectReadinessTracker) -> None:
        """
        :param load_project: the function loading a registered project (by root path) and starting its language servers
        :param tracker: the readiness record to update
        """
        self._load_project = load_project
        self._tracker = tracker
        self._threads: list[threading.Thread] = []

    def start(self, project_root: str, project_name: str) -> None:
        """
        Starts the warm-up of a project in a background thread, unless it is already loading or ready.

        :param project_root: the project's root path
        :param project_name: the project's name
        """
        if self._tracker.begin(project_root, project_name):
            self._threads = [t for t in self._threads if t.is_alive()]
            thread = threading.Thread(target=self._warm_up, args=(project_root,), name=f"warmup:{project_name}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def join(self, timeout: float) -> None:
        """
        Waits for the warm-ups in progress to finish.

        :param timeout: the maximum time to wait in seconds (for all warm-ups together)
        """
        deadline = time.monotonic() + timeout
        for thread in list(self._threads):
            thread.join(max(0.0, deadline - time.monotonic()))

    def _warm_up(self, project_root: str) -> None:
        try:
            project = self._load_project(project_root)
            self._probe_language_servers(project)
        except Exception as e:
            log.error("Warm-up of project %s failed", project_root, exc_info=e)
            self._tracker.finish(project_root, error=f"{type(e).__name__}: {e}")
            return
        log.info("Project %s is warmed up", project_root)
        self._tracker.finish(project_root)

    @staticmethod
    def _probe_language_servers(project: "Project") -> None:
        manager = project.language_server_manager
        if manager is None:
            raise RuntimeError("The project's language servers were not started")

        # one source file per language server: its document symbols and a hover on its first symbol (unlike document
        # symbols, a hover requires the semantic model, e.g. the loaded solution of a C# project)
        language_servers = list(manager.iter_language_servers())
        probed: list["SolidLanguageServer"] = []
        for project_file in project.gather_source_files():
            if len(probed) == len(language_servers):
                break
            relative_path = project_file.rel_path
            if not manager.has_suitable_ls_for_file(relative_path):
                continue
            language_server = manager.get_language_server(relative_path)
            if any(language_server is p for p in probed):
                continue
            probed.append(language_server)
            for symbol in language_server.request_document_symbols(relative_path).iter_symbols():
                selection_range = symbol.get("selectionRange")
                if selection_range is not None:
                    start = selection_range["start"]
                    language_server.request_hover(relative_path, start["line"], start["character"])
                    break


class ProjectServerRegistryFile:
    """
    The file through which clients find a running project server: ``project-server.json`` in Serena's home directory
    for the default port, ``project-server-<port>.json`` for other ports.
    It holds the server's PID, port, start time, version and project readiness, but neither the host (clients connect
    to 127.0.0.1 only) nor the authentication secret.
    A file left behind by a killed server is recognised as stale by its PID.
    """

    def __init__(self, port: int, pid: int, directory: str | None = None) -> None:
        """
        :param port: the server's port
        :param pid: the server's process ID
        :param directory: the directory containing the file; None for Serena's home directory
        """
        self.path = self.path_for(port, directory)
        self._port = port
        self._pid = pid
        self._started_at = datetime.now(UTC).isoformat(timespec="seconds")

    @staticmethod
    def path_for(port: int, directory: str | None = None) -> str:
        """
        :param port: a server's port
        :param directory: the directory containing the file; None for Serena's home directory
        :return: the path of the registry file of the server with the given port
        """
        # the name depends on Serena's standard port rather than on the (overridable) default, such that a server on
        # another port never replaces the registry file of a server on the standard port
        file_name = "project-server.json" if port == SerenaPorts.PROJECT_SERVER_PORT else f"project-server-{port}.json"
        return os.path.join(directory if directory is not None else SerenaPaths().serena_user_home_dir, file_name)

    def write(self, projects: dict[str, dict[str, Any]]) -> None:
        """
        :param projects: the project readiness entries by project root
        """
        content = {
            "pid": self._pid,
            "port": self._port,
            "started_at": self._started_at,
            "version": serena_version(),
            "projects": projects,
        }
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            write_file_atomic(self.path, json.dumps(content, indent=2), encoding="utf-8")
        except OSError as e:
            log.error("Failed to write the project server registry file %s: %s", self.path, e)

    def remove(self) -> None:
        """
        Removes the file, provided that it (still) describes this server.
        """
        try:
            with open(self.path, encoding="utf-8") as f:
                pid = json.load(f).get("pid")
        except (OSError, ValueError):
            return
        if pid == self._pid:
            try:
                os.remove(self.path)
            except OSError as e:
                log.error("Failed to remove the project server registry file %s: %s", self.path, e)


class ExtendedProjectServer(ProjectServer):
    """
    The project server of the Serena fork, adding warm-up, readiness reporting, a registry file, a tool allowlist and
    graceful shutdown to the upstream :class:`ProjectServer`. Without warm-up projects and allowlist, requests are
    handled as upstream, except that a ``/query_project`` request whose body does not match the request model is
    answered with status 400 (instead of 500).

    Routes in addition to the upstream ones:
      * ``GET /heartbeat`` additionally reports ``projects``: the readiness of the warmed-up projects by root path;
      * ``POST /warmup`` with ``{"project": <name or root>}`` starts the warm-up of a registered project (202);
      * ``POST /register`` with ``{"project": <root>}`` registers a project within the writable roots (200);
      * ``POST /unload`` with ``{"project": <name or root>, "forget": <bool>}`` stops a project's language servers
        (saving their caches) and releases the project, which is also removed from the server's registered projects
        if ``forget`` is true (200);
      * ``POST /edit_project`` with the body of ``/query_project`` executes an editing tool (`EDITING_TOOL_NAMES`) in
        a project within the writable roots and returns its result (200);
      * ``POST /shutdown`` stops the server gracefully (202).
    Rejections are reported as plain text: by the allowlist and the writable roots with status 403, for unknown
    projects with 404, for requests conflicting with a project's state (e.g. its unloading) with 409, failures of
    editing tools with 422, and edits whose outcome is unknown (time-out) with 504.
    With writable roots, all writes of edited files of the server's process are confined to them.
    """

    SHUTDOWN_WAIT_SECONDS = 30.0
    """
    the time for which the shutdown waits for warm-ups in progress and, separately, for a request being processed,
    before stopping the loaded projects' language servers anyway
    """
    UNLOAD_WAIT_SECONDS = 60.0
    """the time for which the unloading of a project waits for the requests using it"""
    EDITING_TOOL_NAMES = frozenset(
        ("replace_symbol_body", "insert_after_symbol", "insert_before_symbol", "replace_content", "rename_symbol", "safe_delete_symbol")
    )
    """the tools accepted by ``/edit_project``"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = None,
        warm_projects: Sequence[str] = (),
        allowed_tools: Sequence[str] = (),
        writable_roots: Sequence[str] = (),
    ) -> None:
        """
        :param host: the host address to listen on
        :param port: the port to listen on; if None, use default
        :param warm_projects: the names or root paths of the projects to warm up when the server starts (which cannot
            be unloaded)
        :param allowed_tools: the names of the tools the server accepts; empty for no restriction
        :param writable_roots: the directories within which projects can be registered at runtime; empty for none.
            They require an allowlist (without it, facade methods would edit any project) and a local host.
        :raises ProjectServerSettingsError: if writable roots are given without allowlist or for a non-local host
        """
        if writable_roots and not allowed_tools:
            raise ProjectServerSettingsError("Writable roots (--writable-root) require an allowlist of tools (--allowed-tool)")
        if writable_roots and host not in ("127.0.0.1", "localhost", "::1"):
            raise ProjectServerSettingsError(f"Writable roots (--writable-root) require a local host address, not '{host}'")

        # the routes, which are set up by the base class constructor, depend on the allowlist
        self._allowlist = ToolAllowlist(allowed_tools)
        super().__init__(host=host, port=port)
        self._warm_projects = list(warm_projects)
        self._pinned_roots: set[str] = set()
        self._writable_roots = WritableRoots(writable_roots)
        if not self._writable_roots.is_empty:
            WriteConfinement.confine(self._writable_roots)
        self._usage = ProjectUsageGate()
        self._registration_lock = threading.Lock()
        self._registry = ProjectServerRegistryFile(self._port, os.getpid())
        self._readiness = ProjectReadinessTracker(on_change=self._registry.write)
        self._warmer = ProjectWarmer(self._get_project, self._readiness)
        self._shutdown_scheduled = threading.Event()

    def warm_up(self, project_root_or_name: str) -> str:
        """
        Starts the warm-up of a registered project, unless it is already loading or ready.

        :param project_root_or_name: the project's name or root path
        :return: the project's root path
        :raises ValueError: if the project is not registered (or the name is ambiguous)
        """
        registered_project = self.get_serena_config().get_registered_project(project_root_or_name)
        if registered_project is None:
            raise ValueError(f"Project '{project_root_or_name}' is not registered with Serena.")
        project_root = str(registered_project.project_root)
        self._warmer.start(project_root, registered_project.project_config.project_name)
        return project_root

    def register_project(self, project_root: str) -> tuple[RegisteredProject, bool]:
        """
        Registers a project with the server and in Serena's configuration file, unless it is registered already.
        A registered project of the same name whose root no longer exists (e.g. a removed git worktree) is replaced.

        :param project_root: the project's root path, which must lie within the writable roots and contain a project
            configuration file
        :return: the registered project and whether it was newly registered
        :raises ProjectNotAllowedError: if the root is not within the writable roots
        :raises ProjectNotFoundError: if there is no project configuration file
        :raises ProjectConflictError: if another registered project with an existing root has the same name
        """
        if not self._writable_roots.contains(project_root):
            raise ProjectNotAllowedError(f"Project root '{project_root}' is not within a writable root of this server (--writable-root)")
        serena_config = self.get_serena_config()

        with self._registration_lock:
            # an already registered project is kept
            for registered_project in serena_config.projects:
                if registered_project.matches_root_path(project_root):
                    return registered_project, False

            # the new project's name must identify it unambiguously
            if not os.path.isfile(serena_config.get_project_yml_location(project_root)):
                raise ProjectNotFoundError(f"No project configuration found in '{project_root}'")
            try:
                new_project = RegisteredProject.from_project_root(project_root, serena_config=serena_config)
            except Exception as e:
                raise ProjectRequestRejectedError(f"Invalid project configuration in '{project_root}': {type(e).__name__}: {e}") from e
            stale_projects = []
            for registered_project in serena_config.projects:
                if registered_project.project_name == new_project.project_name:
                    if registered_project.project_root.is_dir():
                        raise ProjectConflictError(
                            f"Project name '{new_project.project_name}' is already used by '{registered_project.project_root}'; "
                            "a different name is required"
                        )
                    stale_projects.append(registered_project)

            # removals replace the project list rather than modify it, as it is read without locking (an append, as
            # by the registration, is safe for readers)
            if stale_projects:
                serena_config.projects = [p for p in serena_config.projects if all(p is not s for s in stale_projects)]
            serena_config.add_registered_project(new_project)
        log.info("Registered project '%s' (%s)", new_project.project_name, new_project.project_root)
        return new_project, True

    def unload_project(self, project_root_or_name: str, forget: bool = False) -> tuple[RegisteredProject, bool]:
        """
        Unloads a project: once no request uses it any more, its language servers are stopped (saving their caches),
        the project is released, and its readiness is no longer reported. Requests for the project meanwhile are
        rejected; afterwards, the project is loaded anew when requested (unless it was forgotten).

        :param project_root_or_name: the project's name or root path (which may no longer exist)
        :param forget: whether to also remove the project from the server's registered projects (the registration
            in Serena's configuration file is left to the caller)
        :return: the project and whether it was loaded
        :raises ProjectNotFoundError: if the project is not registered
        :raises ProjectNotAllowedError: if the project is not within the writable roots
        :raises ProjectConflictError: if the project was warmed up at startup, is being warmed up or still in use
        """
        serena_config = self.get_serena_config()
        try:
            registered_project = serena_config.get_registered_project(project_root_or_name)
        except ValueError as e:
            raise ProjectRequestRejectedError(str(e)) from e
        if registered_project is None:
            # the root of a project may have been removed already (e.g. a removed git worktree)
            normalised_path = os.path.normcase(os.path.abspath(project_root_or_name))
            registered_project = next((p for p in serena_config.projects if os.path.normcase(str(p.project_root)) == normalised_path), None)
        if registered_project is None:
            raise ProjectNotFoundError(f"Project '{project_root_or_name}' is not registered with this server")
        project_root = str(registered_project.project_root)
        if not self._writable_roots.contains(project_root):
            raise ProjectNotAllowedError(f"Project '{project_root}' is not within a writable root of this server (--writable-root)")
        if project_root in self._pinned_roots:
            raise ProjectConflictError(f"Project '{project_root}' is warmed up at startup (--warm) and is not unloaded")

        with self._usage.unloading(project_root, timeout=self.UNLOAD_WAIT_SECONDS):
            if self._readiness.is_loading(project_root):
                raise ProjectConflictError(f"Project '{project_root}' is being warmed up")
            with self._project_load_lock(project_root):
                with self._loaded_projects_lock:
                    project = self._loaded_projects_by_root.pop(project_root, None)
                if project is not None:
                    log.info("Unloading project '%s'", project.project_name)
                    try:
                        project.shutdown(timeout=5)
                    except Exception as e:
                        log.error("Failed to shut down project '%s'", project.project_name, exc_info=e)
                self._readiness.forget(project_root)

                # the registered project holds the project instance; it is replaced by a fresh one (or removed), the
                # project list being replaced rather than modified, as it is read without locking
                with self._registration_lock:
                    remaining_projects = [p for p in serena_config.projects if p is not registered_project]
                    if not forget:
                        remaining_projects.append(RegisteredProject(project_root, registered_project.project_config))
                    serena_config.projects = remaining_projects

        # the released project's caches are freed now rather than at the next collection
        gc.collect()
        return registered_project, project is not None

    def edit_project(self, edit_request: QueryProjectRequest) -> str:
        """
        Executes an editing tool in a project within the writable roots, in the same manner as ``/query_project``
        executes a read-only tool.

        :param edit_request: the request, naming the project (by name or root path), the tool and its parameters
        :return: the tool's result
        :raises ProjectNotAllowedError: if the tool is not one of `EDITING_TOOL_NAMES`, or the project is outside the
            writable roots or read-only
        :raises ProjectNotFoundError: if the project is not registered
        :raises ToolFailedError: if the tool fails
        :raises EditOutcomeUnknownError: if the tool does not complete within the tool timeout
        """
        if edit_request.tool_name not in self.EDITING_TOOL_NAMES:
            raise ProjectNotAllowedError(
                f"Tool '{edit_request.tool_name}' is not accepted by /edit_project; accepted: {', '.join(sorted(self.EDITING_TOOL_NAMES))}"
            )
        try:
            registered_project = self.get_serena_config().get_registered_project(edit_request.project_name)
            params = json.loads(edit_request.tool_params_json)
        except ValueError as e:
            raise ProjectRequestRejectedError(str(e)) from e
        if not isinstance(params, dict):
            raise ProjectRequestRejectedError("The tool parameters (tool_params_json) must be a JSON object")
        if registered_project is None:
            raise ProjectNotFoundError(f"Project '{edit_request.project_name}' is not registered with this server")
        if not self._writable_roots.contains(str(registered_project.project_root)):
            raise ProjectNotAllowedError(
                f"Project '{registered_project.project_root}' is not within a writable root of this server (--writable-root)"
            )
        if registered_project.project_config.read_only:
            raise ProjectNotAllowedError(f"Project '{registered_project.project_root}' is read-only")

        try:
            project = self._get_project(edit_request.project_name)
        except Exception as e:
            raise ProjectRequestFailedError(f"Failed to load project '{registered_project.project_root}': {type(e).__name__}: {e}") from e
        with self._active_project_lock, self._agent.active_project_context(project):
            tool = self._agent.get_tool_by_name(edit_request.tool_name)
            start = time.monotonic()
            try:
                return tool.apply_ex(catch_exceptions=False, **params)
            except ToolCallError as e:
                # a timed-out tool keeps running in the background, possibly applying the edit later
                if time.monotonic() - start >= self.get_serena_config().tool_timeout:
                    raise EditOutcomeUnknownError(f"The outcome of the edit is unknown (it may still be applied): {e}") from e
                raise ToolFailedError(str(e)) from e

    def _project_load_lock(self, project_root: str) -> threading.Lock:
        """
        :param project_root: a project's root path
        :return: the lock under which the project is loaded (see `ProjectServer._get_project`)
        """
        with self._loaded_projects_lock:
            return self._project_load_locks_by_root.setdefault(project_root, threading.Lock())

    def _find_project_root(self, project_root_or_name: str) -> str | None:
        """
        :param project_root_or_name: a project's name or root path
        :return: the root path of the registered project, or None if there is none (or the name is ambiguous)
        """
        try:
            registered_project = self.get_serena_config().get_registered_project(project_root_or_name)
        except ValueError:
            return None
        return None if registered_project is None else str(registered_project.project_root)

    def _process_using_project(self, project_root_or_name: str, view: Callable[[], Any]) -> Any:
        """
        Processes a request concerning a project while using the project (see `ProjectUsageGate`).

        :param project_root_or_name: the name or root path of the project the request concerns
        :param view: the view function processing the request
        :return: the view function's response, or a rejection if the project is being unloaded
        """
        project_root = self._find_project_root(project_root_or_name)
        if project_root is None:
            # the view function reports the unknown project
            return view()
        try:
            with self._usage.use(project_root):
                return view()
        except ProjectRequestRejectedError as e:
            return self._text_response(str(e), e.status)

    def run(self) -> None:
        """
        Runs the server until it is shut down (via ``/shutdown``, Ctrl+C, Ctrl+Break or SIGTERM), warming up the
        configured projects in the background, and finally stops the loaded projects' language servers (saving their
        caches) and removes the registry file. Closing the console is not handled (the server is meant to run without
        one; Windows terminates the process shortly after the close event).

        :raises OSError: if the port is in use, before any project is loaded or the registry file is written
        """
        self._check_port_is_free()
        self._registry.write(self._readiness.snapshot())
        if self._allowlist.tool_names:
            log.info("Accepted tools: %s", ", ".join(self._allowlist.tool_names))
        self._set_termination_signal_handler(signal.default_int_handler)
        for project in self._warm_projects:
            try:
                self._pinned_roots.add(self.warm_up(project))
            except ValueError as e:
                log.error("Cannot warm up project: %s", e)

        try:
            super().run()
        finally:
            # a repeated Ctrl+C or termination request must not interrupt the clean-up
            self._set_termination_signal_handler(signal.SIG_IGN)
            try:
                self._shut_down_projects()
            finally:
                self._registry.remove()

    def _check_port_is_free(self) -> None:
        """
        :raises OSError: if the port is in use (e.g. by another project server, whose registry file must not be
            overwritten)
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((self._host, self._port))
            except OSError as e:
                raise OSError(f"Port {self._port} on {self._host} is already in use: {e}") from e

    @staticmethod
    def _set_termination_signal_handler(handler: Any) -> None:
        for signal_name in ("SIGINT", "SIGBREAK", "SIGTERM"):
            termination_signal = getattr(signal, signal_name, None)
            if termination_signal is not None:
                signal.signal(termination_signal, handler)

    def _shut_down_projects(self) -> None:
        # let warm-ups in progress publish their projects and a request being processed finish; a warm-up still
        # loading afterwards is abandoned, its language servers ending with the process (their input is closed)
        self._warmer.join(timeout=self.SHUTDOWN_WAIT_SECONDS)
        request_finished = self._active_project_lock.acquire(timeout=self.SHUTDOWN_WAIT_SECONDS)
        try:
            with self._loaded_projects_lock:
                projects = list(self._loaded_projects_by_root.values())
                self._loaded_projects_by_root.clear()
            for project in projects:
                log.info("Shutting down project '%s'", project.project_name)
                try:
                    project.shutdown(timeout=5)
                except Exception as e:
                    log.error("Failed to shut down project '%s'", project.project_name, exc_info=e)
        finally:
            if request_finished:
                self._active_project_lock.release()

    @staticmethod
    def _text_response(message: str, status: int) -> Response:
        return Response(message, status=status, mimetype="text/plain")

    @staticmethod
    def _json_response(content: dict[str, Any], status: int) -> Response:
        return Response(json.dumps(content), status=status, mimetype="application/json")

    @staticmethod
    def _project_json_dict(registered_project: RegisteredProject) -> dict[str, str]:
        return {"name": registered_project.project_name, "root": str(registered_project.project_root)}

    def _setup_routes(self) -> None:
        super()._setup_routes()
        app = self._app
        query_project_view = app.view_functions["query_project"]
        call_facade_method_view = app.view_functions["call_facade_method"]

        def heartbeat() -> dict[str, Any]:
            return {"status": "alive", "projects": self._readiness.snapshot()}

        def query_project() -> Any:
            try:
                query_request = QueryProjectRequest.model_validate(request.get_json())
                self._allowlist.check_tool(query_request.tool_name)
            except ValidationError as e:
                return self._text_response(f"Invalid project query: {e}", 400)
            except ToolNotAllowedError as e:
                return self._text_response(str(e), 403)
            return self._process_using_project(query_request.project_name, query_project_view)

        def call_facade_method() -> Any:
            try:
                call_request = CallFacadeMethodRequest.model_validate(request.get_json())
                facade = self._agent.get_repl().entrypoint.get_facade_(call_request.facade_name)
                self._allowlist.check_facade_method(facade.get_method(call_request.method_name))
            except ToolNotAllowedError as e:
                return self._text_response(str(e), 403)
            except HTTPException:
                raise
            except Exception as e:
                return self._text_response(f"{type(e).__name__}: {e}", 400)
            return self._process_using_project(call_request.project_name, call_facade_method_view)

        app.view_functions["heartbeat"] = heartbeat
        app.view_functions["query_project"] = query_project
        app.view_functions["call_facade_method"] = call_facade_method

        def read_project_of_body() -> str | None:
            """
            :return: the request body's non-empty string ``project``, or None if there is none
            """
            project = (request.get_json(silent=True) or {}).get("project")
            return project if isinstance(project, str) and project else None

        @app.route("/warmup", methods=["POST"])
        def warmup() -> Any:
            project = read_project_of_body()
            if project is None:
                return self._text_response('Expected a JSON body {"project": <name or root path>}', 400)
            project_root_or_name: str = project

            def start_warm_up() -> Response:
                try:
                    self.warm_up(project_root_or_name)
                except ValueError as e:
                    return self._text_response(str(e), 404)
                return self._json_response({"projects": self._readiness.snapshot()}, 202)

            return self._process_using_project(project_root_or_name, start_warm_up)

        @app.route("/register", methods=["POST"])
        def register() -> Any:
            project = read_project_of_body()
            if project is None:
                return self._text_response('Expected a JSON body {"project": <root path>}', 400)
            try:
                registered_project, added = self.register_project(project)
            except ProjectRequestRejectedError as e:
                return self._text_response(str(e), e.status)
            return self._json_response({"project": self._project_json_dict(registered_project), "added": added}, 200)

        @app.route("/unload", methods=["POST"])
        def unload() -> Any:
            project = read_project_of_body()
            forget = (request.get_json(silent=True) or {}).get("forget", False)
            if project is None or not isinstance(forget, bool):
                return self._text_response('Expected a JSON body {"project": <name or root path>, "forget": <bool>}', 400)
            try:
                registered_project, was_loaded = self.unload_project(project, forget=forget)
            except ProjectRequestRejectedError as e:
                return self._text_response(str(e), e.status)
            content = {"project": self._project_json_dict(registered_project), "unloaded": was_loaded, "forgotten": forget}
            return self._json_response(content, 200)

        @app.route("/edit_project", methods=["POST"])
        def edit_project() -> Any:
            try:
                edit_request = QueryProjectRequest.model_validate(request.get_json(silent=True))
            except ValidationError as e:
                return self._text_response(f"Invalid project edit: {e}", 400)

            def apply_edit() -> Response:
                try:
                    return self._text_response(self.edit_project(edit_request), 200)
                except ProjectRequestRejectedError as e:
                    return self._text_response(str(e), e.status)

            return self._process_using_project(edit_request.project_name, apply_edit)

        @app.route("/shutdown", methods=["POST"])
        def shutdown() -> Any:
            log.info("Shutdown requested")
            self._schedule_shutdown()
            return self._text_response("Shutting down", 202)

    def _schedule_shutdown(self) -> None:
        # interrupting the main thread (shortly after the response is sent) ends the server loop, upon which run()
        # cleans up; a repeated request must not interrupt the clean-up
        if not self._shutdown_scheduled.is_set():
            self._shutdown_scheduled.set()
            threading.Timer(0.2, _thread.interrupt_main).start()


class ProjectEditError(Exception):
    """
    Raised when the project server does not apply an edit; the message gives the server's reason.
    """


class ProjectEditClient(ProjectServerClient):
    """
    Client for the ``/edit_project`` route of a running :class:`ExtendedProjectServer`.
    """

    TIMEOUT_MARGIN_SECONDS = 30.0
    """
    the time by which the client waits longer than the server's tool timeout, such that the server's report of a
    timed-out edit reaches the client
    """

    def __init__(self, serena_config: SerenaConfig, host: str = "127.0.0.1", port: int | None = None) -> None:
        """
        :param serena_config: Serena's configuration, providing the authentication secret and the tool timeout
        :param host: the host address of the project server
        :param port: the port of the project server; if None, use default
        :raises ConnectionError: if the project server is not reachable
        """
        super().__init__(serena_config, host=host, port=port)
        self._edit_timeout = serena_config.tool_timeout + self.TIMEOUT_MARGIN_SECONDS

    def edit_project(self, project_name: str, tool_name: str, tool_params_json: str) -> str:
        """
        Edits a project by executing an editing tool in its context.

        :param project_name: the name or root path of the project
        :param tool_name: the name of the editing tool
        :param tool_params_json: the parameters to pass to the tool, encoded as a JSON string
        :return: the tool's result
        :raises ProjectEditError: if the server rejects the edit, the tool fails or the outcome of the edit is unknown
        """
        payload = QueryProjectRequest(project_name=project_name, tool_name=tool_name, tool_params_json=tool_params_json).model_dump()
        try:
            response = requests_lib.post(f"{self._base_url}/edit_project", json=payload, headers=self._headers, timeout=self._edit_timeout)
        except requests_lib.Timeout as e:
            raise ProjectEditError(
                f"The project server did not answer within {self._edit_timeout} seconds; the outcome of the edit is unknown "
                "(read the file before repeating the edit)"
            ) from e
        if response.status_code == 200:
            return response.text
        hint = " (read the file before repeating the edit)" if response.status_code == 504 else ""
        raise ProjectEditError(f"The project server did not apply the edit ({response.status_code}): {response.text}{hint}")
