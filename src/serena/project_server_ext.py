# SPDX-License-Identifier: GPL-3.0-or-later
"""
Extensions of the Serena fork for a long-running project server: projects are warmed up at startup and their readiness
is reported by the heartbeat, a registry file lets clients find the server, an allowlist restricts the tools the server
accepts, and a shutdown route stops the server gracefully (saving the language servers' caches).
"""

import _thread
import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from flask import Response, request
from werkzeug.exceptions import HTTPException

from serena import serena_version
from serena.config.serena_config import SerenaPaths
from serena.project_server import CallFacadeMethodRequest, ProjectServer, QueryProjectRequest
from serena.util.file_system import write_file_atomic

if TYPE_CHECKING:
    from serena.project import Project
    from serena.repl.facade import FacadeMethod
    from solidlsp.ls import SolidLanguageServer

log = logging.getLogger(__name__)


class ToolNotAllowedError(Exception):
    """
    Raised when a request asks the project server for a tool or facade method outside of its allowlist.
    """


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
        self._entries: dict[str, dict[str, Any]] = {}
        self._start_times: dict[str, float] = {}

    def begin(self, project_root: str, project_name: str) -> bool:
        """
        Records the start of the warm-up of a project, unless the project is already loading or ready.

        :param project_root: the project's root path
        :param project_name: the project's name
        :return: whether the warm-up shall be started
        """
        with self._lock:
            entry = self._entries.get(project_root)
            if entry is not None and entry["state"] != ProjectState.FAILED.value:
                return False
            self._start_times[project_root] = time.monotonic()
            self._entries[project_root] = {"name": project_name, "state": ProjectState.LOADING.value, "seconds": None, "error": None}
            self._on_change(self._snapshot())
            return True

    def finish(self, project_root: str, error: str | None = None) -> None:
        """
        Records the end of the warm-up of a project.

        :param project_root: the project's root path
        :param error: the error that made the warm-up fail; None if the project is ready
        """
        with self._lock:
            entry = self._entries[project_root]
            entry["state"] = (ProjectState.READY if error is None else ProjectState.FAILED).value
            entry["seconds"] = round(time.monotonic() - self._start_times[project_root], 1)
            entry["error"] = error
            self._on_change(self._snapshot())

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """
        :return: the readiness entries (name, state, seconds, error) by project root
        """
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        return {root: dict(entry) for root, entry in self._entries.items()}


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

    def start(self, project_root: str, project_name: str) -> None:
        """
        Starts the warm-up of a project in a background thread, unless it is already loading or ready.

        :param project_root: the project's root path
        :param project_name: the project's name
        """
        if self._tracker.begin(project_root, project_name):
            threading.Thread(target=self._warm_up, args=(project_root,), name=f"warmup:{project_name}", daemon=True).start()

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
        file_name = "project-server.json" if port == ProjectServer.PORT else f"project-server-{port}.json"
        self.path = os.path.join(directory if directory is not None else SerenaPaths().serena_user_home_dir, file_name)
        self._port = port
        self._pid = pid
        self._started_at = datetime.now(UTC).isoformat(timespec="seconds")

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
    graceful shutdown to the upstream :class:`ProjectServer`; without warm-up projects and allowlist, requests are
    handled exactly as upstream.

    Routes in addition to the upstream ones:
      * ``GET /heartbeat`` additionally reports ``projects``: the readiness of the warmed-up projects by root path;
      * ``POST /warmup`` with ``{"project": <name or root>}`` starts the warm-up of a registered project (202);
      * ``POST /shutdown`` stops the server gracefully (202).
    Rejections by the allowlist are reported with status 403 and malformed ``/query_project`` requests with status
    400, each with the reason as plain text.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int | None = None, warm_projects: Sequence[str] = (), allowed_tools: Sequence[str] = ()
    ) -> None:
        """
        :param host: the host address to listen on
        :param port: the port to listen on; if None, use default
        :param warm_projects: the names or root paths of the projects to warm up when the server starts
        :param allowed_tools: the names of the tools the server accepts; empty for no restriction
        """
        # the routes, which are set up by the base class constructor, depend on the allowlist
        self._allowlist = ToolAllowlist(allowed_tools)
        super().__init__(host=host, port=port)
        self._warm_projects = list(warm_projects)
        self._registry = ProjectServerRegistryFile(self._port, os.getpid())
        self._readiness = ProjectReadinessTracker(on_change=self._registry.write)
        self._warmer = ProjectWarmer(self._get_project, self._readiness)

    def warm_up(self, project_root_or_name: str) -> None:
        """
        Starts the warm-up of a registered project, unless it is already loading or ready.

        :param project_root_or_name: the project's name or root path
        :raises ValueError: if the project is not registered (or the name is ambiguous)
        """
        registered_project = self.get_serena_config().get_registered_project(project_root_or_name)
        if registered_project is None:
            raise ValueError(f"Project '{project_root_or_name}' is not registered with Serena.")
        self._warmer.start(str(registered_project.project_root), registered_project.project_config.project_name)

    def run(self) -> None:
        """
        Runs the server until it is shut down (via ``/shutdown``, Ctrl+C, Ctrl+Break or SIGTERM), warming up the
        configured projects in the background, and finally stops the loaded projects' language servers (saving their
        caches) and removes the registry file.
        """
        self._registry.write(self._readiness.snapshot())
        if self._allowlist.tool_names:
            log.info("Accepted tools: %s", ", ".join(self._allowlist.tool_names))
        for termination_signal in (getattr(signal, "SIGBREAK", None), signal.SIGTERM):
            if termination_signal is not None:
                signal.signal(termination_signal, signal.default_int_handler)
        for project in self._warm_projects:
            try:
                self.warm_up(project)
            except ValueError as e:
                log.error("Cannot warm up project: %s", e)

        try:
            super().run()
        finally:
            self._shut_down_projects()
            self._registry.remove()

    def _shut_down_projects(self) -> None:
        with self._loaded_projects_lock:
            projects = list(self._loaded_projects_by_root.values())
            self._loaded_projects_by_root.clear()
        for project in projects:
            log.info("Shutting down project '%s'", project.project_name)
            try:
                project.shutdown(timeout=5)
            except Exception as e:
                log.error("Failed to shut down project '%s'", project.project_name, exc_info=e)

    @staticmethod
    def _text_response(message: str, status: int) -> Response:
        return Response(message, status=status, mimetype="text/plain")

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
                return query_project_view()
            except ToolNotAllowedError as e:
                return self._text_response(str(e), 403)
            except HTTPException:
                raise
            except Exception as e:
                log.warning("Rejected project query: %s", e)
                return self._text_response(f"{type(e).__name__}: {e}", 400)

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
            return call_facade_method_view()

        app.view_functions["heartbeat"] = heartbeat
        app.view_functions["query_project"] = query_project
        app.view_functions["call_facade_method"] = call_facade_method

        @app.route("/warmup", methods=["POST"])
        def warmup() -> Any:
            body = request.get_json(silent=True) or {}
            project = body.get("project")
            if not isinstance(project, str) or not project:
                return self._text_response('Expected a JSON body {"project": <name or root path>}', 400)
            try:
                self.warm_up(project)
            except ValueError as e:
                return self._text_response(str(e), 404)
            return Response(json.dumps({"projects": self._readiness.snapshot()}), status=202, mimetype="application/json")

        @app.route("/shutdown", methods=["POST"])
        def shutdown() -> Any:
            log.info("Shutdown requested")
            self._schedule_shutdown()
            return self._text_response("Shutting down", 202)

    @staticmethod
    def _schedule_shutdown() -> None:
        # interrupting the main thread (shortly after the response is sent) ends the server loop, upon which run() cleans up
        threading.Timer(0.2, _thread.interrupt_main).start()
