"""Tests for the project server extensions of the Serena fork (tool allowlist, warm-up and readiness, registry file,
graceful shutdown), exercised through the HTTP routes of a server whose agent is mocked.
"""

import json
import os
import pickle
import shutil
import signal
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from flask import Flask

from serena.config.serena_config import SerenaConfig
from serena.constants import SERENA_MANAGED_DIR_NAME
from serena.project_server import ProjectServer
from serena.project_server_ext import (
    ExtendedProjectServer,
    ProjectReadinessTracker,
    ProjectServerRegistryFile,
    ProjectServerSettingsError,
    ProjectUsageGate,
    ProjectWarmer,
    ToolAllowlist,
    WritableRoots,
)
from serena.repl.facade import FacadeMethodInfo
from serena.tools import FindSymbolTool, RenameSymbolTool

SECRET = "test-shared-secret"
AUTH = {"Authorization": f"Bearer {SECRET}"}


def _make_server(
    registry_dir: Path, allowed_tools: tuple[str, ...] = (), mock_handlers: bool = True, writable_roots: tuple[str, ...] = ()
) -> ExtendedProjectServer:
    """
    Creates a server with a mocked agent (no language servers), as the upstream project server tests do; unless
    `mock_handlers` is False, the upstream query and facade call handlers are replaced by stubs echoing the request.
    """
    server = ExtendedProjectServer.__new__(ExtendedProjectServer)
    server._agent = MagicMock()
    server._agent.serena_config.auth_secret = SECRET
    server._loaded_projects_by_root = {}
    server._project_load_locks_by_root = {}
    server._active_project_lock = threading.Lock()
    server._loaded_projects_lock = threading.Lock()
    server._host = "127.0.0.1"
    server._port = 24299
    server._app = Flask(__name__)
    server._allowlist = ToolAllowlist(allowed_tools)
    server._warm_projects = []
    server._pinned_roots = set()
    server._writable_roots = WritableRoots(writable_roots)
    server._usage = ProjectUsageGate()
    server._registration_lock = threading.Lock()
    server._registry = ProjectServerRegistryFile(server._port, os.getpid(), directory=str(registry_dir))
    server._readiness = ProjectReadinessTracker(on_change=server._registry.write)
    server._warmer = ProjectWarmer(server._get_project, server._readiness)
    server._shutdown_scheduled = threading.Event()
    server._setup_routes()
    if mock_handlers:
        server._query_project = lambda req: f"queried {req.tool_name}"
        server._call_facade_method = lambda req: f"called {req.method_name}"
    return server


def _register(server: ExtendedProjectServer, *project_roots: str) -> None:
    """Registers projects (named after their root's last component) with the server's mocked configuration."""
    registered = {
        key: SimpleNamespace(project_root=Path(root), project_config=SimpleNamespace(project_name=Path(root).name))
        for root in project_roots
        for key in (root, Path(root).name)
    }
    _mocked_agent(server).serena_config.get_registered_project.side_effect = lambda key: registered.get(key)


def _mocked_agent(server: ExtendedProjectServer) -> MagicMock:
    return cast(MagicMock, server._agent)


def _query(server: ExtendedProjectServer, tool_name: str) -> Any:
    body = {"project_name": "p", "tool_name": tool_name, "tool_params_json": "{}"}
    with server._app.test_client() as client:
        return client.post("/query_project", json=body, headers=AUTH)


def _call(server: ExtendedProjectServer, method_name: str, info: FacadeMethodInfo) -> Any:
    _mocked_agent(server).get_repl().entrypoint.get_facade_.return_value.get_method.return_value = SimpleNamespace(
        info=info, qualified_name=f"lsp.{method_name}"
    )
    body = {"project_name": "p", "facade_name": "lsp", "method_name": method_name, "args": [], "kwargs": {}}
    with server._app.test_client() as client:
        return client.post("/call_facade_method", json=body, headers=AUTH)


def _wait_until_settled(tracker: ProjectReadinessTracker, root: str) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        entry = tracker.snapshot().get(root)
        if entry is not None and entry["state"] != "loading":
            return entry
        time.sleep(0.01)
    raise AssertionError(f"warm-up of {root} did not finish")


class TestToolAllowlist:
    ALLOWED = ("find_symbol", "get_symbols_overview", "find_referencing_symbols", "find_declaration")

    @pytest.mark.parametrize("tool_name", ["rename_symbol", "replace_in_files", "find_implementations"])
    def test_query_for_a_tool_outside_the_allowlist_is_rejected_with_its_reason(self, tmp_path: Path, tool_name: str) -> None:
        response = _query(_make_server(tmp_path, self.ALLOWED), tool_name)
        assert response.status_code == 403
        assert f"Tool '{tool_name}' is not allowed" in response.get_data(as_text=True)

    def test_query_for_an_allowed_tool_is_executed(self, tmp_path: Path) -> None:
        response = _query(_make_server(tmp_path, self.ALLOWED), "find_symbol")
        assert response.status_code == 200
        assert response.get_data(as_text=True) == "queried find_symbol"

    def test_editing_facade_method_is_rejected_even_if_its_tool_is_listed(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, ("rename_symbol",))
        response = _call(
            server, "rename_symbol", FacadeMethodInfo(name="rename_symbol", can_edit=True, corresponding_tool=RenameSymbolTool)
        )
        assert response.status_code == 403
        assert "lsp.rename_symbol" in response.get_data(as_text=True)

    def test_facade_method_without_a_corresponding_tool_is_rejected(self, tmp_path: Path) -> None:
        response = _call(_make_server(tmp_path, self.ALLOWED), "project_context", FacadeMethodInfo(name="project_context"))
        assert response.status_code == 403

    def test_facade_method_of_an_allowed_tool_is_executed(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, self.ALLOWED)
        response = _call(server, "find_symbol", FacadeMethodInfo(name="find_symbol", corresponding_tool=FindSymbolTool))
        assert response.status_code == 200
        assert pickle.loads(response.data) == "called find_symbol"

    def test_without_allowlist_facade_methods_are_executed_as_upstream(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        response = _call(
            server, "rename_symbol", FacadeMethodInfo(name="rename_symbol", can_edit=True, corresponding_tool=RenameSymbolTool)
        )
        assert response.status_code == 200
        assert pickle.loads(response.data) == "called rename_symbol"

    def test_malformed_query_is_rejected_with_its_reason(self, tmp_path: Path) -> None:
        with _make_server(tmp_path)._app.test_client() as client:
            response = client.post("/query_project", json={"tool_name": "find_symbol"}, headers=AUTH)
        assert response.status_code == 400
        assert "project_name" in response.get_data(as_text=True)

    def test_facade_call_for_an_unknown_facade_is_rejected_with_its_reason(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path, self.ALLOWED)
        _mocked_agent(server).get_repl().entrypoint.get_facade_.side_effect = ValueError("No facade 'nope'")
        body = {"project_name": "p", "facade_name": "nope", "method_name": "x", "args": [], "kwargs": {}}
        with server._app.test_client() as client:
            response = client.post("/call_facade_method", json=body, headers=AUTH)
        assert response.status_code == 400
        assert "No facade 'nope'" in response.get_data(as_text=True)

    def test_requests_still_require_authentication(self, tmp_path: Path) -> None:
        with _make_server(tmp_path, self.ALLOWED)._app.test_client() as client:
            assert client.post("/query_project", json={}).status_code == 401
            assert client.post("/warmup", json={"project": "p"}).status_code == 401
            assert client.post("/shutdown").status_code == 401


class _StubLanguageServer:
    def __init__(self) -> None:
        self.requests: list[tuple[Any, ...]] = []

    def request_document_symbols(self, relative_path: str) -> Any:
        self.requests.append(("document_symbols", relative_path))
        symbols = [{"name": "Foo", "selectionRange": {"start": {"line": 3, "character": 4}}}]
        return SimpleNamespace(iter_symbols=lambda: iter(symbols))

    def request_hover(self, relative_path: str, line: int, column: int) -> None:
        self.requests.append(("hover", relative_path, line, column))


def _stub_project(language_server: _StubLanguageServer, files: list[str]) -> Any:
    manager = SimpleNamespace(
        iter_language_servers=lambda: iter([language_server]),
        has_suitable_ls_for_file=lambda path: path.endswith(".cs"),
        get_language_server=lambda path: language_server,
    )
    return SimpleNamespace(language_server_manager=manager, gather_source_files=lambda: [SimpleNamespace(rel_path=f) for f in files])


class TestWarmUp:
    def test_warmed_up_project_is_reported_ready_after_probing_its_language_server(self) -> None:
        language_server = _StubLanguageServer()
        tracker = ProjectReadinessTracker(on_change=lambda snapshot: None)
        ProjectWarmer(lambda root: _stub_project(language_server, ["README.md", "A.cs", "B.cs"]), tracker).start("/repo", "repo")

        entry = _wait_until_settled(tracker, "/repo")
        assert entry["state"] == "ready"
        assert entry["name"] == "repo"
        assert language_server.requests == [("document_symbols", "A.cs"), ("hover", "A.cs", 3, 4)]

    def test_failed_load_is_reported_with_its_error(self) -> None:
        def fail(root: str) -> Any:
            raise ValueError("cannot load")

        tracker = ProjectReadinessTracker(on_change=lambda snapshot: None)
        ProjectWarmer(fail, tracker).start("/repo", "repo")

        entry = _wait_until_settled(tracker, "/repo")
        assert entry["state"] == "failed"
        assert entry["error"] == "ValueError: cannot load"

    def test_project_already_loading_is_not_loaded_again_and_failed_one_is_retried(self) -> None:
        tracker = ProjectReadinessTracker(on_change=lambda snapshot: None)
        assert tracker.begin("/repo", "repo")
        assert not tracker.begin("/repo", "repo")
        tracker.finish("/repo", error="boom")
        assert tracker.begin("/repo", "repo")

    def test_hanging_probe_of_a_cold_project_does_not_delay_queries_for_another_project(self, tmp_path: Path) -> None:
        # queries run under the server's process-wide lock (upstream _query_project); the probe must not take it
        server = _make_server(tmp_path, mock_handlers=False)
        _register(server, "/ready", "/cold")
        server._loaded_projects_by_root[str(Path("/ready"))] = MagicMock()
        _mocked_agent(server).get_tool_by_name.return_value.apply_ex.return_value = "found"

        release = threading.Event()

        class _HangingLanguageServer(_StubLanguageServer):
            def request_hover(self, relative_path: str, line: int, column: int) -> None:
                release.wait(timeout=10)

        cold_project = _stub_project(_HangingLanguageServer(), ["A.cs"])
        server._loaded_projects_by_root[str(Path("/cold"))] = cold_project
        server.warm_up("/cold")
        try:
            assert server._readiness.snapshot()[str(Path("/cold"))]["state"] == "loading"
            start = time.monotonic()
            with server._app.test_client() as client:
                response = client.post(
                    "/query_project", json={"project_name": "ready", "tool_name": "find_symbol", "tool_params_json": "{}"}, headers=AUTH
                )
            assert response.get_data(as_text=True) == "found"
            assert time.monotonic() - start < 2
            assert server._readiness.snapshot()[str(Path("/cold"))]["state"] == "loading"
        finally:
            release.set()

    def test_heartbeat_reports_project_readiness(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        server._readiness.begin("/repo", "repo")
        server._readiness.finish("/repo")
        with server._app.test_client() as client:
            heartbeat = client.get("/heartbeat", headers=AUTH).get_json()
        assert heartbeat["status"] == "alive"
        assert heartbeat["projects"]["/repo"]["state"] == "ready"

    def test_warmup_route_starts_registered_projects_only(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        started: list[tuple[str, str]] = []
        server._warmer = SimpleNamespace(start=lambda root, name: started.append((root, name)))
        registered = SimpleNamespace(project_root=Path("/repo"), project_config=SimpleNamespace(project_name="repo"))
        _mocked_agent(server).serena_config.get_registered_project.side_effect = lambda key: registered if key == "repo" else None

        with server._app.test_client() as client:
            assert client.post("/warmup", json={"project": "repo"}, headers=AUTH).status_code == 202
            assert client.post("/warmup", json={"project": "unknown"}, headers=AUTH).status_code == 404
            assert client.post("/warmup", json={}, headers=AUTH).status_code == 400
        assert started == [(str(Path("/repo")), "repo")]


class TestRegistryFile:
    def test_registry_describes_the_server_and_follows_project_readiness(self, tmp_path: Path) -> None:
        registry = ProjectServerRegistryFile(ProjectServer.PORT, 4242, directory=str(tmp_path))
        tracker = ProjectReadinessTracker(on_change=registry.write)
        tracker.begin("/repo", "repo")
        content = json.loads(Path(registry.path).read_text(encoding="utf-8"))
        assert Path(registry.path).name == "project-server.json"
        assert (content["pid"], content["port"]) == (4242, ProjectServer.PORT)
        assert content["projects"]["/repo"]["state"] == "loading"
        assert "auth_secret" not in json.dumps(content)

        tracker.finish("/repo")
        assert json.loads(Path(registry.path).read_text(encoding="utf-8"))["projects"]["/repo"]["state"] == "ready"

    def test_non_default_port_has_its_own_registry_file(self, tmp_path: Path) -> None:
        assert Path(ProjectServerRegistryFile(24226, 1, directory=str(tmp_path)).path).name == "project-server-24226.json"

    def test_registry_of_another_server_is_not_removed(self, tmp_path: Path) -> None:
        own = ProjectServerRegistryFile(ProjectServer.PORT, 1, directory=str(tmp_path))
        other = ProjectServerRegistryFile(ProjectServer.PORT, 2, directory=str(tmp_path))
        own.write({})
        other.write({})
        own.remove()
        assert os.path.exists(other.path)
        other.remove()
        assert not os.path.exists(other.path)


class TestShutdown:
    def test_run_shuts_down_loaded_projects_and_removes_the_registry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _make_server(tmp_path)
        project = MagicMock()
        server._loaded_projects_by_root["/repo"] = project
        registry_present_while_running: list[bool] = []
        monkeypatch.setattr(ProjectServer, "run", lambda self: registry_present_while_running.append(os.path.exists(server._registry.path)))
        monkeypatch.setattr("serena.project_server_ext.signal.signal", lambda *args: None)

        server.run()

        assert registry_present_while_running == [True]
        project.shutdown.assert_called_once()
        assert not os.path.exists(server._registry.path)

    def test_failing_server_loop_and_project_still_shut_down_the_others(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _make_server(tmp_path)
        failing, healthy = MagicMock(), MagicMock()
        failing.shutdown.side_effect = RuntimeError("cannot stop")
        server._loaded_projects_by_root.update({"/failing": failing, "/healthy": healthy})
        handlers: list[Any] = []

        def fail(self: ProjectServer) -> None:
            raise RuntimeError("server loop failed")

        monkeypatch.setattr(ProjectServer, "run", fail)
        monkeypatch.setattr("serena.project_server_ext.signal.signal", lambda signal_number, handler: handlers.append(handler))

        with pytest.raises(RuntimeError, match="server loop failed"):
            server.run()

        healthy.shutdown.assert_called_once()
        assert not os.path.exists(server._registry.path)
        # termination signals are ignored during the clean-up
        assert handlers[-1] is signal.SIG_IGN

    def test_server_on_a_busy_port_leaves_the_other_servers_registry_alone(self, tmp_path: Path) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other_server:
            other_server.bind(("127.0.0.1", 0))
            other_server.listen()
            server = _make_server(tmp_path)
            server._port = other_server.getsockname()[1]
            server._registry = ProjectServerRegistryFile(server._port, os.getpid() + 1, directory=str(tmp_path))
            server._registry.write({})
            registry_content = Path(server._registry.path).read_text(encoding="utf-8")
            server._registry = ProjectServerRegistryFile(server._port, os.getpid(), directory=str(tmp_path))

            with pytest.raises(OSError, match="already in use"):
                server.run()

        assert Path(server._registry.path).read_text(encoding="utf-8") == registry_content

    def test_shutdown_stops_a_project_whose_warm_up_was_still_loading(self, tmp_path: Path) -> None:
        server = _make_server(tmp_path)
        project = MagicMock()

        def load_slowly(root: str) -> Any:
            time.sleep(0.3)
            server._loaded_projects_by_root[root] = project
            return _stub_project(_StubLanguageServer(), [])

        server._warmer = ProjectWarmer(load_slowly, server._readiness)
        server._warmer.start("/slow", "slow")
        server._shut_down_projects()
        project.shutdown.assert_called_once()

    def test_shutdown_route_schedules_the_shutdown_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _make_server(tmp_path)
        timers: list[Any] = []

        class _RecordingTimer:
            def __init__(self, interval: float, function: Any) -> None:
                timers.append(function)

            def start(self) -> None:
                pass

        monkeypatch.setattr("serena.project_server_ext.threading.Timer", _RecordingTimer)
        with server._app.test_client() as client:
            assert client.post("/shutdown", headers=AUTH).status_code == 202
            assert client.post("/shutdown", headers=AUTH).status_code == 202
        assert len(timers) == 1


class TestSettings:
    def test_writable_roots_require_an_allowlist_and_a_local_host(self) -> None:
        with pytest.raises(ProjectServerSettingsError, match="allowed-tool"):
            ExtendedProjectServer(writable_roots=("/worktrees",))
        with pytest.raises(ProjectServerSettingsError, match="local host"):
            ExtendedProjectServer(host="0.0.0.0", allowed_tools=("find_symbol",), writable_roots=("/worktrees",))

    def test_writable_roots_contain_paths_strictly_inside_them(self, tmp_path: Path) -> None:
        roots = WritableRoots([str(tmp_path / "worktrees")])

        assert roots.contains(str(tmp_path / "worktrees" / "wt1"))
        assert not roots.contains(str(tmp_path / "worktrees"))
        assert not roots.contains(str(tmp_path / "worktrees" / ".." / "main"))
        assert not roots.contains(str(tmp_path / "main"))
        assert not WritableRoots().contains(str(tmp_path / "worktrees" / "wt1"))


class TestRegistrationAndUnloading:
    """
    Registration and unloading of projects at runtime, with Serena's configuration in a temporary file.
    """

    @pytest.fixture
    def server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExtendedProjectServer:
        config_path = tmp_path / "serena_config.yml"
        config_path.write_text("projects: []\n", encoding="utf-8")
        monkeypatch.setattr(SerenaConfig, "_determine_config_file_path", classmethod(lambda cls: str(config_path)))
        self.worktrees = tmp_path / "worktrees"
        self.worktrees.mkdir()

        server = _make_server(tmp_path, allowed_tools=("find_symbol",), writable_roots=(str(self.worktrees),))
        serena_config = SerenaConfig.from_config_file()
        serena_config.auth_secret = SECRET
        _mocked_agent(server).serena_config = serena_config
        self.started_warm_ups: list[str] = []
        server._warmer = SimpleNamespace(start=lambda root, name: self.started_warm_ups.append(root))
        return server

    def _make_project(self, dir_name: str, project_name: str | None = None) -> Path:
        project_dir = self.worktrees / dir_name
        (project_dir / SERENA_MANAGED_DIR_NAME).mkdir(parents=True)
        (project_dir / SERENA_MANAGED_DIR_NAME / "project.yml").write_text(
            f'project_name: "{project_name or dir_name}"\nlanguage_servers: []\n', encoding="utf-8"
        )
        return project_dir

    @staticmethod
    def _registered_roots_in_file() -> list[Path]:
        return [Path(project.project_root) for project in SerenaConfig.from_config_file(generate_if_missing=False).projects]

    @staticmethod
    def _post(server: ExtendedProjectServer, route: str, body: dict[str, Any]) -> Any:
        with server._app.test_client() as client:
            return client.post(route, json=body, headers=AUTH)

    def _load(self, server: ExtendedProjectServer, project_dir: Path) -> MagicMock:
        """Publishes a mocked loaded instance of a registered project, which reports itself ready."""
        project = MagicMock()
        root = str(project_dir.resolve())
        server._loaded_projects_by_root[root] = project
        server._readiness.begin(root, project_dir.name)
        server._readiness.finish(root)
        return project

    def test_registered_project_is_known_to_the_server_and_written_to_the_configuration_once(self, server: ExtendedProjectServer) -> None:
        project_dir = self._make_project("wt1")

        first = self._post(server, "/register", {"project": str(project_dir)})
        second = self._post(server, "/register", {"project": str(project_dir)})

        assert first.status_code == 200
        assert first.get_json() == {"project": {"name": "wt1", "root": str(project_dir.resolve())}, "added": True}
        assert second.status_code == 200
        assert second.get_json()["added"] is False
        assert self._registered_roots_in_file() == [project_dir.resolve()]
        assert self._post(server, "/warmup", {"project": str(project_dir)}).status_code == 202
        assert self.started_warm_ups == [str(project_dir.resolve())]

    def test_registration_outside_the_writable_roots_or_without_configuration_is_rejected(
        self, server: ExtendedProjectServer, tmp_path: Path
    ) -> None:
        outside = tmp_path / "main"
        (outside / SERENA_MANAGED_DIR_NAME).mkdir(parents=True)
        (self.worktrees / "bare").mkdir()

        assert self._post(server, "/register", {"project": str(outside)}).status_code == 403
        assert self._post(server, "/register", {"project": str(self.worktrees / "bare")}).status_code == 404
        assert self._post(server, "/register", {}).status_code == 400
        assert self._registered_roots_in_file() == []

    def test_name_of_an_existing_project_conflicts_but_one_of_a_removed_root_is_replaced(self, server: ExtendedProjectServer) -> None:
        first = self._make_project("wave1", project_name="ews@integration")
        second = self._make_project("wave2", project_name="ews@integration")
        assert self._post(server, "/register", {"project": str(first)}).status_code == 200

        conflict = self._post(server, "/register", {"project": str(second)})
        shutil.rmtree(first)
        replacement = self._post(server, "/register", {"project": str(second)})

        assert conflict.status_code == 409
        assert "ews@integration" in conflict.get_data(as_text=True)
        assert replacement.status_code == 200
        assert self._registered_roots_in_file() == [second.resolve()]
        registered = server.get_serena_config().get_registered_project("ews@integration")
        assert registered is not None
        assert registered.project_root == second.resolve()

    def test_unloaded_project_is_shut_down_and_no_longer_reported(self, server: ExtendedProjectServer) -> None:
        project_dir = self._make_project("wt1")
        self._post(server, "/register", {"project": str(project_dir)})
        project = self._load(server, project_dir)

        response = self._post(server, "/unload", {"project": "wt1"})

        assert response.status_code == 200
        assert response.get_json() == {"project": {"name": "wt1", "root": str(project_dir.resolve())}, "unloaded": True, "forgotten": False}
        project.shutdown.assert_called_once()
        with server._app.test_client() as client:
            assert str(project_dir.resolve()) not in client.get("/heartbeat", headers=AUTH).get_json()["projects"]
        assert str(project_dir.resolve()) not in json.loads(Path(server._registry.path).read_text(encoding="utf-8"))["projects"]
        assert server.get_serena_config().get_registered_project("wt1") is not None

    def test_forgotten_project_is_unknown_to_the_server_and_dropped_from_the_configuration_with_the_next_change(
        self, server: ExtendedProjectServer
    ) -> None:
        forgotten = self._make_project("wt1")
        other = self._make_project("wt2")
        self._post(server, "/register", {"project": str(forgotten)})
        self._load(server, forgotten)
        projects_before = iter(server.get_serena_config().projects)

        response = self._post(server, "/unload", {"project": str(forgotten), "forget": True})
        self._post(server, "/register", {"project": str(other)})

        assert response.get_json()["forgotten"] is True
        assert self._post(server, "/unload", {"project": str(forgotten)}).status_code == 404
        assert self._registered_roots_in_file() == [other.resolve()]
        # a reader iterating the registered projects during the unloading still sees all of them
        assert [p.project_root for p in projects_before] == [forgotten.resolve()]

    def test_project_warmed_up_at_startup_or_being_warmed_up_is_not_unloaded(self, server: ExtendedProjectServer) -> None:
        pinned = self._make_project("pinned")
        warming = self._make_project("warming")
        for project_dir in (pinned, warming):
            self._post(server, "/register", {"project": str(project_dir)})
        server._pinned_roots.add(str(pinned.resolve()))
        server._readiness.begin(str(warming.resolve()), "warming")

        assert self._post(server, "/unload", {"project": "pinned"}).status_code == 409
        assert self._post(server, "/unload", {"project": "warming"}).status_code == 409
        assert self._post(server, "/unload", {"project": "unknown"}).status_code == 404

    def test_unloading_waits_for_requests_using_the_project_and_rejects_new_ones(
        self, server: ExtendedProjectServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_dir = self._make_project("wt1")
        self._post(server, "/register", {"project": str(project_dir)})
        project = self._load(server, project_dir)
        entered, release = threading.Event(), threading.Event()
        num_queries = [0]

        def query(req: Any) -> str:
            num_queries[0] += 1
            if num_queries[0] == 1:
                entered.set()
                release.wait(timeout=10)
            return "queried"

        monkeypatch.setattr(server, "_query_project", query)
        responses: dict[str, Any] = {}
        body = {"project_name": "wt1", "tool_name": "find_symbol", "tool_params_json": "{}"}
        running_query = threading.Thread(target=lambda: responses.update(running=self._post(server, "/query_project", body)))
        running_query.start()
        assert entered.wait(timeout=5)
        unloading = threading.Thread(target=lambda: responses.update(unload=self._post(server, "/unload", {"project": "wt1"})))
        unloading.start()

        # a query arriving while the unloading waits is rejected, and the project is not shut down meanwhile
        deadline = time.monotonic() + 5
        while self._post(server, "/query_project", body).status_code != 409:
            assert time.monotonic() < deadline
        project.shutdown.assert_not_called()

        release.set()
        running_query.join(timeout=5)
        unloading.join(timeout=5)
        assert responses["running"].get_data(as_text=True) == "queried"
        assert responses["unload"].status_code == 200
        project.shutdown.assert_called_once()

    def test_queries_for_other_projects_are_answered_while_a_project_shuts_down(
        self, server: ExtendedProjectServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unloaded_dir = self._make_project("wt1")
        other_dir = self._make_project("wt2")
        for project_dir in (unloaded_dir, other_dir):
            self._post(server, "/register", {"project": str(project_dir)})
        unloaded = self._load(server, unloaded_dir)
        self._load(server, other_dir)
        shutting_down, release = threading.Event(), threading.Event()
        unloaded.shutdown.side_effect = lambda timeout: (shutting_down.set(), release.wait(timeout=10))
        monkeypatch.setattr(server, "_query_project", lambda req: f"queried {req.project_name}")

        unloading = threading.Thread(target=lambda: self._post(server, "/unload", {"project": "wt1"}))
        unloading.start()
        try:
            assert shutting_down.wait(timeout=5)
            start = time.monotonic()
            response = self._post(server, "/query_project", {"project_name": "wt2", "tool_name": "find_symbol", "tool_params_json": "{}"})
            assert response.get_data(as_text=True) == "queried wt2"
            assert time.monotonic() - start < 2
        finally:
            release.set()
            unloading.join(timeout=5)
