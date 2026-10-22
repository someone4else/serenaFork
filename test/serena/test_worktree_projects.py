"""
Tests for the preparation and release of git worktrees as Serena projects (`serena project prepare-worktree` and
`serena project release-worktree`), with Serena's configuration in a temporary directory.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import psutil
import pytest
import requests
from click.testing import CliRunner

from serena.cli import ProjectCommands
from serena.config.serena_config import SerenaConfig
from serena.constants import SERENA_MANAGED_DIR_NAME
from serena.project_server_ext import ProjectServerRegistryFile
from serena.project_server_port import ProjectServerPort
from serena.util.yaml import load_yaml
from serena.worktree_projects import ProjectServerAdminClient, ProjectServerState, WorktreeProjectError, WorktreeProjects

MAIN_PROJECT_YML = """\
# the main project of the repository
project_name: "main"
language_servers:
- csharp
ignored_paths:
- "**/bin"
"""


class _RecordingProjectServer(ProjectServerAdminClient):
    """
    A stand-in for the project server's admin client, answering each route with the given sequence of statuses (or
    raising the given exceptions).
    """

    def __init__(self, statuses_by_route: dict[str, list[int | Exception]]) -> None:  # the base class (an HTTP client) is not initialised
        self.statuses_by_route = statuses_by_route
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, route: str, body: dict[str, Any]) -> Any:
        self.requests.append((route, body))
        statuses = self.statuses_by_route[route]
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        if isinstance(status, Exception):
            raise status
        return SimpleNamespace(status_code=status, text=f"answer {status}", json=dict)


class TestWorktreeProjects:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.config_path = tmp_path / "home" / "serena_config.yml"
        self.config_path.parent.mkdir()
        self.config_path.write_text("projects: []\n", encoding="utf-8")
        monkeypatch.setattr(SerenaConfig, "_determine_config_file_path", classmethod(lambda cls: str(self.config_path)))
        # the commands must never reach a project server running on this machine
        monkeypatch.setattr(ProjectServerAdminClient, "find_running", classmethod(lambda cls, serena_config: None))

        self.main = tmp_path / "repo"
        main_data = self.main / SERENA_MANAGED_DIR_NAME
        (main_data / "cache" / "csharp").mkdir(parents=True)
        (main_data / "memories").mkdir()
        (main_data / "project.yml").write_text(MAIN_PROJECT_YML, encoding="utf-8")
        (main_data / "project.local.yml").write_text('project_name: "local"\n', encoding="utf-8")
        (main_data / "memories" / "note.md").write_text("note", encoding="utf-8")
        (main_data / "cache" / "csharp" / "raw_document_symbols.pkl").write_bytes(b"raw symbols")
        (main_data / "cache" / "csharp" / "document_symbols.pkl").write_bytes(b"symbols with absolute paths")
        self.main_yml_bytes = (main_data / "project.yml").read_bytes()
        self.worktrees = tmp_path / "wt" / "waves"
        self.worktrees.mkdir(parents=True)
        self.runner = CliRunner()

    def _make_worktree(self, name: str) -> Path:
        worktree = self.worktrees / name
        worktree.mkdir()
        (worktree / "Program.cs").write_text("class Program {}\n", encoding="utf-8")
        return worktree

    @staticmethod
    def _registered_roots() -> list[Path]:
        return [Path(p.project_root) for p in SerenaConfig.from_config_file(generate_if_missing=False).projects]

    def _prepare(self, worktree: Path, *args: str) -> Any:
        return self.runner.invoke(ProjectCommands.prepare_worktree, [str(worktree), "--from", str(self.main), "--json", *args])

    def test_preparation_copies_configuration_and_raw_caches_and_leaves_the_main_project_alone(self) -> None:
        worktree = self._make_worktree("integration")

        result = self._prepare(worktree)

        # without a project server, the project is prepared but not served
        assert result.exit_code == 3, result.output
        report = json.loads(result.output)
        assert report["name"] == "main@waves-integration"
        assert (report["copied"], report["registered"], report["project_server"]) == (True, True, "absent")
        data = worktree / SERENA_MANAGED_DIR_NAME
        project_yml = load_yaml(str(data / "project.yml"))
        assert project_yml["project_name"] == "main@waves-integration"
        assert list(project_yml["language_servers"]) == ["csharp"]
        assert list(project_yml["ignored_paths"]) == ["**/bin"]
        assert (data / "cache" / "csharp" / "raw_document_symbols.pkl").read_bytes() == b"raw symbols"
        assert not (data / "cache" / "csharp" / "document_symbols.pkl").exists()
        assert not (data / "project.local.yml").exists()
        assert not (data / "memories" / "note.md").exists()
        assert (self.main / SERENA_MANAGED_DIR_NAME / "project.yml").read_bytes() == self.main_yml_bytes
        assert self._registered_roots() == [worktree.resolve()]

    def test_repeated_preparation_keeps_the_prepared_worktree(self) -> None:
        worktree = self._make_worktree("feature-1")
        self._prepare(worktree, "--name", "repo@feature-1")
        cache = worktree / SERENA_MANAGED_DIR_NAME / "cache" / "csharp" / "raw_document_symbols.pkl"
        cache.write_bytes(b"symbols updated in the worktree")

        repeated = self._prepare(worktree)
        renamed = self._prepare(worktree, "--name", "other")

        assert json.loads(repeated.output)["name"] == "repo@feature-1"
        assert json.loads(repeated.output)["copied"] is False
        assert cache.read_bytes() == b"symbols updated in the worktree"
        assert renamed.exit_code == 1
        assert "already prepared as project 'repo@feature-1'" in renamed.output
        assert self._registered_roots() == [worktree.resolve()]

    def test_name_of_an_existing_project_conflicts_but_one_of_a_removed_worktree_is_reused(self) -> None:
        first = self._make_worktree("wave1")
        second = self._make_worktree("wave2")
        self._prepare(first, "--name", "repo@integration")

        conflict = self._prepare(second, "--name", "repo@integration")
        for path in sorted(first.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        first.rmdir()
        reuse = self._prepare(second, "--name", "repo@integration")

        assert conflict.exit_code == 1
        assert "already used" in conflict.output
        assert json.loads(reuse.output)["name"] == "repo@integration"
        assert self._registered_roots() == [second.resolve()]

    def test_release_unregisters_the_project_and_optionally_deletes_its_data(self) -> None:
        kept = self._make_worktree("kept")
        deleted = self._make_worktree("deleted")
        for worktree in (kept, deleted):
            self._prepare(worktree)

        kept_result = self.runner.invoke(ProjectCommands.release_worktree, [str(kept), "--json"])
        deleted_result = self.runner.invoke(ProjectCommands.release_worktree, [str(deleted), "--delete-data", "--json"])

        assert kept_result.exit_code == 0, kept_result.output
        assert json.loads(kept_result.output)["registered"] is False
        assert (kept / SERENA_MANAGED_DIR_NAME / "project.yml").exists()
        assert deleted_result.exit_code == 0, deleted_result.output
        assert not (deleted / SERENA_MANAGED_DIR_NAME).exists()
        assert self._registered_roots() == []

    def test_running_project_server_registers_and_warms_up_the_prepared_project(self) -> None:
        worktree = self._make_worktree("served")
        server = _RecordingProjectServer({"/register": [200], "/warmup": [202]})
        projects = WorktreeProjects(SerenaConfig.from_config_file(), server)

        report = projects.prepare(str(worktree), str(self.main))

        assert report.project_server == ProjectServerState.ANNOUNCED
        assert server.requests == [("/register", {"project": str(worktree.resolve())}), ("/warmup", {"project": str(worktree.resolve())})]

    def test_rejection_by_the_project_server_is_reported(self) -> None:
        worktree = self._make_worktree("rejected")
        server = _RecordingProjectServer({"/register": [403]})
        projects = WorktreeProjects(SerenaConfig.from_config_file(), server)

        report = projects.prepare(str(worktree), str(self.main), warm_up=False)

        assert report.project_server == ProjectServerState.REJECTED
        assert report.message == "403: answer 403"
        assert self._registered_roots() == [worktree.resolve()]

    def test_release_retries_while_the_project_server_is_busy_with_the_project(self, monkeypatch: pytest.MonkeyPatch) -> None:
        worktree = self._make_worktree("busy")
        self._prepare(worktree)
        monkeypatch.setattr(WorktreeProjects, "UNLOAD_RETRY_INTERVAL_SECONDS", 0.01)
        server = _RecordingProjectServer({"/unload": [409, 409, 200]})
        projects = WorktreeProjects(SerenaConfig.from_config_file(), server)

        report = projects.release(str(worktree))

        assert report.project_server == ProjectServerState.UNLOADED
        assert server.requests == [("/unload", {"project": str(worktree.resolve()), "forget": True})] * 3
        assert self._registered_roots() == []

    def test_release_fails_and_keeps_the_registration_if_the_project_server_stays_busy(self) -> None:
        worktree = self._make_worktree("stuck")
        self._prepare(worktree)
        projects = WorktreeProjects(SerenaConfig.from_config_file(), _RecordingProjectServer({"/unload": [409]}))

        with pytest.raises(WorktreeProjectError, match="409"):
            projects.release(str(worktree), timeout=0)

        assert self._registered_roots() == [worktree.resolve()]

    def test_unresponsive_project_server_fails_the_release_and_is_reported_by_the_preparation(self) -> None:
        released = self._make_worktree("released")
        self._prepare(released)
        unresponsive = {route: [ConnectionError("no response")] for route in ("/register", "/warmup", "/unload")}

        with pytest.raises(WorktreeProjectError, match="no response"):
            WorktreeProjects(SerenaConfig.from_config_file(), _RecordingProjectServer(unresponsive)).release(str(released))
        report = WorktreeProjects(SerenaConfig.from_config_file(), _RecordingProjectServer(unresponsive)).prepare(
            str(self._make_worktree("prepared")), str(self.main)
        )

        assert set(self._registered_roots()) == {released.resolve(), (self.worktrees / "prepared").resolve()}
        assert (report.project_server, report.message) == (ProjectServerState.REJECTED, "no response")

    def test_interrupted_preparation_is_completed_by_the_next_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        worktree = self._make_worktree("interrupted")

        def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        with monkeypatch.context() as patch:
            patch.setattr("serena.worktree_projects.save_yaml", fail)
            assert self._prepare(worktree).exit_code != 0
        completed = self._prepare(worktree, "--name", "repo@interrupted")

        assert json.loads(completed.output)["copied"] is True
        assert load_yaml(str(worktree / SERENA_MANAGED_DIR_NAME / "project.yml"))["project_name"] == "repo@interrupted"
        assert (worktree / SERENA_MANAGED_DIR_NAME / "cache" / "csharp" / "raw_document_symbols.pkl").read_bytes() == b"raw symbols"
        assert not list((worktree / SERENA_MANAGED_DIR_NAME).rglob("*.tmp"))


class TestFindRunningProjectServer:
    """
    Finding the running project server via its registry file in Serena's home directory.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.registry_path = tmp_path / "project-server.json"
        monkeypatch.setattr(ProjectServerRegistryFile, "path_for", staticmethod(lambda port, directory=None: str(self.registry_path)))
        self.serena_config = cast(SerenaConfig, SimpleNamespace(auth_secret="secret", tool_timeout=240))
        self.heartbeats: list[Any] = []

        def heartbeat(url: str, headers: dict[str, str], timeout: float) -> Any:
            self.heartbeats.append((url, headers))
            if self.heartbeat_answer is None:
                raise requests.ConnectionError("refused")
            return SimpleNamespace(raise_for_status=self.heartbeat_answer)

        monkeypatch.setattr("serena.project_server.requests_lib.get", heartbeat)
        self.heartbeat_answer: Any = lambda: None

    def _write_registry(self, content: str) -> None:
        self.registry_path.write_text(content, encoding="utf-8")

    def _raise_unauthorized(self) -> None:
        raise requests.HTTPError("401 Unauthorized")

    def test_live_server_answering_its_heartbeat_is_found(self) -> None:
        self._write_registry(json.dumps({"pid": os.getpid(), "port": 24299}))

        client = ProjectServerAdminClient.find_running(self.serena_config)

        assert client is not None
        assert self.heartbeats == [("http://127.0.0.1:24299/heartbeat", {"Authorization": "Bearer secret"})]

    @pytest.mark.parametrize(
        "case",
        ["no_registry", "broken_registry", "dead_process", "no_heartbeat", "unauthorized"],
    )
    def test_stale_or_unreachable_server_is_not_found(self, case: str) -> None:
        dead_pid = max(psutil.pids()) + 100000
        if case == "broken_registry":
            self._write_registry("{not json")
        elif case != "no_registry":
            self._write_registry(json.dumps({"pid": dead_pid if case == "dead_process" else os.getpid(), "port": 24299}))
        if case == "no_heartbeat":
            self.heartbeat_answer = None
        elif case == "unauthorized":
            self.heartbeat_answer = self._raise_unauthorized

        assert ProjectServerAdminClient.find_running(self.serena_config) is None


class TestProjectServerPort:
    def test_environment_variable_overrides_the_default_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ProjectServerPort.ENVIRONMENT_VARIABLE, raising=False)
        default = ProjectServerPort.default()
        monkeypatch.setenv(ProjectServerPort.ENVIRONMENT_VARIABLE, "24226")

        assert default == 24225
        assert ProjectServerPort.default() == 24226

    @pytest.mark.parametrize("value", ["port", "0", "70000"])
    def test_invalid_port_is_rejected(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv(ProjectServerPort.ENVIRONMENT_VARIABLE, value)

        with pytest.raises(ValueError, match=ProjectServerPort.ENVIRONMENT_VARIABLE):
            ProjectServerPort.default()
