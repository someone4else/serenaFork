"""
Tests for the session side of editing other projects via the project server: the `edit_project` tool and the
`worktree-projects` mode.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from serena.agent import ToolSet
from serena.config.context_mode import SerenaAgentMode
from serena.project_server_ext import ProjectEditError
from serena.tools import EditProjectTool


class _RecordingEditClient:
    """A stand-in for the project server's edit client, recording the edits."""

    edits: list[tuple[str, str, str]] = []
    error: Exception | None = None

    def __init__(self, serena_config: Any) -> None:
        pass

    def edit_project(self, project_name: str, tool_name: str, tool_params_json: str) -> str:
        if self.error is not None:
            raise self.error
        self.edits.append((project_name, tool_name, tool_params_json))
        return "edited"


@pytest.fixture
def edit_client(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingEditClient]:
    monkeypatch.setattr(_RecordingEditClient, "edits", [])
    monkeypatch.setattr(_RecordingEditClient, "error", None)
    monkeypatch.setattr("serena.project_server_ext.ProjectEditClient", _RecordingEditClient)
    return _RecordingEditClient


def _tool(active_tool_names: set[str], lsp: bool = True) -> EditProjectTool:
    agent = MagicMock()
    agent.get_language_backend.return_value.is_lsp.return_value = lsp
    agent.get_active_tools.return_value.contains_tool_name.side_effect = lambda name: name in active_tool_names
    return EditProjectTool(agent)


class TestEditProjectTool:
    def test_edit_is_delegated_to_the_project_server(self, edit_client: type[_RecordingEditClient]) -> None:
        result = _tool({"replace_symbol_body"}).apply("C:/wt/wt1", "replace_symbol_body", '{"name_path": "A"}')

        assert result == "edited"
        assert edit_client.edits == [("C:/wt/wt1", "replace_symbol_body", '{"name_path": "A"}')]

    def test_edit_the_session_may_not_perform_itself_is_refused(self, edit_client: type[_RecordingEditClient]) -> None:
        with pytest.raises(ValueError, match="not active in this session"):
            _tool({"find_symbol"}).apply("C:/wt/wt1", "replace_symbol_body", "{}")

        assert edit_client.edits == []

    def test_edit_requires_the_language_server_backend(self, edit_client: type[_RecordingEditClient]) -> None:
        with pytest.raises(ValueError, match="language server backend"):
            _tool({"replace_symbol_body"}, lsp=False).apply("C:/wt/wt1", "replace_symbol_body", "{}")

    def test_rejection_by_the_project_server_reaches_the_agent(self, edit_client: type[_RecordingEditClient]) -> None:
        edit_client.error = ProjectEditError("The project server did not apply the edit (403): outside the writable roots")

        with pytest.raises(ProjectEditError, match="403"):
            _tool({"replace_symbol_body"}).apply("C:/main", "replace_symbol_body", "{}")


class TestWorktreeProjectsMode:
    def test_mode_enables_reading_and_editing_other_projects(self) -> None:
        tool_set = ToolSet.default().apply(SerenaAgentMode.load("worktree-projects"))

        for tool_name in ("list_queryable_projects", "query_project", "edit_project"):
            assert tool_set.includes_name(tool_name)

    def test_sessions_without_the_mode_cannot_edit_other_projects(self) -> None:
        assert not ToolSet.default().includes_name("edit_project")
        assert not ToolSet.default().apply(SerenaAgentMode.load("query-projects")).includes_name("edit_project")
