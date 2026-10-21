"""Unit tests for symbol lookup error semantics (no language server required)."""

import logging
from collections.abc import Iterator

import pytest

from serena.agent import SerenaAgent
from serena.config.serena_config import SerenaConfig
from serena.symbol import LanguageServerSymbolRetriever, SymbolRetrievalError
from serena.tools.tools_base import Tool, ToolMarkerDoesNotRequireActiveProject


class _StubSymbol:
    def __init__(self, name_path: str) -> None:
        self._name_path = name_path

    def get_name_path(self) -> str:
        return self._name_path

    def to_dict(self, kind: bool = False, relative_path: bool = False) -> dict:
        return {"name_path": self._name_path}


def _retriever_with_find_result(result: list) -> LanguageServerSymbolRetriever:
    """Build a retriever without running __init__ and stub its `find` to a fixed result."""
    retriever = LanguageServerSymbolRetriever.__new__(LanguageServerSymbolRetriever)
    retriever.find = lambda *args, **kwargs: result
    return retriever


class TestFindUniqueErrors:
    def test_symbol_retrieval_error_is_value_error(self) -> None:
        # Backward compatibility: callers that catch ValueError keep working.
        assert issubclass(SymbolRetrievalError, ValueError)

    def test_no_match_raises_symbol_retrieval_error(self) -> None:
        retriever = _retriever_with_find_result([])
        with pytest.raises(SymbolRetrievalError, match="No symbol matching"):
            retriever.find_unique("Foo/bar", within_relative_path="some/file.py")

    def test_multiple_matches_raises_symbol_retrieval_error(self) -> None:
        # Two candidates, neither an exact name-path match for the pattern -> ambiguous.
        retriever = _retriever_with_find_result([_StubSymbol("A/Foo/bar"), _StubSymbol("B/Foo/bar")])
        with pytest.raises(SymbolRetrievalError, match="Found multiple"):
            retriever.find_unique("Foo/bar")

    def test_single_match_is_returned(self) -> None:
        only = _StubSymbol("A/Foo/bar")
        retriever = _retriever_with_find_result([only])
        assert retriever.find_unique("Foo/bar") is only


class TestToolLoggingOfLookupErrors:
    """Expected lookup failures are reported to the caller and logged concisely; genuine errors keep their traceback."""

    @pytest.fixture
    def agent(self) -> Iterator[SerenaAgent]:
        agent = SerenaAgent(serena_config=SerenaConfig(log_level=logging.INFO).with_headless_mode_overrides())
        yield agent
        agent.on_shutdown(timeout=5)

    @staticmethod
    def _failing_tool(agent: SerenaAgent, error: Exception, monkeypatch: pytest.MonkeyPatch) -> Tool:
        class FailingTool(Tool, ToolMarkerDoesNotRequireActiveProject):
            def apply(self) -> str:
                raise error

        tool = FailingTool(agent)
        monkeypatch.setattr(tool, "is_active", lambda: True)
        return tool

    def test_symbol_retrieval_error_is_logged_as_warning_without_traceback(
        self, agent: SerenaAgent, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = self._failing_tool(agent, SymbolRetrievalError("No symbol matching 'Foo' found"), monkeypatch)
        with caplog.at_level(logging.INFO, logger="serena.tools.tools_base"):
            result = tool.apply_ex(log_call=False)
        assert "No symbol matching 'Foo' found" in result
        records = [r for r in caplog.records if "No symbol matching" in r.getMessage()]
        assert [r.levelno for r in records] == [logging.WARNING]
        assert records[0].exc_info is None

    def test_other_errors_are_logged_as_errors_with_traceback(
        self, agent: SerenaAgent, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        tool = self._failing_tool(agent, ValueError("broken"), monkeypatch)
        with caplog.at_level(logging.INFO, logger="serena.tools.tools_base"):
            tool.apply_ex(log_call=False)
        records = [r for r in caplog.records if "broken" in r.getMessage()]
        assert [r.levelno for r in records] == [logging.ERROR]
        assert records[0].exc_info is not None
