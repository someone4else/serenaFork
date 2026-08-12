"""Unit tests for symbol lookup error semantics (no language server required)."""

import pytest

from serena.symbol import LanguageServerSymbolRetriever, SymbolRetrievalError


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
    retriever.find = lambda *args, **kwargs: result  # type: ignore[method-assign]
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
