"""Tests for multi-language server symbol retrieval optimization."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from serena.ls_manager import LanguageServerManager
from serena.symbol import LanguageServerSymbolRetriever
from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from test.conftest import get_repo_path, start_ls_context

log = logging.getLogger(__name__)


@pytest.mark.python
@pytest.mark.typescript
class TestMultiLanguageSymbolRetrieval:
    """Test that symbol retrieval optimizes language server queries in multi-LS setups."""

    def test_find_symbol_with_file_path_queries_only_matching_ls(self):
        """
        When a specific file path is provided to find(), only the language server
        that supports that file type should be queried.
        """
        # Create language servers for Python and TypeScript
        python_repo = str(get_repo_path(Language.PYTHON))
        typescript_repo = str(get_repo_path(Language.TYPESCRIPT))

        with start_ls_context(Language.PYTHON, python_repo) as python_ls:
            with start_ls_context(Language.TYPESCRIPT, typescript_repo) as typescript_ls:
                # Create a LanguageServerManager with both servers
                ls_manager = LanguageServerManager({Language.PYTHON: python_ls, Language.TYPESCRIPT: typescript_ls})

                symbol_retriever = LanguageServerSymbolRetriever(ls_manager)

                # Mock the request_full_symbol_tree method to track calls
                python_mock = MagicMock(wraps=python_ls.request_full_symbol_tree)
                typescript_mock = MagicMock(wraps=typescript_ls.request_full_symbol_tree)

                with patch.object(python_ls, "request_full_symbol_tree", python_mock):
                    with patch.object(typescript_ls, "request_full_symbol_tree", typescript_mock):
                        # Find symbols in a Python file
                        # This should only call request_full_symbol_tree on Python LS
                        python_symbols = symbol_retriever.find("UserService", within_relative_path="test_repo/services.py")

                        # Verify Python LS was called
                        assert python_mock.call_count == 1, "Python LS should be called for .py file"
                        # Verify TypeScript LS was NOT called
                        assert typescript_mock.call_count == 0, "TypeScript LS should NOT be called for .py file"
                        # Verify that we got valid symbols from Python file
                        assert len(python_symbols) > 0, "Should find symbols in Python file"
                        assert any("UserService" in s.name for s in python_symbols), "Should find UserService symbol"

                        # Reset mocks
                        python_mock.reset_mock()
                        typescript_mock.reset_mock()

                        # Now find symbols in a TypeScript file
                        # This should only call request_full_symbol_tree on TypeScript LS
                        ts_symbols = symbol_retriever.find("helper", within_relative_path="index.ts")

                        # Verify TypeScript LS was called
                        assert typescript_mock.call_count == 1, "TypeScript LS should be called for .ts file"
                        # Verify Python LS was NOT called
                        assert python_mock.call_count == 0, "Python LS should NOT be called for .ts file"
                        # Verify that we got valid symbols from TypeScript file
                        # (may be 0 if no matching symbols, but should not error)
                        assert isinstance(ts_symbols, list), "Should return a list of symbols"

    def test_find_symbol_with_directory_path_queries_all_ls(self):
        """
        When a directory path (or no path) is provided to find(), all language servers
        should be queried since a directory may contain files for multiple languages.
        """
        # Create language servers for Python and TypeScript
        python_repo = str(get_repo_path(Language.PYTHON))
        typescript_repo = str(get_repo_path(Language.TYPESCRIPT))

        with start_ls_context(Language.PYTHON, python_repo) as python_ls:
            with start_ls_context(Language.TYPESCRIPT, typescript_repo) as typescript_ls:
                # Create a LanguageServerManager with both servers
                ls_manager = LanguageServerManager({Language.PYTHON: python_ls, Language.TYPESCRIPT: typescript_ls})

                symbol_retriever = LanguageServerSymbolRetriever(ls_manager)

                # Mock the request_full_symbol_tree method to track calls
                python_mock = MagicMock(wraps=python_ls.request_full_symbol_tree)
                typescript_mock = MagicMock(wraps=typescript_ls.request_full_symbol_tree)

                with patch.object(python_ls, "request_full_symbol_tree", python_mock):
                    with patch.object(typescript_ls, "request_full_symbol_tree", typescript_mock):
                        # Find symbols in a directory (or with no path)
                        # This should call request_full_symbol_tree on BOTH language servers
                        symbol_retriever.find("something", within_relative_path="test_repo")

                        # Verify both language servers were called
                        assert python_mock.call_count == 1, "Python LS should be called for directory search"
                        assert typescript_mock.call_count == 1, "TypeScript LS should be called for directory search"

                        # Reset mocks
                        python_mock.reset_mock()
                        typescript_mock.reset_mock()

                        # Test with no path specified
                        symbol_retriever.find("something")

                        # Verify both language servers were called again
                        assert python_mock.call_count == 1, "Python LS should be called when no path specified"
                        assert typescript_mock.call_count == 1, "TypeScript LS should be called when no path specified"

    def test_log_level_for_ignored_file(self, caplog):
        """
        Verify that when a file is ignored by a language server, the log is at DEBUG level,
        not ERROR level, which is expected behavior in multi-LS setups.
        """
        # Create language servers for Python and TypeScript
        python_repo = str(get_repo_path(Language.PYTHON))

        with start_ls_context(Language.PYTHON, python_repo) as python_ls:
            # Set log level to DEBUG to capture debug logs
            with caplog.at_level(logging.DEBUG, logger="solidlsp.ls"):
                # Try to request symbols for a TypeScript file from Python LS
                # This should log at DEBUG level, not ERROR
                result = python_ls.request_full_symbol_tree(within_relative_path="some_file.ts")

                # Verify the result is empty (as expected)
                assert result == [], "Should return empty list for unsupported file type"

                # Verify the log was at DEBUG level
                debug_logs = [record for record in caplog.records if record.levelno == logging.DEBUG]
                error_logs = [record for record in caplog.records if record.levelno == logging.ERROR]

                # There should be a debug log about the ignored file
                ignored_logs = [log for log in debug_logs if "ignored" in log.message.lower()]
                assert len(ignored_logs) > 0, "Should have debug log about ignored file"

                # There should be NO error logs about the ignored file
                ignored_error_logs = [log for log in error_logs if "ignored" in log.message.lower()]
                assert len(ignored_error_logs) == 0, "Should NOT have error log about ignored file"
