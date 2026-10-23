# SPDX-License-Identifier: MIT

"""Unit tests: stopping a language server is bounded by the stop timeout, not by the request timeout.

The shutdown handshake awaits the reply to the shutdown request. That reply is awaited for no longer than the
timeout passed to ``stop``, which is a stop budget of a few seconds, not the request timeout configured for
ordinary requests (minutes). An unresponsive server therefore neither delays the stop up to the request timeout
nor leaves a background thread that fails later with an uncaught exception, whose traceback ends up on stderr.

No language markers: the server speaks no LSP and runs as a plain child process.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_process import DEFAULT_LS_REQUEST_TIMEOUT, StdioLanguageServer
from solidlsp.util.subprocess_util import ManagedSubprocess

STOP_TIMEOUT: float = 0.3
"""the stop timeout used here; it is far below the request timeout of ordinary requests"""

REQUEST_TIMEOUT: float = 1.0
"""a request timeout standing in for the minutes-long one of a production installation"""

OBSERVATION_SECONDS: float = 1.0
"""the time after a stop in which a shutdown sequence that still awaits the reply would fail"""


@dataclass
class _UnresponsiveServer:
    """A stdio language server whose process never answers, so the reply to the shutdown request never arrives."""

    server: StdioLanguageServer
    process: subprocess.Popen[bytes]


def _start_unresponsive_server() -> _UnresponsiveServer:
    """Starts a language server over a child process that neither reads nor writes anything."""
    server = StdioLanguageServer(
        process_launch_info=MagicMock(),
        ls_id=LanguageServerId.PYTHON,
        determine_log_level=lambda _line: logging.INFO,
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    server._process = ManagedSubprocess(process, name="Test ls", start_new_session=False)
    return _UnresponsiveServer(server=server, process=process)


@pytest.fixture()
def unresponsive_server() -> _UnresponsiveServer:
    """An unresponsive language server, stopped by the test and killed afterwards if it is still alive."""
    started = _start_unresponsive_server()
    yield started
    if started.process.poll() is None:
        started.process.kill()
        started.process.wait(timeout=10)


def test_stop_is_bounded_by_the_stop_timeout(unresponsive_server: _UnresponsiveServer) -> None:
    """An unanswered shutdown request is given up on within the stop timeout, and the process is terminated."""
    server = unresponsive_server.server
    server.set_request_timeout(DEFAULT_LS_REQUEST_TIMEOUT)

    started_at = time.monotonic()
    server.stop(timeout=STOP_TIMEOUT)
    elapsed = time.monotonic() - started_at

    # the stop timeout leaves room for terminating the process, whereas the request timeout is two orders of magnitude larger
    assert elapsed < 1.5, (
        f"stop took {elapsed:.2f}s; the reply to the shutdown request must be awaited for no longer than the {STOP_TIMEOUT}s stop timeout"
    )
    assert unresponsive_server.process.poll() is not None, "the language server process must be terminated"


def test_stop_leaves_no_uncaught_exception_in_a_background_thread(
    unresponsive_server: _UnresponsiveServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unanswered shutdown request fails within the stop rather than later, as a traceback on stderr."""
    uncaught: list[str] = []

    def record_uncaught(args: threading.ExceptHookArgs) -> None:
        thread_name = args.thread.name if args.thread is not None else "<unknown>"
        uncaught.append(f"{thread_name}: {args.exc_type.__name__}: {args.exc_value}")

    monkeypatch.setattr(threading, "excepthook", record_uncaught)
    server = unresponsive_server.server

    # the request timeout must be short enough for a shutdown sequence that awaits it to fail within the test
    server.set_request_timeout(REQUEST_TIMEOUT)

    server.stop(timeout=STOP_TIMEOUT)

    # a shutdown sequence that still awaits the reply outlives the stop; give it the time in which it would fail
    time.sleep(OBSERVATION_SECONDS)
    assert uncaught == [], f"uncaught exception(s) in background threads: {uncaught}"
