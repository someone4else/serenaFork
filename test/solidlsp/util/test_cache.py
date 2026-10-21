"""Tests for saving symbol caches that several processes (e.g. a long-running project server and interactive sessions)
share: saves are atomic, a cache file held open by a reader does not block the writer indefinitely, and leftovers of
killed writers are cleaned up.
"""

import glob
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from serena.ls_manager import LanguageServerManager
from solidlsp.ls import SolidLanguageServer
from solidlsp.util.cache import CacheFileBusyError, CacheSaveSchedule, load_cache, save_cache

VERSION = ("v", 1)


def _temp_files(cache_path: Path) -> list[str]:
    return glob.glob(str(cache_path) + ".*.tmp")


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_concurrent_saves_leave_a_complete_cache_and_no_temporary_files(tmp_path: Path) -> None:
    cache_path = tmp_path / "document_symbols.pkl"
    payloads = [{"writer": i, "data": "x" * 200_000} for i in range(6)]

    def save_repeatedly(payload: dict[str, Any]) -> None:
        for _ in range(5):
            try:
                save_cache(str(cache_path), VERSION, payload, replace_timeout=5.0)
            except CacheFileBusyError:
                pass  # another writer's replacement was in progress; the cache file then holds that writer's data

    threads = [threading.Thread(target=save_repeatedly, args=(p,)) for p in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert load_cache(str(cache_path), VERSION) in payloads
    assert _temp_files(cache_path) == []


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace a file that is open in another process")
def test_save_into_a_file_held_open_by_a_reader_fails_in_time_and_leaves_the_cache_intact(tmp_path: Path) -> None:
    cache_path = tmp_path / "document_symbols.pkl"
    save_cache(str(cache_path), VERSION, "old")

    with open(cache_path, "rb"):
        start = time.monotonic()
        with pytest.raises(CacheFileBusyError):
            save_cache(str(cache_path), VERSION, "new", replace_timeout=0.3)
        assert time.monotonic() - start < 3

    assert load_cache(str(cache_path), VERSION) == "old"
    assert _temp_files(cache_path) == []
    save_cache(str(cache_path), VERSION, "new")
    assert load_cache(str(cache_path), VERSION) == "new"


def test_temporary_files_of_finished_writers_are_removed_and_those_of_running_ones_kept(tmp_path: Path) -> None:
    cache_path = tmp_path / "document_symbols.pkl"
    abandoned = tmp_path / f"document_symbols.pkl.{_dead_pid()}.abc.tmp"
    in_progress = tmp_path / f"document_symbols.pkl.{os.getppid()}.def.tmp"
    abandoned.write_bytes(b"partial")
    in_progress.write_bytes(b"partial")

    save_cache(str(cache_path), VERSION, "data")

    assert not abandoned.exists()
    assert in_progress.exists()


def test_cache_save_schedule_pauses_regular_saves_after_a_busy_cache_file() -> None:
    now = [100.0]
    schedule = CacheSaveSchedule(pause_seconds=60, clock=lambda: now[0])
    assert schedule.is_due()
    schedule.pause()
    now[0] += 59
    assert not schedule.is_due()
    now[0] += 1
    assert schedule.is_due()


class _CacheOnlyLanguageServer(SolidLanguageServer):
    """A language server whose symbol caches are populated by the test instead of a server process."""

    def _create_base_initialize_params(self) -> Any:
        raise NotImplementedError

    def _start_server(self) -> None:
        raise NotImplementedError

    @classmethod
    def with_cache_dir(cls, cache_dir: Path, clock: Callable[[], float] = time.monotonic) -> "_CacheOnlyLanguageServer":
        ls = cls.__new__(cls)
        ls.cache_dir = cache_dir
        ls._ls_specific_raw_document_symbols_cache_version = 1
        ls._raw_document_symbols_cache = {"a.py": ("hash", None)}
        ls._raw_document_symbols_cache_is_modified = True
        ls._document_symbols_cache = {}
        ls._document_symbols_cache_is_modified = False
        ls._cache_save_schedule = CacheSaveSchedule(clock=clock)
        return ls


class _BusyCacheFiles:
    """Makes replacing cache files fail as if another process held them open (on any OS), a given number of times."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, failures: int) -> None:
        self.remaining_failures = failures
        self.attempts = 0
        real_replace = os.replace

        def replace(source: str, target: str) -> None:
            self.attempts += 1
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise PermissionError(13, "The process cannot access the file because it is being used by another process")
            real_replace(source, target)

        monkeypatch.setattr("solidlsp.util.cache.os.replace", replace)


def test_busy_cache_file_fails_the_save_within_the_timeout_without_leftovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_path = tmp_path / "document_symbols.pkl"
    save_cache(str(cache_path), VERSION, "old")
    _BusyCacheFiles(monkeypatch, failures=1000)

    start = time.monotonic()
    with pytest.raises(CacheFileBusyError):
        save_cache(str(cache_path), VERSION, "new", replace_timeout=0.3)

    assert time.monotonic() - start < 3
    assert load_cache(str(cache_path), VERSION) == "old"
    assert _temp_files(cache_path) == []


def test_language_server_resumes_regular_saves_after_the_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    ls = _CacheOnlyLanguageServer.with_cache_dir(tmp_path, clock=lambda: now[0])
    cache_path = tmp_path / SolidLanguageServer.RAW_DOCUMENT_SYMBOL_CACHE_FILENAME
    busy = _BusyCacheFiles(monkeypatch, failures=1000)

    ls.save_cache()
    attempts_after_busy_save = busy.attempts
    ls.save_cache()
    assert busy.attempts == attempts_after_busy_save  # paused: no new attempt (and no new dump)

    busy.remaining_failures = 0
    now[0] += 61
    ls.save_cache()
    assert load_cache(str(cache_path), ls._raw_document_symbols_cache_version()) == {"a.py": ("hash", None)}
    assert not ls._raw_document_symbols_cache_is_modified


def test_final_save_waits_longer_for_a_busy_cache_file_than_a_regular_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # replacing fails for about 1.5 s (30 attempts at the retry interval of 50 ms)
    cache_path = tmp_path / SolidLanguageServer.RAW_DOCUMENT_SYMBOL_CACHE_FILENAME
    ls = _CacheOnlyLanguageServer.with_cache_dir(tmp_path)
    _BusyCacheFiles(monkeypatch, failures=30)
    ls.save_cache()
    assert ls._raw_document_symbols_cache_is_modified

    final_ls = _CacheOnlyLanguageServer.with_cache_dir(tmp_path)
    _BusyCacheFiles(monkeypatch, failures=30)
    final_ls.save_cache(final=True)
    assert not final_ls._raw_document_symbols_cache_is_modified
    assert load_cache(str(cache_path), final_ls._raw_document_symbols_cache_version()) == {"a.py": ("hash", None)}


def test_stopping_a_language_server_saves_its_cache_as_the_final_save() -> None:
    ls = MagicMock()
    ls.is_running.return_value = True
    LanguageServerManager._stop_language_server(ls, save_cache=True)
    ls.save_cache.assert_called_once_with(final=True)


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace a file that is open in another process")
def test_language_server_retries_a_busy_cache_only_after_the_pause_or_when_stopping(tmp_path: Path) -> None:
    ls = _CacheOnlyLanguageServer.with_cache_dir(tmp_path)
    cache_path = tmp_path / SolidLanguageServer.RAW_DOCUMENT_SYMBOL_CACHE_FILENAME
    save_cache(str(cache_path), "outdated", {})

    with open(cache_path, "rb"):
        ls.save_cache()
        modified_after_busy_save = ls._raw_document_symbols_cache_is_modified
        written_at = cache_path.stat().st_mtime_ns

        # a regular save during the pause does not write the cache again
        ls.save_cache()
        assert cache_path.stat().st_mtime_ns == written_at
        assert _temp_files(cache_path) == []

    assert modified_after_busy_save
    # the final save before stopping is attempted despite the pause
    ls.save_cache(final=True)
    assert load_cache(str(cache_path), ls._raw_document_symbols_cache_version()) == {"a.py": ("hash", None)}
