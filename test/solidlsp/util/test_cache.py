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
from pathlib import Path
from typing import Any

import pytest

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
    def with_cache_dir(cls, cache_dir: Path) -> "_CacheOnlyLanguageServer":
        ls = cls.__new__(cls)
        ls.cache_dir = cache_dir
        ls._ls_specific_raw_document_symbols_cache_version = 1
        ls._raw_document_symbols_cache = {"a.py": ("hash", None)}
        ls._raw_document_symbols_cache_is_modified = True
        ls._document_symbols_cache = {}
        ls._document_symbols_cache_is_modified = False
        ls._cache_save_schedule = CacheSaveSchedule()
        return ls


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
