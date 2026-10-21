# SPDX-License-Identifier: MIT

import glob
import logging
import os
import pickle
import tempfile
import time
from collections.abc import Callable
from typing import Any, Optional

import psutil
from sensai.util.pickle import load_pickle

from solidlsp.util.pickle_util import SafePickleLoader

log = logging.getLogger(__name__)


class CacheFileBusyError(OSError):
    """
    Raised when a cache file could not be replaced within the allowed time because another process holds it open (on
    Windows, a file that is open in another process cannot be replaced); the cache file is left unchanged.
    """


def load_cache(path: str, version: Any, loader: SafePickleLoader | None = None) -> Optional[Any]:
    if loader is None:
        data = load_pickle(path)
    else:
        data = loader.load(path)
    if not isinstance(data, dict) or "__cache_version" not in data:
        log.info("Cache is outdated (expected version %s). Ignoring cache at %s", version, path)
        return None
    saved_version = data["__cache_version"]
    if saved_version != version:
        log.info("Cache is outdated (expected version %s, got %s). Ignoring cache at %s", version, saved_version, path)
        return None
    return data["obj"]


def save_cache(path: str, version: Any, obj: Any, replace_timeout: float = 1.0) -> None:
    """
    Saves an object as a versioned cache file atomically: the data is written to a temporary file in the same directory,
    which then replaces the cache file. Readers therefore never see a partially written file, and processes saving the
    same cache concurrently (e.g. several Serena instances working on the same project) never interleave their data;
    the last one to replace the file wins.

    :param path: the path of the cache file
    :param version: the cache version, which `load_cache` compares against
    :param obj: the object to save
    :param replace_timeout: the time in seconds for which replacing the cache file is retried while another process
        holds it open
    :raises CacheFileBusyError: if the cache file could not be replaced in time
    """
    _remove_abandoned_temp_files(path)

    fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=f"{os.path.basename(path)}.{os.getpid()}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"__cache_version": version, "obj": obj}, f, protocol=pickle.HIGHEST_PROTOCOL)
        _replace_with_retry(temp_path, path, replace_timeout)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _replace_with_retry(source: str, target: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError as e:
            if time.monotonic() >= deadline:
                raise CacheFileBusyError(f"Cache file {target} is in use by another process") from e
            time.sleep(0.05)


def _remove_abandoned_temp_files(path: str) -> None:
    """
    Removes the temporary files that processes which no longer exist (e.g. killed while saving) left behind for the
    given cache file; their names contain the ID of the writing process.
    """
    prefix = os.path.basename(path) + "."
    for temp_path in glob.glob(glob.escape(path) + ".*.tmp"):
        pid_str = os.path.basename(temp_path)[len(prefix) :].split(".", 1)[0]
        if not pid_str.isdigit() or int(pid_str) == os.getpid() or psutil.pid_exists(int(pid_str)):
            continue
        try:
            os.remove(temp_path)
            log.info("Removed abandoned temporary cache file %s", temp_path)
        except OSError:
            pass


class CacheSaveSchedule:
    """
    Decides whether saving a cache shall be attempted: after the cache file was found busy, regular saves are paused for
    a while, as the process holding the file open (typically while loading a large cache) would make every attempt write
    the entire cache in vain.
    """

    def __init__(self, pause_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        """
        :param pause_seconds: the duration of the pause after a save failed because the cache file was busy
        :param clock: the monotonic clock (in seconds)
        """
        self._pause_seconds = pause_seconds
        self._clock = clock
        self._paused_until = float("-inf")

    def is_due(self) -> bool:
        """
        :return: whether a regular save shall be attempted
        """
        return self._clock() >= self._paused_until

    def pause(self) -> None:
        """
        Pauses regular saves after a save failed because the cache file was busy.
        """
        self._paused_until = self._clock() + self._pause_seconds
