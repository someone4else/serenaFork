"""The retention of the date-stamped log directories in Serena's user home."""

from datetime import date, timedelta

import pytest

from serena.log_retention import LogRetention

TODAY = date(2026, 10, 23)


def _create_day_directories(logs_root, days: int) -> list[str]:
    names = []
    for days_ago in range(days):
        name = (TODAY - timedelta(days=days_ago)).isoformat()
        (logs_root / name).mkdir()
        (logs_root / name / "session_1.txt").write_text("log", encoding="utf-8")
        names.append(name)
    return names


def test_directories_of_days_beyond_the_retention_are_removed(tmp_path) -> None:
    names = _create_day_directories(tmp_path, days=20)

    removed = LogRetention(days=14).prune(str(tmp_path), today=TODAY)

    # the retention period includes the day it starts at, so the directories of the last 15 days remain
    assert removed == sorted(names[15:])
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(names[:15])


def test_entries_which_the_log_file_creation_did_not_produce_are_kept(tmp_path) -> None:
    (tmp_path / "2026-10-23").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "2026-1-5").mkdir()
    (tmp_path / "2026-10-01.txt").write_text("a log file directly in the root", encoding="utf-8")

    removed = LogRetention(days=1).prune(str(tmp_path), today=TODAY)

    assert removed == []
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(["2026-10-23", "notes", "2026-1-5", "2026-10-01.txt"])


def test_a_retention_period_below_one_day_is_rejected() -> None:
    with pytest.raises(ValueError):
        LogRetention(days=0)


def test_a_missing_log_directory_is_tolerated(tmp_path) -> None:
    missing_root = tmp_path / "absent"

    assert LogRetention(days=14).prune(str(missing_root), today=TODAY) == []
