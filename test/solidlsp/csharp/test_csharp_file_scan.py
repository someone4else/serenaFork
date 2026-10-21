"""Tests for the file scan with which the C# language server locates the solution of a repository (no language server
required).
"""

import os
from pathlib import Path

import pytest

from solidlsp.language_servers.csharp_language_server import breadth_first_file_scan, find_solution_or_project_file


def test_ignore_predicate_receives_paths_relative_to_the_root(tmp_path: Path) -> None:
    (tmp_path / "src" / "App").mkdir(parents=True)
    (tmp_path / "src" / "App" / "App.csproj").write_text("<Project />", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "Stale.csproj").write_text("<Project />", encoding="utf-8")
    seen: list[str] = []

    def is_ignored(relative_path: str) -> bool:
        seen.append(relative_path)
        return relative_path == "bin"

    files = list(breadth_first_file_scan(str(tmp_path), is_ignored))

    assert files == [str(tmp_path / "src" / "App" / "App.csproj")]
    assert sorted(seen) == sorted(["bin", "src", os.path.join("src", "App"), os.path.join("src", "App", "App.csproj")])


@pytest.mark.skipif(os.name != "nt", reason="reserved device names exist on Windows only")
def test_file_with_a_reserved_device_name_does_not_prevent_finding_the_solution(tmp_path: Path) -> None:
    # a stray file named "nul" (e.g. created by a shell redirection in another environment) resolves to the device
    # path \\.\nul, for which computing a path relative to the repository root fails
    with open("\\\\?\\" + str(tmp_path / "nul"), "w", encoding="utf-8") as f:
        f.write("stray output")
    (tmp_path / "App").mkdir()
    (tmp_path / "App" / "App.sln").write_text("", encoding="utf-8")

    assert find_solution_or_project_file(str(tmp_path)) == str(tmp_path / "App" / "App.sln")
    assert str(tmp_path / "App" / "App.sln") in list(breadth_first_file_scan(str(tmp_path), lambda _: False))
