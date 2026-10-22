"""
Tests for the fork's writing of edited files: preservation of line endings (`SerenaConfig.preserve_line_endings`) and
the confinement of a process's writes to writable roots.
"""

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from serena.code_editor import EditedFileContext, LanguageServerCodeEditor
from serena.config.serena_config import LineEnding, SerenaConfig
from serena.edited_file_writer import EditedFileWriter, WritableRoots, WriteConfinement, WriteOutsideWritableRootsError
from serena.language_backend import BuiltinLanguageBackend
from serena.project import Project
from serena.repl.api.edit_api import EditApi
from serena.repl.api.fs_api import FsApi
from serena.symbol import LanguageServerSymbolRetriever
from solidlsp.ls_config import LanguageServerId
from test.conftest import get_repo_path


def _write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


class TestEditedFileWriter:
    def test_existing_crlf_file_keeps_crlf(self, tmp_path: Path) -> None:
        path = _write_bytes(tmp_path / "a.cs", b"class A\r\n{\r\n}\r\n")

        EditedFileWriter("utf-8", "\n", preserve_line_endings=True).write(path, "class A\n{\n    int x;\n}\n")

        assert Path(path).read_bytes() == b"class A\r\n{\r\n    int x;\r\n}\r\n"

    def test_existing_lf_file_keeps_lf(self, tmp_path: Path) -> None:
        path = _write_bytes(tmp_path / "a.ts", b"a\nb\n")

        EditedFileWriter("utf-8", "\r\n", preserve_line_endings=True).write(path, "a\nc\nb\n")

        assert Path(path).read_bytes() == b"a\nc\nb\n"

    @pytest.mark.parametrize(
        ("original", "expected"),
        [
            pytest.param(b"a\r\nb\r\nc\nd", b"x\r\ny\r\n", id="crlf_prevails"),
            pytest.param(b"a\r\nb\nc\nd", b"x\ny\n", id="lf_prevails"),
            pytest.param(b"a\r\nb\nc", b"x\r\ny\r\n", id="tie_is_crlf"),
            pytest.param(b"\xff\xfe invalid\r\nbytes\r\nc\n", b"x\r\ny\r\n", id="invalid_bytes"),
        ],
    )
    def test_existing_file_gets_the_prevailing_line_ending(self, tmp_path: Path, original: bytes, expected: bytes) -> None:
        path = _write_bytes(tmp_path / "mixed.txt", original)

        EditedFileWriter("utf-8", "\n", preserve_line_endings=True).write(path, "x\ny\n")

        assert Path(path).read_bytes() == expected

    @pytest.mark.parametrize(
        "original",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"single line", id="single_line"),
            pytest.param(b"old\rmac\r", id="lone_cr"),
        ],
    )
    def test_existing_file_without_line_breaks_gets_the_configured_line_ending(self, tmp_path: Path, original: bytes) -> None:
        path = _write_bytes(tmp_path / "no_breaks.txt", original)

        EditedFileWriter("utf-8", "\r\n", preserve_line_endings=True).write(path, "first\nsecond\n")

        assert Path(path).read_bytes() == b"first\r\nsecond\r\n"

    def test_new_file_gets_the_configured_line_ending(self, tmp_path: Path) -> None:
        path = str(tmp_path / "new.txt")

        EditedFileWriter("utf-8", "\r\n", preserve_line_endings=True).write(path, "first\nsecond\n")

        assert Path(path).read_bytes() == b"first\r\nsecond\r\n"

    def test_crlf_in_the_contents_is_not_doubled(self, tmp_path: Path) -> None:
        path = _write_bytes(tmp_path / "a.cs", b"a\r\nb\r\n")

        EditedFileWriter("utf-8", "\n", preserve_line_endings=True).write(path, "a\nnew\r\nlines\r\nb\n")

        assert Path(path).read_bytes() == b"a\r\nnew\r\nlines\r\nb\r\n"

    def test_legacy_encoding_is_kept(self, tmp_path: Path) -> None:
        word = b"\xf1\xeb\xee\xe2\xee".decode("cp1251")
        path = _write_bytes(tmp_path / "cp1251.txt", f"{word}\r\nend\r\n".encode("cp1251"))

        EditedFileWriter("cp1251", "\n", preserve_line_endings=True).write(path, f"{word}\nnew\nend\n")

        assert Path(path).read_bytes() == f"{word}\r\nnew\r\nend\r\n".encode("cp1251")

    def test_file_with_byte_order_mark_keeps_its_line_endings(self, tmp_path: Path) -> None:
        bom = b"\xef\xbb\xbf".decode("utf-8")
        path = _write_bytes(tmp_path / "bom.cs", b"\xef\xbb\xbfclass A\n{\n}\n")

        EditedFileWriter("utf-8", "\r\n", preserve_line_endings=True).write(path, f"{bom}class B\n{{\n}}\n")

        assert Path(path).read_bytes() == b"\xef\xbb\xbfclass B\n{\n}\n"

    def test_without_preservation_the_configured_line_ending_applies(self, tmp_path: Path) -> None:
        path = _write_bytes(tmp_path / "a.cs", b"a\r\nb\r\n")

        EditedFileWriter("utf-8", "\n", preserve_line_endings=False).write(path, "a\nc\nb\n")

        assert Path(path).read_bytes() == b"a\nc\nb\n"


class TestWriteConfinement:
    @pytest.fixture
    def roots(self, tmp_path: Path) -> Iterator[Path]:
        """
        A writable root containing a worktree, beside a main directory outside of it, the process being confined to
        the writable root for the duration of the test.
        """
        (tmp_path / "worktrees" / "wt1").mkdir(parents=True)
        (tmp_path / "main").mkdir()
        (tmp_path / "main" / "File.cs").write_bytes(b"main\n")
        WriteConfinement.confine(WritableRoots([str(tmp_path / "worktrees")]))
        try:
            yield tmp_path
        finally:
            WriteConfinement.confine(None)

    def test_confined_writes_inside_the_roots_succeed_and_outside_fail(self, roots: Path) -> None:
        writer = EditedFileWriter("utf-8", "\n", preserve_line_endings=False)

        writer.write(str(roots / "worktrees" / "wt1" / "File.cs"), "worktree\n")
        for outside in (roots / "main" / "File.cs", roots / "worktrees" / "wt1" / ".." / ".." / "main" / "File.cs"):
            with pytest.raises(WriteOutsideWritableRootsError):
                writer.write(str(outside), "overwritten\n")

        assert (roots / "worktrees" / "wt1" / "File.cs").read_bytes() == b"worktree\n"
        assert (roots / "main" / "File.cs").read_bytes() == b"main\n"

    @pytest.mark.skipif(os.name != "nt", reason="directory junctions exist on Windows only")
    def test_confined_write_through_a_junction_to_outside_fails(self, roots: Path) -> None:
        junction = roots / "worktrees" / "wt1" / "linked"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(roots / "main")], check=True, capture_output=True)

        with pytest.raises(WriteOutsideWritableRootsError):
            EditedFileWriter("utf-8", "\n", preserve_line_endings=False).write(str(junction / "File.cs"), "overwritten\n")

        assert (roots / "main" / "File.cs").read_bytes() == b"main\n"

    def test_confined_writes_through_symbolic_links_to_outside_fail(self, roots: Path) -> None:
        linked_dir = roots / "worktrees" / "wt1" / "linked_dir"
        linked_file = roots / "worktrees" / "wt1" / "Linked.cs"
        try:
            os.symlink(roots / "main", linked_dir, target_is_directory=True)
            os.symlink(roots / "main" / "File.cs", linked_file)
        except OSError:
            pytest.skip("creating symbolic links is not permitted")
        writer = EditedFileWriter("utf-8", "\n", preserve_line_endings=False)

        for outside in (linked_dir / "File.cs", linked_dir / "New.cs", linked_file):
            with pytest.raises(WriteOutsideWritableRootsError):
                writer.write(str(outside), "overwritten\n")

        assert (roots / "main" / "File.cs").read_bytes() == b"main\n"
        assert not (roots / "main" / "New.cs").exists()

    def test_confined_file_rename_to_outside_fails(self, roots: Path) -> None:
        (roots / "worktrees" / "wt1" / "Old.cs").write_bytes(b"old\n")
        editor = SimpleNamespace(project_root=str(roots / "worktrees" / "wt1"), _relative_path_from_uri=lambda uri: uri)
        rename = LanguageServerCodeEditor.EditOperationRenameFile(cast(Any, editor), "Old.cs", os.path.join("..", "..", "main", "New.cs"))

        with pytest.raises(WriteOutsideWritableRootsError):
            rename.apply()

        assert (roots / "worktrees" / "wt1" / "Old.cs").exists()
        assert not (roots / "main" / "New.cs").exists()

    def test_unconfined_process_writes_anywhere(self, tmp_path: Path) -> None:
        path = tmp_path / "anywhere.txt"

        EditedFileWriter("utf-8", "\n", preserve_line_endings=False).write(str(path), "written\n")

        assert path.read_bytes() == b"written\n"


class TestCreateTextFile:
    @pytest.mark.parametrize("preserve_line_endings", [True, False])
    def test_line_endings_of_new_and_overwritten_files(self, tmp_path: Path, preserve_line_endings: bool) -> None:
        (tmp_path / "existing.txt").write_bytes(b"old\r\ncontent\r\n")
        serena_config = SerenaConfig(
            gui_log_window=False, web_dashboard=False, line_ending=LineEnding.LF, preserve_line_endings=preserve_line_endings
        )
        agent = MagicMock()
        agent.get_active_project_or_raise.return_value = Project.load(str(tmp_path), serena_config=serena_config)
        api = FsApi(agent)

        api.create_text_file("new.txt", "a\nb\n")
        api.create_text_file("existing.txt", "new\ncontent\n")

        assert (tmp_path / "new.txt").read_bytes() == b"a\nb\n"
        expected_overwritten = b"new\r\ncontent\r\n" if preserve_line_endings else b"new\ncontent\n"
        assert (tmp_path / "existing.txt").read_bytes() == expected_overwritten


@pytest.mark.python
class TestLanguageServerEdits:
    """
    Edits through the language server of a project whose line ending convention is LF, in a copy of the Python test
    repository in which `variables.py` and `models.py` use CRLF and `services.py` uses LF.
    """

    VARIABLES_PATH = str(Path("test_repo") / "variables.py")
    MODELS_PATH = str(Path("test_repo") / "models.py")
    SERVICES_PATH = str(Path("test_repo") / "services.py")

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        """
        The copy of the Python test repository with the line endings described in the class docstring.
        """
        repo = tmp_path / "repo"
        shutil.copytree(get_repo_path(LanguageServerId.PYTHON), repo)
        for rel_path in (self.VARIABLES_PATH, self.MODELS_PATH, self.SERVICES_PATH):
            file_path = repo / rel_path
            lf_data = file_path.read_bytes().replace(b"\r\n", b"\n")
            file_path.write_bytes(lf_data if rel_path == self.SERVICES_PATH else lf_data.replace(b"\n", b"\r\n"))
        assert b"User" in (repo / self.SERVICES_PATH).read_bytes()
        return repo

    @staticmethod
    @contextmanager
    def _project(repo: Path, preserve_line_endings: bool | None) -> Iterator[Project]:
        """
        Provides the project of the repository copy with started language servers.

        :param repo: the repository copy
        :param preserve_line_endings: the global setting; None for its default
        """
        settings = {} if preserve_line_endings is None else {"preserve_line_endings": preserve_line_endings}
        serena_config = SerenaConfig(line_ending=LineEnding.LF, **settings).with_headless_mode_overrides()
        project = Project.load(str(repo), serena_config=serena_config)
        project.create_language_server_manager()
        try:
            yield project
        finally:
            project.shutdown(timeout=5)

    @staticmethod
    def _assert_line_endings(path: Path, crlf: bool) -> None:
        data = path.read_bytes()
        assert data.count(b"\n") > 0
        assert data.count(b"\r\n") == (data.count(b"\n") if crlf else 0)
        assert b"\r\r\n" not in data

    def test_symbol_and_content_edits_keep_crlf(self, repo: Path) -> None:
        with self._project(repo, preserve_line_endings=True) as project:
            editor = LanguageServerCodeEditor(LanguageServerSymbolRetriever(project))

            editor.replace_body(
                "VariableContainer/modify_instance_var", self.VARIABLES_PATH, "def modify_instance_var(self):\n        pass"
            )
            self._assert_line_endings(repo / self.VARIABLES_PATH, crlf=True)

            with EditedFileContext(self.VARIABLES_PATH, editor) as context:
                context.set_updated_content(context.get_original_content().replace("pass", "return None"))
            self._assert_line_endings(repo / self.VARIABLES_PATH, crlf=True)
            assert b"return None" in (repo / self.VARIABLES_PATH).read_bytes()

    def test_multi_file_edits_keep_the_line_endings_of_each_file(self, repo: Path) -> None:
        with self._project(repo, preserve_line_endings=True) as project:
            LanguageServerCodeEditor(LanguageServerSymbolRetriever(project)).rename_symbol("User", self.MODELS_PATH, "RenamedUser")

            agent = MagicMock()
            agent.get_active_project_or_raise.return_value = project
            agent.get_language_backend.return_value = BuiltinLanguageBackend.LSP.get_instance()
            EditApi(agent).replace_in_files("RenamedUser", "Member", "literal", relative_path="test_repo")

        for rel_path, crlf in ((self.MODELS_PATH, True), (self.SERVICES_PATH, False)):
            data = (repo / rel_path).read_bytes()
            assert b"Member" in data
            assert b"RenamedUser" not in data
            self._assert_line_endings(repo / rel_path, crlf=crlf)

    def test_by_default_edits_apply_the_configured_line_ending(self, repo: Path) -> None:
        with self._project(repo, preserve_line_endings=None) as project:
            editor = LanguageServerCodeEditor(LanguageServerSymbolRetriever(project))

            editor.replace_body(
                "VariableContainer/modify_instance_var", self.VARIABLES_PATH, "def modify_instance_var(self):\n        pass"
            )

        self._assert_line_endings(repo / self.VARIABLES_PATH, crlf=False)
