import os
import shutil
import tempfile
from pathlib import Path

from serena.config.serena_config import ProjectConfig, SerenaConfig
from serena.project import Project
from solidlsp.ls_config import LanguageServerId


class TestConfigIgnoresPrecedeGitignoreNegation:
    """Configured ignored_paths must take precedence over .gitignore negations.

    A .gitignore may re-include a whole directory with a negation (e.g. "!kept/"), which under
    a naive flat pathspec would also re-include build outputs inside it (e.g. "kept/bin/"),
    defeating a configured "**/bin/**" ignore. Configured ignores are applied last so they win.
    """

    def setup_method(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)
        # A .gitignore that ignores bin/ everywhere but re-includes the whole "kept" tree.
        (self.root / ".gitignore").write_text("[Bb]in/\n!kept/\n", encoding="utf-8")
        os.makedirs(self.root / "kept" / "bin", exist_ok=True)
        (self.root / "kept" / "src.py").write_text("x = 1\n", encoding="utf-8")
        (self.root / "kept" / "bin" / "artifact.bin").write_bytes(b"\x00\x01\x02")

    def teardown_method(self) -> None:
        shutil.rmtree(self.test_dir)

    def _project(self) -> Project:
        config = ProjectConfig(
            project_name="test_project",
            language_servers=[LanguageServerId.PYTHON],
            ignored_paths=["**/bin", "**/bin/**"],
            ignore_all_files_in_gitignore=True,
        )
        serena_config = SerenaConfig().with_headless_mode_overrides()
        return Project(project_root=str(self.root), project_config=config, serena_config=serena_config)

    def test_config_bin_ignore_wins_over_gitignore_negation(self) -> None:
        project = self._project()
        # bin dir and its contents are ignored despite the "!kept/" negation re-including the tree
        assert project.is_ignored_path(str(self.root / "kept" / "bin"))
        assert project.is_ignored_path(str(self.root / "kept" / "bin" / "artifact.bin"))
        # non-build source inside the re-included tree stays searchable
        assert not project.is_ignored_path(str(self.root / "kept"))
        assert not project.is_ignored_path(str(self.root / "kept" / "src.py"))
