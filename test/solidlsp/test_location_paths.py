"""The paths under which language server locations of files inside and outside the workspace are addressed.

The conversion is exercised directly, without a running language server: a response entry of a definition,
reference or implementation request is passed to the conversion that all of these requests share.
"""

import os
from pathlib import Path
from typing import Self

from solidlsp import SolidLanguageServer, ls_types

_RANGE: ls_types.Range = {
    "start": {"line": 0, "character": 0},
    "end": {"line": 1, "character": 0},
}


class _UnstartedLanguageServer(SolidLanguageServer):
    """A language server that is never started; it serves to convert location responses."""

    def _start_server(self) -> None:
        raise NotImplementedError

    def _create_base_initialize_params(self) -> dict:
        return {}

    @classmethod
    def _build(cls, repository_root_path: Path, additional_workspace_folders: tuple[Path, ...] = ()) -> Self:
        """
        :param repository_root_path: the root of the analysed repository
        :param additional_workspace_folders: further workspace folders, which lie outside the repository root
        :return: an instance that cannot execute requests but converts their responses
        """
        language_server = object.__new__(cls)
        language_server.repository_root_path = str(repository_root_path)
        workspace_folders = [str(repository_root_path)] + [str(folder) for folder in additional_workspace_folders]
        language_server._abs_workspace_folders_all = workspace_folders
        return language_server


def _convert(language_server: SolidLanguageServer, file_path: Path) -> ls_types.Location | None:
    """
    :param language_server: the language server that reported the file
    :param file_path: the file to report a location in
    :return: the location created from the response entry for the start of the file
    """
    request = SolidLanguageServer.DefinitionLocationRequest(language_server, file_path.name, 0, 0)
    request.skip_ignored_paths = False
    return request.convert_location_item({"uri": file_path.as_uri(), "range": _RANGE})


def test_location_in_project_file_is_addressed_relative_to_repository_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source_file = root / "src" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def function():\n    pass\n", encoding="utf-8")

    location = _convert(_UnstartedLanguageServer._build(root), source_file)

    assert location is not None
    assert location["relativePath"] == os.path.join("src", "module.py")


def test_location_outside_all_workspace_folders_is_addressed_absolutely(tmp_path: Path) -> None:
    """A declaration in a file that the language server generated from package metadata, which lies in a
    temporary directory outside the workspaces, is returned instead of being rejected.
    """
    root = tmp_path / "project"
    root.mkdir()
    metadata_file = tmp_path / "metadata" / "ObjectExtensions.cs"
    metadata_file.parent.mkdir()
    metadata_file.write_text("public static class ObjectExtensions { }\n", encoding="utf-8")

    location = _convert(_UnstartedLanguageServer._build(root), metadata_file)

    assert location is not None
    path = location["relativePath"]
    assert path is not None
    assert os.path.isabs(path)
    assert Path(path).resolve() == metadata_file.resolve()


def test_location_in_additional_workspace_folder_is_addressed_relative_to_repository_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    additional_folder = tmp_path / "shared"
    additional_folder.mkdir()
    source_file = additional_folder / "shared_module.py"
    source_file.write_text("def function():\n    pass\n", encoding="utf-8")
    language_server = _UnstartedLanguageServer._build(root, additional_workspace_folders=(additional_folder,))

    location = _convert(language_server, source_file)

    assert location is not None
    path = location["relativePath"]
    assert path is not None
    assert Path(path).parts == ("..", "shared", "shared_module.py")
