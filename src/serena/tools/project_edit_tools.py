# SPDX-License-Identifier: GPL-3.0-or-later
"""
The Serena fork's tool for editing projects other than the active one (e.g. git worktrees) via the project server.
"""

from serena.tools import Tool, ToolMarkerCanEdit, ToolMarkerDoesNotRequireActiveProject, ToolMarkerOptional


class EditProjectTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject, ToolMarkerCanEdit):
    """
    Tool for editing a project other than the active one (e.g. a git worktree), by executing an editing tool in its
    context via the project server.
    """

    def apply(self, project_name: str, tool_name: str, tool_params_json: str) -> str:
        """
        Edits another project (e.g. a git worktree registered with the project server) by executing an editing tool
        in its context, with the same parameters as for the active project. Applicable tools: replace_symbol_body,
        insert_after_symbol, insert_before_symbol, replace_content, rename_symbol, safe_delete_symbol.
        If the result says that the outcome of the edit is unknown, read the file before repeating the edit.

        :param project_name: the root path of the project to edit (or its name)
        :param tool_name: the name of the editing tool to execute
        :param tool_params_json: the parameters to pass to the tool, encoded as a JSON string
        """
        # the project server client is imported lazily, as its module builds upon the tools package
        from serena.project_server_ext import ProjectEditClient

        if not self.agent.get_language_backend().is_lsp():
            raise ValueError("Editing other projects requires the language server backend")
        # an edit that the session may not perform itself (e.g. in planning mode) may not be delegated either
        if not self.agent.get_active_tools().contains_tool_name(tool_name):
            raise ValueError(f"Tool '{tool_name}' is not active in this session and therefore cannot be applied to another project")
        return ProjectEditClient(self.agent.serena_config).edit_project(project_name, tool_name, tool_params_json)
