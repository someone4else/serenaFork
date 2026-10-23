import logging

import pytest

from serena.config.serena_config import SerenaConfig
from serena.mcp import SerenaMCPFactory
from serena.tools.symbol_tools import GetDiagnosticsForFileTool, GetDiagnosticsForSymbolTool
from serena.tools.tools_base import Tool, ToolRegistry
from solidlsp.ls_diagnostic_severity import DiagnosticSeverityThreshold


@pytest.mark.parametrize("context", ("chatgpt", "codex", "oaicompat-agent"))
def test_all_tool_parameters_have_type(context):
    """
    For every tool exposed by Serena, ensure that the generated
    Open‑AI schema contains a ``type`` entry for each parameter.
    """
    cfg = SerenaConfig(log_level=logging.ERROR).with_headless_mode_overrides()
    registry = ToolRegistry()
    cfg.included_optional_tools = tuple(registry.get_tool_names_optional())
    factory = SerenaMCPFactory(transport="stdio", context=context)
    # Initialize the agent so that the tools are available
    factory.agent = factory._create_serena_agent(cfg)
    tools = list(factory._iter_tools())

    for tool in tools:
        mcp_tool = factory.make_mcp_tool(tool, openai_tool_compatible=True)
        params = mcp_tool.parameters

        # Collect any parameter that lacks a type
        issues = []
        print(f"Checking tool {tool}")

        if "properties" not in params:
            issues.append(f"Tool {tool.get_name()!r} missing properties section")
        else:
            for pname, prop in params["properties"].items():
                if "type" not in prop:
                    issues.append(f"Tool {tool.get_name()!r} parameter {pname!r} missing 'type'")
        if issues:
            raise AssertionError("\n".join(issues))


@pytest.mark.parametrize(
    "tool_class, required_arguments",
    [
        (GetDiagnosticsForFileTool, {"relative_path": "module.py"}),
        (GetDiagnosticsForSymbolTool, {"name_path": "module/function"}),
    ],
)
def test_diagnostics_tool_accepts_a_severity_name(tool_class: type[Tool], required_arguments: dict[str, str]) -> None:
    """A diagnostics request states the severity threshold either by its LSP code or by its name, both denoting
    the same threshold.
    """
    arguments_model = tool_class.get_apply_fn_metadata_from_cls().arg_model

    by_name = arguments_model.model_validate(required_arguments | {"min_severity": "Warning"}).model_dump()
    by_code = arguments_model.model_validate(required_arguments | {"min_severity": 2}).model_dump()

    assert DiagnosticSeverityThreshold.parse(by_name["min_severity"]) == DiagnosticSeverityThreshold.parse(by_code["min_severity"])
