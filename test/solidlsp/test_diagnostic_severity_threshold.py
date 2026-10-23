"""The severity threshold that a diagnostics request applies, stated either by the LSP severity code or by its name.

The threshold is exercised directly, without a running language server.
"""

import pytest

from solidlsp.ls_diagnostic_severity import DiagnosticSeverityThreshold
from solidlsp.lsp_protocol_handler.lsp_types import DiagnosticSeverity


@pytest.mark.parametrize("name, code", [("Error", 1), ("Warning", 2), ("Information", 3), ("Hint", 4)])
def test_severity_is_stated_by_code_or_by_name(name: str, code: int) -> None:
    threshold = DiagnosticSeverityThreshold.parse(code)

    assert DiagnosticSeverityThreshold.parse(name) == threshold
    assert DiagnosticSeverityThreshold.parse(name.lower()) == threshold
    assert DiagnosticSeverityThreshold.parse(name.upper()) == threshold
    assert DiagnosticSeverityThreshold.parse(str(code)) == threshold


def test_threshold_includes_diagnostics_of_its_severity_and_of_more_severe_ones() -> None:
    threshold = DiagnosticSeverityThreshold.parse("Warning")

    assert threshold.admits(DiagnosticSeverity.Error)
    assert threshold.admits(DiagnosticSeverity.Warning)
    assert not threshold.admits(DiagnosticSeverity.Information)
    assert not threshold.admits(DiagnosticSeverity.Hint)


def test_threshold_includes_a_diagnostic_that_states_no_severity() -> None:
    threshold = DiagnosticSeverityThreshold.parse("Error")

    assert threshold.admits(None)


def test_severity_that_denotes_no_lsp_level_is_rejected_naming_every_accepted_form() -> None:
    with pytest.raises(ValueError) as exception_info:
        DiagnosticSeverityThreshold.parse("Critical")

    message = str(exception_info.value)
    for severity in DiagnosticSeverity:
        assert severity.name in message
        assert str(severity.value) in message
    assert "Critical" in message
