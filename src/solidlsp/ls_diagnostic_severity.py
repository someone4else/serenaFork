# SPDX-License-Identifier: MIT
"""
The severity threshold by which a language server request filters the diagnostics of a document.
"""

from dataclasses import dataclass
from typing import Self

from solidlsp.lsp_protocol_handler.lsp_types import DiagnosticSeverity


@dataclass(frozen=True)
class DiagnosticSeverityThreshold:
    """
    A threshold that a diagnostics request applies to the severity of diagnostics: a diagnostic is included
    if it is of the threshold's severity or of a more severe one.

    The severity is stated by the client either as the numeric LSP severity code or as the name of the LSP
    severity, where names are matched case-insensitively.
    """

    severity: DiagnosticSeverity
    """the numeric code of the least severe severity that the threshold includes"""

    @classmethod
    def parse(cls, value: int | str) -> Self:
        """
        Parses a severity threshold as stated in a diagnostics request.

        :param value: the numeric code of the least severe severity to include (1=Error, 2=Warning,
            3=Information, 4=Hint), its decimal representation or the corresponding LSP severity name
        :return: the severity threshold
        """
        # reduce the stated value to the lower-cased name or code it denotes
        normalized_value = value.strip().lower() if isinstance(value, str) else str(value)

        # match it against the severity names and codes, which the LSP severity type alone declares
        for severity in DiagnosticSeverity:
            if normalized_value in (severity.name.lower(), str(severity.value)):
                return cls(severity)

        # state both accepted forms when the value denotes no LSP severity
        accepted_values = ", ".join(f"{severity.value}/{severity.name}" for severity in DiagnosticSeverity)
        raise ValueError(f"min_severity must be one of {accepted_values}, got {value}")

    def admits(self, severity: int | None) -> bool:
        """
        Determines whether a diagnostic of the given severity is included by the threshold.

        :param severity: the numeric severity code of the diagnostic; ``None`` if the diagnostic states no
            severity, which the threshold includes unconditionally
        :return: whether the diagnostic is included
        """
        return severity is None or int(severity) <= self.severity
