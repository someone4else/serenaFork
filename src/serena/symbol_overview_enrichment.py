# SPDX-License-Identifier: GPL-3.0-or-later
"""
Enrichment of a file's symbols overview (an extension of the Serena fork): structural containers are seen through,
callables show their signatures, classes, interfaces and structs show their base types, and constructors are labelled
as such.
"""

import logging
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from serena.repl.api.lsp_api import LspSymbolsOverviewRenderer, SymbolOutputParams
from serena.symbol import LanguageServerSymbol, SymbolDictGrouper
from solidlsp.ls_types import SymbolKind
from solidlsp.ls_utils import FileUtils

if TYPE_CHECKING:
    from serena.agent import SerenaAgent

log = logging.getLogger(__name__)

SymbolPredicate = Callable[[LanguageServerSymbol], bool]


@dataclass(frozen=True)
class SymbolPosition:
    """
    The position of a symbol's identifier within its file, which identifies the symbol across the (freshly created)
    symbol wrappers that the symbol tree yields upon each traversal.
    """

    line: int
    column: int

    @classmethod
    def of(cls, symbol: LanguageServerSymbol) -> "SymbolPosition | None":
        """
        :param symbol: the symbol
        :return: the symbol's position, or None if the symbol has no precise location
        """
        if symbol.line is None or symbol.column is None:
            return None
        return cls(symbol.line, symbol.column)


class BaseTypeClauseExtractor:
    """
    Extracts the base type clause (e.g. ``: BaseClass, IInterface``) of a class, interface or struct from its source
    declaration.

    Supported declaration forms include access modifiers and keywords such as ``abstract``, ``sealed`` or ``final``,
    generic type parameters (also with constraints containing colons, as in Swift), primary constructors (Kotlin, C#
    records), ``where`` constraints, nested generic base types, ``record class``/``record struct`` as well as
    declarations spanning several lines.
    Only files of languages in which a colon introduces the base type list of a brace-delimited declaration are
    considered (C#, C++, Kotlin, Swift): in Python, for instance, the very same colon opens the class body, and a
    brace further down the body (a dict literal) would otherwise splice the body into the symbol name.
    """

    MAX_DECLARATION_LINES = 15
    """the maximum number of source lines that are scanned for the opening brace of a declaration"""

    ENRICHED_KINDS = frozenset({SymbolKind.Class, SymbolKind.Interface, SymbolKind.Struct})
    """the kinds of symbols whose base types are extracted"""

    COLON_BASE_LIST_EXTENSIONS = frozenset({".cs", ".cpp", ".cc", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".kt", ".kts", ".swift"})
    """the extensions of the source files whose declarations introduce base types with a colon"""

    _BRACKET_PAIRS = {"<": ">", "(": ")"}
    _DECLARATION_KEYWORDS_AFTER_NAME = frozenset({"final"})

    def __init__(self, source: str) -> None:
        """
        :param source: the content of the file declaring the symbols (with line endings normalised to LF)
        """
        # LSP line numbers count line breaks only, unlike str.splitlines, which also splits at form feeds etc.
        self._lines = source.split("\n")

    @classmethod
    def applies_to(cls, relative_path: str) -> bool:
        """
        :param relative_path: the path of a source file
        :return: whether the file's language introduces base types with a colon
        """
        return os.path.splitext(relative_path)[1].lower() in cls.COLON_BASE_LIST_EXTENSIONS

    def extract(self, symbol: LanguageServerSymbol) -> str | None:
        """
        :param symbol: a symbol declared in the file
        :return: the base type clause (``: BaseClass, IInterface``), or None if the symbol is not of an enriched kind,
            has no base types or its declaration could not be parsed
        """
        if symbol.symbol_kind not in self.ENRICHED_KINDS or symbol.line is None or symbol.line >= len(self._lines):
            return None

        # collect the declaration lines up to the opening brace (none in sight: not a brace-language declaration)
        declaration_parts: list[str] = []
        for line in self._lines[symbol.line : symbol.line + self.MAX_DECLARATION_LINES]:
            declaration_parts.append(line)
            if "{" in line:
                break
        else:
            return None

        # collapse continuation lines, so that a multi-line declaration yields the same clause as a single-line one
        declaration = re.sub(r"\s+", " ", " ".join(declaration_parts))
        declaration = declaration[: declaration.find("{")]
        declaration = re.sub(r"\bwhere\b.*$", "", declaration, flags=re.DOTALL).strip()

        match = re.search(rf"\b(?:class|interface|struct|record(?:\s+(?:class|struct))?)\s+{re.escape(symbol.name)}\b", declaration)
        if match is None:
            return None
        bases = self._base_type_list_after_name(declaration, match.end())
        return ": " + bases if bases else None

    @classmethod
    def _base_type_list_after_name(cls, declaration: str, position: int) -> str | None:
        """
        :param declaration: the declaration (on a single line, without its body)
        :param position: the position right after the declared type's name
        :return: the base type list, which follows the colon after the generic type parameters, the primary constructor
            and keywords such as ``final``; None if there is no such colon
        """
        # skip bracketed groups (which may contain colons themselves) and keywords between the name and the colon
        while True:
            while position < len(declaration) and declaration[position].isspace():
                position += 1
            if position >= len(declaration):
                return None
            opening = declaration[position]
            if opening in cls._BRACKET_PAIRS:
                position = cls._skip_bracketed_group(declaration, position)
                if position < 0:
                    return None
                continue
            keyword = re.match(r"\w+", declaration[position:])
            if keyword is not None and keyword.group(0) in cls._DECLARATION_KEYWORDS_AFTER_NAME:
                position += len(keyword.group(0))
                continue
            break

        # a single colon (unlike the scope operator "::") introduces the base types
        if declaration[position] != ":" or declaration[position : position + 2] == "::":
            return None
        bases = declaration[position + 1 :].strip().rstrip(",").strip()
        return bases or None

    @classmethod
    def _skip_bracketed_group(cls, text: str, position: int) -> int:
        """
        :param text: the text
        :param position: the position of an opening bracket
        :return: the position after the matching closing bracket (nested brackets of either kind are balanced), or -1
            if the group is not closed
        """
        expected_closings: list[str] = []
        for index in range(position, len(text)):
            char = text[index]
            if char in cls._BRACKET_PAIRS:
                expected_closings.append(cls._BRACKET_PAIRS[char])
            elif expected_closings and char == expected_closings[-1]:
                expected_closings.pop()
                if not expected_closings:
                    return index + 1
        return -1


class SymbolOverviewEnricher:
    """
    Converts the top-level symbols of a file into the dictionaries of an enriched symbols overview.

    Compared to the plain overview, the enriched one
      * sees through purely structural containers (namespaces, modules, packages) and through a lone class wrapping
        the content of a file, such that the requested depth applies to the meaningful content;
      * appends the language server's ``detail`` (the signature) to the names of callables;
      * appends the base type clause to the names of classes, interfaces and structs;
      * labels methods named like their enclosing class (constructors in C# and similar languages) as constructors.

    The base type clauses are taken from the source when the enricher is created (in the process holding the project),
    such that the enricher itself only holds data and can be pickled along with the symbols.
    """

    _TRANSPARENT_CONTAINER_KINDS = frozenset({SymbolKind.Namespace, SymbolKind.Module, SymbolKind.Package})
    _DETAIL_INCLUDED_KINDS = frozenset({SymbolKind.Method, SymbolKind.Function, SymbolKind.Constructor})
    _CONSTRUCTOR_KIND_NAME = "Constructor"

    def __init__(self, base_type_clauses: dict[SymbolPosition, str]) -> None:
        """
        :param base_type_clauses: the base type clauses of the file's symbols by the symbols' positions
        """
        self._base_type_clauses = base_type_clauses

    @classmethod
    def for_file(
        cls, symbols: Sequence[LanguageServerSymbol], relative_path: str, abs_path: str, encoding: str
    ) -> "SymbolOverviewEnricher":
        """
        :param symbols: the top-level symbols of a file
        :param relative_path: the path of the file relative to the project root
        :param abs_path: the absolute path of the file
        :param encoding: the project's encoding; files that cannot be decoded with it are read with a detected one
        :return: an enricher for the overview of the file; without base type clauses if the file's language does not
            introduce base types with a colon or the file cannot be read
        """
        if not BaseTypeClauseExtractor.applies_to(relative_path):
            return cls({})
        try:
            source = FileUtils.read_file(abs_path, encoding)
        except Exception as e:
            log.warning("Cannot read %s for its base type clauses: %s", relative_path, e)
            return cls({})
        return cls.from_source(symbols, source)

    @classmethod
    def from_source(cls, symbols: Sequence[LanguageServerSymbol], source: str) -> "SymbolOverviewEnricher":
        """
        :param symbols: the top-level symbols of a file of a language introducing base types with a colon
        :param source: the content of the file (with line endings normalised to LF)
        :return: an enricher for the overview of the file
        """
        extractor = BaseTypeClauseExtractor(source)
        base_type_clauses: dict[SymbolPosition, str] = {}

        def collect(symbol: LanguageServerSymbol) -> None:
            position = SymbolPosition.of(symbol)
            if position is not None:
                clause = extractor.extract(symbol)
                if clause is not None:
                    base_type_clauses[position] = clause
            for child in symbol.iter_children():
                collect(child)

        for symbol in symbols:
            collect(symbol)
        return cls(base_type_clauses)

    def to_dicts(self, symbol: LanguageServerSymbol, output_params: SymbolOutputParams) -> list[LanguageServerSymbol.OutputDict]:
        """
        :param symbol: a top-level symbol of the file
        :param output_params: the output parameters of the overview; the depth is adjusted for structural containers
        :return: the enriched dictionaries representing the symbol
        """
        include = output_params.child_inclusion_predicate or (lambda _: True)

        # regular symbol: emitted with the requested depth
        if symbol.symbol_kind not in self._TRANSPARENT_CONTAINER_KINDS:
            return [self._to_enriched_dict(symbol, output_params, output_params.depth, include)]

        # a chain of transparent containers (e.g. Namespace A -> Namespace A.B) collapses into the innermost one
        children = [c for c in symbol.iter_children() if include(c)]
        if len(children) == 1 and children[0].symbol_kind in self._TRANSPARENT_CONTAINER_KINDS:
            return self.to_dicts(children[0], output_params)

        # neither the container nor a lone wrapping class consumes the requested depth
        depth = output_params.depth + 1 + self._count_singleton_wrapper_depth(children, include)
        return [self._to_enriched_dict(symbol, output_params, depth, include)]

    def _count_singleton_wrapper_depth(self, children: list[LanguageServerSymbol], include: SymbolPredicate) -> int:
        """
        :param children: the included children of a transparent container
        :param include: the child inclusion predicate
        :return: 1 if the children consist of a single non-transparent symbol with included children of its own (e.g. a
            lone class in a C# namespace), which then acts as a mere wrapper; 0 otherwise. Transparent children are
            handled by the recursion in `to_dicts`, and a wrapping class is compensated for one level only, as it is
            a meaningful symbol in its own right.
        """
        if len(children) != 1:
            return 0
        only_child = children[0]
        if only_child.symbol_kind in self._TRANSPARENT_CONTAINER_KINDS:
            return 0
        return 1 if any(include(c) for c in only_child.iter_children()) else 0

    def _to_enriched_dict(
        self, symbol: LanguageServerSymbol, output_params: SymbolOutputParams, depth: int, include: SymbolPredicate
    ) -> LanguageServerSymbol.OutputDict:
        """
        :param symbol: the symbol
        :param output_params: the output parameters of the overview
        :param depth: the depth to use instead of the one in the output parameters
        :param include: the child inclusion predicate of the output parameters (or one including all children)
        :return: the enriched dictionary representing the symbol
        """
        output_dict = symbol.to_dict(
            name_path=output_params.name_path,
            name=output_params.name,
            kind=output_params.kind,
            location=output_params.location,
            relative_path=output_params.relative_path,
            body_location=output_params.body_location,
            depth=depth,
            body=output_params.include_body,
            children_body=output_params.children_body,
            children_name=output_params.children_name,
            children_name_path=output_params.children_name_path,
            child_inclusion_predicate=output_params.child_inclusion_predicate,
        )
        self._enrich(symbol, output_dict, include, parent=None)
        return output_dict

    def _enrich(
        self,
        symbol: LanguageServerSymbol,
        output_dict: LanguageServerSymbol.OutputDict,
        include: SymbolPredicate,
        parent: LanguageServerSymbol | None,
    ) -> None:
        """
        Enriches the given dictionary (as produced by `to_dict`) in place, recursing into its children.

        :param symbol: the symbol from which the dictionary was created
        :param output_dict: the dictionary
        :param include: the child inclusion predicate passed to `to_dict`, such that the included children of the
            symbol correspond one-to-one (and in order) to the children in the dictionary
        :param parent: the parent symbol, if any
        """
        # signature for callables, base types for classes, interfaces and structs
        if "name" in output_dict:
            if symbol.symbol_kind in self._DETAIL_INCLUDED_KINDS:
                detail = symbol.symbol_root.get("detail") or ""
                if detail:
                    output_dict["name"] = detail if self._detail_repeats_name(symbol.name, detail) else f"{symbol.name} {detail}"
            else:
                position = SymbolPosition.of(symbol)
                clause = self._base_type_clauses.get(position) if position is not None else None
                if clause is not None:
                    output_dict["name"] = f"{symbol.name} {clause}"

        # constructors, which C# language servers report as methods named like the enclosing class
        if (
            "kind" in output_dict
            and symbol.symbol_kind == SymbolKind.Method
            and parent is not None
            and parent.symbol_kind == SymbolKind.Class
            and parent.name == symbol.name
        ):
            output_dict["kind"] = self._CONSTRUCTOR_KIND_NAME

        # children
        if "children" in output_dict:
            child_symbols = [c for c in symbol.iter_children() if include(c)]
            for child_symbol, child_dict in zip(child_symbols, output_dict["children"], strict=True):
                self._enrich(child_symbol, child_dict, include, parent=symbol)

    @staticmethod
    def _detail_repeats_name(name: str, detail: str) -> bool:
        """
        :param name: a symbol's name
        :param detail: the symbol's ``detail``
        :return: whether the detail already starts with the name, as is the case for language servers reporting a
            full signature (``Add(int, int) : int``) rather than its tail (``(int, int) : int``); the name must not be
            followed by an identifier character, such that "Get" is not considered repeated by a detail on "GetAll"
        """
        if not detail.startswith(name):
            return False
        remainder = detail[len(name) :]
        return not remainder or not (remainder[0].isalnum() or remainder[0] == "_")


class EnrichedSymbolsOverviewRenderer(LspSymbolsOverviewRenderer):
    """
    Renders a file's symbols overview enriched by a `SymbolOverviewEnricher`, keeping the fallbacks of the plain
    overview (depth-0 overview, symbol counts by kind) for the case where the length limit is exceeded.
    """

    def __init__(
        self,
        agent: "SerenaAgent",
        max_answer_chars: int,
        output_params: SymbolOutputParams,
        enricher: SymbolOverviewEnricher,
        grouper: SymbolDictGrouper | None = None,
    ):
        """
        :param agent: the agent (not retained)
        :param max_answer_chars: the maximum number of characters; -1 for the configured default
        :param output_params: the output parameters of the overview
        :param enricher: the enricher for the file whose overview is rendered
        :param grouper: the grouper for the symbol dictionaries
        """
        super().__init__(agent, max_answer_chars, output_params, grouper=grouper)
        self._enricher = enricher

    def symbol_dicts_(
        self, symbols: list[LanguageServerSymbol], info_by_symbol: dict[LanguageServerSymbol, str]
    ) -> list[LanguageServerSymbol.OutputDict]:
        """
        :param symbols: the top-level symbols of the file
        :param info_by_symbol: not supported (a symbols overview never includes hover info); must be empty
        :return: the enriched dictionaries of the symbols, which need not correspond one-to-one to the symbols, as a
            chain of structural containers is represented by the innermost container
        """
        if info_by_symbol:
            raise ValueError("The enriched symbols overview does not support per-symbol info")
        symbol_dicts: list[LanguageServerSymbol.OutputDict] = []
        for symbol in symbols:
            symbol_dicts.extend(self._enricher.to_dicts(symbol, self._output_params))
        return symbol_dicts
