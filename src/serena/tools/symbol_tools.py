"""
Language server-related tools
"""

import copy
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from serena.symbol import LanguageServerSymbol, LanguageServerSymbolDictGrouper
from serena.tools import (
    SUCCESS_RESULT,
    EditingToolWithDiagnostics,
    Tool,
    ToolMarkerSymbolicEdit,
    ToolMarkerSymbolicRead,
)
from serena.tools.tools_base import ToolMarkerOptional
from serena.util.ls_diagnostics import GroupedDiagnostics
from serena.util.text_utils import find_text_coordinates
from solidlsp.ls_types import SymbolKind

if TYPE_CHECKING:
    from serena.project import Project

log = logging.getLogger(__name__)

# Symbol kinds that are purely structural containers (namespaces, modules, packages).
# In languages like C#/Java these wrap the "real" symbols (classes, interfaces, etc.)
# and provide no useful information at depth=0 on their own.
_TRANSPARENT_CONTAINER_KINDS = frozenset({SymbolKind.Namespace, SymbolKind.Module, SymbolKind.Package})

# Symbol kinds for which the LSP 'detail' field shall be appended to the name in the
# symbols overview output. For callables this is the signature (e.g. "(int a, int b): int").
_DETAIL_INCLUDED_KINDS = frozenset({SymbolKind.Method, SymbolKind.Function, SymbolKind.Constructor})

# Symbol kinds for which source-code-based inheritance info shall be extracted
# and appended to the name in the symbols overview output.
_INHERITANCE_ENRICHED_KINDS = frozenset({SymbolKind.Class, SymbolKind.Interface, SymbolKind.Struct})

# Maximum number of source lines to scan when looking for the opening brace '{' of a
# class/interface/struct declaration (handles multi-line declarations with long generic
# parameter lists or where-constraints).
_MAX_DECLARATION_LINES = 15


class RestartLanguageServerTool(Tool, ToolMarkerOptional):
    """Restarts the language server(s)."""

    def apply(self) -> str:
        """Use this tool only on explicit user request or after confirmation.
        It may be necessary to restart the language server if it hangs.
        """
        self.agent.reset_language_server_manager()
        return SUCCESS_RESULT


class GetSymbolsOverviewTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets an overview of the top-level symbols defined in a given file.
    """

    symbol_dict_grouper = LanguageServerSymbolDictGrouper(["kind"], ["kind"], collapse_singleton=True)

    def apply(self, relative_path: str, depth: int = -1, max_answer_chars: int = -1) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param depth: depth up to which descendants shall be retrieved.
            Default (-1) results in a language specific choice: 1 for java and kotlin and 0 for other languages
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :return: a JSON object containing symbols grouped by kind in a compact format.
        """
        # Note: file system sync not required (relevant file is opened in the language server explicitly)

        if depth == -1:
            if relative_path.endswith((".java", ".kt")):
                depth = 1
            else:
                depth = 0

        result = self.get_symbol_overview(relative_path, depth=depth)

        # capture kind names and depth-0 snapshots before grouping, which mutates the dicts
        kind_names = [d.get("kind", "unknown") for d in result]
        if depth > 0:
            depth_0_result = [d.copy() for d in result]
            for d in depth_0_result:
                d.pop("children", None)

        compact_result = self.symbol_dict_grouper.group(result)
        result_json_str = self._to_json(compact_result)

        # shortened result closures
        def make_kind_counts() -> str:
            return f"Symbol counts by kind:\n{self._to_json(Counter(kind_names))}"

        if depth == 0:
            shortened_results = [make_kind_counts]
        else:

            def make_depth_0_result() -> str:
                compact_depth_0_result = self.symbol_dict_grouper.group(depth_0_result)
                return "Depth 0 overview:\n" + self._to_json(compact_depth_0_result)

            shortened_results = [make_depth_0_result, make_kind_counts]

        return self._limit_length(result_json_str, max_answer_chars, shortened_result_factories=shortened_results)

    def get_symbol_overview(self, relative_path: str, depth: int = 0) -> list[LanguageServerSymbol.OutputDict]:
        """
        :param relative_path: relative path to a source file
        :param depth: the depth up to which descendants shall be retrieved
        :return: a list of symbol dictionaries representing the symbol overview of the file
        """
        symbol_retriever = self.create_language_server_symbol_retriever()

        # The symbol overview is capable of working with both files and directories,
        # but we want to ensure that the user provides a file path.
        file_path = os.path.join(self.project.project_root, relative_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory {relative_path} does not exist in the project.")
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file path, but got a directory path: {relative_path}. ")
        if not symbol_retriever.can_analyze_file(relative_path):
            raise ValueError(
                f"Cannot extract symbols from file {relative_path}. Active language servers: {[l.value for l in self.agent.get_active_language_server_ids()]}"
            )

        symbols = symbol_retriever.get_symbol_overview(relative_path)[relative_path]

        def child_inclusion_predicate(s: LanguageServerSymbol) -> bool:
            return not s.is_low_level()

        symbol_dicts: list[LanguageServerSymbol.OutputDict] = []
        for symbol in symbols:
            symbol_dicts.extend(
                _flatten_transparent_containers_to_dicts(
                    symbol,
                    depth=depth,
                    child_inclusion_predicate=child_inclusion_predicate,
                    project=self.project,
                )
            )
        return symbol_dicts


def _extract_inheritance_from_source(symbol: LanguageServerSymbol, project: "Project") -> str | None:
    """
    Extracts the inheritance clause from the source declaration of a Class/Interface/Struct symbol.

    Reads the source file starting at the symbol's declared line and uses a regex to locate the
    inheritance part (e.g. ``: BaseClass, IInterface``). Multi-line declarations as well as
    C#/Java-style generic type parameters and ``where`` constraints are handled.

    Supported C# declaration forms (access modifiers, ``abstract``, ``sealed``, etc. are all
    tolerated):

    - ``public class MyClass : BaseClass, IFoo``
    - ``public class MyClass<T> : Bar<T> where T : IComparable``
    - ``public class MyClass : Bar<List<int>, Dictionary<string, int>>``
    - ``public record class Foo : Bar``
    - ``public record struct Foo : IFoo``

    Only brace-delimited declarations are considered, i.e. the declaration must be followed by an
    opening ``{`` within the scanned lines. This restricts the extraction to languages where a
    colon introduces the base type list (C#, C++, ...). In Python the very same colon opens the
    class body instead, so its declarations are deliberately left alone.

    :param symbol: the symbol to inspect; must have ``relative_path`` and ``line`` set
    :param project: the project instance used to read the source file
    :return: the inheritance suffix string like ``: BaseClass, IInterface``, or None if the symbol
        has no base types or the declaration could not be parsed
    """
    if symbol.relative_path is None or symbol.line is None:
        return None
    try:
        lines = project.read_file(symbol.relative_path).splitlines()
        start_line = symbol.line
        if start_line >= len(lines):
            return None

        # collect up to _MAX_DECLARATION_LINES lines from the declaration line onwards until '{' is found
        declaration_parts: list[str] = []
        for i in range(start_line, min(start_line + _MAX_DECLARATION_LINES, len(lines))):
            declaration_parts.append(lines[i])
            if "{" in lines[i]:
                break
        else:
            # no opening brace in sight: not a brace-language declaration (see the docstring)
            return None

        # collapse the indentation of continuation lines, so that a declaration spanning several
        # lines yields the same suffix as the equivalent single-line declaration
        declaration = re.sub(r"\s+", " ", " ".join(declaration_parts))

        # truncate at the opening brace (start of the body)
        declaration = declaration[: declaration.find("{")]

        # remove 'where' constraint clauses (e.g. 'where T : IComparable')
        declaration = re.sub(r"\bwhere\b.*$", "", declaration, flags=re.DOTALL).strip()

        # Look for the inheritance clause after the type name (and optional generics):
        # [^:]* greedily consumes everything that is not ':' (i.e. the optional generic type
        # parameters), then the ':' introduces the base type list.
        name_pattern = re.escape(symbol.name)
        m = re.search(
            rf"\b(?:class|interface|struct|record(?:\s+(?:class|struct))?)\s+{name_pattern}\b[^:]*:\s*(.+)",
            declaration,
        )
        if m:
            bases = m.group(1).strip().rstrip(",").strip()
            if bases:
                return ": " + bases
    except Exception:
        log.debug("Failed to extract inheritance from source for symbol %s", symbol.name, exc_info=True)
    return None


def _detail_repeats_name(name: str, detail: str) -> bool:
    """
    :param name: the symbol's name
    :param detail: the symbol's LSP ``detail`` field
    :return: whether the detail already starts with the symbol's name, as is the case for language
        servers that report a full signature (``Add(int, int) : int``) rather than just the
        parameter/return part (``(int, int) : int``). The character following the name must not be
        part of an identifier, so that the name "Get" is not considered to be repeated by a detail
        describing "GetAll".
    """
    if not detail.startswith(name):
        return False
    remainder = detail[len(name) :]
    return not remainder or not (remainder[0].isalnum() or remainder[0] == "_")


def _count_singleton_wrapper_depth(
    children: list[LanguageServerSymbol],
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
) -> int:
    """
    Returns 1 if the given children list represents a singleton non-transparent wrapper (e.g. a lone
    Class inside a Namespace in a typical C# file), 0 otherwise.

    A singleton wrapper is a non-transparent symbol that is the *only* visible child at its level and
    itself has visible descendants. In that case the symbol acts as a structural wrapper and should
    not consume the user's depth budget.

    Transparent containers are deliberately excluded here because they are already handled by the
    recursion in :func:`_flatten_transparent_containers_to_dicts`. The compensation is capped at +1:
    a Class is a meaningful symbol (unlike a Namespace), so we do not recurse further.
    """
    if len(children) != 1:
        return 0

    only_child = children[0]

    # transparent containers are handled by the caller's recursion
    if only_child.symbol_kind in _TRANSPARENT_CONTAINER_KINDS:
        return 0

    # check whether the sole non-transparent child itself has visible descendants
    has_grandchildren = any(child_inclusion_predicate(c) for c in only_child.iter_children())
    if not has_grandchildren:
        return 0  # leaf symbol; no extra depth needed

    # the single non-transparent child (e.g. a Class) acts as a wrapper -> +1
    return 1


def _enhance_symbol_dict(
    symbol: LanguageServerSymbol,
    output_dict: LanguageServerSymbol.OutputDict,
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
    parent_symbol: LanguageServerSymbol | None = None,
    project: "Project | None" = None,
) -> None:
    """
    Recursively enhances an OutputDict (as produced by ``to_dict``) in-place with:

    1. **Signatures** - for callable symbols (Method, Function, Constructor) the LSP ``detail``
       field (e.g. ``(int a, int b): int``) is appended to the ``name`` entry, so the LM sees full
       signatures instead of bare names. Language servers that already repeat the symbol name in
       the detail (``Add(int, int) : int``) do not get it prepended a second time.
    2. **Inheritance info** - for Class/Interface/Struct symbols the inheritance clause is extracted
       from the source file and appended to the name as a ``: BaseClass, IInterface`` suffix.
    3. **Constructor detection** - in C# (and similar OO languages) the language server returns
       constructors as ``Method`` symbols whose name matches the enclosing class name. Such entries
       are relabelled to ``kind="Constructor"``, so the LM can tell them apart from regular methods.
    4. **Recursion** - the *symbol* tree is walked in parallel with the *dict* tree, so that each
       enhancement is applied with access to the full symbol metadata (including the LSP ``detail``
       field and the parent reference).

    :param symbol: the symbol whose data was used to build ``output_dict``
    :param output_dict: the dict produced by ``symbol.to_dict()``, modified in-place
    :param child_inclusion_predicate: the predicate that was passed to ``to_dict``, such that the
        child symbol list matches the ``children`` list in the dict
    :param parent_symbol: the parent symbol, used for constructor detection (None for root symbols)
    :param project: the project instance used to read source files for inheritance extraction
        (Class/Interface/Struct symbols only); if None, no inheritance info is added
    """
    # 1. append the LSP detail (signature) to the name for callable kinds
    if symbol.symbol_kind in _DETAIL_INCLUDED_KINDS:
        detail: str = symbol.symbol_root.get("detail", "") or ""
        if detail and "name" in output_dict:
            output_dict["name"] = detail if _detail_repeats_name(symbol.name, detail) else f"{symbol.name} {detail}"

    # 2. append source-based inheritance info for Class/Interface/Struct
    elif symbol.symbol_kind in _INHERITANCE_ENRICHED_KINDS and project is not None:
        suffix = _extract_inheritance_from_source(symbol, project)
        if suffix and "name" in output_dict:
            output_dict["name"] = f"{symbol.name} {suffix}"

    # 3. detect C# constructors (emitted as Method symbols with the enclosing class' name) and relabel them
    if (
        symbol.symbol_kind == SymbolKind.Method
        and "kind" in output_dict
        and parent_symbol is not None
        and parent_symbol.symbol_kind == SymbolKind.Class
        and parent_symbol.name == symbol.name
    ):
        output_dict["kind"] = "Constructor"

    # 4. recurse into the children
    if "children" not in output_dict:
        return
    child_dicts = output_dict["children"]
    # The child dicts correspond 1-to-1 (in order) to the children that passed the inclusion
    # predicate, because to_dict() builds them in iteration order.
    child_symbols = [c for c in symbol.iter_children() if child_inclusion_predicate(c)]
    for child_sym, child_dict in zip(child_symbols, child_dicts, strict=True):
        _enhance_symbol_dict(child_sym, child_dict, child_inclusion_predicate, parent_symbol=symbol, project=project)


def _flatten_transparent_containers_to_dicts(
    symbol: LanguageServerSymbol,
    depth: int,
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
    project: "Project | None" = None,
) -> list[LanguageServerSymbol.OutputDict]:
    """
    Converts a symbol to one or more OutputDicts, automatically "seeing through" purely structural
    containers (Namespace, Module, Package), such that the user always sees meaningful content even
    at depth=0.

    For transparent containers the strategy is:

    - The container itself is emitted with an effective depth that compensates for all "wrapper"
      levels between the container and the first level of real content, specifically

      * ``+1`` for the transparent container itself (it is structural, not meaningful, and should
        not consume the user's depth budget),
      * ``+1`` (via :func:`_count_singleton_wrapper_depth`) if the transparent container's only
        child is a *non-transparent* singleton wrapper (e.g. a lone Class in a C# namespace). That
        class also acts purely as a structural wrapper in single-class files and should not consume
        the user's depth budget either.

    - If a transparent container has *exactly one* child which is also transparent, we recurse and
      flatten further, so the user does not see a chain of nested namespaces.

    For all other symbols the behaviour is the same as a plain ``to_dict`` call.
    """
    if symbol.symbol_kind not in _TRANSPARENT_CONTAINER_KINDS:
        # regular symbol; emit it as-is and apply the enhancements
        output_dict = symbol.to_dict(
            name_path=False,
            name=True,
            depth=depth,
            kind=True,
            relative_path=False,
            location=False,
            child_inclusion_predicate=child_inclusion_predicate,
        )
        _enhance_symbol_dict(symbol, output_dict, child_inclusion_predicate, project=project)
        return [output_dict]

    children = [c for c in symbol.iter_children() if child_inclusion_predicate(c)]

    # If the container has exactly one child which is *also* a transparent container, skip the outer
    # wrapper entirely and recurse on the inner one. This collapses chains like
    # Namespace(A) -> Namespace(A.B) -> Class(Foo) into a single entry "Namespace A.B -> {Class Foo}".
    if len(children) == 1 and children[0].symbol_kind in _TRANSPARENT_CONTAINER_KINDS:
        return _flatten_transparent_containers_to_dicts(
            children[0],
            depth=depth,
            child_inclusion_predicate=child_inclusion_predicate,
            project=project,
        )

    # The container has substantive children; emit it with depth + 1 + extra_depth, such that neither
    # the transparent container nor a singleton non-transparent wrapper (e.g. a lone class in a C#
    # file) consumes the user's depth budget.
    extra_depth = _count_singleton_wrapper_depth(children, child_inclusion_predicate)
    output_dict = symbol.to_dict(
        name_path=False,
        name=True,
        depth=depth + 1 + extra_depth,
        kind=True,
        relative_path=False,
        location=False,
        child_inclusion_predicate=child_inclusion_predicate,
    )
    _enhance_symbol_dict(symbol, output_dict, child_inclusion_predicate, project=project)
    return [output_dict]


class FindSymbolTool(Tool, ToolMarkerSymbolicRead):
    """
    Performs a global (or local) search using the language server backend.
    """

    # group children by kind, keeping just the name (the parent's name_path makes it unambiguous);
    # we don't group the top-level result list because many tests rely on it being a flat list of symbol dicts
    symbol_dict_grouper = LanguageServerSymbolDictGrouper([], ["kind"], collapse_singleton=True)

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path_pattern: str,
        depth: int = 0,
        relative_path: str = "",
        include_body: bool = False,
        include_info: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        substring_matching: bool = False,
        max_matches: int = -1,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds symbols and code entities (classes, methods, etc.) based on the given name path pattern.
        The returned symbol information can be used for edits or further queries.
        Specify `depth > 0` to also retrieve children/descendants (e.g., methods of a class).

        A name path is a path in the symbol tree *within a source file*.
        For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
        If a symbol is overloaded (e.g., in Java), a 0-based index is appended (e.g. "MyClass/my_method[0]") to
        uniquely identify it.

        To search for a symbol, you provide a name path pattern that is used to match against name paths.
        It can be
         * a simple name (e.g. "method"), which will match any symbol with that name
         * a relative path like "class/method", which will match any symbol with that name path suffix
         * an absolute name path "/class/method" (absolute name path), which requires an exact match of the full name path within the source file.
        Append an index `[i]` to match a specific overload only, e.g. "MyClass/my_method[1]".

        :param name_path_pattern: the name path matching pattern (see above)
        :param depth: depth up to which descendants shall be retrieved (e.g. use 1 to also retrieve immediate children;
            for the case where the symbol is a class, this will return its methods).
            Ignored if `include_body=True`. Default 0.
        :param relative_path: (optional) restrict search to this file or directory. If None, searches entire codebase.
            If a directory is passed, the search will be restricted to the files in that directory.
            If a file is passed, the search will be restricted to that file.
        :param include_body: whether to include the symbol's source code. Use judiciously.
        :param include_info: whether to include additional info (hover-like, typically including docstring and signature),
            about the symbol (ignored if include_body is True). Info is never included for child symbols.
            Note: Depending on the language, this can be slow (e.g., C/C++).
        :param include_kinds: (optional) limits results to the given LSP symbol kinds (integers)
        :param exclude_kinds: (optional) list of LSP symbol kinds (integers) to exclude.
        :param substring_matching: If True, use substring matching for the last element of the pattern, such that
            "Foo/get" would match "Foo/getValue" and "Foo/getData".
        :param max_matches: maximum number of permitted matches. If exceeded, a shortened result is returned
             which allows refining the search. -1 (default) means no limit. Set to 1 if you search for a single symbol.
        :param max_answer_chars: max result length; -1 for default
        :return: symbols (with locations) matching the name.
        """
        # Note: file system sync not required; the symbol finder opens all relevant source files explicitly in the case of changes

        if include_body:
            depth = 0  # ignore user-specified depth if include_body is True
        assert max_matches != 0, "max_matches must be > 0 or equal to -1."
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        symbols = symbol_retriever.find(
            name_path_pattern,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
            substring_matching=substring_matching,
            within_relative_path=relative_path,
        )
        n_matches = len(symbols)

        def create_short_result_relative_path_to_name_paths() -> str:
            relative_path_to_name_paths: defaultdict[str, list[str]] = defaultdict(list)
            for s in symbols:
                relative_path_to_name_paths[s.location.relative_path or "unknown"].append(s.get_name_path())
            return f"Shortened result:\n{self._to_json(relative_path_to_name_paths)}"

        if 0 < max_matches < n_matches:
            return f"Matched {n_matches}>{max_matches=} symbols.\n" + create_short_result_relative_path_to_name_paths()

        symbol_dicts = [
            s.to_dict(
                kind=True,
                name_path=True,
                name=False,
                relative_path=True,
                body_location=True,
                depth=depth,
                body=include_body,
                children_name=True,
                children_name_path=False,
            )
            for s in symbols
        ]
        if not include_body and include_info:
            info_by_symbol = symbol_retriever.request_info_for_symbol_batch(symbols)
            for s, s_dict in zip(symbols, symbol_dicts, strict=True):
                if symbol_info := info_by_symbol.get(s):
                    # In python 3.15 we could specify extra_items=True in the TypedDict definition,
                    # https://peps.python.org/pep-0728/
                    # If we ever upgrade to 3.15, we can remove the type: ignore[typeddict-unknown-key]
                    s_dict["info"] = symbol_info

        grouped_symbol_dicts = self.symbol_dict_grouper.group(symbol_dicts)
        result = self._to_json(grouped_symbol_dicts)
        return self._limit_length(result, max_answer_chars, shortened_result_factories=[create_short_result_relative_path_to_name_paths])

    @classmethod
    def get_param_aliases(cls) -> dict[str, str]:
        return {"name_path": "name_path_pattern"}


class FindReferencingSymbolsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that reference the given symbol using the language server backend
    """

    symbol_dict_grouper = LanguageServerSymbolDictGrouper(["relative_path", "kind"], ["kind"], collapse_singleton=True)

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds references to the symbol at the given `name_path`. The result will contain metadata about the referencing symbols
        as well as a short code snippet around the reference.

        :param name_path: name path of the symbol
        :param relative_path: the relative path to the file containing the symbol for which to find references.
        :param include_kinds: (optional) limits results to the given LSP symbol kinds (integers)
        :param exclude_kinds: optional list of LSP symbol kinds (integers) to exclude.
        :param max_answer_chars: max result length; -1 for default
        :return: a list of JSON objects with the symbols referencing the requested symbol
        """
        # file system sync needed for case where symbol finder does not perform a global search, updating everything
        if relative_path:
            self.project.ls_sync_file_system_changes()

        include_body = False  # It is probably never a good idea to include the body of the referencing symbols
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None

        symbol_retriever = self.create_language_server_symbol_retriever()
        references_in_symbols = symbol_retriever.find_referencing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )

        reference_dicts = []
        for ref in references_in_symbols:
            ref_dict_orig = ref.symbol.to_dict(kind=True, relative_path=True, depth=0, body=include_body, body_location=True)
            ref_dict = dict(ref_dict_orig)
            if not include_body:
                ref_relative_path = ref.symbol.location.relative_path
                assert ref_relative_path is not None, f"Referencing symbol {ref.symbol.name} has no relative path, this is likely a bug."
                content_around_ref = self.project.retrieve_content_around_line(
                    relative_file_path=ref_relative_path, line=ref.line, context_lines_before=1, context_lines_after=1
                )
                ref_dict["content_around_reference"] = content_around_ref.to_display_string()
            reference_dicts.append(ref_dict)

        # capture lightweight reference data before grouping
        ref_summaries = []
        for ref, d in zip(references_in_symbols, reference_dicts, strict=True):
            ref_summaries.append(
                {
                    "name_path": d.get("name_path"),
                    "kind": d.get("kind"),
                    "relative_path": d.get("relative_path"),
                    "reference_line": ref.line,
                }
            )

        result = self.symbol_dict_grouper.group(reference_dicts)

        # shortened result closures, from least to most aggressive shortening
        def make_refs_without_context() -> str:
            """References with name_path and reference line, without surrounding code lines"""
            grouped = self.symbol_dict_grouper.group(copy.deepcopy(ref_summaries))
            return f"References without surrounding lines:\n{self._to_json(grouped)}"

        def make_per_file_counts() -> str:
            counts = Counter(str(r["relative_path"]) for r in ref_summaries)
            return f"Reference counts per file:\n{self._to_json(counts)}"

        def make_summary() -> str:
            return f"Found {len(ref_summaries)} references."

        shortened_results = [make_refs_without_context, make_per_file_counts, make_summary]

        result_json = self._to_json(result)
        return self._limit_length(result_json, max_answer_chars, shortened_result_factories=shortened_results)


class FindImplementationsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that implement the given symbol using the language server backend.
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_info: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds implementations of the symbol at the given `name_path`.

        :param name_path: the symbol's name path
        :param relative_path: the relative path to the file containing the symbol for which to find implementations.
            Note that here you can't pass a directory but must pass a file.
        :param include_info: whether to include additional info (hover-like, typically including docstring and signature),
            about the implementing symbols.
        :param include_kinds: (optional) limits results to the given LSP symbol kinds (integers)
        :param exclude_kinds: (optional) list of LSP symbol kinds (integers) to exclude.
        :param max_answer_chars: max result length; -1 for default
        :return: a list of JSON objects with the symbols implementing the requested symbol
        """
        self.project.ls_sync_file_system_changes()

        include_body = False
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()

        implementing_symbols = symbol_retriever.find_implementing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )

        symbol_dicts = [
            dict(s.to_dict(kind=True, relative_path=True, depth=0, body=include_body, body_location=True)) for s in implementing_symbols
        ]
        if include_info:
            info_by_symbol = symbol_retriever.request_info_for_symbol_batch(implementing_symbols)
            for s, s_dict in zip(implementing_symbols, symbol_dicts, strict=True):
                if symbol_info := info_by_symbol.get(s):
                    s_dict["info"] = symbol_info
                    s_dict.pop("name", None)  # name is included in the info

        result = self._to_json(symbol_dicts)
        return self._limit_length(result, max_answer_chars)


class FindDeclarationTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds the declaration/definition of a symbol
    """

    def apply(
        self,
        relative_path: str,
        regex: str,
        containing_symbol_name_path: str | None = None,
        include_body: bool = False,
        include_info: bool = False,
    ) -> str:
        r"""
        Finds the declaration of a symbol.

        :param relative_path: the relative path to the source file containing the symbol for which to find the declaration.
        :param regex: a regular expression with one group, where the group matches the symbol for which to perform the lookup.
            For example, to find the declaration of the `process` method in a call like `obj.process()`,
            pass an expression like "obj\.(process)\(process_input_arg=37\)".
            Prefer regexes with sufficiently large context around the group to render the match unambiguous.
            Uses Python syntax with MULTILINE and DOTALL flags enabled.
        :param containing_symbol_name_path: optional name path of a containing symbol whose body shall be searched instead of the full file.
        :param include_body: whether to include the symbol's body in the result. Default False.
        :param include_info: whether to include additional info (hover-like). Default False.
        """
        self.project.ls_sync_file_system_changes()

        symbol_retriever = self.create_language_server_symbol_retriever()
        relative_path = self._sanitize_input_param(relative_path)
        regex = self._sanitize_input_param(regex)

        # find relevant location for lookup
        editor = self.create_code_editor()
        if not containing_symbol_name_path:
            content = editor.read_file(relative_path)
            coords = find_text_coordinates(content, regex, require_unique=True)
            assert coords is not None
        else:
            symbol = symbol_retriever.find_unique(name_path_pattern=containing_symbol_name_path, within_relative_path=relative_path)
            body_line_numers = symbol.get_body_line_numbers_or_raise()
            content = editor.read_file(relative_path, lines=body_line_numers)
            coords = find_text_coordinates(content, regex, require_unique=True)
            assert coords is not None
            coords.line += body_line_numers[0]

        # retrieve declaration
        defining_symbol = symbol_retriever.find_declaration(
            relative_file_path=relative_path,
            line=coords.line,
            column=coords.col,
            include_body=include_body,
        )
        if defining_symbol is None:
            raise ValueError(
                f"No symbol declaration found at the location of the regex match. Location: {relative_path}:{coords.line}:{coords.col}."
            )

        # create output
        symbol_dict = self._defining_symbol_to_result_dict(
            symbol_retriever,
            defining_symbol,
            include_body,
            include_info,
        )
        result = self._to_json(symbol_dict)
        return result

    @staticmethod
    def _defining_symbol_to_result_dict(
        symbol_retriever: Any,
        defining_symbol: LanguageServerSymbol,
        include_body: bool,
        include_info: bool,
    ) -> dict[str, Any]:
        symbol_dict = dict(defining_symbol.to_dict(kind=True, relative_path=True, depth=0, body=include_body, body_location=True))
        if not include_body and include_info:
            if symbol_info := symbol_retriever.request_info_for_symbol(defining_symbol):
                symbol_dict["info"] = symbol_info
                symbol_dict.pop("name", None)
        return symbol_dict


class GetDiagnosticsForFileTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets diagnostics for a file, optionally restricted to a line range, grouped by file, severity, and containing symbol.
    """

    FILE_LEVEL_DIAGNOSTIC_BUCKET = "<file>"

    def apply(
        self,
        relative_path: str,
        start_line: int = 0,
        end_line: int = -1,
        min_severity: int = 4,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Gets diagnostics for a file. Diagnostics are grouped as `relative_path -> severity -> name_path -> diagnostics_results`.
        If a diagnostic cannot be mapped to a symbol, it is grouped under the special name path `<file>`.

        :param relative_path: the relative path to the file to inspect.
        :param start_line: the first 0-based line to include. Defaults to 0.
        :param end_line: the last 0-based line to include. Defaults to -1, which means until the end of the file.
        :param min_severity: minimum LSP severity to include, where 1=Error, 2=Warning, 3=Information, 4=Hint.
            Diagnostics with lower-or-equal numeric severity are returned.
        :param max_answer_chars: max result length; -1 for default
        :return: grouped diagnostics for the requested file.
        """
        self.project.ls_sync_file_system_changes()

        symbol_retriever = self.create_language_server_symbol_retriever()
        diagnostics = symbol_retriever.get_file_diagnostics(
            relative_file_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            min_severity=min_severity,
        )

        grouped_diagnostics = GroupedDiagnostics()
        for diagnostic in diagnostics:
            diag_range = diagnostic["range"]["start"]
            name_path = self.FILE_LEVEL_DIAGNOSTIC_BUCKET
            owner_symbol = symbol_retriever.find_diagnostic_owner_symbol(
                relative_file_path=relative_path,
                line=diag_range["line"],
                column=diag_range["character"],
            )
            if owner_symbol is not None:
                name_path = owner_symbol.get_name_path()
            grouped_diagnostics.add(relative_path, name_path, diagnostic)

        result = self._to_json(grouped_diagnostics.get_dict())
        return self._limit_length(result, max_answer_chars)


class GetDiagnosticsForSymbolTool(Tool, ToolMarkerSymbolicRead, ToolMarkerOptional):
    """
    Gets diagnostics for a symbol and, optionally, for symbols that reference it.
    """

    def apply(
        self,
        name_path: str,
        reference_file: str = "",
        check_symbol_references: bool = False,
        min_severity: int = 4,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Gets diagnostics for the specified symbol. When `check_symbol_references` is true, diagnostics for all
        referencing symbols are also included. The result is grouped as
        `relative_path -> severity -> name_path -> diagnostics_results`.

        :param name_path: the name path of the symbol to inspect.
        :param reference_file: optional file path used to disambiguate the symbol search.
        :param check_symbol_references: whether to additionally collect diagnostics for symbols that reference the symbol.
        :param min_severity: minimum LSP severity to include, where 1=Error, 2=Warning, 3=Information, 4=Hint.
            Diagnostics with lower-or-equal numeric severity are returned.
        :param max_answer_chars: max result length; -1 for default
        :return: grouped diagnostics for the requested symbol and, optionally, its referencing symbols.
        """
        self.project.ls_sync_file_system_changes()

        symbol_retriever = self.create_language_server_symbol_retriever()
        diagnostics_by_symbol = symbol_retriever.get_symbol_diagnostics(
            name_path=name_path,
            reference_file=reference_file or None,
            check_symbol_references=check_symbol_references,
            min_severity=min_severity,
        )

        grouped_diagnostics = GroupedDiagnostics()
        for symbol, diagnostics in diagnostics_by_symbol.items():
            relative_path = symbol.relative_path
            if relative_path is None:
                continue
            symbol_name_path = symbol.get_name_path()
            for diagnostic in diagnostics:
                grouped_diagnostics.add(relative_path, symbol_name_path, diagnostic)

        result = self._to_json(grouped_diagnostics.get_dict())
        return self._limit_length(result, max_answer_chars)


class ReplaceSymbolBodyTool(EditingToolWithDiagnostics):
    """
    Replaces the full definition of a symbol using the language server backend.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        r"""
        Replaces the body of the given symbol.

        IMPORTANT: Only replace symbol bodies if you have previously made a retrieval with include_body=True and thus know what
        constitutes the body!

        :param name_path: name path of the symbol whose body to replace
        :param relative_path: the relative path to the file containing the symbol
        :param body: the new symbol body. The symbol body is the definition of a symbol
            in the programming language, including e.g. the signature line for functions.
            Depending on the language, it may or may not include a preceding docstring or other preceding annotations.
        """
        with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
            code_editor = self.create_code_editor()
            code_editor.replace_body(
                name_path,
                relative_file_path=relative_path,
                body=body,
            )
            return diagnostics_context.format_result(SUCCESS_RESULT)


class InsertAfterSymbolTool(EditingToolWithDiagnostics):
    """
    Inserts content after the end of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Use this to insert code after a class/method/function definition.
        Don't use to insert after assignments (constants, fields).

        :param name_path: name path of the symbol after which to insert content
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted. The inserted code shall begin with the next line after
            the symbol.
        """
        with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
            code_editor = self.create_code_editor()
            code_editor.insert_after_symbol(name_path, relative_file_path=relative_path, body=body)
            return diagnostics_context.format_result(SUCCESS_RESULT)


class InsertBeforeSymbolTool(EditingToolWithDiagnostics):
    """
    Inserts content before the beginning of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given content before the beginning of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment; or
        a new import statement before the first symbol in the file.

        :param name_path: name path of the symbol before which to insert content
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted before the line in which the referenced symbol is defined
        """
        with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
            code_editor = self.create_code_editor()
            code_editor.insert_before_symbol(name_path, relative_file_path=relative_path, body=body)
            return diagnostics_context.format_result(SUCCESS_RESULT)


class RenameSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Renames a symbol throughout the codebase using language server refactoring capabilities.
    For JB, we use a separate tool.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> str:
        """
        Renames the symbol with the given `name_path` to `new_name` throughout the entire codebase.
        Note: for languages with method overloading, like Java, name_path may have to include a method's
        signature to uniquely identify a method.

        :param name_path: name path of the symbol to rename
        :param relative_path: the relative path to the file containing the symbol to rename
        :param new_name: the new name for the symbol
        :return: result summary indicating success or failure
        """
        self.project.ls_sync_file_system_changes()
        code_editor = self.create_ls_code_editor()
        status_message = code_editor.rename_symbol(name_path, relative_path=relative_path, new_name=new_name)
        return status_message


class SafeDeleteSymbol(Tool, ToolMarkerSymbolicEdit):
    def apply(
        self,
        name_path_pattern: str,
        relative_path: str,
    ) -> str:
        """
        Deletes the symbol if it is safe to do so (i.e., if there are no references to it)
        or returns a list of references to it.

        :param name_path_pattern: name path of the symbol to delete
        :param relative_path: the relative path to the file containing the symbol to delete
        """
        self.project.ls_sync_file_system_changes()

        ls_symbol_retriever = self.create_language_server_symbol_retriever()
        symbol = ls_symbol_retriever.find_unique(name_path_pattern, substring_matching=False, within_relative_path=relative_path)
        symbol_rel_path = symbol.relative_path
        assert symbol_rel_path is not None, f"Symbol {name_path_pattern} has no relative path, this is likely a bug."
        assert symbol_rel_path == relative_path, f"Symbol {name_path_pattern} is not in the expected relative path {relative_path}."
        symbol_name_path = symbol.get_name_path()

        symbol_line = symbol.line
        symbol_col = symbol.column
        assert symbol_line is not None and symbol_col is not None, (
            f"Symbol {name_path_pattern} has no identifier position, this is likely a bug."
        )
        lang_server = ls_symbol_retriever.get_language_server(symbol_rel_path)
        references_locations = lang_server.request_references(symbol_rel_path, symbol_line, symbol_col)
        file_to_lines: dict[str, list[int]] = defaultdict(list)
        if references_locations:
            for ref_loc in references_locations:
                ref_relative_path = ref_loc.get("relativePath")
                if ref_relative_path is None:
                    continue
                file_to_lines[ref_relative_path].append(ref_loc["range"]["start"]["line"])
        if file_to_lines:
            return f"Cannot delete, the symbol {symbol_name_path} is referenced in: {self._to_json(file_to_lines)}"
        code_editor = self.create_ls_code_editor()
        code_editor.delete_symbol(symbol_name_path, relative_file_path=symbol_rel_path)
        return SUCCESS_RESULT
