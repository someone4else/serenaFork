"""
Language server-related tools
"""

import os
import re
from collections.abc import Callable, Sequence

from serena.symbol import LanguageServerSymbol, LanguageServerSymbolDictGrouper
from serena.tools import (
    SUCCESS_RESULT,
    Tool,
    ToolMarkerSymbolicEdit,
    ToolMarkerSymbolicRead,
)
from serena.tools.tools_base import ToolMarkerOptional
from solidlsp.ls_types import SymbolKind

# Symbol kinds that are purely structural containers (namespaces, modules, packages).
# In languages like C#/Java these wrap the "real" symbols (classes, interfaces, etc.)
# and provide no useful information at depth=0 on their own.
_TRANSPARENT_CONTAINER_KINDS = frozenset({SymbolKind.Namespace, SymbolKind.Module, SymbolKind.Package})

# Symbol kinds for which the LSP 'detail' field should be appended to the name in the
# symbols overview output. For callables this is the signature (e.g. "(int a, int b): int").
_DETAIL_INCLUDED_KINDS = frozenset({SymbolKind.Method, SymbolKind.Function, SymbolKind.Constructor})

# Symbol kinds for which hover-based information (e.g. inheritance) should be retrieved
# and appended to the name in the symbols overview output.
_HOVER_ENRICHED_KINDS = frozenset({SymbolKind.Class, SymbolKind.Interface, SymbolKind.Struct})


class RestartLanguageServerTool(Tool, ToolMarkerOptional):
    """Restarts the language server, may be necessary when edits not through Serena happen."""

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

    def apply(self, relative_path: str, depth: int = 0, max_answer_chars: int = -1) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param depth: depth up to which descendants of top-level symbols shall be retrieved
            (e.g. 1 retrieves immediate children). Default 0.
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :return: a JSON object containing symbols grouped by kind in a compact format.
        """
        result = self.get_symbol_overview(relative_path, depth=depth)
        compact_result = self.symbol_dict_grouper.group(result)
        result_json_str = self._to_json(compact_result)
        return self._limit_length(result_json_str, max_answer_chars)

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
                f"Cannot extract symbols from file {relative_path}. Active languages: {[l.value for l in self.agent.get_active_lsp_languages()]}"
            )

        symbols = symbol_retriever.get_symbol_overview(relative_path)[relative_path]

        def child_inclusion_predicate(s: LanguageServerSymbol) -> bool:
            return not s.is_low_level()

        # Batch hover requests for Class/Interface/Struct symbols to retrieve inheritance info.
        hover_symbols = _collect_symbols_for_hover(symbols, child_inclusion_predicate)
        hover_info: dict[str, str | None] | None = None
        if hover_symbols:
            raw_hover = symbol_retriever.request_info_for_symbol_batch(hover_symbols)
            hover_info = {_hover_key(sym): info for sym, info in raw_hover.items()}

        symbol_dicts: list[LanguageServerSymbol.OutputDict] = []
        for symbol in symbols:
            symbol_dicts.extend(
                _flatten_transparent_containers_to_dicts(
                    symbol,
                    depth=depth,
                    child_inclusion_predicate=child_inclusion_predicate,
                    hover_info=hover_info,
                )
            )
        return symbol_dicts


def _hover_key(symbol: LanguageServerSymbol) -> str:
    """Return a stable string key for a symbol suitable for use in the hover_info dict.

    Uses the symbol's file location (path:line:column) for uniqueness, falling back to
    name+kind when location attributes are absent.  The fallback is only reached for
    symbols that lack a recorded location; ``request_info_for_symbol_batch`` skips such
    symbols and omits them from the result dict, so the fallback key will never match an
    entry in the hover_info dict (the lookup returns ``None``), which is the correct
    behaviour for symbols without location data.
    """
    path = symbol.relative_path
    line = symbol.line
    column = symbol.column
    if path is not None and line is not None and column is not None:
        return f"{path}:{line}:{column}"
    return f"{symbol.name}:{symbol.symbol_kind.name}"


def _collect_symbols_for_hover(
    symbols: list[LanguageServerSymbol],
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
) -> list[LanguageServerSymbol]:
    """Recursively collect all Class/Interface/Struct symbols from the symbol tree.

    These are the symbols for which hover-based inheritance info will be requested.
    Only symbols passing the inclusion predicate are visited/collected.

    :param symbols: list of top-level (or current-level) symbols to inspect
    :param child_inclusion_predicate: predicate that filters which children are traversed
    :return: flat list of symbols whose kind is in ``_HOVER_ENRICHED_KINDS``
    """
    result: list[LanguageServerSymbol] = []
    for sym in symbols:
        if sym.symbol_kind in _HOVER_ENRICHED_KINDS:
            result.append(sym)
        children = [c for c in sym.iter_children() if child_inclusion_predicate(c)]
        result.extend(_collect_symbols_for_hover(children, child_inclusion_predicate))
    return result


def _extract_inheritance_from_hover(hover_text: str) -> str | None:
    """Extract the C#-style inheritance suffix from hover text.

    Given hover text like ``class MyClass : BaseClass, IInterface``, returns
    ``: BaseClass, IInterface``.  Returns ``None`` if no inheritance marker is found.

    Only the first non-empty line of the (stripped) hover text is examined, so
    multi-line docstrings or signatures do not pollute the result.

    :param hover_text: raw hover text returned by the language server
    :return: the inheritance suffix (e.g. ``: BaseClass, IInterface``) or ``None``
    """
    # Strip markdown code fences (e.g. ```csharp\n...\n```)
    text = re.sub(r"```[^\n]*\n?", "", hover_text)
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Find the C#-style ' : ' inheritance marker
        idx = line.find(" : ")
        if idx != -1:
            return line[idx:].strip()
        # Only inspect the very first non-empty line
        break
    return None


def _count_singleton_wrapper_depth(
    children: list[LanguageServerSymbol],
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
) -> int:
    """
    Returns 1 if the given children list represents a singleton non-transparent
    wrapper (e.g. a lone Class inside a Namespace in a typical C# file), 0 otherwise.

    A singleton wrapper is a non-transparent symbol that is the *only* visible child
    at its level and itself has visible descendants.  In that case the symbol acts as
    a structural wrapper and should not consume the user's depth budget.

    Transparent containers are deliberately excluded here because they are already
    handled by the recursion in ``_flatten_transparent_containers_to_dicts``.
    The compensation is capped at +1: Class is a meaningful symbol (unlike Namespace),
    so we do not recurse further.
    """
    if len(children) != 1:
        return 0

    only_child = children[0]

    # Transparent containers are handled by the main function's recursion.
    if only_child.symbol_kind in _TRANSPARENT_CONTAINER_KINDS:
        return 0

    # Check whether the sole non-transparent child itself has visible descendants.
    has_grandchildren = any(child_inclusion_predicate(c) for c in only_child.iter_children())
    if not has_grandchildren:
        return 0  # Leaf symbol - no extra depth needed.

    # The single non-transparent child (e.g. a Class) acts as a wrapper -> +1.
    return 1


def _enhance_symbol_dict(
    symbol: LanguageServerSymbol,
    output_dict: LanguageServerSymbol.OutputDict,
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
    parent_symbol: LanguageServerSymbol | None = None,
    hover_info: dict[str, str | None] | None = None,
) -> None:
    """
    Recursively enhances an OutputDict (produced by ``to_dict``) in-place with:

    1. **Signatures** – for callable symbols (Method, Function, Constructor) the LSP
       ``detail`` field (e.g. ``(int a, int b): int``) is appended to the ``name``
       entry so the LLM sees full signatures instead of bare names.

    2. **Inheritance info** – for Class/Interface/Struct symbols the hover text
       (pre-fetched via ``request_info_for_symbol_batch``) is parsed to extract
       the C#-style ``: BaseClass, IInterface`` suffix and appended to the name.

    3. **Constructor detection** – in C# (and similar OO languages) the language
       server returns constructors as ``Method`` symbols whose name matches the
       enclosing class name.  This function relabels those entries to
       ``kind="Constructor"`` so the LLM can distinguish them from regular methods.

    4. **Recursion** – the function walks the *symbol* tree (``LanguageServerSymbol``)
       in parallel with the *dict* tree so that each enhancement is applied with
       access to the full symbol metadata (including the LSP ``detail`` field and
       the parent reference).

    :param symbol: the ``LanguageServerSymbol`` whose data was used to build
        ``output_dict``.
    :param output_dict: the dict produced by ``symbol.to_dict()``, modified in-place.
    :param child_inclusion_predicate: the same predicate that was passed to
        ``to_dict`` so that the child symbol list matches the ``children`` list in
        the dict.
    :param parent_symbol: the parent ``LanguageServerSymbol``, used for constructor
        detection (``None`` for root-level symbols).
    :param hover_info: optional pre-fetched hover info dict (location key → text), used to
        enrich Class/Interface/Struct symbols with inheritance information.
    """
    # --- 1. Append LSP detail (signature) to the name for callable kinds ---
    if symbol.symbol_kind in _DETAIL_INCLUDED_KINDS:
        detail: str = symbol.symbol_root.get("detail", "") or ""
        if detail and "name" in output_dict:
            output_dict["name"] = f"{symbol.name} {detail}"

    # --- 2. Append hover-based inheritance info for Class/Interface/Struct ---
    elif symbol.symbol_kind in _HOVER_ENRICHED_KINDS and hover_info is not None:
        info = hover_info.get(_hover_key(symbol))
        if info:
            inheritance = _extract_inheritance_from_hover(info)
            if inheritance and "name" in output_dict:
                output_dict["name"] = f"{symbol.name} {inheritance}"

    # --- 3. Detect C# constructors and relabel them ---
    # C# constructors are emitted as Method symbols with the same name as the
    # enclosing class.  We relabel them as "Constructor" for clarity.
    if (
        symbol.symbol_kind == SymbolKind.Method
        and "kind" in output_dict
        and parent_symbol is not None
        and parent_symbol.symbol_kind == SymbolKind.Class
        and parent_symbol.name == symbol.name
    ):
        output_dict["kind"] = "Constructor"

    # --- 4. Recurse into children ---
    if "children" not in output_dict:
        return
    child_dicts = output_dict["children"]
    # The child dicts correspond 1-to-1 (in order) to the children that passed the
    # inclusion predicate, because to_dict() builds them in iteration order.
    child_symbols = [c for c in symbol.iter_children() if child_inclusion_predicate(c)]
    for child_sym, child_dict in zip(child_symbols, child_dicts, strict=True):
        _enhance_symbol_dict(child_sym, child_dict, child_inclusion_predicate, parent_symbol=symbol, hover_info=hover_info)


def _flatten_transparent_containers_to_dicts(
    symbol: LanguageServerSymbol,
    depth: int,
    child_inclusion_predicate: Callable[[LanguageServerSymbol], bool],
    hover_info: dict[str, str | None] | None = None,
) -> list[LanguageServerSymbol.OutputDict]:
    """
    Converts a symbol to one or more OutputDicts, automatically "seeing through"
    purely structural containers (Namespace, Module, Package) so that the user
    always sees meaningful content even at depth=0.

    For transparent containers the strategy is:
      - The container itself is emitted with an effective depth that compensates
        for all "wrapper" levels between the container and the first level of real
        content.  Specifically:
          * ``+1`` for the transparent container itself (it is structural, not
            meaningful, and should not consume the user's depth budget).
          * ``+1`` (via ``_count_singleton_wrapper_depth``) if the transparent
            container's only child is a *non-transparent* singleton wrapper (e.g.
            a lone Class in a C# Namespace).  That Class also acts purely as a
            structural wrapper in single-class files and should not consume the
            user's depth budget either.
      - If a transparent container has *exactly one* child and that child is also
        transparent, we recurse and flatten further so the user doesn't see a
        chain of nested namespaces.

    For all other symbols the behaviour is identical to the original code.
    """
    if symbol.symbol_kind not in _TRANSPARENT_CONTAINER_KINDS:
        # Normal symbol - emit as before, then apply enhancements
        output_dict = symbol.to_dict(
            name_path=False,
            name=True,
            depth=depth,
            kind=True,
            relative_path=False,
            location=False,
            child_inclusion_predicate=child_inclusion_predicate,
        )
        _enhance_symbol_dict(symbol, output_dict, child_inclusion_predicate, hover_info=hover_info)
        return [output_dict]

    # --- Transparent container handling ---

    children = [c for c in symbol.iter_children() if child_inclusion_predicate(c)]

    # If the container has exactly one child that is *also* a transparent container,
    # skip the outer wrapper entirely and recurse on the inner one.
    # This collapses chains like  Namespace(A) -> Namespace(A.B) -> Class(Foo)
    # into a single entry "Namespace A.B -> { Class Foo }".
    if len(children) == 1 and children[0].symbol_kind in _TRANSPARENT_CONTAINER_KINDS:
        return _flatten_transparent_containers_to_dicts(
            children[0],
            depth=depth,
            child_inclusion_predicate=child_inclusion_predicate,
            hover_info=hover_info,
        )

    # The container has substantive children - emit it with depth + 1 + extra_depth
    # so that neither the transparent container nor a singleton non-transparent
    # wrapper (e.g. a lone Class in a C# file) consumes the user's depth budget.
    extra_depth = _count_singleton_wrapper_depth(children, child_inclusion_predicate)
    effective_depth = depth + 1 + extra_depth
    output_dict = symbol.to_dict(
        name_path=False,
        name=True,
        depth=effective_depth,
        kind=True,
        relative_path=False,
        location=False,
        child_inclusion_predicate=child_inclusion_predicate,
    )
    _enhance_symbol_dict(symbol, output_dict, child_inclusion_predicate, hover_info=hover_info)
    return [output_dict]


class FindSymbolTool(Tool, ToolMarkerSymbolicRead):
    """
    Performs a global (or local) search using the language server backend.
    """

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
        max_answer_chars: int = -1,
    ) -> str:
        """
        Retrieves information on all symbols/code entities (classes, methods, etc.) based on the given name path pattern.
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
            for the case where the symbol is a class, this will return its methods). Default 0.
        :param relative_path: Optional. Restrict search to this file or directory. If None, searches entire codebase.
            If a directory is passed, the search will be restricted to the files in that directory.
            If a file is passed, the search will be restricted to that file.
            If you have some knowledge about the codebase, you should use this parameter, as it will significantly
            speed up the search as well as reduce the number of results.
        :param include_body: whether to include the symbol's source code. Use judiciously.
        :param include_info: whether to include additional info (hover-like, typically including docstring and signature),
            about the symbol (ignored if include_body is True). Info is never included for child symbols.
            Note: Depending on the language, this can be slow (e.g., C/C++).
        :param include_kinds: List of LSP symbol kind integers to include.
            If not provided, all kinds are included.
        :param exclude_kinds: Optional. List of LSP symbol kind integers to exclude. Takes precedence over `include_kinds`.
            If not provided, no kinds are excluded.
        :param substring_matching: If True, use substring matching for the last element of the pattern, such that
            "Foo/get" would match "Foo/getValue" and "Foo/getData".
        :param max_answer_chars: Max characters for the JSON result. If exceeded, no content is returned.
            -1 means the default value from the config will be used.
        :return: a list of symbols (with locations) matching the name.
        """
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
        symbol_dicts = [dict(s.to_dict(kind=True, relative_path=True, body_location=True, depth=depth, body=include_body)) for s in symbols]
        if not include_body and include_info:
            info_by_symbol = symbol_retriever.request_info_for_symbol_batch(symbols)
            for s, s_dict in zip(symbols, symbol_dicts, strict=True):
                if symbol_info := info_by_symbol.get(s):
                    s_dict["info"] = symbol_info
                    s_dict.pop("name", None)  # name is included in the info
        result = self._to_json(symbol_dicts)
        return self._limit_length(result, max_answer_chars)


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

        :param name_path: for finding the symbol to find references for, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol for which to find references.
            Note that here you can't pass a directory but must pass a file.
        :param include_kinds: same as in the `find_symbol` tool.
        :param exclude_kinds: same as in the `find_symbol` tool.
        :param max_answer_chars: same as in the `find_symbol` tool.
        :return: a list of JSON objects with the symbols referencing the requested symbol
        """
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

        result = self.symbol_dict_grouper.group(reference_dicts)  # type: ignore

        result_json = self._to_json(result)
        return self._limit_length(result_json, max_answer_chars)


class ReplaceSymbolBodyTool(Tool, ToolMarkerSymbolicEdit):
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
        Replaces the body of the symbol with the given `name_path`.

        The tool shall be used to replace symbol bodies that have been previously retrieved
        (e.g. via `find_symbol`).
        IMPORTANT: Do not use this tool if you do not know what exactly constitutes the body of the symbol.

        :param name_path: for finding the symbol to replace, same logic as in the `find_symbol` tool.
        :param relative_path: the relative path to the file containing the symbol
        :param body: the new symbol body. The symbol body is the definition of a symbol
            in the programming language, including e.g. the signature line for functions.
            IMPORTANT: The body does NOT include any preceding docstrings/comments or imports, in particular.
        """
        code_editor = self.create_code_editor()
        code_editor.replace_body(
            name_path,
            relative_file_path=relative_path,
            body=body,
        )
        return SUCCESS_RESULT


class InsertAfterSymbolTool(Tool, ToolMarkerSymbolicEdit):
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
        Inserts the given body/content after the end of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment.

        :param name_path: name path of the symbol after which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted. The inserted code shall begin with the next line after
            the symbol.
        """
        code_editor = self.create_code_editor()
        code_editor.insert_after_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class InsertBeforeSymbolTool(Tool, ToolMarkerSymbolicEdit):
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

        :param name_path: name path of the symbol before which to insert content (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted before the line in which the referenced symbol is defined
        """
        code_editor = self.create_code_editor()
        code_editor.insert_before_symbol(name_path, relative_file_path=relative_path, body=body)
        return SUCCESS_RESULT


class RenameSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Renames a symbol throughout the codebase using language server refactoring capabilities.
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

        :param name_path: name path of the symbol to rename (definitions in the `find_symbol` tool apply)
        :param relative_path: the relative path to the file containing the symbol to rename
        :param new_name: the new name for the symbol
        :return: result summary indicating success or failure
        """
        code_editor = self.create_code_editor()
        status_message = code_editor.rename_symbol(name_path, relative_file_path=relative_path, new_name=new_name)
        return status_message
