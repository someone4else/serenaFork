"""Unit tests for the symbols overview enrichment (no language server required).

The overview post-processing sees through structural containers, appends signatures and
inheritance clauses to symbol names and relabels constructors; these tests exercise that logic
on hand-built symbol trees.
"""

from typing import Any

from serena.symbol import LanguageServerSymbol
from serena.tools.symbol_tools import (
    _count_singleton_wrapper_depth,
    _extract_inheritance_from_source,
    _flatten_transparent_containers_to_dicts,
)
from solidlsp.ls_types import SymbolKind


def _symbol(
    name: str,
    kind: SymbolKind,
    children: list[dict[str, Any]] | None = None,
    line: int = 0,
    detail: str | None = None,
    relative_path: str = "Foo.cs",
) -> dict[str, Any]:
    """Builds a raw symbol dict of the shape the language servers deliver."""
    symbol: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "children": children or [],
        "selectionRange": {"start": {"line": line, "character": 0}, "end": {"line": line, "character": len(name)}},
        "location": {"relativePath": relative_path},
    }
    if detail is not None:
        symbol["detail"] = detail
    return symbol


def _flatten(root: dict[str, Any], depth: int, project: Any = None) -> list[LanguageServerSymbol.OutputDict]:
    return _flatten_transparent_containers_to_dicts(
        LanguageServerSymbol(root),
        depth=depth,
        child_inclusion_predicate=lambda s: not s.is_low_level(),
        project=project,
    )


class _StubProject:
    """Stands in for a Project, serving a single source file."""

    def __init__(self, content: str) -> None:
        self._content = content

    def read_file(self, relative_path: str) -> str:
        return self._content


class TestExtractInheritanceFromSource:
    @staticmethod
    def _extract(source: str, name: str, kind: SymbolKind = SymbolKind.Class, line: int = 0) -> str | None:
        symbol = LanguageServerSymbol(_symbol(name, kind, line=line))
        return _extract_inheritance_from_source(symbol, _StubProject(source))

    def test_base_class_and_interface(self):
        assert self._extract("public class Foo : Bar, IBaz\n{\n}\n", "Foo") == ": Bar, IBaz"

    def test_no_base_types_yields_none(self):
        assert self._extract("public class Foo\n{\n}\n", "Foo") is None

    def test_generic_parameters_and_where_constraint_are_stripped(self):
        source = "public class Foo<T> : Bar<T> where T : IComparable\n{\n}\n"
        assert self._extract(source, "Foo") == ": Bar<T>"

    def test_nested_generic_base_type(self):
        source = "public class Foo : Bar<List<int>, Dictionary<string, int>>\n{\n}\n"
        assert self._extract(source, "Foo") == ": Bar<List<int>, Dictionary<string, int>>"

    def test_record_class_and_record_struct(self):
        assert self._extract("public record class Foo : Bar\n{\n}\n", "Foo") == ": Bar"
        assert self._extract("public record struct Foo : IFoo\n{\n}\n", "Foo", kind=SymbolKind.Struct) == ": IFoo"

    def test_interface_declaration(self):
        assert self._extract("public interface IFoo : IDisposable\n{\n}\n", "IFoo", kind=SymbolKind.Interface) == ": IDisposable"

    def test_multi_line_declaration(self):
        source = "public sealed class Foo\n    : Bar,\n      IBaz\n{\n}\n"
        assert self._extract(source, "Foo") == ": Bar, IBaz"

    def test_declaration_of_a_different_symbol_is_not_matched(self):
        # The regex is anchored on the symbol's own name, so a neighbouring declaration is ignored.
        source = "public class Other : Bar\n{\n}\n\npublic class Foo\n{\n}\n"
        assert self._extract(source, "Foo", line=4) is None

    def test_symbol_line_beyond_end_of_file(self):
        assert self._extract("public class Foo : Bar\n", "Foo", line=99) is None


class TestCountSingletonWrapperDepth:
    @staticmethod
    def _children(*roots: dict[str, Any]) -> list[LanguageServerSymbol]:
        return [LanguageServerSymbol(r) for r in roots]

    def test_lone_class_with_members_is_a_wrapper(self):
        children = self._children(_symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)]))
        assert _count_singleton_wrapper_depth(children, lambda s: not s.is_low_level()) == 1

    def test_lone_class_without_members_is_not_a_wrapper(self):
        children = self._children(_symbol("Foo", SymbolKind.Class))
        assert _count_singleton_wrapper_depth(children, lambda s: not s.is_low_level()) == 0

    def test_several_children_are_not_a_wrapper(self):
        children = self._children(
            _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)]),
            _symbol("Baz", SymbolKind.Class, [_symbol("Qux", SymbolKind.Method)]),
        )
        assert _count_singleton_wrapper_depth(children, lambda s: not s.is_low_level()) == 0

    def test_transparent_child_is_left_to_the_caller(self):
        children = self._children(_symbol("Inner", SymbolKind.Namespace, [_symbol("Foo", SymbolKind.Class)]))
        assert _count_singleton_wrapper_depth(children, lambda s: not s.is_low_level()) == 0


class TestFlattenTransparentContainers:
    def test_lone_class_in_a_namespace_shows_its_members_at_depth_0(self):
        # A typical C# file: namespace -> class -> methods. Neither the namespace nor the lone class
        # is meaningful on its own, so at depth 0 the user must still see the methods.
        root = _symbol(
            "MyApp",
            SymbolKind.Namespace,
            [_symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method), _symbol("Baz", SymbolKind.Method)])],
        )
        (result,) = _flatten(root, depth=0)
        assert result["name"] == "MyApp"
        (class_dict,) = result["children"]
        assert class_dict["name"] == "Foo"
        assert [c["name"] for c in class_dict["children"]] == ["Bar", "Baz"]

    def test_several_classes_in_a_namespace_stop_at_the_classes(self):
        # Without a singleton wrapper only the namespace level is compensated, so the members of the
        # classes stay hidden at depth 0.
        root = _symbol(
            "MyApp",
            SymbolKind.Namespace,
            [
                _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)]),
                _symbol("Qux", SymbolKind.Class, [_symbol("Quux", SymbolKind.Method)]),
            ],
        )
        (result,) = _flatten(root, depth=0)
        assert [c["name"] for c in result["children"]] == ["Foo", "Qux"]
        assert all("children" not in c for c in result["children"])

    def test_chain_of_namespaces_is_collapsed(self):
        root = _symbol(
            "A",
            SymbolKind.Namespace,
            [_symbol("A.B", SymbolKind.Namespace, [_symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)])])],
        )
        (result,) = _flatten(root, depth=0)
        # the outer namespace is dropped entirely
        assert result["name"] == "A.B"
        assert [c["name"] for c in result["children"]] == ["Foo"]

    def test_non_transparent_root_keeps_the_requested_depth(self):
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)])
        (result,) = _flatten(root, depth=0)
        assert result["name"] == "Foo"
        assert "children" not in result


class TestSymbolDictEnrichment:
    def test_method_detail_is_appended_to_the_name(self):
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Add", SymbolKind.Method, detail="(int, int) : int")], line=0)
        (result,) = _flatten(root, depth=1)
        assert result["children"][0]["name"] == "Add (int, int) : int"

    def test_constructor_is_relabelled(self):
        # C# language servers emit constructors as methods named like the enclosing class.
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Foo", SymbolKind.Method), _symbol("Bar", SymbolKind.Method)])
        (result,) = _flatten(root, depth=1)
        assert [(c["name"], c["kind"]) for c in result["children"]] == [("Foo", "Constructor"), ("Bar", "Method")]

    def test_inheritance_is_appended_to_the_class_name(self):
        project = _StubProject("namespace MyApp;\n\npublic class Foo : Bar, IBaz\n{\n}\n")
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method, line=4)], line=2)
        (result,) = _flatten(root, depth=1, project=project)
        assert result["name"] == "Foo : Bar, IBaz"

    def test_no_project_means_no_inheritance_lookup(self):
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)])
        (result,) = _flatten(root, depth=1, project=None)
        assert result["name"] == "Foo"
