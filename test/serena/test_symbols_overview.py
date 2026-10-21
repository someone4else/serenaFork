"""Tests for the enriched symbols overview of the Serena fork.

The overview sees through structural containers, appends signatures and base type clauses to symbol names and
relabels constructors. The unit tests exercise this on hand-built symbol trees; the language server tests check the
overview tool end to end.
"""

import json
import logging
import pickle
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from serena.agent import SerenaAgent
from serena.config.serena_config import ProjectConfig, RegisteredProject, SerenaConfig
from serena.project import Project
from serena.repl.api.lsp_api import LspSymbolCollection, SymbolOutputParams
from serena.symbol import LanguageServerSymbol
from serena.symbol_overview_enrichment import BaseTypeClauseExtractor, EnrichedSymbolsOverviewRenderer, SymbolOverviewEnricher
from serena.tools import GetSymbolsOverviewTool
from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_types import SymbolKind
from test.conftest import get_repo_path, language_server_tests_enabled


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


def _is_not_low_level(symbol: LanguageServerSymbol) -> bool:
    return not symbol.is_low_level()


def _overview_params(depth: int) -> SymbolOutputParams:
    return SymbolOutputParams(
        name_path=False,
        name=True,
        depth=depth,
        kind=True,
        relative_path=False,
        location=False,
        child_inclusion_predicate=_is_not_low_level,
    )


def _overview(root: dict[str, Any], depth: int, source: str = "") -> list[LanguageServerSymbol.OutputDict]:
    symbol = LanguageServerSymbol(root)
    return SymbolOverviewEnricher.from_source([symbol], source).to_dicts(symbol, _overview_params(depth))


class TestBaseTypeClauseExtractor:
    @staticmethod
    def _extract(source: str, name: str, kind: SymbolKind = SymbolKind.Class, line: int = 0) -> str | None:
        return BaseTypeClauseExtractor(source).extract(LanguageServerSymbol(_symbol(name, kind, line=line)))

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

    def test_brace_on_the_same_line(self):
        assert self._extract("public class Foo : Bar {\n}\n", "Foo") == ": Bar"

    def test_cpp_access_specifier_is_kept(self):
        assert self._extract("class Foo : public Bar {\n};\n", "Foo") == ": public Bar"

    def test_only_languages_introducing_base_types_with_a_colon_are_considered(self):
        # In Python the colon opens the class body rather than a base type list, and a brace further down the body
        # (a dict literal) must not make the body count as a base type clause
        assert BaseTypeClauseExtractor.applies_to("Foo.cs")
        assert BaseTypeClauseExtractor.applies_to("foo.hpp")
        assert not BaseTypeClauseExtractor.applies_to("models.py")
        assert not BaseTypeClauseExtractor.applies_to("Foo.java")

    def test_kotlin_primary_constructor_is_skipped(self):
        assert self._extract("class Foo(val x: Int, y: String) : Bar(x), Baz {\n}\n", "Foo") == ": Bar(x), Baz"

    def test_swift_generic_constraint_is_skipped(self):
        assert self._extract("struct Box<T: Hashable>: Container {\n}\n", "Box", kind=SymbolKind.Struct) == ": Container"

    def test_cpp_specialization_with_scope_operators(self):
        assert self._extract("class Foo<std::string> : public Bar<std::string> {\n};\n", "Foo") == ": public Bar<std::string>"

    def test_cpp_final_class(self):
        assert self._extract("class Foo final : public Bar {\n};\n", "Foo") == ": public Bar"

    def test_csharp_positional_record(self):
        assert self._extract("public record Foo(int X, string Y) : Bar(X)\n{\n}\n", "Foo") == ": Bar(X)"

    def test_kotlin_class_without_base_types(self):
        assert self._extract("class Foo(val x: Int) {\n}\n", "Foo") is None

    def test_declaration_without_a_brace_is_skipped(self):
        assert self._extract("public class Foo : Bar\n" + "\n" * BaseTypeClauseExtractor.MAX_DECLARATION_LINES, "Foo") is None

    def test_declaration_of_a_different_symbol_is_not_matched(self):
        source = "public class Other : Bar\n{\n}\n\npublic class Foo\n{\n}\n"
        assert self._extract(source, "Foo", line=4) is None

    def test_symbol_line_beyond_end_of_file(self):
        assert self._extract("public class Foo : Bar\n", "Foo", line=99) is None

    def test_methods_are_not_enriched(self):
        assert self._extract("public void Foo() : Bar\n{\n}\n", "Foo", kind=SymbolKind.Method) is None


class TestStructuralContainers:
    def test_lone_class_in_a_namespace_shows_its_members_at_depth_0(self):
        # A typical C# file: namespace -> class -> methods. Neither the namespace nor the lone class is meaningful
        # on its own, so at depth 0 the user must still see the methods.
        root = _symbol(
            "MyApp",
            SymbolKind.Namespace,
            [_symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method), _symbol("Baz", SymbolKind.Method)])],
        )
        (result,) = _overview(root, depth=0)
        assert result["name"] == "MyApp"
        (class_dict,) = result["children"]
        assert class_dict["name"] == "Foo"
        assert [c["name"] for c in class_dict["children"]] == ["Bar", "Baz"]

    def test_several_classes_in_a_namespace_stop_at_the_classes(self):
        root = _symbol(
            "MyApp",
            SymbolKind.Namespace,
            [
                _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)]),
                _symbol("Qux", SymbolKind.Class, [_symbol("Quux", SymbolKind.Method)]),
            ],
        )
        (result,) = _overview(root, depth=0)
        assert [c["name"] for c in result["children"]] == ["Foo", "Qux"]
        assert all("children" not in c for c in result["children"])

    def test_lone_class_without_members_is_not_expanded_further(self):
        root = _symbol("MyApp", SymbolKind.Namespace, [_symbol("Foo", SymbolKind.Class)])
        (result,) = _overview(root, depth=0)
        assert [c["name"] for c in result["children"]] == ["Foo"]

    def test_chain_of_namespaces_is_collapsed(self):
        root = _symbol(
            "A",
            SymbolKind.Namespace,
            [_symbol("A.B", SymbolKind.Namespace, [_symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)])])],
        )
        (result,) = _overview(root, depth=0)
        assert result["name"] == "A.B"
        assert [c["name"] for c in result["children"]] == ["Foo"]
        assert [c["name"] for c in result["children"][0]["children"]] == ["Bar"]

    def test_non_transparent_root_keeps_the_requested_depth(self):
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Bar", SymbolKind.Method)])
        (result,) = _overview(root, depth=0)
        assert result["name"] == "Foo"
        assert "children" not in result


class TestNameEnrichment:
    def test_method_detail_is_appended_to_the_name(self):
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Add", SymbolKind.Method, detail="(int, int) : int")])
        (result,) = _overview(root, depth=1)
        assert result["children"][0]["name"] == "Add (int, int) : int"

    def test_detail_that_already_contains_the_name_is_not_duplicated(self):
        # some C# servers report the full signature in 'detail' rather than just its tail
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Run", SymbolKind.Method, detail="Run(CancellationToken) : Task")])
        (result,) = _overview(root, depth=1)
        assert result["children"][0]["name"] == "Run(CancellationToken) : Task"

    def test_detail_of_a_longer_name_is_still_appended(self):
        # "Get" is a prefix of "GetAll", but the detail describes a different identifier
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Get", SymbolKind.Method, detail="GetAll() : Task")])
        (result,) = _overview(root, depth=1)
        assert result["children"][0]["name"] == "Get GetAll() : Task"

    def test_constructor_is_relabelled(self):
        # C# language servers emit constructors as methods named like the enclosing class
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Foo", SymbolKind.Method), _symbol("Bar", SymbolKind.Method)])
        (result,) = _overview(root, depth=1)
        assert [(c["name"], c["kind"]) for c in result["children"]] == [("Foo", "Constructor"), ("Bar", "Method")]

    def test_base_types_are_appended_to_the_class_name(self):
        source = "namespace MyApp;\n\npublic class Foo : Bar, IBaz\n{\n    public void Run() { }\n}\n"
        root = _symbol("Foo", SymbolKind.Class, [_symbol("Run", SymbolKind.Method, line=4)], line=2)
        (result,) = _overview(root, depth=1, source=source)
        assert result["name"] == "Foo : Bar, IBaz"

    def test_base_types_of_a_class_inside_a_namespace(self):
        source = "namespace MyApp\n{\n    public class Foo : IBaz\n    {\n        public void Run() { }\n    }\n}\n"
        root = _symbol(
            "MyApp", SymbolKind.Namespace, [_symbol("Foo", SymbolKind.Class, [_symbol("Run", SymbolKind.Method, line=4)], line=2)]
        )
        (result,) = _overview(root, depth=0, source=source)
        assert result["children"][0]["name"] == "Foo : IBaz"


class TestEnricherForFile:
    @staticmethod
    def _class_symbol() -> LanguageServerSymbol:
        return LanguageServerSymbol(_symbol("Foo", SymbolKind.Class, [_symbol("Run", SymbolKind.Method, line=3)], line=1))

    def test_file_in_another_encoding_than_the_projects_is_read_with_the_detected_one(self, tmp_path):
        # a legacy C# file with a Russian comment, while the project is configured for UTF-8
        source = "// Класс для проверки кодировки\npublic class Foo : Bar\n{\n    public void Run() { }\n}\n"
        (tmp_path / "Foo.cs").write_bytes(source.encode("cp1251"))
        symbol = self._class_symbol()
        enricher = SymbolOverviewEnricher.for_file([symbol], "Foo.cs", str(tmp_path / "Foo.cs"), "utf-8")
        (result,) = enricher.to_dicts(symbol, _overview_params(depth=1))
        assert result["name"] == "Foo : Bar"

    def test_files_of_other_languages_are_not_read(self, tmp_path):
        symbol = LanguageServerSymbol(_symbol("Foo", SymbolKind.Class, line=0, relative_path="foo.py"))
        enricher = SymbolOverviewEnricher.for_file([symbol], "foo.py", str(tmp_path / "missing.py"), "utf-8")
        (result,) = enricher.to_dicts(symbol, _overview_params(depth=0))
        assert result["name"] == "Foo"

    def test_unreadable_file_yields_an_overview_without_base_types(self, tmp_path):
        symbol = self._class_symbol()
        enricher = SymbolOverviewEnricher.for_file([symbol], "Foo.cs", str(tmp_path / "missing.cs"), "utf-8")
        (result,) = enricher.to_dicts(symbol, _overview_params(depth=1))
        assert result["name"] == "Foo"


class TestEnrichedSymbolsOverviewRenderer:
    _AGENT: Any = SimpleNamespace(serena_config=SimpleNamespace(default_max_tool_answer_chars=100_000))

    def _collection(self) -> LspSymbolCollection:
        source = "namespace MyApp\n{\n    public class Foo : IBaz\n    {\n        public Foo() { }\n    }\n}\n"
        root = LanguageServerSymbol(
            _symbol("MyApp", SymbolKind.Namespace, [_symbol("Foo", SymbolKind.Class, [_symbol("Foo", SymbolKind.Method, line=4)], line=2)])
        )
        enricher = SymbolOverviewEnricher.from_source([root], source)
        renderer = EnrichedSymbolsOverviewRenderer(self._AGENT, -1, _overview_params(depth=0), enricher)
        return LspSymbolCollection([root], renderer)

    def test_rendered_overview_is_enriched(self):
        (namespace,) = json.loads(self._collection().represent())
        (class_dict,) = namespace["children"]
        assert class_dict["name"] == "Foo : IBaz"
        assert class_dict["children"] == [{"name": "Foo", "kind": "Constructor"}]

    def test_overview_renders_identically_after_pickling(self):
        # facade results are pickled by the project server and rendered by the client
        collection = self._collection()
        assert pickle.loads(pickle.dumps(collection)).represent() == collection.represent()


@pytest.fixture
def serena_agent(request: pytest.FixtureRequest) -> Iterator[SerenaAgent]:
    language = LanguageServerId(request.param)
    if not language_server_tests_enabled(language):
        pytest.skip(f"Tests for language {language} are not enabled.")

    # an agent for the language's test repository, registered in an in-memory configuration
    serena_config = SerenaConfig(log_level=logging.ERROR, tool_timeout=600).with_headless_mode_overrides()
    project = Project(
        project_root=str(get_repo_path(language)),
        project_config=ProjectConfig(
            project_name=f"test_repo_{language}",
            language_servers=[language],
            ignored_paths=[],
            excluded_tools=[],
            ignore_all_files_in_gitignore=True,
            initial_prompt="",
            encoding="utf-8",
        ),
        serena_config=serena_config,
    )
    serena_config.projects = [RegisteredProject.from_project_instance(project)]
    agent = SerenaAgent(project=f"test_repo_{language}", serena_config=serena_config)
    agent.execute_task(lambda: None)

    yield agent

    agent.on_shutdown(timeout=5)


@pytest.mark.csharp
@pytest.mark.parametrize("serena_agent", [LanguageServerId.CSHARP], indirect=True)
def test_csharp_overview_shows_base_types_and_constructors(serena_agent: SerenaAgent):
    overview = serena_agent.get_tool(GetSymbolsOverviewTool).apply_ex(relative_path="Models/Person.cs")
    assert "Person" in overview
    assert "Constructor" in overview

    greeter_overview = serena_agent.get_tool(GetSymbolsOverviewTool).apply_ex(relative_path="Services/ConsoleGreeter.cs")
    assert "ConsoleGreeter : IGreeter" in greeter_overview


@pytest.mark.csharp
@pytest.mark.parametrize("serena_agent", [LanguageServerId.CSHARP], indirect=True)
def test_csharp_facade_overview_survives_pickling(serena_agent: SerenaAgent):
    # the project server returns facade results pickled, and the client renders them
    method = serena_agent.get_repl().entrypoint.get_facade_("lsp").get_method("get_symbols_overview")
    result = serena_agent.execute_task(lambda: method("Models/Person.cs"))
    rendered = result.represent()
    assert "IsAdult() : bool" in rendered
    assert pickle.loads(pickle.dumps(result)).represent() == rendered


@pytest.mark.python
@pytest.mark.parametrize("serena_agent", [LanguageServerId.PYTHON], indirect=True)
def test_python_overview_leaves_class_names_alone(serena_agent: SerenaAgent):
    overview = serena_agent.get_tool(GetSymbolsOverviewTool).apply_ex(relative_path="test_repo/models.py", depth=1)
    # grouped by kind; a class with children is represented as a dict mapping its name to its children
    class_names = [name for entry in json.loads(overview)["Class"] for name in ([entry] if isinstance(entry, str) else entry)]
    assert "User" in class_names
    assert all(":" not in name for name in class_names)
