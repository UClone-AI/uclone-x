"""Unit tests for Code Intelligence subsystem (ASTParser and SymbolGraph)."""

from pathlib import Path

import pytest

from uclone_x.code_intel import (
    ASTParser,
    IndexFreshness,
    SymbolGraph,
    SymbolGraphProtocol,
    SymbolKind,
    SymbolLocation,
    SymbolNode,
)

# =============================================================================
# Python AST Parsing Tests
# =============================================================================

PYTHON_SAMPLE = '''"""Sample module docstring."""

from typing import Optional, List
import os

GLOBAL_MAX: int = 100
DEFAULT_NAME = "uclone"

class BaseWorker:
    """Base worker class."""
    worker_id: str = "worker-0"

    def __init__(self, name: str) -> None:
        """Init worker."""
        self.name = name

    def execute(self) -> bool:
        """Execute task."""
        return True

async def run_pipeline(items: List[str], max_retries: int = 3) -> int:
    """Run pipeline asynchronously."""
    return len(items)
'''


@pytest.mark.parametrize("use_tree_sitter", [True, False])
def test_ast_parser_python_symbol_extraction(use_tree_sitter: bool) -> None:
    """Verify Python AST parser extracts classes, methods, functions, variables, and docstrings."""
    parser = ASTParser(use_tree_sitter=use_tree_sitter)
    file_path = Path("src/worker.py")

    symbols = parser.parse_file(file_path, PYTHON_SAMPLE)
    assert len(symbols) > 0

    sym_by_name = {s.name: s for s in symbols}

    # Module
    assert "worker" in sym_by_name
    mod_sym = sym_by_name["worker"]
    assert mod_sym.kind == SymbolKind.MODULE
    assert mod_sym.docstring is not None
    assert "Sample module docstring." in mod_sym.docstring

    # Variables
    assert "GLOBAL_MAX" in sym_by_name or any("GLOBAL_MAX" in s.name for s in symbols)
    assert "DEFAULT_NAME" in sym_by_name or any("DEFAULT_NAME" in s.name for s in symbols)

    # Class
    assert "BaseWorker" in sym_by_name
    cls_sym = sym_by_name["BaseWorker"]
    assert cls_sym.kind == SymbolKind.CLASS
    assert cls_sym.docstring == "Base worker class."
    assert "execute" in cls_sym.children or "__init__" in cls_sym.children

    # Methods
    method_names = [s.name for s in symbols if s.kind == SymbolKind.METHOD]
    assert any("execute" in m for m in method_names)
    assert any("__init__" in m for m in method_names)

    # Function
    assert "run_pipeline" in sym_by_name
    fn_sym = sym_by_name["run_pipeline"]
    assert fn_sym.kind == SymbolKind.FUNCTION
    assert fn_sym.docstring == "Run pipeline asynchronously."
    assert fn_sym.location.file_path == file_path
    assert fn_sym.location.start_line > 0


def test_ast_parser_python_syntax_error_graceful() -> None:
    """Invalid Python syntax should return empty tuple rather than crashing."""
    parser = ASTParser()
    file_path = Path("bad.py")
    symbols = parser.parse_file(file_path, "def broken(: return }")
    assert isinstance(symbols, tuple)


def test_ast_parser_unsupported_extension() -> None:
    """Unsupported file extension returns empty tuple."""
    parser = ASTParser()
    symbols = parser.parse_file(Path("data.txt"), "some random text")
    assert symbols == ()


def test_ast_parser_empty_content() -> None:
    """Empty or whitespace-only content produces empty or module-only tuple."""
    parser = ASTParser()
    assert parser.parse_file(Path("empty.py"), "") == ()
    assert parser.parse_file(Path("blank.py"), "   \n\t  ") == ()


# =============================================================================
# TypeScript & JavaScript AST Parsing Tests
# =============================================================================

TS_SAMPLE = """
import { Request, Response } from 'express';
import * as path from 'path';

export const API_VERSION: string = 'v1';

export interface UserServiceConfig {
    timeoutMs: number;
    retries: number;
}

export class UserService {
    private isReady: boolean = true;

    constructor(private config: UserServiceConfig) {}

    public async findUser(id: string): Promise<User> {
        return { id, name: 'Alice' };
    }
}

export function createUserService(config: UserServiceConfig): UserService {
    return new UserService(config);
}

export const helperFn = async (x: number) => x * 2;
"""


def test_ast_parser_typescript_extraction() -> None:
    """Verify TypeScript symbols (interface, class, method, function, variable) extraction."""
    parser = ASTParser()
    file_path = Path("src/service.ts")

    symbols = parser.parse_file(file_path, TS_SAMPLE)
    assert len(symbols) > 0

    sym_by_name = {s.name: s for s in symbols}

    # Interface
    assert "UserServiceConfig" in sym_by_name
    iface = sym_by_name["UserServiceConfig"]
    assert iface.kind == SymbolKind.INTERFACE

    # Class
    assert "UserService" in sym_by_name
    cls_sym = sym_by_name["UserService"]
    assert cls_sym.kind == SymbolKind.CLASS

    # Function
    assert "createUserService" in sym_by_name
    fn_sym = sym_by_name["createUserService"]
    assert fn_sym.kind == SymbolKind.FUNCTION

    # Variable
    assert "API_VERSION" in sym_by_name
    assert sym_by_name["API_VERSION"].kind == SymbolKind.VARIABLE


def test_ast_parser_javascript_extraction() -> None:
    """Verify JavaScript file parsing."""
    parser = ASTParser()
    js_code = """
    class AppController {
        handle() {
            return 200;
        }
    }

    function initApp() {
        return new AppController();
    }
    """
    symbols = parser.parse_file(Path("app.js"), js_code)
    assert any(s.name == "AppController" and s.kind == SymbolKind.CLASS for s in symbols)
    assert any(s.name == "initApp" and s.kind == SymbolKind.FUNCTION for s in symbols)


# =============================================================================
# Imports and References Extraction Tests
# =============================================================================


def test_ast_parser_extract_imports() -> None:
    """Verify extraction of Python and TypeScript imports."""
    parser = ASTParser()

    py_imports = parser.extract_imports(Path("test.py"), PYTHON_SAMPLE)
    assert "os" in py_imports
    assert any("typing" in imp for imp in py_imports)

    ts_imports = parser.extract_imports(Path("test.ts"), TS_SAMPLE)
    assert "express" in ts_imports or "path" in ts_imports


def test_ast_parser_extract_references() -> None:
    """Verify identifier extraction without keywords."""
    parser = ASTParser()
    refs = parser.extract_references(Path("test.py"), PYTHON_SAMPLE)
    assert "BaseWorker" in refs
    assert "run_pipeline" in refs
    assert "def" not in refs
    assert "class" not in refs


def test_ast_parser_extract_symbol_references_locations() -> None:
    """Verify accurate location extraction for symbol references."""
    parser = ASTParser()
    file_path = Path("app.py")
    content = "user = User()\nprint(user.name)\n"
    ref_map = parser.extract_symbol_references(file_path, content)

    assert "User" in ref_map
    assert "user" in ref_map
    user_locs = ref_map["user"]
    assert len(user_locs) == 2
    assert user_locs[0].start_line == 1
    assert user_locs[1].start_line == 2


# =============================================================================
# SymbolGraph Tests
# =============================================================================


def test_symbol_graph_protocol_conformance() -> None:
    """Verify SymbolGraph conforms to SymbolGraphProtocol."""
    graph = SymbolGraph()
    assert isinstance(graph, SymbolGraphProtocol)


def test_symbol_graph_empty_index_semantics() -> None:
    """Empty symbol graph must return UNAVAILABLE freshness per P6."""
    graph = SymbolGraph()
    res = graph.find_definitions("NonExistent")
    assert res.freshness == IndexFreshness.UNAVAILABLE
    assert res.locations == ()
    assert res.reason is not None

    res_ref = graph.find_references("NonExistent")
    assert res_ref.freshness == IndexFreshness.UNAVAILABLE
    assert res_ref.locations == ()


def test_symbol_graph_index_and_find_definitions() -> None:
    """Verify indexing symbols and querying definitions."""
    graph = SymbolGraph()
    parser = ASTParser()

    file_a = Path("src/module_a.py")
    code_a = """
class DataService:
    def fetch(self) -> dict:
        return {}

def global_helper() -> str:
    return "ok"
"""
    symbols_a = parser.parse_file(file_a, code_a)
    graph.index_symbols(file_a, symbols_a)

    # Existing definition
    lookup = graph.find_definitions("DataService")
    assert lookup.freshness == IndexFreshness.FRESH
    assert len(lookup.locations) == 1
    assert lookup.locations[0].file_path == file_a
    assert lookup.indexed_at_ns is not None

    # Method lookup by qualified or short name
    lookup_method = graph.find_definitions("DataService.fetch")
    assert lookup_method.freshness == IndexFreshness.FRESH
    assert len(lookup_method.locations) >= 1

    lookup_short = graph.find_definitions("fetch")
    assert lookup_short.freshness == IndexFreshness.FRESH
    assert len(lookup_short.locations) >= 1

    # Non-existent definition on fresh index returns 0 locations but FRESH
    lookup_missing = graph.find_definitions("NoSuchClass")
    assert lookup_missing.freshness == IndexFreshness.FRESH
    assert lookup_missing.locations == ()


def test_symbol_graph_idempotent_reindexing() -> None:
    """Re-indexing a file replaces old symbols without leaking duplicates."""
    graph = SymbolGraph()
    file_path = Path("src/dynamic.py")

    loc1 = SymbolLocation(file_path=file_path, start_line=1, start_col=0, end_line=1, end_col=10)
    sym1 = SymbolNode(name="OldSymbol", kind=SymbolKind.FUNCTION, location=loc1)
    graph.index_symbols(file_path, (sym1,))

    assert len(graph.find_definitions("OldSymbol").locations) == 1

    # Re-index with new symbol
    loc2 = SymbolLocation(file_path=file_path, start_line=1, start_col=0, end_line=1, end_col=10)
    sym2 = SymbolNode(name="NewSymbol", kind=SymbolKind.FUNCTION, location=loc2)
    graph.index_symbols(file_path, (sym2,))

    # Old symbol must be evicted
    assert graph.find_definitions("OldSymbol").locations == ()
    assert len(graph.find_definitions("NewSymbol").locations) == 1


def test_symbol_graph_invalidate_file() -> None:
    """Invalidating a file drops all its symbols and references."""
    graph = SymbolGraph()
    file_a = Path("src/a.py")
    file_b = Path("src/b.py")

    loc_a = SymbolLocation(file_path=file_a, start_line=1, start_col=0, end_line=2, end_col=0)
    loc_b = SymbolLocation(file_path=file_b, start_line=1, start_col=0, end_line=2, end_col=0)

    graph.index_symbols(file_a, (SymbolNode(name="SymA", kind=SymbolKind.CLASS, location=loc_a),))
    graph.index_symbols(file_b, (SymbolNode(name="SymB", kind=SymbolKind.CLASS, location=loc_b),))

    assert len(graph.find_definitions("SymA").locations) == 1
    assert len(graph.find_definitions("SymB").locations) == 1

    graph.invalidate_file(file_a)

    assert graph.find_definitions("SymA").locations == ()
    assert len(graph.find_definitions("SymB").locations) == 1

    # Invalidate file_b -> graph becomes empty
    graph.invalidate_file(file_b)
    assert graph.find_definitions("SymB").freshness == IndexFreshness.UNAVAILABLE


def test_symbol_graph_find_references_and_staleness() -> None:
    """Verify references querying and staleness markers."""
    graph = SymbolGraph()
    file_path = Path("src/caller.py")

    loc = SymbolLocation(file_path=file_path, start_line=5, start_col=4, end_line=5, end_col=12)
    sym = SymbolNode(name="Caller", kind=SymbolKind.CLASS, location=loc)
    graph.index_symbols(file_path, (sym,))

    ref_loc = SymbolLocation(
        file_path=file_path, start_line=10, start_col=8, end_line=10, end_col=16
    )
    graph.index_references(file_path, {"TargetService": (ref_loc,)})

    # Fresh references query
    refs = graph.find_references("TargetService")
    assert refs.freshness == IndexFreshness.FRESH
    assert len(refs.locations) == 1
    assert refs.locations[0] == ref_loc

    # Mark stale
    graph.mark_stale("File modified on disk")
    refs_stale = graph.find_references("TargetService")
    assert refs_stale.freshness == IndexFreshness.STALE
    assert refs_stale.reason == "File modified on disk"
    assert len(refs_stale.locations) == 1

    defs_stale = graph.find_definitions("Caller")
    assert defs_stale.freshness == IndexFreshness.STALE

    # Mark fresh again
    graph.mark_fresh()
    assert graph.find_definitions("Caller").freshness == IndexFreshness.FRESH


def test_symbol_graph_index_file_extended() -> None:
    """Verify end-to-end index_file method with cross-file dependencies and hierarchy."""
    graph = SymbolGraph()
    file_path = Path("src/pipeline.py")
    code = """
import os
import sys

class Pipeline:
    def step_one(self) -> None:
        pass
    def step_two(self) -> None:
        pass

def run() -> None:
    p = Pipeline()
    p.step_one()
"""
    graph.index_file(file_path, code)

    # Check definitions
    assert len(graph.find_definitions("Pipeline").locations) == 1
    assert len(graph.find_definitions("run").locations) == 1

    # Check hierarchy
    hierarchy = graph.get_symbol_hierarchy("Pipeline")
    assert "step_one" in hierarchy
    assert "step_two" in hierarchy

    # Check dependencies
    deps = graph.get_file_dependencies(file_path)
    assert "os" in deps
    assert "sys" in deps

    # Check symbols in file
    syms = graph.get_symbols_in_file(file_path)
    assert len(syms) >= 2

    # Check all symbols
    all_syms = graph.get_all_symbols()
    assert len(all_syms) == len(syms)

    # Clear graph
    graph.clear()
    assert graph.find_definitions("Pipeline").freshness == IndexFreshness.UNAVAILABLE


# =============================================================================
# Tree-sitter grammar source selection (#1100)
# =============================================================================


def test_the_language_pack_is_preferred_over_the_legacy_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`tree-sitter-language-pack` is consulted first; the old bundle is not touched.

    It is the only one of the two with a cp313 wheel, so an environment holding both must
    resolve through the pack rather than through the shared object the bundle ships.

    Killed by: src/uclone_x/code_intel/ast_parser.py :: parser = self._load_from_language_pack(lang_name) or self._load_from_language_bundle(
    Becomes: parser = self._load_from_language_bundle(lang_name) or self._load_from_language_pack(
    """
    sentinel = object()
    bundle_calls: list[str] = []

    def from_pack(self: ASTParser, lang: str) -> object:
        return sentinel

    def from_bundle(self: ASTParser, lang: str) -> object | None:
        bundle_calls.append(lang)
        return None

    monkeypatch.setattr(ASTParser, "_load_from_language_pack", from_pack)
    monkeypatch.setattr(ASTParser, "_load_from_language_bundle", from_bundle)

    parser = ASTParser()
    assert parser._get_tree_sitter_parser("python") is sentinel  # pyright: ignore[reportPrivateUsage]
    assert bundle_calls == []


def test_the_legacy_bundle_still_answers_when_the_pack_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An environment installed before #1100 keeps working without a re-install.

    Killed by: src/uclone_x/code_intel/ast_parser.py :: parser = self._load_from_language_pack(lang_name) or self._load_from_language_bundle(
    Becomes: parser = self._load_from_language_pack(lang_name) or self._load_from_language_pack(
    """
    sentinel = object()

    def from_pack(self: ASTParser, lang: str) -> object | None:
        return None

    def from_bundle(self: ASTParser, lang: str) -> object:
        return sentinel

    monkeypatch.setattr(ASTParser, "_load_from_language_pack", from_pack)
    monkeypatch.setattr(ASTParser, "_load_from_language_bundle", from_bundle)

    parser = ASTParser()
    assert parser._get_tree_sitter_parser("go") is sentinel  # pyright: ignore[reportPrivateUsage]


def test_a_failing_grammar_source_is_attempted_once_and_then_reported_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither source available means `None`, cached, without re-importing on every call.

    Killed by: src/uclone_x/code_intel/ast_parser.py :: self._ts_load_attempted[lang_name] = True
    Becomes: self._ts_load_attempted[lang_name] = False
    """
    attempts: list[str] = []

    def from_pack(self: ASTParser, lang: str) -> object | None:
        attempts.append(lang)
        return None

    def from_bundle(self: ASTParser, lang: str) -> object | None:
        return None

    monkeypatch.setattr(ASTParser, "_load_from_language_pack", from_pack)
    monkeypatch.setattr(ASTParser, "_load_from_language_bundle", from_bundle)

    parser = ASTParser()
    assert parser.is_tree_sitter_available("rust") is False
    assert parser.is_tree_sitter_available("rust") is False
    assert attempts == ["rust"]


def test_module_presence_rejects_a_sys_modules_none_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module withheld by `sys.modules[name] = None` counts as absent, not present.

    That sentinel is how `tests/unit/test_kernel_dependency_footprint.py` withholds an
    optional distribution from a subprocess, so a presence check that believed it would
    report an extra as installed in exactly the environment that lacks it (P6).

    Killed by: src/uclone_x/code_intel/ast_parser.py :: importlib.import_module(module_name)
    Becomes: pass
    """
    import sys

    from uclone_x.code_intel.ast_parser import module_present

    monkeypatch.setitem(sys.modules, "uclone_x_absent_probe", None)
    assert module_present("uclone_x_absent_probe") is False
    assert module_present("uclone_x.code_intel.ast_parser") is True


class _Node:
    """The two attributes of a tree-sitter node that docstring detection reads."""

    def __init__(self, type_: str, *children: "_Node") -> None:
        self.type = type_
        self.children = list(children)


@pytest.mark.parametrize(
    "statement",
    [
        pytest.param(_Node("string"), id="language-pack-grammar"),
        pytest.param(_Node("expression_statement", _Node("string")), id="legacy-bundle-grammar"),
    ],
)
def test_docstring_is_found_in_either_grammar_shape(statement: _Node) -> None:
    """Both grammars `ASTParser` loads are read, not only the one the dev venv happens to hold.

    The public CI syncs `tree-sitter-language-pack`, whose tree-sitter-python puts a
    docstring's `string` directly in the block; the legacy bundle wraps it in an
    `expression_statement`. Reading only the wrapped form dropped every docstring and the
    module symbol there, while the suite stayed green wherever the legacy bundle was
    installed, so the shapes are pinned here without either grammar.

    Killed by: src/uclone_x/code_intel/ast_parser.py :: if statement.type == "string":
    Becomes: if statement.type == "strinG":
    """
    from uclone_x.code_intel.ast_parser import (
        _docstring_node,  # pyright: ignore[reportPrivateUsage]
    )

    found = _docstring_node(statement)
    assert found is not None and found.type == "string"


def test_a_statement_that_is_not_a_bare_string_has_no_docstring() -> None:
    """An expression statement that starts with something else is not a docstring."""
    from uclone_x.code_intel.ast_parser import (
        _docstring_node,  # pyright: ignore[reportPrivateUsage]
    )

    assert _docstring_node(_Node("expression_statement", _Node("call"))) is None
    assert _docstring_node(_Node("assignment")) is None
