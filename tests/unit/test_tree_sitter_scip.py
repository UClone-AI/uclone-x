"""Comprehensive unit tests for Tree-sitter multi-language AST parser and SCIP indexer."""

from pathlib import Path

import pytest

import uclone_x
from uclone_x.code_intel import (
    ASTParser,
    IndexFreshness,
    SCIPDocument,
    SCIPIndex,
    SCIPIndexer,
    SCIPIndexerProtocol,
    SCIPIndexGenerator,
    SCIPSymbolRole,
    SymbolGraph,
    SymbolGraphProtocol,
    SymbolKind,
    format_scip_symbol,
)

# =============================================================================
# Multi-Language Code Samples
# =============================================================================

PYTHON_SAMPLE = '''"""Module docstring for python sample."""

from typing import Optional, List
import os

GLOBAL_LIMIT: int = 100

class DataProcessor:
    """Processes datasets."""
    batch_size: int = 32

    def __init__(self, name: str) -> None:
        """Initialize processor."""
        self.name = name

    def process(self, items: List[str]) -> bool:
        """Process batch of items."""
        return len(items) > 0

async def fetch_data(source: str) -> str:
    """Fetch data from remote source."""
    return f"data from {source}"
'''

TYPESCRIPT_SAMPLE = """/**
 * Module comment for TypeScript service
 */

import { Config } from './config';
import * as fs from 'fs';

export const API_PORT: number = 3000;

/** Configuration interface */
export interface ServiceConfig {
    host: string;
    port: number;
}

/** Core user service */
export class UserService {
    private isRunning: boolean = true;

    constructor(private config: ServiceConfig) {}

    public async getUser(id: string): Promise<User> {
        return { id, name: 'Alice' };
    }
}

export function createService(cfg: ServiceConfig): UserService {
    return new UserService(cfg);
}

export const helperArrow = async (x: number) => x * 10;
"""

RUST_SAMPLE = """//! Rust crate top-level documentation

use std::collections::HashMap;
use std::sync::Arc;

pub const MAX_WORKERS: usize = 16;
pub static IS_INITIALIZED: bool = true;

/// User account entity
pub struct UserAccount {
    pub id: u64,
    pub username: String,
}

/// Account state enum
pub enum AccountState {
    Active,
    Suspended,
}

/// Operations for accounts
pub trait AccountRepository {
    fn find_by_id(&self, id: u64) -> Option<UserAccount>;
}

impl UserAccount {
    /// Create new account
    pub fn new(id: u64, username: String) -> Self {
        Self { id, username }
    }

    pub fn is_valid(&self) -> bool {
        self.id > 0
    }
}

pub fn calculate_checksum(data: &[u8]) -> u32 {
    42
}
"""

GO_SAMPLE = """// Package auth provides authentication primitives.
package auth

import (
    "fmt"
    "time"
)

const DefaultTimeout = 30 * time.Second
var GlobalStatus = "online"

// AuthSession holds user session data.
type AuthSession struct {
    Token string
    UserID int64
}

// TokenValidator interface
type TokenValidator interface {
    Validate(token string) bool
}

// NewSession creates an auth session.
func NewSession(userID int64, token string) *AuthSession {
    return &AuthSession{UserID: userID, Token: token}
}

// IsExpired checks session validity.
func (s *AuthSession) IsExpired() bool {
    return len(s.Token) == 0
}
"""


# =============================================================================
# 1. ASTParser Multi-Language Parsing Tests
# =============================================================================


@pytest.mark.parametrize("use_tree_sitter", [True, False])
def test_ast_parser_python_full(use_tree_sitter: bool) -> None:
    """Verify Python AST extraction in both Tree-sitter and AST fallback modes."""
    parser = ASTParser(use_tree_sitter=use_tree_sitter)
    file_path = Path("src/processor.py")
    symbols = parser.parse_file(file_path, PYTHON_SAMPLE)

    assert len(symbols) > 0
    names = {s.name: s for s in symbols}

    assert "processor" in names
    assert names["processor"].kind == SymbolKind.MODULE
    assert names["processor"].docstring is not None
    assert "Module docstring" in names["processor"].docstring

    assert "DataProcessor" in names
    cls_node = names["DataProcessor"]
    assert cls_node.kind == SymbolKind.CLASS
    assert cls_node.docstring == "Processes datasets."
    assert "process" in cls_node.children or "__init__" in cls_node.children

    assert any(s.name == "DataProcessor.process" and s.kind == SymbolKind.METHOD for s in symbols)
    assert "fetch_data" in names
    assert names["fetch_data"].kind == SymbolKind.FUNCTION


@pytest.mark.parametrize("use_tree_sitter", [True, False])
def test_ast_parser_typescript_full(use_tree_sitter: bool) -> None:
    """Verify TypeScript extraction in Tree-sitter and regex fallback modes."""
    parser = ASTParser(use_tree_sitter=use_tree_sitter)
    file_path = Path("src/service.ts")
    symbols = parser.parse_file(file_path, TYPESCRIPT_SAMPLE)

    assert len(symbols) > 0
    names = {s.name: s for s in symbols}

    assert "ServiceConfig" in names
    assert names["ServiceConfig"].kind == SymbolKind.INTERFACE

    assert "UserService" in names
    assert names["UserService"].kind == SymbolKind.CLASS

    assert "createService" in names
    assert names["createService"].kind == SymbolKind.FUNCTION

    assert "API_PORT" in names
    assert names["API_PORT"].kind == SymbolKind.VARIABLE


@pytest.mark.parametrize("use_tree_sitter", [True, False])
def test_ast_parser_rust_full(use_tree_sitter: bool) -> None:
    """Verify Rust extraction (struct, enum, trait, impl method, function, const, docstrings)."""
    parser = ASTParser(use_tree_sitter=use_tree_sitter)
    file_path = Path("src/account.rs")
    symbols = parser.parse_file(file_path, RUST_SAMPLE)

    assert len(symbols) > 0
    names = {s.name: s for s in symbols}

    # Struct
    assert "UserAccount" in names
    struct_sym = names["UserAccount"]
    assert struct_sym.kind == SymbolKind.CLASS
    if use_tree_sitter:
        assert struct_sym.docstring is not None
        assert "User account entity" in struct_sym.docstring

    # Enum
    assert "AccountState" in names
    assert names["AccountState"].kind == SymbolKind.CLASS

    # Trait
    assert "AccountRepository" in names
    assert names["AccountRepository"].kind == SymbolKind.INTERFACE

    # Impl method
    assert "UserAccount.new" in names
    assert names["UserAccount.new"].kind == SymbolKind.METHOD

    # Standalone function
    assert "calculate_checksum" in names
    assert names["calculate_checksum"].kind == SymbolKind.FUNCTION

    # Constant / static
    assert "MAX_WORKERS" in names
    assert names["MAX_WORKERS"].kind == SymbolKind.VARIABLE


@pytest.mark.parametrize("use_tree_sitter", [True, False])
def test_ast_parser_go_full(use_tree_sitter: bool) -> None:
    """Verify Go extraction (package, struct, interface, method with receiver, function, const)."""
    parser = ASTParser(use_tree_sitter=use_tree_sitter)
    file_path = Path("auth/session.go")
    symbols = parser.parse_file(file_path, GO_SAMPLE)

    assert len(symbols) > 0
    names = {s.name: s for s in symbols}

    # Package / Module
    assert "auth" in names
    assert names["auth"].kind == SymbolKind.MODULE

    # Struct
    assert "AuthSession" in names
    struct_sym = names["AuthSession"]
    assert struct_sym.kind == SymbolKind.CLASS

    # Interface
    assert "TokenValidator" in names
    assert names["TokenValidator"].kind == SymbolKind.INTERFACE

    # Function
    assert "NewSession" in names
    assert names["NewSession"].kind == SymbolKind.FUNCTION

    # Method with receiver type
    assert "AuthSession.IsExpired" in names
    assert names["AuthSession.IsExpired"].kind == SymbolKind.METHOD

    # Const / Var
    assert "DefaultTimeout" in names
    assert names["DefaultTimeout"].kind == SymbolKind.VARIABLE


def test_ast_parser_syntax_errors_graceful() -> None:
    """AST parser must safely handle syntax errors across all languages without crashing."""
    parser = ASTParser()

    assert isinstance(parser.parse_file(Path("bad.py"), "def ::: broken (("), tuple)
    assert isinstance(parser.parse_file(Path("bad.ts"), "export class { { broken"), tuple)
    assert isinstance(parser.parse_file(Path("bad.rs"), "pub struct { pub fn ::: }"), tuple)
    assert isinstance(parser.parse_file(Path("bad.go"), "func ::: ((( {"), tuple)
    assert parser.parse_file(Path("unsupported.xyz"), "some content") == ()
    assert parser.parse_file(Path("empty.py"), "") == ()


def test_ast_parser_imports_and_references() -> None:
    """Verify import and reference extraction across Python, TS, Rust, and Go."""
    parser = ASTParser()

    # Python
    py_imports = parser.extract_imports(Path("app.py"), PYTHON_SAMPLE)
    assert "os" in py_imports
    py_refs = parser.extract_references(Path("app.py"), PYTHON_SAMPLE)
    assert "DataProcessor" in py_refs
    assert "class" not in py_refs

    # TypeScript
    ts_imports = parser.extract_imports(Path("app.ts"), TYPESCRIPT_SAMPLE)
    assert "fs" in ts_imports
    ts_refs = parser.extract_references(Path("app.ts"), TYPESCRIPT_SAMPLE)
    assert "UserService" in ts_refs
    assert "interface" not in ts_refs

    # Rust
    rs_imports = parser.extract_imports(Path("app.rs"), RUST_SAMPLE)
    assert any("std::collections::HashMap" in imp for imp in rs_imports)
    rs_refs = parser.extract_references(Path("app.rs"), RUST_SAMPLE)
    assert "UserAccount" in rs_refs
    assert "struct" not in rs_refs

    # Go
    go_imports = parser.extract_imports(Path("app.go"), GO_SAMPLE)
    assert "fmt" in go_imports
    assert "time" in go_imports
    go_refs = parser.extract_references(Path("app.go"), GO_SAMPLE)
    assert "AuthSession" in go_refs
    assert "package" not in go_refs


# =============================================================================
# 2. SCIP Symbol Formatting and Document Envelopes
# =============================================================================


def test_format_scip_symbol() -> None:
    """Verify deterministic SCIP symbol descriptors per spec."""
    s_class = format_scip_symbol("python", Path("src/core.py"), "Engine", SymbolKind.CLASS)
    assert s_class == "scip-python local 0.1.0 src/core.py/Engine#"

    s_fn = format_scip_symbol("rust", Path("src/lib.rs"), "init", SymbolKind.FUNCTION)
    assert s_fn == "scip-rust local 0.1.0 src/lib.rs/init()."

    s_var = format_scip_symbol("go", Path("main.go"), "Version", SymbolKind.VARIABLE)
    assert s_var == "scip-go local 0.1.0 main.go/Version:"

    s_mod = format_scip_symbol("typescript", Path("src/index.ts"), "auth", SymbolKind.MODULE)
    assert s_mod == "scip-typescript local 0.1.0 src/index.ts/auth/"


def test_scip_indexer_protocol_conformance() -> None:
    """Verify SCIPIndexer and SCIPIndexGenerator conform to SCIPIndexerProtocol."""
    indexer = SCIPIndexer()
    assert isinstance(indexer, SCIPIndexerProtocol)

    generator = SCIPIndexGenerator()
    assert isinstance(generator, SCIPIndexerProtocol)


def test_scip_indexer_single_file_and_caching() -> None:
    """Verify indexing single file, symbol/occurrence generation, and hash caching."""
    indexer = SCIPIndexer(project_root=Path("."))
    file_path = Path("src/account.rs")

    doc1 = indexer.index_file(file_path, RUST_SAMPLE)
    assert isinstance(doc1, SCIPDocument)
    assert doc1.language == "rust"
    assert len(doc1.symbols) > 0
    assert len(doc1.occurrences) > 0

    # Ensure definition occurrence has DEFINITION role
    def_occs = [o for o in doc1.occurrences if o.symbol_roles & int(SCIPSymbolRole.DEFINITION)]
    assert len(def_occs) > 0

    # Test Incremental Cache: Re-indexing same content returns exact cached instance
    doc2 = indexer.index_file(file_path, RUST_SAMPLE)
    assert doc1 is doc2

    # Modifying content creates updated document
    modified_code = RUST_SAMPLE + "\npub fn extra_helper() {}\n"
    doc3 = indexer.index_file(file_path, modified_code)
    assert doc3 is not doc1
    assert any("extra_helper" in s.symbol for s in doc3.symbols)


def test_scip_indexer_cross_file_definitions_and_references() -> None:
    """Verify cross-file definition and reference lookups across multiple languages."""
    indexer = SCIPIndexer()

    file_py = Path("src/server.py")
    code_py = "class ApiServer:\n    def start(self): pass\n"

    file_ts = Path("src/client.ts")
    code_ts = (
        "import { ApiServer } from './server';\nconst server = new ApiServer();\nserver.start();\n"
    )

    indexer.index_file(file_py, code_py)
    indexer.index_file(file_ts, code_ts)

    # Find definition in Python
    lookup_def = indexer.find_definition("ApiServer")
    assert lookup_def.freshness == IndexFreshness.FRESH
    assert len(lookup_def.locations) == 1
    assert lookup_def.locations[0].file_path == file_py

    # Find method definition
    lookup_method = indexer.find_definition("ApiServer.start")
    assert lookup_method.freshness == IndexFreshness.FRESH
    assert len(lookup_method.locations) == 1

    # Find references across files
    lookup_ref = indexer.find_references("ApiServer")
    assert lookup_ref.freshness == IndexFreshness.FRESH
    assert any(loc.file_path == file_ts for loc in lookup_ref.locations)

    # Missing symbol on fresh index
    missing = indexer.find_definition("NonExistentSymbol")
    assert missing.freshness == IndexFreshness.FRESH
    assert missing.locations == ()


def test_scip_indexer_file_invalidation() -> None:
    """Invalidating a file drops its cached doc, definitions, and references."""
    indexer = SCIPIndexer()
    f1 = Path("f1.go")
    f2 = Path("f2.go")

    indexer.index_file(f1, "package main\ntype User struct {}\n")
    indexer.index_file(f2, "package main\ntype Order struct {}\n")

    assert len(indexer.find_definition("User").locations) == 1
    assert len(indexer.find_definition("Order").locations) == 1

    indexer.invalidate_file(f1)

    assert indexer.find_definition("User").locations == ()
    assert len(indexer.find_definition("Order").locations) == 1
    assert indexer.get_document(f1) is None

    # Invalidate second file -> index becomes empty / UNAVAILABLE
    indexer.invalidate_file(f2)
    assert indexer.find_definition("Order").freshness == IndexFreshness.UNAVAILABLE


def test_scip_indexer_staleness_and_provenance() -> None:
    """Verify explicit IndexFreshness states per Principle 6."""
    indexer = SCIPIndexer()

    # Empty index -> UNAVAILABLE
    assert indexer.find_definition("Foo").freshness == IndexFreshness.UNAVAILABLE
    empty_index = indexer.generate_index()
    assert empty_index.freshness == IndexFreshness.UNAVAILABLE
    assert empty_index.reason is not None

    # Indexed -> FRESH
    indexer.index_file(Path("lib.rs"), "pub fn compute() -> i32 { 100 }")
    assert indexer.find_definition("compute").freshness == IndexFreshness.FRESH
    fresh_index = indexer.generate_index()
    assert fresh_index.freshness == IndexFreshness.FRESH

    # Mark Stale -> STALE with explicit reason
    indexer.mark_stale("File modified on disk externally")
    stale_lookup = indexer.find_definition("compute")
    assert stale_lookup.freshness == IndexFreshness.STALE
    assert stale_lookup.reason == "File modified on disk externally"
    assert len(stale_lookup.locations) == 1

    stale_index = indexer.generate_index()
    assert stale_index.freshness == IndexFreshness.STALE
    assert stale_index.reason == "File modified on disk externally"

    # Mark Fresh -> FRESH
    indexer.mark_fresh()
    assert indexer.find_definition("compute").freshness == IndexFreshness.FRESH

    # Clear -> UNAVAILABLE
    indexer.clear()
    assert indexer.find_definition("compute").freshness == IndexFreshness.UNAVAILABLE


def test_scip_indexer_export_json_and_dict() -> None:
    """Verify exporting SCIP index as standard dictionary and JSON."""
    indexer = SCIPIndexer(project_root=Path("/workspace"))
    indexer.index_file(Path("main.py"), PYTHON_SAMPLE)
    indexer.index_file(Path("service.ts"), TYPESCRIPT_SAMPLE)

    d = indexer.export_scip_dict()
    assert "metadata" in d
    assert "documents" in d
    assert len(d["documents"]) == 2
    assert d["freshness"] == "fresh"

    json_str = indexer.export_scip_json()
    assert isinstance(json_str, str)
    assert '"tool_info_name": "uclone-x"' in json_str
    assert '"language": "python"' in json_str
    assert '"language": "typescript"' in json_str


def test_symbol_graph_export_scip_index() -> None:
    """Verify SymbolGraph export_scip_index and find_definition singular alias."""
    graph = SymbolGraph()
    assert isinstance(graph, SymbolGraphProtocol)

    f_py = Path("calc.py")
    graph.index_file(f_py, "class Calculator:\n    def add(self, a, b): return a + b\n")

    # Singular alias find_definition
    lookup = graph.find_definition("Calculator")
    assert lookup.freshness == IndexFreshness.FRESH
    assert len(lookup.locations) == 1

    # Export SCIP Index
    scip_idx = graph.export_scip_index(project_root=Path("."))
    assert isinstance(scip_idx, SCIPIndex)
    assert scip_idx.freshness == IndexFreshness.FRESH
    assert len(scip_idx.documents) == 1
    assert scip_idx.documents[0].language == "python"
    assert any("Calculator" in sym.symbol for sym in scip_idx.documents[0].symbols)


def test_scip_indexer_metadata_reports_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCIP index metadata reports the live package version, not a copy of it.

    Asserting only that `tool_info_version == uclone_x.__version__` would pass
    against a literal that happens to be correct today -- which is exactly the state
    #1131 describes. Moving `__version__` to a value no literal in the tree carries
    separates "reads the declaration" from "restates today's value".

    Killed by: src/uclone_x/code_intel/scip.py :: tool_info_version=uclone_x.__version__,
    Becomes: tool_info_version="0.0.0",
    """
    monkeypatch.setattr(uclone_x, "__version__", "9.8.7-probe")

    indexer = SCIPIndexer()
    indexer.index_file(Path("probe.py"), "def f():\n    return 1\n")

    assert indexer.generate_index().metadata.tool_info_version == "9.8.7-probe"


def test_symbol_graph_export_reports_package_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """SymbolGraph's exported SCIP metadata reports the live package version.

    Killed by: src/uclone_x/code_intel/symbol_graph.py :: tool_info_version=uclone_x.__version__,
    Becomes: tool_info_version="0.0.0",
    """
    monkeypatch.setattr(uclone_x, "__version__", "9.8.7-probe")

    graph = SymbolGraph()
    graph.index_file(Path("probe.py"), "class Probe:\n    pass\n")

    assert graph.export_scip_index(project_root=Path(".")).metadata.tool_info_version == (
        "9.8.7-probe"
    )
