"""Protocols for Tree-sitter parsing, symbol graphs, SCIP indexer, and LSP client.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed; structural conformance is enforced statically by the bindings in
`tests/unit/test_protocol_conformance.py` (issue 2026-09-02-035).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from uclone_x.code_intel.models import (
    Diagnostic,
    SCIPDocument,
    SCIPIndex,
    SymbolLookup,
    SymbolNode,
)


@runtime_checkable
class ASTParserProtocol(Protocol):
    """Protocol for Tree-sitter based syntax and symbol extraction."""

    def parse_file(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Parse source file into symbol nodes without executing code."""
        ...


@runtime_checkable
class SymbolGraphProtocol(Protocol):
    """Protocol for repository-scale code knowledge graph."""

    def index_symbols(self, file_path: Path, symbols: tuple[SymbolNode, ...]) -> None:
        """Replace the indexed symbols for one file.

        Keyed by file so re-indexing is idempotent. The previous signature took symbols
        alone with no inverse, so an incremental re-index accumulated stale symbols
        with no way to evict them (issue 2026-09-02-036).
        """
        ...

    def invalidate_file(self, file_path: Path) -> None:
        """Drop everything indexed for one file, e.g. when it is deleted."""
        ...

    def find_definitions(self, symbol_name: str) -> SymbolLookup:
        """Locate where a symbol is defined, reporting index freshness with the answer."""
        ...

    def find_references(self, symbol_name: str) -> SymbolLookup:
        """Find all usages of a symbol, reporting index freshness with the answer."""
        ...


@runtime_checkable
class SCIPIndexerProtocol(Protocol):
    """Protocol for SCIP semantic symbol indexing and cross-file navigation."""

    def index_file(self, file_path: Path, content: str | None = None) -> SCIPDocument:
        """Index a single file and return its SCIP document envelope."""
        ...

    def invalidate_file(self, file_path: Path) -> None:
        """Evict a file from the index cache."""
        ...

    def generate_index(self) -> SCIPIndex:
        """Generate full SCIP index envelope across all indexed files."""
        ...

    def find_definition(self, symbol_name: str) -> SymbolLookup:
        """Locate definition of symbol."""
        ...

    def find_references(self, symbol_name: str) -> SymbolLookup:
        """Locate references of symbol."""
        ...


@runtime_checkable
class LSPClientProtocol(Protocol):
    """Protocol for language server interaction and live diagnostics."""

    async def initialize(self, workspace_root: Path) -> None:
        """Start the language server process.

        Raises:
            UCloneXError: If no server is available for the workspace's languages. Per
                P6 this is not a degraded-but-successful start.
        """
        ...

    async def get_diagnostics(self, file_path: Path) -> tuple[Diagnostic, ...]:
        """Fetch compiler and type errors for a file."""
        ...

    async def go_to_definition(self, file_path: Path, line: int, col: int) -> SymbolLookup:
        """Resolve the definition at a cursor location.

        Returns a `SymbolLookup` rather than a list: "no language server is running for
        this file's language" and "this symbol has no definition" are different answers,
        and the specification requires them to stay distinguishable.
        """
        ...
