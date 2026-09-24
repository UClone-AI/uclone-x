"""Data models for code intelligence (AST, Symbol Graph, LSP, SCIP)."""

from __future__ import annotations

from enum import IntFlag, StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Diagnostic",
    "DiagnosticSeverity",
    "IndexFreshness",
    "SCIPDocument",
    "SCIPIndex",
    "SCIPMetadata",
    "SCIPOccurrence",
    "SCIPRelationship",
    "SCIPSymbolInformation",
    "SCIPSymbolRole",
    "SymbolKind",
    "SymbolLocation",
    "SymbolLookup",
    "SymbolNode",
]


class SymbolKind(StrEnum):
    """Classification of code symbols extracted via Tree-sitter or LSP."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    VARIABLE = "variable"
    MODULE = "module"


class DiagnosticSeverity(StrEnum):
    """Severity of a diagnostic, as an enum rather than a commented string."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class IndexFreshness(StrEnum):
    """Whether a lookup's answer can be trusted, and why not when it cannot.

    `code-intelligence-lsp-scip.md` section 4 states the requirement in the contract's
    own docstrings: a query "must raise or return an explicit 'unavailable' marker
    (never a silent None indistinguishable from 'no definition found')", and "must
    surface index staleness explicitly rather than silently serving a stale result".
    A bare `list[SymbolLocation]` cannot express either, because an empty list means
    all three things at once.
    """

    FRESH = "fresh"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class SymbolLocation(BaseModel):
    """Precise source code span location."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    file_path: Path
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class SymbolNode(BaseModel):
    """Symbol representation in the semantic code knowledge graph."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    kind: SymbolKind
    location: SymbolLocation
    signature: str | None = None
    docstring: str | None = None
    children: tuple[str, ...] = Field(default_factory=tuple)


class SymbolLookup(BaseModel):
    """The result of a symbol query, carrying whether it can be believed.

    `freshness` is required and has no default: "the index was never built" and "the
    symbol genuinely has no definition" are different answers, and P6 forbids returning
    a value that cannot be distinguished from the failure case.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    locations: tuple[SymbolLocation, ...] = Field(default_factory=tuple)
    freshness: IndexFreshness
    reason: str | None = Field(
        default=None,
        description="Why the answer is stale or unavailable — which language server is "
        "not running, or which files changed since the index was built.",
    )
    indexed_at_ns: int | None = None


class Diagnostic(BaseModel):
    """Compiler/linter diagnostic emitted by LSP or static analyzer."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    file_path: Path
    line: int
    col: int
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR


# =============================================================================
# SCIP (Source Code Intelligence Protocol) Models
# =============================================================================


class SCIPSymbolRole(IntFlag):
    """Bitmask representing SCIP symbol occurrence roles."""

    NONE = 0
    DEFINITION = 1
    IMPORT = 2
    WRITE_ACCESS = 4
    READ_ACCESS = 8


class SCIPRelationship(BaseModel):
    """Relationship between SCIP symbols."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbol: str
    is_reference: bool = False
    is_implementation: bool = False
    is_type_definition: bool = False
    is_definition: bool = False


class SCIPSymbolInformation(BaseModel):
    """Metadata and relationships associated with a defined SCIP symbol."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbol: str
    documentation: tuple[str, ...] = Field(default_factory=tuple)
    relationships: tuple[SCIPRelationship, ...] = Field(default_factory=tuple)
    kind: SymbolKind = SymbolKind.VARIABLE
    signature: str | None = None
    enclosing_symbol: str | None = None


class SCIPOccurrence(BaseModel):
    """Precise source range occurrence of a symbol in a document."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    range: tuple[int, int, int, int]  # (start_line, start_col, end_line, end_col) 1-based
    symbol: str
    symbol_roles: int = int(SCIPSymbolRole.NONE)
    syntax_kind: str | None = None
    override_documentation: tuple[str, ...] = Field(default_factory=tuple)


class SCIPDocument(BaseModel):
    """Per-file SCIP index envelope containing symbols and occurrences."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    relative_path: Path
    language: str
    symbols: tuple[SCIPSymbolInformation, ...] = Field(default_factory=tuple)
    occurrences: tuple[SCIPOccurrence, ...] = Field(default_factory=tuple)


class SCIPMetadata(BaseModel):
    """Metadata for a SCIP repository index."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tool_info_name: str = "uclone-x"
    tool_info_version: str = "0.1.0"
    project_root: Path = Field(default_factory=lambda: Path("."))
    text_document_encoding: str = "UTF-8"


class SCIPIndex(BaseModel):
    """Full repository SCIP index envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    metadata: SCIPMetadata
    documents: tuple[SCIPDocument, ...] = Field(default_factory=tuple)
    external_symbols: tuple[SCIPSymbolInformation, ...] = Field(default_factory=tuple)
    freshness: IndexFreshness = IndexFreshness.FRESH
    reason: str | None = None
    indexed_at_ns: int | None = None
