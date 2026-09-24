"""Semantic SCIP (Source Code Intelligence Protocol) symbol graph indexer."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uclone_x
from uclone_x.code_intel.ast_parser import ASTParser
from uclone_x.code_intel.models import (
    IndexFreshness,
    SCIPDocument,
    SCIPIndex,
    SCIPMetadata,
    SCIPOccurrence,
    SCIPSymbolInformation,
    SCIPSymbolRole,
    SymbolKind,
    SymbolLocation,
    SymbolLookup,
    SymbolNode,
)
from uclone_x.code_intel.protocols import ASTParserProtocol, SCIPIndexerProtocol

__all__ = [
    "SCIPIndexGenerator",
    "SCIPIndexer",
    "format_scip_symbol",
    "resolve_language",
]

_SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
}


def resolve_language(file_path: Path) -> str:
    """Map file extension to language identifier."""
    suffix = file_path.suffix.lower()
    return _SUPPORTED_EXTENSIONS.get(suffix, "plaintext")


def format_scip_symbol(
    language: str,
    file_path: Path,
    symbol_name: str,
    kind: SymbolKind,
) -> str:
    """Generate a deterministic SCIP symbol string for a code entity.

    Format: `scip-<language> local 0.1.0 <rel_path>/<symbol_name><descriptor_suffix>`
    Suffixes:
      - Class / Interface: `#`
      - Function / Method: `().`
      - Variable: `:`
      - Module: `/`
    """
    rel_str = str(file_path).replace("\\", "/")
    if kind in {SymbolKind.CLASS, SymbolKind.INTERFACE}:
        suffix = "#"
    elif kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}:
        suffix = "()."
    elif kind == SymbolKind.MODULE:
        suffix = "/"
    else:
        suffix = ":"

    return f"scip-{language} local 0.1.0 {rel_str}/{symbol_name}{suffix}"


class SCIPIndexer(SCIPIndexerProtocol):
    """Semantic SCIP-compatible symbol graph indexer and cross-file navigator.

    Conforms to `SCIPIndexerProtocol`. Provides incremental file-level index caching,
    hash-based change detection, standard SCIP document/symbol/occurrence envelope
    generation, cross-file definition & reference queries, and explicit freshness tracking (P6).
    """

    def __init__(
        self,
        project_root: Path = Path("."),
        parser: ASTParserProtocol | None = None,
    ) -> None:
        self._project_root = Path(project_root)
        self._parser = parser or ASTParser()

        # Incremental file-level index cache
        self._documents: dict[Path, SCIPDocument] = {}
        self._file_hashes: dict[Path, str] = {}
        self._file_symbols: dict[Path, tuple[SymbolNode, ...]] = {}

        # Fast lookup indexes
        # Symbol key -> list of definition locations
        self._definitions: dict[str, list[SymbolLocation]] = {}
        self._short_definitions: dict[str, list[SymbolLocation]] = {}
        # Symbol key -> list of reference locations
        self._references: dict[str, list[SymbolLocation]] = {}
        self._short_references: dict[str, list[SymbolLocation]] = {}

        # Symbol string -> SCIPSymbolInformation
        self._symbol_info_map: dict[str, SCIPSymbolInformation] = {}

        self._last_indexed_at_ns: int | None = None
        self._is_stale: bool = False
        self._stale_reason: str | None = None

    @property
    def project_root(self) -> Path:
        """Root directory of the indexed project."""
        return self._project_root

    # =========================================================================
    # Protocol Implementation
    # =========================================================================

    def index_file(self, file_path: Path, content: str | None = None) -> SCIPDocument:
        """Index a single file incrementally with change-detection and SCIP envelopes."""
        norm_path = Path(file_path)
        actual_path = norm_path if norm_path.is_absolute() else self._project_root / norm_path

        if content is None:
            if not actual_path.exists():
                raise FileNotFoundError(f"File not found: {actual_path}")
            file_content = actual_path.read_text(encoding="utf-8", errors="replace")
        else:
            file_content = content

        content_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

        # Incremental Cache Check: if already indexed and hash unchanged, return cached doc
        if norm_path in self._documents and self._file_hashes.get(norm_path) == content_hash:
            return self._documents[norm_path]

        # Invalidate previous index entries for this file before re-indexing
        if norm_path in self._documents:
            self._evict_file_data(norm_path)

        language = resolve_language(norm_path)
        symbols = self._parser.parse_file(norm_path, file_content)

        scip_symbols: list[SCIPSymbolInformation] = []
        occurrences: list[SCIPOccurrence] = []

        # 1. Index Symbol Definitions
        for sym in symbols:
            scip_id = format_scip_symbol(language, norm_path, sym.name, sym.kind)
            doc_tuple = (sym.docstring,) if sym.docstring else ()

            # Detect enclosing symbol if qualified
            enclosing: str | None = None
            if "." in sym.name:
                parent_name = sym.name.rsplit(".", 1)[0]
                enclosing = format_scip_symbol(language, norm_path, parent_name, SymbolKind.CLASS)

            sym_info = SCIPSymbolInformation(
                symbol=scip_id,
                documentation=doc_tuple,
                kind=sym.kind,
                signature=sym.signature,
                enclosing_symbol=enclosing,
            )
            scip_symbols.append(sym_info)
            self._symbol_info_map[scip_id] = sym_info

            # Definition occurrence
            occ_range = (
                sym.location.start_line,
                sym.location.start_col,
                sym.location.end_line,
                sym.location.end_col,
            )
            occurrences.append(
                SCIPOccurrence(
                    range=occ_range,
                    symbol=scip_id,
                    symbol_roles=int(SCIPSymbolRole.DEFINITION),
                )
            )

            # Record definition locations
            self._record_definition(sym.name, sym.location)
            self._record_definition(scip_id, sym.location)
            if "." in sym.name:
                short_name = sym.name.split(".")[-1]
                self._record_short_definition(short_name, sym.location)

        # 2. Index Symbol References
        if isinstance(self._parser, ASTParser):
            ref_map = self._parser.extract_symbol_references(norm_path, file_content)
            for token_name, loc_tuple in ref_map.items():
                for loc in loc_tuple:
                    ref_scip_id = f"scip-{language} local 0.1.0 reference/{token_name}"
                    occ_range = (
                        loc.start_line,
                        loc.start_col,
                        loc.end_line,
                        loc.end_col,
                    )
                    occurrences.append(
                        SCIPOccurrence(
                            range=occ_range,
                            symbol=ref_scip_id,
                            symbol_roles=int(SCIPSymbolRole.READ_ACCESS),
                        )
                    )
                    self._record_reference(token_name, loc)
                    if "." in token_name:
                        self._record_short_reference(token_name.split(".")[-1], loc)

        scip_doc = SCIPDocument(
            relative_path=norm_path,
            language=language,
            symbols=tuple(scip_symbols),
            occurrences=tuple(occurrences),
        )

        # Update Cache
        self._documents[norm_path] = scip_doc
        self._file_hashes[norm_path] = content_hash
        self._file_symbols[norm_path] = symbols
        self._last_indexed_at_ns = time.time_ns()
        self._is_stale = False
        self._stale_reason = None

        return scip_doc

    def invalidate_file(self, file_path: Path) -> None:
        """Evict a file from the index cache when modified or deleted."""
        norm_path = Path(file_path)
        self._evict_file_data(norm_path)
        self._documents.pop(norm_path, None)
        self._file_hashes.pop(norm_path, None)
        self._file_symbols.pop(norm_path, None)

        if not self._documents:
            self._last_indexed_at_ns = None

    def generate_index(self) -> SCIPIndex:
        """Generate a complete SCIP index envelope across all indexed documents.

        `tool_info_version` is read as `uclone_x.__version__` through the module
        rather than via `from uclone_x import __version__`: an import-time binding
        reports the same string but no test can move it, so nothing could show the
        emitted metadata follows the declaration (#1131, following #1121).
        """
        metadata = SCIPMetadata(
            tool_info_name="uclone-x",
            tool_info_version=uclone_x.__version__,
            project_root=self._project_root,
            text_document_encoding="UTF-8",
        )

        if not self._documents:
            return SCIPIndex(
                metadata=metadata,
                documents=(),
                external_symbols=(),
                freshness=IndexFreshness.UNAVAILABLE,
                reason="SCIP index has no indexed documents",
                indexed_at_ns=None,
            )

        freshness = IndexFreshness.STALE if self._is_stale else IndexFreshness.FRESH
        return SCIPIndex(
            metadata=metadata,
            documents=tuple(self._documents.values()),
            external_symbols=tuple(self._symbol_info_map.values()),
            freshness=freshness,
            reason=self._stale_reason if self._is_stale else None,
            indexed_at_ns=self._last_indexed_at_ns,
        )

    def find_definition(self, symbol_name: str) -> SymbolLookup:
        """Locate where a symbol is defined, reporting explicit index freshness."""
        if not self._documents:
            return SymbolLookup(
                locations=(),
                freshness=IndexFreshness.UNAVAILABLE,
                reason="SCIP index is empty or not yet indexed",
                indexed_at_ns=None,
            )

        locs = self._resolve_definition_locations(symbol_name)
        freshness = IndexFreshness.STALE if self._is_stale else IndexFreshness.FRESH
        return SymbolLookup(
            locations=locs,
            freshness=freshness,
            reason=self._stale_reason if self._is_stale else None,
            indexed_at_ns=self._last_indexed_at_ns,
        )

    def find_definitions(self, symbol_name: str) -> SymbolLookup:
        """Plural alias matching `find_definition`."""
        return self.find_definition(symbol_name)

    def find_references(self, symbol_name: str) -> SymbolLookup:
        """Find all usages/references of a symbol across indexed documents."""
        if not self._documents:
            return SymbolLookup(
                locations=(),
                freshness=IndexFreshness.UNAVAILABLE,
                reason="SCIP index is empty or not yet indexed",
                indexed_at_ns=None,
            )

        locs = self._resolve_reference_locations(symbol_name)
        freshness = IndexFreshness.STALE if self._is_stale else IndexFreshness.FRESH
        return SymbolLookup(
            locations=locs,
            freshness=freshness,
            reason=self._stale_reason if self._is_stale else None,
            indexed_at_ns=self._last_indexed_at_ns,
        )

    # =========================================================================
    # Extended Capabilities
    # =========================================================================

    def index_directory(
        self,
        dir_path: Path | None = None,
        extensions: Sequence[str] | None = None,
    ) -> tuple[SCIPDocument, ...]:
        """Scan and index all supported source files in a directory."""
        target_dir = Path(dir_path) if dir_path is not None else self._project_root
        allowed_exts = set(extensions) if extensions else set(_SUPPORTED_EXTENSIONS.keys())

        docs: list[SCIPDocument] = []
        for path in target_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_exts:
                # Avoid hidden directories or build artifacts (.git, .venv, node_modules)
                parts = path.parts
                if any(
                    p.startswith(".")
                    or p in {"node_modules", "target", "__pycache__", "dist", "build"}
                    for p in parts
                ):
                    continue
                try:
                    rel_path = path.relative_to(self._project_root)
                except ValueError:
                    rel_path = path
                doc = self.index_file(rel_path)
                docs.append(doc)

        return tuple(docs)

    def get_document(self, file_path: Path) -> SCIPDocument | None:
        """Return the cached SCIP document for a file if indexed."""
        return self._documents.get(Path(file_path))

    def get_all_documents(self) -> tuple[SCIPDocument, ...]:
        """Return all indexed SCIP documents."""
        return tuple(self._documents.values())

    def mark_stale(self, reason: str | None = None) -> None:
        """Explicitly mark index as stale with a diagnostic reason (Principle 6)."""
        self._is_stale = True
        self._stale_reason = reason or "SCIP index marked stale due to filesystem changes"

    def mark_fresh(self) -> None:
        """Clear staleness marker."""
        self._is_stale = False
        self._stale_reason = None

    def clear(self) -> None:
        """Reset all internal SCIP index state and cached documents."""
        self._documents.clear()
        self._file_hashes.clear()
        self._file_symbols.clear()
        self._definitions.clear()
        self._short_definitions.clear()
        self._references.clear()
        self._short_references.clear()
        self._symbol_info_map.clear()
        self._last_indexed_at_ns = None
        self._is_stale = False
        self._stale_reason = None

    def export_scip_dict(self) -> dict[str, Any]:
        """Export index as a structured dictionary matching standard SCIP schemas."""
        scip_index = self.generate_index()
        return scip_index.model_dump(mode="json")

    def export_scip_json(self, indent: int = 2) -> str:
        """Export index as formatted JSON string."""
        return json.dumps(self.export_scip_dict(), indent=indent)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _record_definition(self, key: str, loc: SymbolLocation) -> None:
        if key not in self._definitions:
            self._definitions[key] = []
        if loc not in self._definitions[key]:
            self._definitions[key].append(loc)

    def _record_short_definition(self, key: str, loc: SymbolLocation) -> None:
        if key not in self._short_definitions:
            self._short_definitions[key] = []
        if loc not in self._short_definitions[key]:
            self._short_definitions[key].append(loc)

    def _record_reference(self, key: str, loc: SymbolLocation) -> None:
        if key not in self._references:
            self._references[key] = []
        if loc not in self._references[key]:
            self._references[key].append(loc)

    def _record_short_reference(self, key: str, loc: SymbolLocation) -> None:
        if key not in self._short_references:
            self._short_references[key] = []
        if loc not in self._short_references[key]:
            self._short_references[key].append(loc)

    def _resolve_definition_locations(self, symbol_name: str) -> tuple[SymbolLocation, ...]:
        if symbol_name in self._definitions:
            return tuple(self._definitions[symbol_name])
        if symbol_name in self._short_definitions:
            return tuple(self._short_definitions[symbol_name])
        # Try stripping punctuation or suffix
        cleaned = symbol_name.rstrip("#().:/")
        if cleaned in self._definitions:
            return tuple(self._definitions[cleaned])
        return ()

    def _resolve_reference_locations(self, symbol_name: str) -> tuple[SymbolLocation, ...]:
        short_name = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
        refs: list[SymbolLocation] = []

        if symbol_name in self._references:
            refs.extend(self._references[symbol_name])
        elif short_name in self._references:
            refs.extend(self._references[short_name])
        elif short_name in self._short_references:
            refs.extend(self._short_references[short_name])

        deduped: list[SymbolLocation] = []
        for r in refs:
            if r not in deduped:
                deduped.append(r)
        return tuple(deduped)

    def _evict_file_data(self, file_path: Path) -> None:
        """Remove definition and reference locations belonging to a specific file."""
        for sym_name, locs in list(self._definitions.items()):
            filtered = [loc for loc in locs if loc.file_path != file_path]
            if filtered:
                self._definitions[sym_name] = filtered
            else:
                self._definitions.pop(sym_name, None)

        for sym_name, locs in list(self._short_definitions.items()):
            filtered = [loc for loc in locs if loc.file_path != file_path]
            if filtered:
                self._short_definitions[sym_name] = filtered
            else:
                self._short_definitions.pop(sym_name, None)

        for sym_name, locs in list(self._references.items()):
            filtered = [loc for loc in locs if loc.file_path != file_path]
            if filtered:
                self._references[sym_name] = filtered
            else:
                self._references.pop(sym_name, None)

        for sym_name, locs in list(self._short_references.items()):
            filtered = [loc for loc in locs if loc.file_path != file_path]
            if filtered:
                self._short_references[sym_name] = filtered
            else:
                self._short_references.pop(sym_name, None)


SCIPIndexGenerator = SCIPIndexer
