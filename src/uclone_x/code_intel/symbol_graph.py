"""In-memory repository symbol knowledge graph and indexer."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    SymbolLocation,
    SymbolLookup,
    SymbolNode,
)
from uclone_x.code_intel.protocols import ASTParserProtocol, SymbolGraphProtocol
from uclone_x.code_intel.scip import format_scip_symbol, resolve_language

__all__ = ["SymbolGraph"]


class SymbolGraph(SymbolGraphProtocol):
    """In-memory semantic code knowledge graph.

    Conforms to `SymbolGraphProtocol`. Provides multi-file symbol indexing,
    fast definition lookup, cross-file reference tracking, symbol hierarchy,
    dependency resolution, SCIP index export, and strict freshness semantics.
    """

    def __init__(self) -> None:
        # Indexed files and their defined symbol nodes
        self._file_symbols: dict[Path, tuple[SymbolNode, ...]] = {}
        # Exact symbol name -> list of definition locations
        self._definitions: dict[str, list[SymbolLocation]] = {}
        # Short / unqualified symbol name -> list of definition locations
        self._short_definitions: dict[str, list[SymbolLocation]] = {}
        # Symbol name -> list of reference locations
        self._references: dict[str, list[SymbolLocation]] = {}
        # Per-file reference maps: file_path -> (symbol_name -> tuple of locations)
        self._file_references: dict[Path, dict[str, tuple[SymbolLocation, ...]]] = {}
        # Per-file dependency maps: file_path -> set of imported module names
        self._file_dependencies: dict[Path, set[str]] = {}
        # Symbol name -> tuple of child names
        self._symbol_hierarchy: dict[str, tuple[str, ...]] = {}

        self._last_indexed_at_ns: int | None = None
        self._is_stale: bool = False
        self._stale_reason: str | None = None

    # =========================================================================
    # Protocol Implementation
    # =========================================================================

    def index_symbols(self, file_path: Path, symbols: tuple[SymbolNode, ...]) -> None:
        """Replace the indexed symbols for one file (idempotent)."""
        norm_path = Path(file_path)

        # Invalidate previous entries for this file if re-indexing
        if norm_path in self._file_symbols:
            self._evict_file_definitions(norm_path)

        self._file_symbols[norm_path] = symbols

        for node in symbols:
            # Index by exact name
            if node.name not in self._definitions:
                self._definitions[node.name] = []
            if node.location not in self._definitions[node.name]:
                self._definitions[node.name].append(node.location)

            # If qualified (e.g. MyClass.my_method), also index short name
            if "." in node.name:
                short_name = node.name.split(".")[-1]
                if short_name not in self._short_definitions:
                    self._short_definitions[short_name] = []
                if node.location not in self._short_definitions[short_name]:
                    self._short_definitions[short_name].append(node.location)

            # Record symbol hierarchy
            if node.children:
                self._symbol_hierarchy[node.name] = node.children

        self._last_indexed_at_ns = time.time_ns()
        self._is_stale = False
        self._stale_reason = None

    def invalidate_file(self, file_path: Path) -> None:
        """Drop everything indexed for one file (e.g. when modified or deleted)."""
        norm_path = Path(file_path)
        self._evict_file_definitions(norm_path)
        self._evict_file_references(norm_path)
        self._file_symbols.pop(norm_path, None)
        self._file_dependencies.pop(norm_path, None)

        if not self._file_symbols:
            self._last_indexed_at_ns = None

    def find_definition(self, symbol_name: str) -> SymbolLookup:
        """Locate where a symbol is defined (singular alias to find_definitions)."""
        return self.find_definitions(symbol_name)

    def find_definitions(self, symbol_name: str) -> SymbolLookup:
        """Locate where a symbol is defined, reporting index freshness."""
        if not self._file_symbols:
            return SymbolLookup(
                locations=(),
                freshness=IndexFreshness.UNAVAILABLE,
                reason="Symbol graph is empty or not yet indexed",
                indexed_at_ns=None,
            )

        if self._is_stale:
            locs = self._resolve_definition_locations(symbol_name)
            return SymbolLookup(
                locations=locs,
                freshness=IndexFreshness.STALE,
                reason=self._stale_reason or "Index is stale",
                indexed_at_ns=self._last_indexed_at_ns,
            )

        locs = self._resolve_definition_locations(symbol_name)
        return SymbolLookup(
            locations=locs,
            freshness=IndexFreshness.FRESH,
            indexed_at_ns=self._last_indexed_at_ns,
        )

    def find_references(self, symbol_name: str) -> SymbolLookup:
        """Find all usages of a symbol, reporting index freshness."""
        if not self._file_symbols:
            return SymbolLookup(
                locations=(),
                freshness=IndexFreshness.UNAVAILABLE,
                reason="Symbol graph is empty or not yet indexed",
                indexed_at_ns=None,
            )

        short_name = symbol_name.split(".")[-1] if "." in symbol_name else symbol_name
        refs: list[SymbolLocation] = []

        if symbol_name in self._references:
            refs.extend(self._references[symbol_name])
        elif short_name in self._references:
            refs.extend(self._references[short_name])

        # Deduplicate preserving order
        deduped: list[SymbolLocation] = []
        for r in refs:
            if r not in deduped:
                deduped.append(r)

        if self._is_stale:
            return SymbolLookup(
                locations=tuple(deduped),
                freshness=IndexFreshness.STALE,
                reason=self._stale_reason or "Index is stale",
                indexed_at_ns=self._last_indexed_at_ns,
            )

        return SymbolLookup(
            locations=tuple(deduped),
            freshness=IndexFreshness.FRESH,
            indexed_at_ns=self._last_indexed_at_ns,
        )

    # =========================================================================
    # Extended Capabilities
    # =========================================================================

    def index_file(
        self,
        file_path: Path,
        content: str,
        parser: ASTParserProtocol | None = None,
    ) -> None:
        """Parse source content and index its definitions, references, and imports."""
        active_parser = parser or ASTParser()
        symbols = active_parser.parse_file(file_path, content)
        self.index_symbols(file_path, symbols)

        if isinstance(active_parser, ASTParser):
            ref_dict = active_parser.extract_symbol_references(file_path, content)
            self.index_references(file_path, ref_dict)
            imports = active_parser.extract_imports(file_path, content)
            self.index_dependencies(file_path, imports)

    def index_references(
        self,
        file_path: Path,
        references: Mapping[str, Sequence[SymbolLocation]],
    ) -> None:
        """Index symbol reference locations for a specific file."""
        norm_path = Path(file_path)
        self._evict_file_references(norm_path)

        stored_map: dict[str, tuple[SymbolLocation, ...]] = {}
        for sym_name, locs in references.items():
            t_locs = tuple(locs)
            stored_map[sym_name] = t_locs
            if sym_name not in self._references:
                self._references[sym_name] = []
            for loc in t_locs:
                if loc not in self._references[sym_name]:
                    self._references[sym_name].append(loc)

        self._file_references[norm_path] = stored_map

    def index_dependencies(self, file_path: Path, imported_modules: Sequence[str]) -> None:
        """Record imported modules and dependencies for a file."""
        norm_path = Path(file_path)
        self._file_dependencies[norm_path] = set(imported_modules)

    def get_symbol_hierarchy(self, symbol_name: str) -> tuple[str, ...]:
        """Return child symbol names for a class, interface, or module."""
        return self._symbol_hierarchy.get(symbol_name, ())

    def get_file_dependencies(self, file_path: Path) -> tuple[str, ...]:
        """Return all imported module names for a given file."""
        norm_path = Path(file_path)
        return tuple(sorted(self._file_dependencies.get(norm_path, set())))

    def get_symbols_in_file(self, file_path: Path) -> tuple[SymbolNode, ...]:
        """Return all symbol nodes defined in a specific file."""
        norm_path = Path(file_path)
        return self._file_symbols.get(norm_path, ())

    def get_all_symbols(self) -> tuple[SymbolNode, ...]:
        """Return all indexed symbols across all files."""
        all_syms: list[SymbolNode] = []
        for syms in self._file_symbols.values():
            all_syms.extend(syms)
        return tuple(all_syms)

    def export_scip_index(self, project_root: Path = Path(".")) -> SCIPIndex:
        """Export current symbol graph state as a standard SCIP index envelope.

        `tool_info_version` is read as `uclone_x.__version__` through the module for
        the reason recorded on `SCIPIndexer.generate_index` (#1131).
        """
        metadata = SCIPMetadata(
            tool_info_name="uclone-x",
            tool_info_version=uclone_x.__version__,
            project_root=Path(project_root),
            text_document_encoding="UTF-8",
        )

        if not self._file_symbols:
            return SCIPIndex(
                metadata=metadata,
                documents=(),
                external_symbols=(),
                freshness=IndexFreshness.UNAVAILABLE,
                reason="Symbol graph is empty or not yet indexed",
                indexed_at_ns=None,
            )

        documents: list[SCIPDocument] = []
        all_symbol_info: list[SCIPSymbolInformation] = []

        for f_path, sym_nodes in self._file_symbols.items():
            lang = resolve_language(f_path)
            scip_syms: list[SCIPSymbolInformation] = []
            occurrences: list[SCIPOccurrence] = []

            for sym in sym_nodes:
                scip_id = format_scip_symbol(lang, f_path, sym.name, sym.kind)
                doc_tuple = (sym.docstring,) if sym.docstring else ()
                enclosing = None
                if "." in sym.name:
                    parent_name = sym.name.rsplit(".", 1)[0]
                    enclosing = format_scip_symbol(lang, f_path, parent_name, sym.kind)

                info = SCIPSymbolInformation(
                    symbol=scip_id,
                    documentation=doc_tuple,
                    kind=sym.kind,
                    signature=sym.signature,
                    enclosing_symbol=enclosing,
                )
                scip_syms.append(info)
                all_symbol_info.append(info)

                occurrences.append(
                    SCIPOccurrence(
                        range=(
                            sym.location.start_line,
                            sym.location.start_col,
                            sym.location.end_line,
                            sym.location.end_col,
                        ),
                        symbol=scip_id,
                        symbol_roles=int(SCIPSymbolRole.DEFINITION),
                    )
                )

            # References for this file
            if f_path in self._file_references:
                for token_name, loc_tuple in self._file_references[f_path].items():
                    for loc in loc_tuple:
                        ref_scip_id = f"scip-{lang} local 0.1.0 reference/{token_name}"
                        occurrences.append(
                            SCIPOccurrence(
                                range=(
                                    loc.start_line,
                                    loc.start_col,
                                    loc.end_line,
                                    loc.end_col,
                                ),
                                symbol=ref_scip_id,
                                symbol_roles=int(SCIPSymbolRole.READ_ACCESS),
                            )
                        )

            documents.append(
                SCIPDocument(
                    relative_path=f_path,
                    language=lang,
                    symbols=tuple(scip_syms),
                    occurrences=tuple(occurrences),
                )
            )

        freshness = IndexFreshness.STALE if self._is_stale else IndexFreshness.FRESH
        return SCIPIndex(
            metadata=metadata,
            documents=tuple(documents),
            external_symbols=tuple(all_symbol_info),
            freshness=freshness,
            reason=self._stale_reason if self._is_stale else None,
            indexed_at_ns=self._last_indexed_at_ns,
        )

    def mark_stale(self, reason: str | None = None) -> None:
        """Explicitly mark index as stale with a diagnostic reason."""
        self._is_stale = True
        self._stale_reason = reason or "Index marked stale due to filesystem edits"

    def mark_fresh(self) -> None:
        """Clear staleness marker."""
        self._is_stale = False
        self._stale_reason = None

    def clear(self) -> None:
        """Reset all internal symbol graph state."""
        self._file_symbols.clear()
        self._definitions.clear()
        self._short_definitions.clear()
        self._references.clear()
        self._file_references.clear()
        self._file_dependencies.clear()
        self._symbol_hierarchy.clear()
        self._last_indexed_at_ns = None
        self._is_stale = False
        self._stale_reason = None

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _resolve_definition_locations(self, symbol_name: str) -> tuple[SymbolLocation, ...]:
        """Find definition locations by exact name or short name."""
        if symbol_name in self._definitions:
            return tuple(self._definitions[symbol_name])

        if symbol_name in self._short_definitions:
            return tuple(self._short_definitions[symbol_name])

        return ()

    def _evict_file_definitions(self, file_path: Path) -> None:
        """Remove definition locations belonging to a specific file."""
        for sym_name, locs in list(self._definitions.items()):
            filtered = [loc for loc in locs if loc.file_path != file_path]
            if filtered:
                self._definitions[sym_name] = filtered
            else:
                self._definitions.pop(sym_name, None)

        for short_name, locs in list(self._short_definitions.items()):
            filtered = [loc for loc in locs if loc.file_path != file_path]
            if filtered:
                self._short_definitions[short_name] = filtered
            else:
                self._short_definitions.pop(short_name, None)

        # Evict hierarchy for symbols defined in this file
        if file_path in self._file_symbols:
            for sym in self._file_symbols[file_path]:
                self._symbol_hierarchy.pop(sym.name, None)

    def _evict_file_references(self, file_path: Path) -> None:
        """Remove reference locations belonging to a specific file."""
        if file_path not in self._file_references:
            return

        old_map = self._file_references.pop(file_path)
        for sym_name, locs in old_map.items():
            if sym_name in self._references:
                self._references[sym_name] = [
                    loc for loc in self._references[sym_name] if loc not in locs
                ]
                if not self._references[sym_name]:
                    self._references.pop(sym_name, None)
