"""Abstract Syntax Tree parser supporting Tree-sitter and multi-language fallbacks."""

from __future__ import annotations

import ast
import ctypes
import importlib
import re
import warnings
from pathlib import Path
from typing import Any

from uclone_x.code_intel.models import (
    SymbolKind,
    SymbolLocation,
    SymbolNode,
)
from uclone_x.code_intel.protocols import ASTParserProtocol

__all__ = ["ASTParser"]


def module_present(module_name: str) -> bool:
    """Whether a module can actually be imported, not merely named in `sys.modules`."""
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _clean_docstring(raw: str | None) -> str | None:
    """Normalize docstring text or return None if empty."""
    if not raw:
        return None
    cleaned = raw.strip()
    return cleaned if cleaned else None


def _docstring_node(statement: Any) -> Any:
    """The `string` node a Python statement consists of, or None if it is not a bare string.

    The two grammars this module loads disagree on the shape. The legacy
    `tree_sitter_languages` bundle wraps a docstring in an `expression_statement`; the
    tree-sitter-python that `tree-sitter-language-pack` ships puts the `string` directly
    in the block. Reading only the wrapped form lost every docstring, and the module
    symbol with them, once the `code-intel` extra switched grammars (#1100).
    """
    if statement.type == "string":
        return statement
    if statement.type == "expression_statement" and statement.children:
        first = statement.children[0]
        if first.type == "string":
            return first
    return None


def _extract_preceding_comments(lines: list[str], start_row: int) -> str | None:
    """Extract doc comments immediately preceding a declaration line (0-indexed start_row)."""
    doc_lines: list[str] = []
    idx = start_row - 1
    while idx >= 0:
        line = lines[idx].strip()
        if line.startswith("///"):
            doc_lines.insert(0, line[3:].strip())
            idx -= 1
        elif line.startswith("//!") or line.startswith("//"):
            doc_lines.insert(0, line[2:].strip())
            idx -= 1
        elif line.startswith("*") and not line.startswith("*/"):
            doc_lines.insert(0, line[1:].strip())
            idx -= 1
        elif line.startswith("/**"):
            doc_lines.insert(0, line[3:].rstrip("*/").strip())
            idx -= 1
        elif line.startswith("/*") and line.endswith("*/"):
            doc_lines.insert(0, line[2:-2].strip())
            idx -= 1
        else:
            break
    if doc_lines:
        cleaned = "\n".join(doc_lines).strip()
        return cleaned if cleaned else None
    return None


class ASTParser(ASTParserProtocol):
    """In-process multi-language AST parser using Tree-sitter with robust fallbacks.

    Conforms to `ASTParserProtocol`. Supports Python, TypeScript, JavaScript,
    Rust, and Go. Extracts symbol definitions, signatures, docstrings,
    hierarchical children, imports, and identifier references.
    """

    def __init__(self, use_tree_sitter: bool = True, strict_tree_sitter: bool = False) -> None:
        self._use_tree_sitter = use_tree_sitter
        self._strict_tree_sitter = strict_tree_sitter
        self._ts_parsers: dict[str, Any] = {}
        self._ts_load_attempted: dict[str, bool] = {}
        if strict_tree_sitter:
            self.require_tree_sitter("python")

    def is_tree_sitter_available(self, language: str = "python") -> bool:
        """Check if Tree-sitter grammar and parser are available for a given language."""
        if not self._use_tree_sitter:
            return False
        return self._get_tree_sitter_parser(language) is not None

    def require_tree_sitter(self, language: str = "python") -> Any:
        """Return the Tree-sitter parser for a language, or raise MissingDependencyError (P6)."""
        from uclone_x.errors import MissingDependencyError

        try:
            importlib.import_module("tree_sitter")
        except ImportError as exc:
            raise MissingDependencyError(
                extra="code-intel",
                package="tree-sitter",
                feature=f"Tree-sitter AST parsing for {language}",
            ) from exc
        if not (
            module_present("tree_sitter_language_pack") or module_present("tree_sitter_languages")
        ):
            raise MissingDependencyError(
                extra="code-intel",
                package="tree-sitter-language-pack",
                feature=f"Tree-sitter AST parsing for {language}",
            )
        parser = self._get_tree_sitter_parser(language)
        if parser is None:
            raise MissingDependencyError(
                extra="code-intel",
                package="tree-sitter-language-pack",
                feature=f"Tree-sitter AST parsing for {language}",
            )
        return parser

    def _get_tree_sitter_parser(self, lang_name: str) -> Any:
        """Lazily load and cache Tree-sitter parser for a language."""
        if lang_name in self._ts_parsers:
            return self._ts_parsers[lang_name]

        if self._ts_load_attempted.get(lang_name, False):
            return None

        self._ts_load_attempted[lang_name] = True

        # `tree-sitter-language-pack` first: it is the maintained successor and the only one
        # of the two with a cp313 wheel (#1100). `tree-sitter-languages` stays as a fallback
        # so an environment already holding it keeps working without a re-install.
        parser = self._load_from_language_pack(lang_name) or self._load_from_language_bundle(
            lang_name
        )
        if parser is not None:
            self._ts_parsers[lang_name] = parser
        return parser

    def _load_from_language_pack(self, lang_name: str) -> Any:
        """Load a parser through `tree_sitter_language_pack.get_parser`, or None."""
        # Imported by name rather than statically: the distribution is optional, so a
        # static import would be an unresolved one in every environment without the
        # `code-intel` extra -- including the shared dev venv the type checker reads.
        try:
            pack = importlib.import_module("tree_sitter_language_pack")
            get_parser: Any = pack.get_parser
            return get_parser(lang_name)
        except Exception:
            return None

    def _load_from_language_bundle(self, lang_name: str) -> Any:
        """Load a parser out of the legacy `tree_sitter_languages` shared object, or None.

        Kept for environments that already hold the unmaintained bundle; it ships no wheel
        above cp312, which is why it is no longer what the `code-intel` extra installs.
        """
        try:
            import tree_sitter

            # Imported by name for the same reason as the language pack above, and more so:
            # no extra installs this distribution any more, so a static import is unresolved
            # in every environment synced from the lock file (#1100).
            bundle = importlib.import_module("tree_sitter_languages")
            # A namespace package has no `__file__`, and then there is no directory to search.
            if bundle.__file__ is None:
                return None

            ts_dir = Path(bundle.__file__).parent
            so_files = (
                list(ts_dir.glob("*.so"))
                + list(ts_dir.glob("*.dylib"))
                + list(ts_dir.glob("*.dll"))
            )
            if not so_files:
                return None

            cdll = ctypes.cdll.LoadLibrary(str(so_files[0]))
            func_name = f"tree_sitter_{lang_name}"
            lang_func: Any = getattr(cdll, func_name, None)
            if lang_func is None:
                return None

            lang_func.restype = ctypes.c_void_p
            ptr: int | None = lang_func()
            if not ptr:
                return None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                ts_language = tree_sitter.Language(ptr)  # pyright: ignore[reportDeprecated]

            parser = tree_sitter.Parser()
            parser.language = ts_language
            return parser
        except Exception:
            return None

    def parse_file(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Parse source file into symbol nodes without executing code."""
        if not content or not content.strip():
            return ()

        path_obj = Path(file_path)
        suffix = path_obj.suffix.lower()

        try:
            if suffix in {".py", ".pyi"}:
                return self._parse_python(path_obj, content)
            if suffix in {".ts", ".tsx"}:
                return self._parse_typescript(path_obj, content, is_tsx=suffix == ".tsx")
            if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
                return self._parse_javascript(path_obj, content)
            if suffix == ".rs":
                return self._parse_rust(path_obj, content)
            if suffix == ".go":
                return self._parse_go(path_obj, content)
        except Exception:
            return ()

        return ()

    # =========================================================================
    # Python Parsing (Tree-sitter -> stdlib AST fallback)
    # =========================================================================

    def _parse_python(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Parse Python source into symbols using Tree-sitter or stdlib fallback."""
        if self._use_tree_sitter:
            ts_parser = self._get_tree_sitter_parser("python")
            if ts_parser is not None:
                try:
                    symbols = self._parse_python_tree_sitter(file_path, content, ts_parser)
                    if symbols:
                        return symbols
                except Exception:
                    pass

        return self._parse_python_ast(file_path, content)

    def _parse_python_tree_sitter(
        self, file_path: Path, content: str, parser: Any
    ) -> tuple[SymbolNode, ...]:
        """Extract Python symbols from Tree-sitter AST."""
        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)
        root = tree.root_node
        lines: list[str] = content.splitlines()

        symbols: list[SymbolNode] = []

        # Check module docstring
        module_doc = _docstring_node(root.children[0]) if root.children else None
        if module_doc is not None:
            raw_text: str = module_doc.text.decode("utf-8", errors="replace")
            doc = raw_text.strip("\"' \t\r\n")
            if doc:
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=1,
                    start_col=0,
                    end_line=max(1, len(lines)),
                    end_col=len(lines[-1]) if lines else 0,
                )
                symbols.append(
                    SymbolNode(
                        name=file_path.stem,
                        kind=SymbolKind.MODULE,
                        location=loc,
                        docstring=doc,
                    )
                )

        def _get_ts_docstring(node: Any) -> str | None:
            body = node.child_by_field_name("body")
            if not body or not body.children:
                return None
            for child in body.children:
                sub = _docstring_node(child)
                if sub is not None:
                    raw: str = sub.text.decode("utf-8", errors="replace")
                    return _clean_docstring(raw.strip("\"' \t\r\n"))
                if child.type not in {"comment"}:
                    break
            return None

        def _extract_py_class_children(class_node: Any) -> tuple[list[str], list[SymbolNode]]:
            body = class_node.child_by_field_name("body")
            child_names: list[str] = []
            child_symbols: list[SymbolNode] = []
            if not body:
                return child_names, child_symbols

            class_name_n = class_node.child_by_field_name("name")
            class_name: str = (
                class_name_n.text.decode("utf-8", errors="replace") if class_name_n else ""
            )

            for b_node in body.children:
                curr = b_node
                if curr.type == "decorated_definition":
                    for c in curr.children:
                        if c.type in {"function_definition", "class_definition"}:
                            curr = c
                            break

                if curr.type == "function_definition":
                    fn_name_n = curr.child_by_field_name("name")
                    if fn_name_n:
                        fn_name: str = fn_name_n.text.decode("utf-8", errors="replace")
                        child_names.append(fn_name)
                        doc = _get_ts_docstring(curr)
                        start_r: int = int(curr.start_point.row)
                        start_c: int = int(curr.start_point.column)
                        end_r: int = int(curr.end_point.row)
                        end_c: int = int(curr.end_point.column)
                        loc = SymbolLocation(
                            file_path=file_path,
                            start_line=start_r + 1,
                            start_col=start_c,
                            end_line=end_r + 1,
                            end_col=end_c,
                        )
                        sig_line: str | None = (
                            lines[start_r].strip() if start_r < len(lines) else None
                        )
                        child_symbols.append(
                            SymbolNode(
                                name=f"{class_name}.{fn_name}" if class_name else fn_name,
                                kind=SymbolKind.METHOD,
                                location=loc,
                                signature=sig_line,
                                docstring=doc,
                            )
                        )
                elif curr.type in {"expression_statement", "annotated_assignment", "assignment"}:
                    target = (
                        curr.children[0]
                        if curr.type == "expression_statement" and curr.children
                        else curr
                    )
                    if target.type in {"assignment", "annotated_assignment"}:
                        left = target.child_by_field_name("left")
                        if left and left.type == "identifier":
                            var_name: str = left.text.decode("utf-8", errors="replace")
                            child_names.append(var_name)
                            start_r = int(target.start_point.row)
                            start_c = int(target.start_point.column)
                            end_r = int(target.end_point.row)
                            end_c = int(target.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig_line = lines[start_r].strip() if start_r < len(lines) else None
                            child_symbols.append(
                                SymbolNode(
                                    name=f"{class_name}.{var_name}" if class_name else var_name,
                                    kind=SymbolKind.VARIABLE,
                                    location=loc,
                                    signature=sig_line,
                                )
                            )

            return child_names, child_symbols

        for node in root.children:
            curr = node
            if curr.type == "decorated_definition":
                for c in curr.children:
                    if c.type in {"function_definition", "class_definition"}:
                        curr = c
                        break

            if curr.type == "class_definition":
                name_n = curr.child_by_field_name("name")
                name: str = name_n.text.decode("utf-8", errors="replace") if name_n else "Anonymous"
                doc = _get_ts_docstring(curr)
                start_r = int(curr.start_point.row)
                start_c = int(curr.start_point.column)
                end_r = int(curr.end_point.row)
                end_c = int(curr.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                child_names, inner_symbols = _extract_py_class_children(curr)
                sig_line = lines[start_r].strip() if start_r < len(lines) else None
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=sig_line,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif curr.type == "function_definition":
                name_n = curr.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "anonymous"
                doc = _get_ts_docstring(curr)
                start_r = int(curr.start_point.row)
                start_c = int(curr.start_point.column)
                end_r = int(curr.end_point.row)
                end_c = int(curr.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig_line = lines[start_r].strip() if start_r < len(lines) else None
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=sig_line,
                        docstring=doc,
                    )
                )

            elif curr.type in {"expression_statement", "annotated_assignment", "assignment"}:
                target = (
                    curr.children[0]
                    if curr.type == "expression_statement" and curr.children
                    else curr
                )
                if target.type in {"assignment", "annotated_assignment"}:
                    left = target.child_by_field_name("left")
                    if left and left.type == "identifier":
                        var_name = left.text.decode("utf-8", errors="replace")
                        start_r = int(target.start_point.row)
                        start_c = int(target.start_point.column)
                        end_r = int(target.end_point.row)
                        end_c = int(target.end_point.column)
                        loc = SymbolLocation(
                            file_path=file_path,
                            start_line=start_r + 1,
                            start_col=start_c,
                            end_line=end_r + 1,
                            end_col=end_c,
                        )
                        sig_line = lines[start_r].strip() if start_r < len(lines) else None
                        symbols.append(
                            SymbolNode(
                                name=var_name,
                                kind=SymbolKind.VARIABLE,
                                location=loc,
                                signature=sig_line,
                            )
                        )

        return tuple(symbols)

    def _parse_python_ast(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Extract Python symbols using the standard library `ast` module."""
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return ()

        lines: list[str] = content.splitlines()
        symbols: list[SymbolNode] = []

        # Module docstring
        mod_doc = _clean_docstring(ast.get_docstring(tree))
        if mod_doc:
            loc = SymbolLocation(
                file_path=file_path,
                start_line=1,
                start_col=0,
                end_line=max(1, len(lines)),
                end_col=len(lines[-1]) if lines else 0,
            )
            symbols.append(
                SymbolNode(
                    name=file_path.stem,
                    kind=SymbolKind.MODULE,
                    location=loc,
                    docstring=mod_doc,
                )
            )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = _clean_docstring(ast.get_docstring(node))
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=node.lineno,
                    start_col=node.col_offset,
                    end_line=node.end_lineno or node.lineno,
                    end_col=node.end_col_offset or 0,
                )
                child_names: list[str] = []
                inner_symbols: list[SymbolNode] = []

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child_names.append(item.name)
                        item_doc = _clean_docstring(ast.get_docstring(item))
                        item_loc = SymbolLocation(
                            file_path=file_path,
                            start_line=item.lineno,
                            start_col=item.col_offset,
                            end_line=item.end_lineno or item.lineno,
                            end_col=item.end_col_offset or 0,
                        )
                        sig = (
                            f"{'async ' if isinstance(item, ast.AsyncFunctionDef) else ''}"
                            f"def {item.name}({ast.unparse(item.args)})"
                            f"{' -> ' + ast.unparse(item.returns) if item.returns else ''}:"
                        )
                        inner_symbols.append(
                            SymbolNode(
                                name=f"{node.name}.{item.name}",
                                kind=SymbolKind.METHOD,
                                location=item_loc,
                                signature=sig,
                                docstring=item_doc,
                            )
                        )
                    elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                        target_names = self._get_assign_targets(item)
                        for t_name in target_names:
                            child_names.append(t_name)
                            item_loc = SymbolLocation(
                                file_path=file_path,
                                start_line=item.lineno,
                                start_col=item.col_offset,
                                end_line=item.end_lineno or item.lineno,
                                end_col=item.end_col_offset or 0,
                            )
                            inner_symbols.append(
                                SymbolNode(
                                    name=f"{node.name}.{t_name}",
                                    kind=SymbolKind.VARIABLE,
                                    location=item_loc,
                                    signature=f"{t_name} = ...",
                                )
                            )

                bases_str = (
                    f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
                )
                sig = f"class {node.name}{bases_str}:"
                symbols.append(
                    SymbolNode(
                        name=node.name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = _clean_docstring(ast.get_docstring(node))
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=node.lineno,
                    start_col=node.col_offset,
                    end_line=node.end_lineno or node.lineno,
                    end_col=node.end_col_offset or 0,
                )
                sig = (
                    f"{'async ' if isinstance(node, ast.AsyncFunctionDef) else ''}"
                    f"def {node.name}({ast.unparse(node.args)})"
                    f"{' -> ' + ast.unparse(node.returns) if node.returns else ''}:"
                )
                symbols.append(
                    SymbolNode(
                        name=node.name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                    )
                )

            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_names = self._get_assign_targets(node)
                for t_name in target_names:
                    loc = SymbolLocation(
                        file_path=file_path,
                        start_line=node.lineno,
                        start_col=node.col_offset,
                        end_line=node.end_lineno or node.lineno,
                        end_col=node.end_col_offset or 0,
                    )
                    sig_line = (
                        lines[node.lineno - 1].strip()
                        if node.lineno <= len(lines)
                        else f"{t_name} = ..."
                    )
                    symbols.append(
                        SymbolNode(
                            name=t_name,
                            kind=SymbolKind.VARIABLE,
                            location=loc,
                            signature=sig_line,
                        )
                    )

        return tuple(symbols)

    def _get_assign_targets(self, node: ast.Assign | ast.AnnAssign) -> list[str]:
        """Extract variable names from an assignment node."""
        targets: list[str] = []
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                targets.append(node.target.id)
        else:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        return targets

    # =========================================================================
    # TypeScript & JavaScript Parsing
    # =========================================================================

    def _parse_typescript(
        self, file_path: Path, content: str, is_tsx: bool = False
    ) -> tuple[SymbolNode, ...]:
        """Parse TypeScript source using Tree-sitter or regex fallback."""
        if self._use_tree_sitter:
            lang = "tsx" if is_tsx else "typescript"
            ts_parser = self._get_tree_sitter_parser(lang)
            if ts_parser is not None:
                try:
                    symbols = self._parse_ts_tree_sitter(file_path, content, ts_parser)
                    if symbols:
                        return symbols
                except Exception:
                    pass

        return self._parse_ts_fallback(file_path, content)

    def _parse_javascript(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Parse JavaScript source using Tree-sitter or regex fallback."""
        if self._use_tree_sitter:
            ts_parser = self._get_tree_sitter_parser("javascript")
            if ts_parser is not None:
                try:
                    symbols = self._parse_ts_tree_sitter(file_path, content, ts_parser)
                    if symbols:
                        return symbols
                except Exception:
                    pass

        return self._parse_ts_fallback(file_path, content)

    def _parse_ts_tree_sitter(
        self, file_path: Path, content: str, parser: Any
    ) -> tuple[SymbolNode, ...]:
        """Extract TypeScript/JavaScript symbols from Tree-sitter AST."""
        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)
        root = tree.root_node
        lines: list[str] = content.splitlines()

        symbols: list[SymbolNode] = []

        def _unwrap_export(node: Any) -> Any:
            if node.type == "export_statement":
                for c in node.children:
                    if c.type in {
                        "interface_declaration",
                        "class_declaration",
                        "function_declaration",
                        "lexical_declaration",
                        "variable_declaration",
                        "type_alias_declaration",
                        "enum_declaration",
                    }:
                        return c
            return node

        def _extract_ts_class_children(class_node: Any) -> tuple[list[str], list[SymbolNode]]:
            body = class_node.child_by_field_name("body")
            child_names: list[str] = []
            child_symbols: list[SymbolNode] = []
            if not body:
                return child_names, child_symbols

            class_name_n = class_node.child_by_field_name("name")
            class_name: str = (
                class_name_n.text.decode("utf-8", errors="replace") if class_name_n else ""
            )

            for m in body.children:
                if m.type in {
                    "method_definition",
                    "public_field_definition",
                    "field_definition",
                    "property_signature",
                    "method_signature",
                }:
                    mn = m.child_by_field_name("name")
                    if mn:
                        m_name: str = mn.text.decode("utf-8", errors="replace")
                        child_names.append(m_name)
                        is_method = "method" in m.type
                        start_r: int = int(m.start_point.row)
                        start_c: int = int(m.start_point.column)
                        end_r: int = int(m.end_point.row)
                        end_c: int = int(m.end_point.column)
                        loc = SymbolLocation(
                            file_path=file_path,
                            start_line=start_r + 1,
                            start_col=start_c,
                            end_line=end_r + 1,
                            end_col=end_c,
                        )
                        sig_line: str | None = (
                            lines[start_r].strip() if start_r < len(lines) else None
                        )
                        doc = _extract_preceding_comments(lines, start_r)
                        child_symbols.append(
                            SymbolNode(
                                name=f"{class_name}.{m_name}" if class_name else m_name,
                                kind=SymbolKind.METHOD if is_method else SymbolKind.VARIABLE,
                                location=loc,
                                signature=sig_line,
                                docstring=doc,
                            )
                        )
            return child_names, child_symbols

        for node in root.children:
            curr = _unwrap_export(node)

            if curr.type == "interface_declaration":
                name_n = curr.child_by_field_name("name")
                name: str = name_n.text.decode("utf-8", errors="replace") if name_n else "Interface"
                start_r = int(curr.start_point.row)
                start_c = int(curr.start_point.column)
                end_r = int(curr.end_point.row)
                end_c = int(curr.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                child_names, inner_symbols = _extract_ts_class_children(curr)
                sig_line = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=loc,
                        signature=sig_line,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif curr.type == "class_declaration":
                name_n = curr.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "Class"
                start_r = int(curr.start_point.row)
                start_c = int(curr.start_point.column)
                end_r = int(curr.end_point.row)
                end_c = int(curr.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                child_names, inner_symbols = _extract_ts_class_children(curr)
                sig_line = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=sig_line,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif curr.type == "function_declaration":
                name_n = curr.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "function"
                start_r = int(curr.start_point.row)
                start_c = int(curr.start_point.column)
                end_r = int(curr.end_point.row)
                end_c = int(curr.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig_line = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=sig_line,
                        docstring=doc,
                    )
                )

            elif curr.type in {"lexical_declaration", "variable_declaration"}:
                for d in curr.children:
                    if d.type == "variable_declarator":
                        dn = d.child_by_field_name("name")
                        if dn:
                            var_name: str = dn.text.decode("utf-8", errors="replace")
                            val = d.child_by_field_name("value")
                            is_fn = val is not None and val.type in {"arrow_function", "function"}
                            start_r = int(d.start_point.row)
                            start_c = int(d.start_point.column)
                            end_r = int(d.end_point.row)
                            end_c = int(d.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig_line = lines[start_r].strip() if start_r < len(lines) else None
                            symbols.append(
                                SymbolNode(
                                    name=var_name,
                                    kind=SymbolKind.FUNCTION if is_fn else SymbolKind.VARIABLE,
                                    location=loc,
                                    signature=sig_line,
                                )
                            )

        return tuple(symbols)

    def _parse_ts_fallback(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Regex-based fallback symbol extraction for TypeScript / JavaScript."""
        lines: list[str] = content.splitlines()
        symbols: list[SymbolNode] = []

        interface_re = re.compile(r"^(?:export\s+)?interface\s+([A-Za-z0-9_$]+)")
        class_re = re.compile(r"^(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z0-9_$]+)")
        function_re = re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)")
        const_fn_re = re.compile(
            r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z0-9_$]+)\s*=>"
        )
        var_re = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z0-9_$]+)")

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            doc = _extract_preceding_comments(lines, idx - 1)

            m = interface_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = class_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = function_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = const_fn_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = var_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )

        return tuple(symbols)

    # =========================================================================
    # Rust Parsing (Tree-sitter -> regex fallback)
    # =========================================================================

    def _parse_rust(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Parse Rust source using Tree-sitter or regex fallback."""
        if self._use_tree_sitter:
            ts_parser = self._get_tree_sitter_parser("rust")
            if ts_parser is not None:
                try:
                    symbols = self._parse_rust_tree_sitter(file_path, content, ts_parser)
                    if symbols:
                        return symbols
                except Exception:
                    pass

        return self._parse_rust_fallback(file_path, content)

    def _parse_rust_tree_sitter(
        self, file_path: Path, content: str, parser: Any
    ) -> tuple[SymbolNode, ...]:
        """Extract Rust symbols from Tree-sitter AST."""
        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)
        root = tree.root_node
        lines: list[str] = content.splitlines()

        symbols: list[SymbolNode] = []

        # Check module docstring at top
        mod_doc = _extract_preceding_comments(lines, 1) or (
            lines[0][3:].strip() if lines and lines[0].startswith("//!") else None
        )
        if mod_doc:
            loc = SymbolLocation(
                file_path=file_path,
                start_line=1,
                start_col=0,
                end_line=max(1, len(lines)),
                end_col=len(lines[-1]) if lines else 0,
            )
            symbols.append(
                SymbolNode(
                    name=file_path.stem,
                    kind=SymbolKind.MODULE,
                    location=loc,
                    docstring=mod_doc,
                )
            )

        def _extract_struct_fields(struct_node: Any) -> tuple[list[str], list[SymbolNode]]:
            child_names: list[str] = []
            child_symbols: list[SymbolNode] = []
            name_n = struct_node.child_by_field_name("name")
            struct_name = name_n.text.decode("utf-8", errors="replace") if name_n else ""

            body = struct_node.child_by_field_name("body")
            if body and body.type == "field_declaration_list":
                for field in body.children:
                    if field.type == "field_declaration":
                        fn = field.child_by_field_name("name")
                        if fn:
                            f_name = fn.text.decode("utf-8", errors="replace")
                            child_names.append(f_name)
                            start_r = int(field.start_point.row)
                            start_c = int(field.start_point.column)
                            end_r = int(field.end_point.row)
                            end_c = int(field.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig = lines[start_r].strip() if start_r < len(lines) else None
                            doc = _extract_preceding_comments(lines, start_r)
                            child_symbols.append(
                                SymbolNode(
                                    name=f"{struct_name}.{f_name}" if struct_name else f_name,
                                    kind=SymbolKind.VARIABLE,
                                    location=loc,
                                    signature=sig,
                                    docstring=doc,
                                )
                            )
            return child_names, child_symbols

        def _extract_enum_variants(enum_node: Any) -> tuple[list[str], list[SymbolNode]]:
            child_names: list[str] = []
            child_symbols: list[SymbolNode] = []
            name_n = enum_node.child_by_field_name("name")
            enum_name = name_n.text.decode("utf-8", errors="replace") if name_n else ""

            body = enum_node.child_by_field_name("body")
            if body and body.type == "enum_variant_list":
                for variant in body.children:
                    if variant.type == "enum_variant":
                        vn = variant.child_by_field_name("name")
                        if vn:
                            v_name = vn.text.decode("utf-8", errors="replace")
                            child_names.append(v_name)
                            start_r = int(variant.start_point.row)
                            start_c = int(variant.start_point.column)
                            end_r = int(variant.end_point.row)
                            end_c = int(variant.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig = lines[start_r].strip() if start_r < len(lines) else None
                            child_symbols.append(
                                SymbolNode(
                                    name=f"{enum_name}.{v_name}" if enum_name else v_name,
                                    kind=SymbolKind.VARIABLE,
                                    location=loc,
                                    signature=sig,
                                )
                            )
            return child_names, child_symbols

        def _extract_trait_items(trait_node: Any) -> tuple[list[str], list[SymbolNode]]:
            child_names: list[str] = []
            child_symbols: list[SymbolNode] = []
            name_n = trait_node.child_by_field_name("name")
            trait_name = name_n.text.decode("utf-8", errors="replace") if name_n else ""

            body = trait_node.child_by_field_name("body")
            if body and body.type == "declaration_list":
                for item in body.children:
                    if item.type in {"function_item", "function_signature_item"}:
                        fn = item.child_by_field_name("name")
                        if fn:
                            f_name = fn.text.decode("utf-8", errors="replace")
                            child_names.append(f_name)
                            start_r = int(item.start_point.row)
                            start_c = int(item.start_point.column)
                            end_r = int(item.end_point.row)
                            end_c = int(item.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig = lines[start_r].strip() if start_r < len(lines) else None
                            doc = _extract_preceding_comments(lines, start_r)
                            child_symbols.append(
                                SymbolNode(
                                    name=f"{trait_name}.{f_name}" if trait_name else f_name,
                                    kind=SymbolKind.METHOD,
                                    location=loc,
                                    signature=sig,
                                    docstring=doc,
                                )
                            )
            return child_names, child_symbols

        def _extract_impl_items(impl_node: Any) -> list[SymbolNode]:
            child_symbols: list[SymbolNode] = []
            type_n = impl_node.child_by_field_name("type")
            target_type = (
                type_n.text.decode("utf-8", errors="replace").split("<")[0].strip()
                if type_n
                else ""
            )

            body = impl_node.child_by_field_name("body")
            if body and body.type == "declaration_list":
                for item in body.children:
                    if item.type == "function_item":
                        fn = item.child_by_field_name("name")
                        if fn:
                            f_name = fn.text.decode("utf-8", errors="replace")
                            start_r = int(item.start_point.row)
                            start_c = int(item.start_point.column)
                            end_r = int(item.end_point.row)
                            end_c = int(item.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig = lines[start_r].strip() if start_r < len(lines) else None
                            doc = _extract_preceding_comments(lines, start_r)
                            child_symbols.append(
                                SymbolNode(
                                    name=f"{target_type}.{f_name}" if target_type else f_name,
                                    kind=SymbolKind.METHOD,
                                    location=loc,
                                    signature=sig,
                                    docstring=doc,
                                )
                            )
                    elif item.type == "const_item":
                        cn = item.child_by_field_name("name")
                        if cn:
                            c_name = cn.text.decode("utf-8", errors="replace")
                            start_r = int(item.start_point.row)
                            start_c = int(item.start_point.column)
                            end_r = int(item.end_point.row)
                            end_c = int(item.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig = lines[start_r].strip() if start_r < len(lines) else None
                            child_symbols.append(
                                SymbolNode(
                                    name=f"{target_type}.{c_name}" if target_type else c_name,
                                    kind=SymbolKind.VARIABLE,
                                    location=loc,
                                    signature=sig,
                                )
                            )
            return child_symbols

        for node in root.children:
            if node.type == "struct_item":
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "Struct"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                child_names, inner_symbols = _extract_struct_fields(node)
                sig = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif node.type == "enum_item":
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "Enum"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                child_names, inner_symbols = _extract_enum_variants(node)
                sig = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif node.type == "trait_item":
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "Trait"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                child_names, inner_symbols = _extract_trait_items(node)
                sig = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                        children=tuple(child_names),
                    )
                )
                symbols.extend(inner_symbols)

            elif node.type == "impl_item":
                inner_symbols = _extract_impl_items(node)
                symbols.extend(inner_symbols)

            elif node.type == "function_item":
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "func"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                    )
                )

            elif node.type in {"const_item", "static_item"}:
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "CONST"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig = lines[start_r].strip() if start_r < len(lines) else None
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=loc,
                        signature=sig,
                    )
                )

            elif node.type == "mod_item":
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "mod"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig = lines[start_r].strip() if start_r < len(lines) else None
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.MODULE,
                        location=loc,
                        signature=sig,
                    )
                )

        return tuple(symbols)

    def _parse_rust_fallback(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Regex fallback for Rust syntax parsing."""
        lines: list[str] = content.splitlines()
        symbols: list[SymbolNode] = []

        struct_re = re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?struct\s+([A-Za-z0-9_]+)")
        enum_re = re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?enum\s+([A-Za-z0-9_]+)")
        trait_re = re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?trait\s+([A-Za-z0-9_]+)")
        impl_re = re.compile(r"^impl(?:<[^>]+>)?\s+(?:([A-Za-z0-9_]+)\s+for\s+)?([A-Za-z0-9_]+)")
        fn_re = re.compile(
            r"^(?:pub(?:\([^)]+\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)"
        )
        const_re = re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?(?:const|static)\s+([A-Za-z0-9_]+)")
        mod_re = re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?mod\s+([A-Za-z0-9_]+)")

        current_impl_target: str | None = None
        current_impl_depth = 0

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            doc = _extract_preceding_comments(lines, idx - 1)

            # Track impl block scope
            if current_impl_target is not None:
                if "{" in stripped:
                    current_impl_depth += stripped.count("{")
                if "}" in stripped:
                    current_impl_depth -= stripped.count("}")
                    if current_impl_depth <= 0:
                        current_impl_target = None
                        current_impl_depth = 0

            m_impl = impl_re.match(stripped)
            if m_impl:
                current_impl_target = m_impl.group(2)
                current_impl_depth = stripped.count("{") - stripped.count("}")
                continue

            m = struct_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = enum_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = trait_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = fn_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                if current_impl_target:
                    symbols.append(
                        SymbolNode(
                            name=f"{current_impl_target}.{name}",
                            kind=SymbolKind.METHOD,
                            location=loc,
                            signature=stripped,
                            docstring=doc,
                        )
                    )
                else:
                    symbols.append(
                        SymbolNode(
                            name=name,
                            kind=SymbolKind.FUNCTION,
                            location=loc,
                            signature=stripped,
                            docstring=doc,
                        )
                    )
                continue

            m = const_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = mod_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.MODULE,
                        location=loc,
                        signature=stripped,
                    )
                )

        return tuple(symbols)

    # =========================================================================
    # Go Parsing (Tree-sitter -> regex fallback)
    # =========================================================================

    def _parse_go(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Parse Go source using Tree-sitter or regex fallback."""
        if self._use_tree_sitter:
            ts_parser = self._get_tree_sitter_parser("go")
            if ts_parser is not None:
                try:
                    symbols = self._parse_go_tree_sitter(file_path, content, ts_parser)
                    if symbols:
                        return symbols
                except Exception:
                    pass

        return self._parse_go_fallback(file_path, content)

    def _parse_go_tree_sitter(
        self, file_path: Path, content: str, parser: Any
    ) -> tuple[SymbolNode, ...]:
        """Extract Go symbols from Tree-sitter AST."""
        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)
        root = tree.root_node
        lines: list[str] = content.splitlines()

        symbols: list[SymbolNode] = []

        def _extract_go_receiver_type(recv_node: Any) -> str:
            raw_text = recv_node.text.decode("utf-8", errors="replace").strip("() \t\r\n")
            parts = raw_text.split()
            type_part = parts[-1] if len(parts) >= 2 else (parts[0] if parts else "")
            return type_part.lstrip("*&")

        for node in root.children:
            if node.type == "package_clause":
                for c in node.children:
                    if c.type == "package_identifier":
                        pkg_name = c.text.decode("utf-8", errors="replace")
                        start_r = int(node.start_point.row)
                        start_c = int(node.start_point.column)
                        end_r = int(node.end_point.row)
                        end_c = int(node.end_point.column)
                        loc = SymbolLocation(
                            file_path=file_path,
                            start_line=start_r + 1,
                            start_col=start_c,
                            end_line=end_r + 1,
                            end_col=end_c,
                        )
                        doc = _extract_preceding_comments(lines, start_r)
                        symbols.append(
                            SymbolNode(
                                name=pkg_name,
                                kind=SymbolKind.MODULE,
                                location=loc,
                                signature=lines[start_r].strip() if start_r < len(lines) else None,
                                docstring=doc,
                            )
                        )

            elif node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name_n = child.child_by_field_name("name")
                        type_name = (
                            name_n.text.decode("utf-8", errors="replace") if name_n else "Type"
                        )
                        type_node = child.child_by_field_name("type")

                        start_r = int(child.start_point.row)
                        start_c = int(child.start_point.column)
                        end_r = int(child.end_point.row)
                        end_c = int(child.end_point.column)
                        loc = SymbolLocation(
                            file_path=file_path,
                            start_line=start_r + 1,
                            start_col=start_c,
                            end_line=end_r + 1,
                            end_col=end_c,
                        )
                        doc = _extract_preceding_comments(
                            lines, int(node.start_point.row)
                        ) or _extract_preceding_comments(lines, start_r)
                        sig = lines[start_r].strip() if start_r < len(lines) else None

                        is_struct = type_node and type_node.type == "struct_type"
                        is_interface = type_node and type_node.type == "interface_type"

                        child_names: list[str] = []
                        inner_symbols: list[SymbolNode] = []

                        if is_struct and type_node:
                            for f_decl in type_node.children:
                                if f_decl.type == "field_declaration_list":
                                    for item in f_decl.children:
                                        if item.type == "field_declaration":
                                            fn = item.child_by_field_name("name")
                                            if fn:
                                                f_name = fn.text.decode("utf-8", errors="replace")
                                                child_names.append(f_name)
                                                f_start_r = int(item.start_point.row)
                                                f_start_c = int(item.start_point.column)
                                                f_end_r = int(item.end_point.row)
                                                f_end_c = int(item.end_point.column)
                                                f_loc = SymbolLocation(
                                                    file_path=file_path,
                                                    start_line=f_start_r + 1,
                                                    start_col=f_start_c,
                                                    end_line=f_end_r + 1,
                                                    end_col=f_end_c,
                                                )
                                                inner_symbols.append(
                                                    SymbolNode(
                                                        name=f"{type_name}.{f_name}",
                                                        kind=SymbolKind.VARIABLE,
                                                        location=f_loc,
                                                        signature=lines[f_start_r].strip()
                                                        if f_start_r < len(lines)
                                                        else None,
                                                    )
                                                )
                        elif is_interface and type_node:
                            for elem in type_node.children:
                                if elem.type in {"method_spec_list", "method_spec"}:
                                    specs = (
                                        elem.children if elem.type == "method_spec_list" else [elem]
                                    )
                                    for spec in specs:
                                        if spec.type == "method_spec":
                                            fn = spec.child_by_field_name("name")
                                            if fn:
                                                m_name = fn.text.decode("utf-8", errors="replace")
                                                child_names.append(m_name)
                                                m_start_r = int(spec.start_point.row)
                                                m_start_c = int(spec.start_point.column)
                                                m_end_r = int(spec.end_point.row)
                                                m_end_c = int(spec.end_point.column)
                                                m_loc = SymbolLocation(
                                                    file_path=file_path,
                                                    start_line=m_start_r + 1,
                                                    start_col=m_start_c,
                                                    end_line=m_end_r + 1,
                                                    end_col=m_end_c,
                                                )
                                                inner_symbols.append(
                                                    SymbolNode(
                                                        name=f"{type_name}.{m_name}",
                                                        kind=SymbolKind.METHOD,
                                                        location=m_loc,
                                                        signature=lines[m_start_r].strip()
                                                        if m_start_r < len(lines)
                                                        else None,
                                                    )
                                                )

                        symbols.append(
                            SymbolNode(
                                name=type_name,
                                kind=SymbolKind.INTERFACE if is_interface else SymbolKind.CLASS,
                                location=loc,
                                signature=sig,
                                docstring=doc,
                                children=tuple(child_names),
                            )
                        )
                        symbols.extend(inner_symbols)

            elif node.type == "function_declaration":
                name_n = node.child_by_field_name("name")
                name = name_n.text.decode("utf-8", errors="replace") if name_n else "func"
                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                    )
                )

            elif node.type == "method_declaration":
                recv_node = node.child_by_field_name("receiver")
                recv_type = _extract_go_receiver_type(recv_node) if recv_node else ""
                name_n = node.child_by_field_name("name")
                method_name = name_n.text.decode("utf-8", errors="replace") if name_n else "method"
                full_name = f"{recv_type}.{method_name}" if recv_type else method_name

                start_r = int(node.start_point.row)
                start_c = int(node.start_point.column)
                end_r = int(node.end_point.row)
                end_c = int(node.end_point.column)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=start_r + 1,
                    start_col=start_c,
                    end_line=end_r + 1,
                    end_col=end_c,
                )
                sig = lines[start_r].strip() if start_r < len(lines) else None
                doc = _extract_preceding_comments(lines, start_r)
                symbols.append(
                    SymbolNode(
                        name=full_name,
                        kind=SymbolKind.METHOD,
                        location=loc,
                        signature=sig,
                        docstring=doc,
                    )
                )

            elif node.type in {"const_declaration", "var_declaration"}:
                for child in node.children:
                    if child.type in {"const_spec", "var_spec"}:
                        name_n = child.child_by_field_name("name")
                        if name_n:
                            var_name = name_n.text.decode("utf-8", errors="replace")
                            start_r = int(child.start_point.row)
                            start_c = int(child.start_point.column)
                            end_r = int(child.end_point.row)
                            end_c = int(child.end_point.column)
                            loc = SymbolLocation(
                                file_path=file_path,
                                start_line=start_r + 1,
                                start_col=start_c,
                                end_line=end_r + 1,
                                end_col=end_c,
                            )
                            sig = lines[start_r].strip() if start_r < len(lines) else None
                            symbols.append(
                                SymbolNode(
                                    name=var_name,
                                    kind=SymbolKind.VARIABLE,
                                    location=loc,
                                    signature=sig,
                                )
                            )

        return tuple(symbols)

    def _parse_go_fallback(self, file_path: Path, content: str) -> tuple[SymbolNode, ...]:
        """Regex fallback for Go syntax parsing."""
        lines: list[str] = content.splitlines()
        symbols: list[SymbolNode] = []

        pkg_re = re.compile(r"^package\s+([A-Za-z0-9_]+)")
        struct_re = re.compile(r"^type\s+([A-Za-z0-9_]+)\s+struct")
        interface_re = re.compile(r"^type\s+([A-Za-z0-9_]+)\s+interface")
        type_re = re.compile(r"^type\s+([A-Za-z0-9_]+)\s+")
        method_re = re.compile(r"^func\s+\([^)]*?\*?([A-Za-z0-9_]+)\)\s+([A-Za-z0-9_]+)")
        func_re = re.compile(r"^func\s+([A-Za-z0-9_]+)")
        const_re = re.compile(r"^const\s+([A-Za-z0-9_]+)")
        var_re = re.compile(r"^var\s+([A-Za-z0-9_]+)")

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            doc = _extract_preceding_comments(lines, idx - 1)

            m = pkg_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.MODULE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = struct_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = interface_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.INTERFACE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = method_re.match(stripped)
            if m:
                recv_name = m.group(1)
                fn_name = m.group(2)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(fn_name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=f"{recv_name}.{fn_name}",
                        kind=SymbolKind.METHOD,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = func_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.FUNCTION,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = type_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.CLASS,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = const_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )
                continue

            m = var_re.match(stripped)
            if m:
                name = m.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=idx,
                    start_col=line.find(name),
                    end_line=idx,
                    end_col=len(line),
                )
                symbols.append(
                    SymbolNode(
                        name=name,
                        kind=SymbolKind.VARIABLE,
                        location=loc,
                        signature=stripped,
                        docstring=doc,
                    )
                )

        return tuple(symbols)

    # =========================================================================
    # Imports and References Extraction
    # =========================================================================

    def extract_imports(self, file_path: Path, content: str) -> tuple[str, ...]:
        """Extract imported modules and symbols from a source file."""
        suffix = Path(file_path).suffix.lower()
        imports: list[str] = []

        if suffix in {".py", ".pyi"}:
            try:
                tree = ast.parse(content, filename=str(file_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(f"{module}.{alias.name}" if module else alias.name)
            except SyntaxError:
                for line in content.splitlines():
                    s = line.strip()
                    if s.startswith("import ") or s.startswith("from "):
                        imports.append(s)
        elif suffix in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
            import_re = re.compile(
                r"""import\s+(?:(?:(?:\*\s+as\s+([A-Za-z0-9_$]+)|(?:\{[^}]+\})|([A-Za-z0-9_$]+))\s+from\s+['"]([^'"]+)['"])|['"]([^'"]+)['"])"""
            )
            for m in import_re.finditer(content):
                module = m.group(3) or m.group(4)
                if module:
                    imports.append(module)
        elif suffix == ".rs":
            use_re = re.compile(r"^(?:pub(?:\([^)]+\))?\s+)?use\s+([^;]+);", re.MULTILINE)
            for m in use_re.finditer(content):
                use_path = m.group(1).strip()
                imports.append(use_path)
        elif suffix == ".go":
            single_import_re = re.compile(r"""import\s+["']([^"']+)["']""")
            for m in single_import_re.finditer(content):
                imports.append(m.group(1))

            block_import_re = re.compile(r"""import\s*\(\s*([^)]+)\)""", re.DOTALL)
            for m in block_import_re.finditer(content):
                block = m.group(1)
                for line in block.splitlines():
                    s = line.strip()
                    if s and not s.startswith("//"):
                        pkg_match = re.search(r"""["']([^"']+)["']""", s)
                        if pkg_match:
                            imports.append(pkg_match.group(1))

        return tuple(dict.fromkeys(imports))

    def extract_references(self, file_path: Path, content: str) -> tuple[str, ...]:
        """Extract all identifier tokens / referenced symbols in the source."""
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", content)
        keywords = {
            # Python keywords
            "def",
            "class",
            "async",
            "await",
            "return",
            "import",
            "from",
            "as",
            "if",
            "elif",
            "else",
            "for",
            "while",
            "in",
            "is",
            "not",
            "and",
            "or",
            "try",
            "except",
            "finally",
            "with",
            "raise",
            "pass",
            "break",
            "continue",
            "True",
            "False",
            "None",
            "self",
            "cls",
            # JS/TS keywords
            "function",
            "interface",
            "const",
            "let",
            "var",
            "type",
            "export",
            "default",
            "new",
            "this",
            "super",
            "extends",
            "implements",
            "public",
            "private",
            "protected",
            # Rust keywords
            "struct",
            "enum",
            "trait",
            "impl",
            "pub",
            "use",
            "mod",
            "static",
            "unsafe",
            "where",
            "match",
            "Self",
            "crate",
            "dyn",
            "ref",
            "move",
            # Go keywords
            "package",
            "func",
            "range",
            "switch",
            "case",
            "select",
            "chan",
            "go",
            "defer",
            "map",
            "make",
            "nil",
            "fallthrough",
        }
        filtered = [t for t in tokens if t not in keywords]
        return tuple(dict.fromkeys(filtered))

    def extract_symbol_references(
        self, file_path: Path, content: str
    ) -> dict[str, tuple[SymbolLocation, ...]]:
        """Extract exact locations for all identifier references in the file."""
        lines = content.splitlines()
        ref_map: dict[str, list[SymbolLocation]] = {}
        ident_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

        for line_num, line in enumerate(lines, start=1):
            for match in ident_pattern.finditer(line):
                sym = match.group(1)
                loc = SymbolLocation(
                    file_path=file_path,
                    start_line=line_num,
                    start_col=match.start(),
                    end_line=line_num,
                    end_col=match.end(),
                )
                if sym not in ref_map:
                    ref_map[sym] = []
                ref_map[sym].append(loc)

        return {k: tuple(v) for k, v in ref_map.items()}
