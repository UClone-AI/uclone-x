"""Precision filesystem tools implementing workspace-contained file operations."""

from __future__ import annotations

import fnmatch
import os
import re
import uuid
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PathTraversalError
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext

# ======================================================================================
# 1. FileReadTool
# ======================================================================================


class FileReadParams(BaseModel):
    """Parameters for reading file content within workspace boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        description="Path to the file to read, relative to workspace root or absolute within workspace"
    )
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed starting line number (inclusive)",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed ending line number (inclusive)",
    )
    max_lines: int = Field(
        default=800,
        ge=1,
        le=2000,
        description="Maximum number of lines to return per call (default: 800)",
    )
    max_bytes: int = Field(
        default=45 * 1024,
        ge=1,
        le=1024 * 1024,
        description="Maximum bytes to return (default: 45KB)",
    )


class FileReadTool(BaseTool[FileReadParams]):
    """Reads file content within workspace bounds with line slicing and truncation guards."""

    name: str = "file_read"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Read file contents within the workspace boundary with line slicing "
        "(1-indexed start_line/end_line) and token blowup protection. "
        "Absolute paths inside a read-only folder named in the [Workspace] section also work."
    )

    def run(self, params: FileReadParams, context: ToolContext) -> dict[str, Any]:
        """Read and slice file content within workspace."""
        safe_path, root = self.resolve_read_path(params.path, context)

        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: '{params.path}'")
        if safe_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: '{params.path}'")

        content = safe_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        start_line = params.start_line if params.start_line is not None else 1
        end_line = params.end_line if params.end_line is not None else total_lines

        if params.start_line is not None and params.end_line is not None and end_line < start_line:
            raise ValueError(f"end_line ({end_line}) cannot be less than start_line ({start_line})")

        rel_path = self.display_path(safe_path, root, context)

        if total_lines == 0:
            return {
                "path": rel_path,
                "content": "",
                "total_lines": 0,
                "start_line": 1,
                "end_line": 0,
                "truncated": False,
                "bytes_read": 0,
            }

        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        selected_lines = lines[start_idx:end_idx]

        truncated = False
        if len(selected_lines) > params.max_lines:
            selected_lines = selected_lines[: params.max_lines]
            truncated = True

        sliced_text = "".join(selected_lines)
        encoded_bytes = sliced_text.encode("utf-8")
        if len(encoded_bytes) > params.max_bytes:
            encoded_bytes = encoded_bytes[: params.max_bytes]
            sliced_text = encoded_bytes.decode("utf-8", errors="ignore")
            truncated = True

        actual_end_line = start_idx + len(selected_lines) if selected_lines else start_line - 1

        return {
            "path": rel_path,
            "content": sliced_text,
            "total_lines": total_lines,
            "start_line": start_line,
            "end_line": actual_end_line,
            "truncated": truncated,
            "bytes_read": len(sliced_text.encode("utf-8")),
        }


# ======================================================================================
# 2. FileWriteTool
# ======================================================================================


class FileWriteParams(BaseModel):
    """Parameters for atomically writing file content within workspace boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(description="Target file path within workspace")
    content: str = Field(description="Content to write into the file")
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite if file already exists (default: False)",
    )
    create_parents: bool = Field(
        default=True,
        description="Create parent directories if they do not exist (default: True)",
    )
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class FileWriteTool(BaseTool[FileWriteParams]):
    """Atomically writes content to a file within workspace bounds with overwrite protection."""

    name: str = "file_write"
    writes_files: ClassVar[bool] = True  # creates or overwrites a file on the host (#1167)
    description: str = (
        "Atomically write content to a file within the workspace boundary, "
        "optionally creating parent directories and guarding against accidental overwrite."
    )

    def run(self, params: FileWriteParams, context: ToolContext) -> dict[str, Any]:
        """Write content to file atomically."""
        safe_path = self.resolve_safe_path(params.path, context.require_workspace())

        if safe_path.is_dir():
            raise IsADirectoryError(f"Target path is an existing directory: '{params.path}'")

        existed = safe_path.exists()
        if existed and not params.overwrite:
            raise FileExistsError(
                f"File already exists: '{params.path}'. Set overwrite=True to replace."
            )

        parent = safe_path.parent
        if not parent.exists():
            if params.create_parents:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Parent directory does not exist: '{parent}'")

        # Atomic write via temporary file in same directory
        tmp_file = parent / f".{safe_path.name}.tmp.{uuid.uuid4().hex}"
        try:
            tmp_file.write_text(params.content, encoding=params.encoding)
            os.replace(tmp_file, safe_path)
        except Exception:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise

        rel_path = str(safe_path.relative_to(context.require_workspace().resolve()))
        encoded = params.content.encode(params.encoding)

        return {
            "path": rel_path,
            "bytes_written": len(encoded),
            "lines_written": len(params.content.splitlines()),
            "overwritten": existed,
        }


# ======================================================================================
# 3. FileEditTool
# ======================================================================================


class FileEditParams(BaseModel):
    """Parameters for precision string replacement within workspace files."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(description="Target file path within workspace")
    target_content: str = Field(description="Exact string sequence to replace")
    replacement_content: str = Field(description="Replacement string sequence")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-indexed starting line range hint",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Optional 1-indexed ending line range hint",
    )
    allow_multiple: bool = Field(
        default=False,
        description="Allow replacing multiple occurrences; if False, requires exactly 1 match",
    )
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")


class FileEditTool(BaseTool[FileEditParams]):
    """Precision string replacement within workspace files with occurrence uniqueness check."""

    name: str = "file_edit"
    writes_files: ClassVar[bool] = True  # modifies a file on the host in place (#1167)
    description: str = (
        "Exact string replacement (target_content -> replacement_content) within "
        "workspace files, with unique match validation and optional line range hints."
    )

    def run(self, params: FileEditParams, context: ToolContext) -> dict[str, Any]:
        """Perform exact string replacement in target file."""
        safe_path = self.resolve_safe_path(params.path, context.require_workspace())

        if not safe_path.exists():
            raise FileNotFoundError(f"File not found: '{params.path}'")
        if safe_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: '{params.path}'")

        content = safe_path.read_text(encoding=params.encoding)

        # Handle line range scoping if requested
        if params.start_line is not None or params.end_line is not None:
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            start = params.start_line if params.start_line is not None else 1
            end = params.end_line if params.end_line is not None else total_lines

            if end < start:
                raise ValueError(f"end_line ({end}) cannot be less than start_line ({start})")

            start_idx = max(0, start - 1)
            end_idx = min(total_lines, end)

            before = "".join(lines[:start_idx])
            target_block = "".join(lines[start_idx:end_idx])
            after = "".join(lines[end_idx:])
        else:
            before = ""
            target_block = content
            after = ""

        # Validate occurrence count
        count = target_block.count(params.target_content)
        if count == 0:
            scope_desc = (
                f"lines {params.start_line or 1}-{params.end_line or 'EOF'}"
                if (params.start_line is not None or params.end_line is not None)
                else "the entire file"
            )
            raise ValueError(f"Target content not found in '{params.path}' within {scope_desc}.")

        if count > 1 and not params.allow_multiple:
            raise ValueError(
                f"Target content found {count} times in '{params.path}'. "
                "Exactly 1 match required when allow_multiple is False."
            )

        if params.allow_multiple:
            new_block = target_block.replace(params.target_content, params.replacement_content)
            replacements_made = count
        else:
            new_block = target_block.replace(params.target_content, params.replacement_content, 1)
            replacements_made = 1

        new_content = before + new_block + after

        # Atomic write
        parent = safe_path.parent
        tmp_file = parent / f".{safe_path.name}.tmp.{uuid.uuid4().hex}"
        try:
            tmp_file.write_text(new_content, encoding=params.encoding)
            os.replace(tmp_file, safe_path)
        except Exception:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise

        rel_path = str(safe_path.relative_to(context.require_workspace().resolve()))
        return {
            "path": rel_path,
            "replacements_made": replacements_made,
            "bytes_written": len(new_content.encode(params.encoding)),
        }


# ======================================================================================
# 4. FileSearchTool
# ======================================================================================


class FileSearchParams(BaseModel):
    """Parameters for searching file contents or filenames within workspace."""

    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(
        default="",
        description="Search text or regular expression pattern. If empty, matching files are returned.",
    )
    path: str = Field(
        default=".",
        description="Starting directory or file path within workspace (default: '.')",
    )
    is_regex: bool = Field(
        default=False,
        description="Treat query as regular expression (default: False)",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Perform case-sensitive search (default: False)",
    )
    glob_pattern: str | None = Field(
        default=None,
        description="Optional glob filter for file names (e.g. '*.py')",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of match entries to return (default: 100)",
    )
    ignore_hidden: bool = Field(
        default=True,
        description="Ignore hidden files and directories (default: True)",
    )
    ignore_patterns: tuple[str, ...] = Field(
        default=(
            "__pycache__",
            ".git",
            ".venv",
            "node_modules",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
        ),
        description="Patterns to ignore",
    )


class FileSearchTool(BaseTool[FileSearchParams]):
    """Fast pattern/regex search or glob match across workspace files with ignore rules."""

    name: str = "file_search"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Fast pattern/regex search or glob match across workspace files "
        "with configurable ignore rules and match limits. "
        "Absolute paths inside a read-only folder named in the [Workspace] section also work."
    )

    def run(self, params: FileSearchParams, context: ToolContext) -> dict[str, Any]:
        """Search for pattern across workspace files."""
        safe_path, root = self.resolve_read_path(params.path, context)

        if not safe_path.exists():
            raise FileNotFoundError(f"Search path does not exist: '{params.path}'")

        compiled_re = None
        if params.query:
            flags = 0 if params.case_sensitive else re.IGNORECASE
            try:
                if params.is_regex:
                    compiled_re = re.compile(params.query, flags=flags)
                else:
                    compiled_re = re.compile(re.escape(params.query), flags=flags)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{params.query}': {e}") from e

        # Gather files to search
        candidate_files: list[Path] = []
        if safe_path.is_file():
            candidate_files = [safe_path]
        else:
            for root_dir, dirnames, filenames in os.walk(safe_path, followlinks=False):
                # Prune ignored directories in place
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not (params.ignore_hidden and d.startswith("."))
                    and not any(fnmatch.fnmatch(d, pat) for pat in params.ignore_patterns)
                ]

                for fn in filenames:
                    if params.ignore_hidden and fn.startswith("."):
                        continue
                    if any(fnmatch.fnmatch(fn, pat) for pat in params.ignore_patterns):
                        continue
                    if params.glob_pattern and not fnmatch.fnmatch(fn, params.glob_pattern):
                        continue

                    f_path = Path(root_dir) / fn
                    if f_path.is_file():
                        try:
                            safe_f = self.resolve_safe_path(f_path, root)
                            candidate_files.append(safe_f)
                        except PathTraversalError:
                            continue

        matches: list[dict[str, Any]] = []
        truncated = False

        if not params.query:
            # Empty query: return matching candidate files directly without inspecting content
            for f in candidate_files:
                rel_p = self.display_path(f, root, context)
                matches.append(
                    {
                        "path": rel_p,
                        "line_number": None,
                        "line_content": None,
                    }
                )
                if len(matches) >= params.max_results:
                    truncated = len(candidate_files) > params.max_results
                    break
        elif compiled_re is not None:
            for f in candidate_files:
                try:
                    # Fast check to skip binary files
                    with open(f, "rb") as bf:
                        head = bf.read(1024)
                        if b"\x00" in head:
                            continue

                    with open(f, encoding="utf-8", errors="replace") as fh:
                        for line_no, line in enumerate(fh, start=1):
                            if compiled_re.search(line):
                                rel_p = self.display_path(f, root, context)
                                matches.append(
                                    {
                                        "path": rel_p,
                                        "line_number": line_no,
                                        "line_content": line.rstrip("\r\n"),
                                    }
                                )
                                if len(matches) >= params.max_results:
                                    truncated = True
                                    break
                except Exception:
                    continue

                if truncated:
                    break

        return {
            "query": params.query,
            "is_regex": params.is_regex,
            "files_searched": len(candidate_files),
            "total_matches": len(matches),
            "truncated": truncated,
            "matches": matches,
        }


# ======================================================================================
# 5. DirectoryListTool
# ======================================================================================


class DirectoryListParams(BaseModel):
    """Parameters for listing files and directories within workspace."""

    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(
        default=".",
        description="Directory path within workspace to list (default: '.')",
    )
    glob_pattern: str | None = Field(
        default=None,
        description="Optional glob filter for entries (e.g. '*.log')",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of entries to return (default: 100)",
    )
    ignore_hidden: bool = Field(
        default=True,
        description="Ignore hidden files and directories starting with '.' (default: True)",
    )
    recursive: bool = Field(
        default=False,
        description="Whether to recurse into subdirectories (default: False)",
    )


class DirectoryListTool(BaseTool[DirectoryListParams]):
    """Enumerate workspace files and directories with optional glob pattern and recursion."""

    name: str = "directory_list"
    writes_files: ClassVar[bool] = False  # writes no file on the host (#1167)
    description: str = (
        "Enumerate workspace files and directories within workspace boundaries. "
        "Supports optional glob filtering, recursion, and hidden file exclusions. "
        "Absolute paths inside a read-only folder named in the [Workspace] section also work."
    )

    def run(self, params: DirectoryListParams, context: ToolContext) -> dict[str, Any]:
        """List entries in directory within workspace."""
        safe_path, root = self.resolve_read_path(params.path, context)

        if not safe_path.exists():
            raise FileNotFoundError(f"Directory does not exist: '{params.path}'")
        if not safe_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: '{params.path}'")

        entries: list[dict[str, Any]] = []
        truncated = False

        if not params.recursive:
            try:
                raw_entries = sorted(safe_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except Exception as e:
                raise OSError(f"Failed to list directory '{params.path}': {e}") from e

            for entry in raw_entries:
                if params.ignore_hidden and entry.name.startswith("."):
                    continue
                if params.glob_pattern and not fnmatch.fnmatch(entry.name, params.glob_pattern):
                    continue

                try:
                    safe_entry = self.resolve_safe_path(entry, root)
                except PathTraversalError:
                    continue

                rel_p = self.display_path(safe_entry, root, context)
                is_dir = safe_entry.is_dir()
                size_bytes = safe_entry.stat().st_size if not is_dir else 0

                entries.append(
                    {
                        "path": rel_p,
                        "name": safe_entry.name,
                        "is_dir": is_dir,
                        "size_bytes": size_bytes,
                    }
                )
                if len(entries) >= params.max_results:
                    truncated = True
                    break
        else:
            for root_dir, dirnames, filenames in os.walk(safe_path, followlinks=False):
                if params.ignore_hidden:
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]

                # Sort directory names and file names for determinism
                dirnames.sort()
                filenames.sort()

                for dn in dirnames:
                    if params.glob_pattern and not fnmatch.fnmatch(dn, params.glob_pattern):
                        continue
                    d_path = Path(root_dir) / dn
                    try:
                        safe_d = self.resolve_safe_path(d_path, root)
                    except PathTraversalError:
                        continue

                    rel_p = self.display_path(safe_d, root, context)
                    entries.append(
                        {
                            "path": rel_p,
                            "name": safe_d.name,
                            "is_dir": True,
                            "size_bytes": 0,
                        }
                    )
                    if len(entries) >= params.max_results:
                        truncated = True
                        break

                if truncated:
                    break

                for fn in filenames:
                    if params.ignore_hidden and fn.startswith("."):
                        continue
                    if params.glob_pattern and not fnmatch.fnmatch(fn, params.glob_pattern):
                        continue
                    f_path = Path(root_dir) / fn
                    try:
                        safe_f = self.resolve_safe_path(f_path, root)
                    except PathTraversalError:
                        continue

                    rel_p = self.display_path(safe_f, root, context)
                    entries.append(
                        {
                            "path": rel_p,
                            "name": safe_f.name,
                            "is_dir": False,
                            "size_bytes": safe_f.stat().st_size,
                        }
                    )
                    if len(entries) >= params.max_results:
                        truncated = True
                        break

                if truncated:
                    break

        return {
            "path": self.display_path(safe_path, root, context),
            "total_entries": len(entries),
            "truncated": truncated,
            "entries": entries,
        }
