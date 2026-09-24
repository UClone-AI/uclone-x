"""Comprehensive unit tests for BaseTool and precision filesystem tools suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from uclone_x.sandbox.models import IsolationLevel, WorkspaceIsolation
from uclone_x.tools import (
    BaseTool,
    DirectoryListTool,
    FileEditTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
    ToolContext,
    ToolRegistry,
    ToolResultStatus,
    create_default_registry,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provide an isolated workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def tool_context(workspace: Path) -> ToolContext:
    """Provide a standard ToolContext bound to the test workspace."""
    return ToolContext(
        agent_id="test_agent",
        session_id="test_session",
        workspace_root=workspace,
        isolation=WorkspaceIsolation(),
    )


# ======================================================================================
# 1. BaseTool Abstraction & Schema Tests
# ======================================================================================


class DummyParams(BaseModel):
    message: str = Field(description="A dummy message")
    count: int = Field(default=1, ge=1, description="Count multiplier")


class DummyTool(BaseTool[DummyParams]):
    name = "dummy_tool"
    description = "A dummy tool for testing BaseTool generic abstraction"

    def run(self, params: DummyParams, context: ToolContext) -> dict[str, Any]:
        return {"result": params.message * params.count, "agent": context.agent_id}


class AsyncDummyTool(BaseTool[DummyParams]):
    name = "async_dummy_tool"
    description = "Async dummy tool"

    async def run(self, params: DummyParams, context: ToolContext) -> dict[str, Any]:
        return {"async_result": params.message.upper()}


class ContextFirstTool(BaseTool[DummyParams]):
    name = "context_first_tool"
    description = "Tool with context and params"

    def run(self, params: DummyParams, context: ToolContext) -> dict[str, Any]:
        return {"inverted": f"{context.session_id}:{params.message}"}


@pytest.mark.asyncio
async def test_basetool_schema_and_execution(tool_context: ToolContext) -> None:
    """BaseTool automatically generates JSON schema from generic parameter model."""
    tool = DummyTool()
    assert tool.name == "dummy_tool"
    assert tool.description == "A dummy tool for testing BaseTool generic abstraction"

    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "message" in schema["properties"]
    assert "count" in schema["properties"]
    assert "message" in schema["required"]

    # Positional execution
    res = await tool.execute({"message": "hi", "count": 3}, tool_context)
    assert res.success is True
    assert res.status is ToolResultStatus.SUCCESS
    assert res.output == {"result": "hihihi", "agent": "test_agent"}
    assert res.execution_time_ms >= 0.0
    assert res.isolation_level is IsolationLevel.WORKSPACE
    assert res.provenance is not None
    assert res.provenance.requested.model == "dummy_tool"


@pytest.mark.asyncio
async def test_basetool_async_run_and_context_first(tool_context: ToolContext) -> None:
    """BaseTool supports async run and flexible (context, params) signature."""
    async_tool = AsyncDummyTool()
    res_async = await async_tool.execute({"message": "hello"}, tool_context)
    assert res_async.success is True
    assert res_async.output == {"async_result": "HELLO"}

    ctx_tool = ContextFirstTool()
    res_ctx = await ctx_tool.execute({"message": "payload"}, tool_context)
    assert res_ctx.success is True
    assert res_ctx.output == {"inverted": "test_session:payload"}


@pytest.mark.asyncio
async def test_basetool_parameter_validation_errors(tool_context: ToolContext) -> None:
    """BaseTool catches parameter validation errors and returns structured error result."""
    tool = DummyTool()

    # Missing required parameter 'message'
    res_missing = await tool.execute({"count": 5}, tool_context)
    assert res_missing.success is False
    assert res_missing.status is ToolResultStatus.ERROR
    assert "Parameter validation failed" in str(res_missing.error)
    assert res_missing.output is None

    # Invalid type for 'count'
    res_invalid = await tool.execute({"message": "test", "count": "not_an_int"}, tool_context)
    assert res_invalid.success is False
    assert "Parameter validation failed" in str(res_invalid.error)

    # Value constraint violation (count < 1)
    res_ge = await tool.execute({"message": "test", "count": 0}, tool_context)
    assert res_ge.success is False
    assert "Parameter validation failed" in str(res_ge.error)


@pytest.mark.asyncio
async def test_basetool_missing_context() -> None:
    """BaseTool rejects execution when ToolContext is absent."""
    tool = DummyTool()
    # No context provided
    res = await tool.execute({"message": "hello"})
    assert res.success is False
    assert "requires a valid ToolContext" in str(res.error)


@pytest.mark.asyncio
async def test_basetool_flexible_calling_conventions(tool_context: ToolContext) -> None:
    """BaseTool handles keyword arguments and context-first calls seamlessly."""
    tool = DummyTool()

    # execute(context, message="a", count=2)
    res1 = await tool.execute(tool_context, message="a", count=2)
    assert res1.success is True
    assert res1.output == {"result": "aa", "agent": "test_agent"}

    # execute(context=tool_context, message="b")
    res2 = await tool.execute(context=tool_context, message="b")
    assert res2.success is True
    assert res2.output == {"result": "b", "agent": "test_agent"}


def test_basetool_missing_params_type_raises() -> None:
    """Instantiating a BaseTool without generic type or params_type raises TypeError."""

    class BadTool(BaseTool[DummyParams]):
        name = "bad"
        description = "bad"

        def run(self, params: DummyParams, context: ToolContext) -> Any:
            return None

    # Artificially remove resolved params_type
    tool = BadTool()
    tool.params_type = None

    # Subclass with no annotations at all
    class UntypedTool(BaseTool):  # type: ignore[type-arg]
        name = "untyped"
        description = "untyped"

        def run(self, params: Any, context: ToolContext) -> Any:
            return None

    bad = UntypedTool()
    with pytest.raises(TypeError, match="must define `params_type` or inherit from `BaseTool"):
        _ = bad.parameters_schema


# ======================================================================================
# 2. FileReadTool Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_file_read_full_content(workspace: Path, tool_context: ToolContext) -> None:
    """FileReadTool reads full content of existing workspace file."""
    file_path = workspace / "sample.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\n")

    tool = FileReadTool()
    res = await tool.execute({"path": "sample.txt"}, tool_context)

    assert res.success is True
    assert res.status is ToolResultStatus.SUCCESS
    data = cast(dict[str, Any], res.output)
    assert data["path"] == "sample.txt"
    assert data["content"] == "Line 1\nLine 2\nLine 3\n"
    assert data["total_lines"] == 3
    assert data["start_line"] == 1
    assert data["end_line"] == 3
    assert data["truncated"] is False
    assert data["bytes_read"] == len(b"Line 1\nLine 2\nLine 3\n")


@pytest.mark.asyncio
async def test_file_read_line_slicing(workspace: Path, tool_context: ToolContext) -> None:
    """FileReadTool supports 1-indexed line slicing with start_line and end_line."""
    lines = [f"Row {i}\n" for i in range(1, 21)]
    (workspace / "multi.txt").write_text("".join(lines))

    tool = FileReadTool()

    # Slice lines 5 to 8 (inclusive)
    res = await tool.execute(
        {"path": "multi.txt", "start_line": 5, "end_line": 8},
        tool_context,
    )
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["start_line"] == 5
    assert data["end_line"] == 8
    assert data["total_lines"] == 20
    assert data["content"] == "Row 5\nRow 6\nRow 7\nRow 8\n"

    # Slice from line 18 to end
    res_end = await tool.execute(
        {"path": "multi.txt", "start_line": 18},
        tool_context,
    )
    assert res_end.success is True
    data_end = cast(dict[str, Any], res_end.output)
    assert data_end["start_line"] == 18
    assert data_end["end_line"] == 20
    assert data_end["content"] == "Row 18\nRow 19\nRow 20\n"


@pytest.mark.asyncio
async def test_file_read_invalid_slice_range(workspace: Path, tool_context: ToolContext) -> None:
    """FileReadTool rejects end_line < start_line."""
    (workspace / "test.txt").write_text("hello\nworld\n")
    tool = FileReadTool()

    res = await tool.execute(
        {"path": "test.txt", "start_line": 5, "end_line": 2},
        tool_context,
    )
    assert res.success is False
    assert "end_line (2) cannot be less than start_line (5)" in str(res.error)


@pytest.mark.asyncio
async def test_file_read_truncation_limits(workspace: Path, tool_context: ToolContext) -> None:
    """FileReadTool enforces max_lines and max_bytes caps with truncated=True flag."""
    lines = [f"Line {i:03d} - content padding text\n" for i in range(1, 101)]
    (workspace / "large.txt").write_text("".join(lines))

    tool = FileReadTool()

    # Truncate by max_lines
    res_lines = await tool.execute(
        {"path": "large.txt", "max_lines": 10},
        tool_context,
    )
    assert res_lines.success is True
    data_lines = cast(dict[str, Any], res_lines.output)
    assert data_lines["truncated"] is True
    assert str(data_lines["content"]).count("\n") == 10
    assert data_lines["end_line"] == 10

    # Truncate by max_bytes
    res_bytes = await tool.execute(
        {"path": "large.txt", "max_bytes": 50},
        tool_context,
    )
    assert res_bytes.success is True
    data_bytes = cast(dict[str, Any], res_bytes.output)
    assert data_bytes["truncated"] is True
    assert data_bytes["bytes_read"] <= 50


@pytest.mark.asyncio
async def test_file_read_empty_file(workspace: Path, tool_context: ToolContext) -> None:
    """FileReadTool safely handles empty files."""
    (workspace / "empty.txt").write_text("")
    tool = FileReadTool()

    res = await tool.execute({"path": "empty.txt"}, tool_context)
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["content"] == ""
    assert data["total_lines"] == 0
    assert data["bytes_read"] == 0


@pytest.mark.asyncio
async def test_file_read_not_found_and_directory(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileReadTool returns descriptive error on missing file or directory target."""
    tool = FileReadTool()

    # Missing file
    res_missing = await tool.execute({"path": "does_not_exist.txt"}, tool_context)
    assert res_missing.success is False
    assert "FileNotFoundError" in str(res_missing.error)

    # Target is a directory
    sub_dir = workspace / "subdir"
    sub_dir.mkdir()
    res_dir = await tool.execute({"path": "subdir"}, tool_context)
    assert res_dir.success is False
    assert "IsADirectoryError" in str(res_dir.error)


@pytest.mark.asyncio
async def test_file_read_path_traversal_rejection(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileReadTool blocks path traversal attempts outside workspace."""
    tool = FileReadTool()

    # Relative traversal
    res_rel = await tool.execute({"path": "../outside.txt"}, tool_context)
    assert res_rel.success is False
    assert "Path traversal violation" in str(res_rel.error)

    # Absolute traversal
    res_abs = await tool.execute({"path": "/etc/passwd"}, tool_context)
    assert res_abs.success is False
    assert "Path traversal violation" in str(res_abs.error)


# ======================================================================================
# 3. FileWriteTool Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_file_write_new_file_and_parents(workspace: Path, tool_context: ToolContext) -> None:
    """FileWriteTool atomically creates new file and parent directories."""
    tool = FileWriteTool()
    target = "nested/sub/dir/output.txt"

    res = await tool.execute(
        {"path": target, "content": "Atomic content\nLine 2\n", "create_parents": True},
        tool_context,
    )
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["path"] == target
    assert data["overwritten"] is False
    assert data["lines_written"] == 2

    # Verify physical file
    written_file = workspace / "nested" / "sub" / "dir" / "output.txt"
    assert written_file.exists()
    assert written_file.read_text() == "Atomic content\nLine 2\n"


@pytest.mark.asyncio
async def test_file_write_overwrite_protection(workspace: Path, tool_context: ToolContext) -> None:
    """FileWriteTool guards against accidental overwrite when overwrite=False."""
    target_file = workspace / "existing.txt"
    target_file.write_text("original content")

    tool = FileWriteTool()

    # Attempt overwrite with overwrite=False (default)
    res_blocked = await tool.execute(
        {"path": "existing.txt", "content": "new content", "overwrite": False},
        tool_context,
    )
    assert res_blocked.success is False
    assert "FileExistsError" in str(res_blocked.error)
    assert target_file.read_text() == "original content"

    # Explicit overwrite=True
    res_ok = await tool.execute(
        {"path": "existing.txt", "content": "new content", "overwrite": True},
        tool_context,
    )
    assert res_ok.success is True
    data = cast(dict[str, Any], res_ok.output)
    assert data["overwritten"] is True
    assert target_file.read_text() == "new content"


@pytest.mark.asyncio
async def test_file_write_create_parents_false(workspace: Path, tool_context: ToolContext) -> None:
    """FileWriteTool raises FileNotFoundError when parent missing and create_parents=False."""
    tool = FileWriteTool()
    res = await tool.execute(
        {"path": "missing_dir/file.txt", "content": "data", "create_parents": False},
        tool_context,
    )
    assert res.success is False
    assert "Parent directory does not exist" in str(res.error)


@pytest.mark.asyncio
async def test_file_write_path_traversal_rejection(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileWriteTool rejects paths that escape workspace boundary."""
    tool = FileWriteTool()

    res = await tool.execute(
        {"path": "../../escape.txt", "content": "malicious"},
        tool_context,
    )
    assert res.success is False
    assert "Path traversal violation" in str(res.error)


# ======================================================================================
# 4. FileEditTool Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_file_edit_unique_replacement(workspace: Path, tool_context: ToolContext) -> None:
    """FileEditTool performs exact string replacement on a unique match."""
    target_file = workspace / "code.py"
    target_file.write_text("def add(a, b):\n    return a - b\n")

    tool = FileEditTool()
    res = await tool.execute(
        {
            "path": "code.py",
            "target_content": "return a - b",
            "replacement_content": "return a + b",
        },
        tool_context,
    )

    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["replacements_made"] == 1
    assert target_file.read_text() == "def add(a, b):\n    return a + b\n"


@pytest.mark.asyncio
async def test_file_edit_multiple_matches_rejection(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileEditTool rejects multiple occurrences when allow_multiple=False."""
    target_file = workspace / "dup.txt"
    target_file.write_text("foo bar foo baz foo\n")

    tool = FileEditTool()

    # Rejection with allow_multiple=False
    res_err = await tool.execute(
        {
            "path": "dup.txt",
            "target_content": "foo",
            "replacement_content": "qux",
            "allow_multiple": False,
        },
        tool_context,
    )
    assert res_err.success is False
    assert "Target content found 3 times" in str(res_err.error)
    assert "allow_multiple is False" in str(res_err.error)
    assert target_file.read_text() == "foo bar foo baz foo\n"

    # Success with allow_multiple=True
    res_ok = await tool.execute(
        {
            "path": "dup.txt",
            "target_content": "foo",
            "replacement_content": "qux",
            "allow_multiple": True,
        },
        tool_context,
    )
    assert res_ok.success is True
    data = cast(dict[str, Any], res_ok.output)
    assert data["replacements_made"] == 3
    assert target_file.read_text() == "qux bar qux baz qux\n"


@pytest.mark.asyncio
async def test_file_edit_target_not_found(workspace: Path, tool_context: ToolContext) -> None:
    """FileEditTool raises error when target_content is not found."""
    target_file = workspace / "sample.txt"
    target_file.write_text("alpha beta gamma\n")

    tool = FileEditTool()
    res = await tool.execute(
        {"path": "sample.txt", "target_content": "delta", "replacement_content": "omega"},
        tool_context,
    )
    assert res.success is False
    assert "Target content not found" in str(res.error)


@pytest.mark.asyncio
async def test_file_edit_line_range_scoping(workspace: Path, tool_context: ToolContext) -> None:
    """FileEditTool scopes replacement to specified 1-indexed line range."""
    content = "Line 1: item\nLine 2: item\nLine 3: item\nLine 4: item\n"
    target_file = workspace / "scoped.txt"
    target_file.write_text(content)

    tool = FileEditTool()

    # Scope to lines 2..2: exactly 1 match in scope even though 4 exist in file
    res = await tool.execute(
        {
            "path": "scoped.txt",
            "target_content": "item",
            "replacement_content": "REPLACED",
            "start_line": 2,
            "end_line": 2,
            "allow_multiple": False,
        },
        tool_context,
    )
    assert res.success is True
    expected = "Line 1: item\nLine 2: REPLACED\nLine 3: item\nLine 4: item\n"
    assert target_file.read_text() == expected


@pytest.mark.asyncio
async def test_file_edit_invalid_line_range(workspace: Path, tool_context: ToolContext) -> None:
    """FileEditTool rejects end_line < start_line."""
    (workspace / "test.txt").write_text("a\nb\n")
    tool = FileEditTool()
    res = await tool.execute(
        {
            "path": "test.txt",
            "target_content": "a",
            "replacement_content": "b",
            "start_line": 4,
            "end_line": 2,
        },
        tool_context,
    )
    assert res.success is False
    assert "end_line (2) cannot be less than start_line (4)" in str(res.error)


@pytest.mark.asyncio
async def test_file_edit_path_traversal_rejection(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileEditTool blocks path traversal outside workspace."""
    tool = FileEditTool()
    res = await tool.execute(
        {"path": "../escape.txt", "target_content": "a", "replacement_content": "b"},
        tool_context,
    )
    assert res.success is False
    assert "Path traversal violation" in str(res.error)


# ======================================================================================
# 5. FileSearchTool Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_file_search_text_and_regex(workspace: Path, tool_context: ToolContext) -> None:
    """FileSearchTool searches text and regular expressions across workspace files."""
    (workspace / "mod1.py").write_text("def calculate_tax(income):\n    return income * 0.2\n")
    (workspace / "mod2.py").write_text("def calculate_discount(price):\n    return price * 0.1\n")
    (workspace / "notes.txt").write_text("Tax rates:\n- calculate_tax formula\n")

    tool = FileSearchTool()

    # Plain substring search
    res_plain = await tool.execute({"query": "calculate_tax"}, tool_context)
    assert res_plain.success is True
    data_plain = cast(dict[str, Any], res_plain.output)
    assert data_plain["total_matches"] == 2
    matches_plain = cast(list[dict[str, Any]], data_plain["matches"])
    paths = [m["path"] for m in matches_plain]
    assert "mod1.py" in paths
    assert "notes.txt" in paths

    # Regex search: functions starting with calculate_
    res_regex = await tool.execute(
        {"query": r"def calculate_\w+", "is_regex": True},
        tool_context,
    )
    assert res_regex.success is True
    data_regex = cast(dict[str, Any], res_regex.output)
    assert data_regex["total_matches"] == 2
    matches_regex = cast(list[dict[str, Any]], data_regex["matches"])
    matched_lines = [m["line_content"] for m in matches_regex]
    assert "def calculate_tax(income):" in matched_lines
    assert "def calculate_discount(price):" in matched_lines


@pytest.mark.asyncio
async def test_file_search_glob_and_ignore_rules(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileSearchTool respects glob filter and ignores hidden/cache directories."""
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("find_me = True\n")
    (workspace / "src" / "main.json").write_text('{"find_me": true}\n')

    # Ignored directories
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("find_me = True\n")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "cached.pyc").write_text("find_me = True\n")

    tool = FileSearchTool()

    # Filter with glob '*.py'
    res = await tool.execute(
        {"query": "find_me", "glob_pattern": "*.py"},
        tool_context,
    )
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["total_matches"] == 1
    matches = cast(list[dict[str, Any]], data["matches"])
    assert matches[0]["path"] == "src/main.py"


@pytest.mark.asyncio
async def test_file_search_case_sensitivity_and_limits(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileSearchTool respects case_sensitive setting and max_results limit."""
    content = "\n".join([f"Target {i} target" for i in range(1, 20)])
    (workspace / "case.txt").write_text(content)

    tool = FileSearchTool()

    # Case-sensitive search for uppercase "Target"
    res_case = await tool.execute(
        {"query": "Target", "case_sensitive": True, "max_results": 5},
        tool_context,
    )
    assert res_case.success is True
    data_case = cast(dict[str, Any], res_case.output)
    assert data_case["total_matches"] == 5
    assert data_case["truncated"] is True


@pytest.mark.asyncio
async def test_file_search_invalid_regex(workspace: Path, tool_context: ToolContext) -> None:
    """FileSearchTool reports invalid regex patterns gracefully."""
    tool = FileSearchTool()
    res = await tool.execute(
        {"query": "[unclosed bracket", "is_regex": True},
        tool_context,
    )
    assert res.success is False
    assert "Invalid regex pattern" in str(res.error)


@pytest.mark.asyncio
async def test_file_search_path_traversal_rejection(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileSearchTool blocks search paths escaping workspace."""
    tool = FileSearchTool()
    res = await tool.execute({"query": "secret", "path": "../.."}, tool_context)
    assert res.success is False
    assert "Path traversal violation" in str(res.error)


@pytest.mark.asyncio
async def test_file_search_single_file(workspace: Path, tool_context: ToolContext) -> None:
    """FileSearchTool can search within a single target file directly."""
    target = workspace / "single.txt"
    target.write_text("alpha beta gamma\n")

    tool = FileSearchTool()
    res = await tool.execute({"query": "beta", "path": "single.txt"}, tool_context)
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["total_matches"] == 1
    matches = cast(list[dict[str, Any]], data["matches"])
    assert matches[0]["path"] == "single.txt"


@pytest.mark.asyncio
async def test_file_search_hidden_and_binary(workspace: Path, tool_context: ToolContext) -> None:
    """FileSearchTool skips hidden files when ignore_hidden=True and skips binary files."""
    (workspace / ".hidden.txt").write_text("secret_keyword\n")
    (workspace / "binary.bin").write_bytes(b"secret_keyword\x00extra_binary_bytes")
    (workspace / "normal.txt").write_text("secret_keyword\n")

    tool = FileSearchTool()
    res = await tool.execute({"query": "secret_keyword", "ignore_hidden": True}, tool_context)
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["total_matches"] == 1
    matches = cast(list[dict[str, Any]], data["matches"])
    assert matches[0]["path"] == "normal.txt"


@pytest.mark.asyncio
async def test_file_search_non_existent_path(workspace: Path, tool_context: ToolContext) -> None:
    """FileSearchTool reports error if search path does not exist."""
    tool = FileSearchTool()
    res = await tool.execute({"query": "test", "path": "missing_dir"}, tool_context)
    assert res.success is False
    assert "Search path does not exist" in str(res.error)


@pytest.mark.asyncio
async def test_file_search_reports_files_searched_count(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileSearchTool reports files_searched in payload, distinguishing empty results.

    Killed by: src/uclone_x/tools/builtin/filesystem.py :: "files_searched": len(candidate_files),
    Becomes: "files_searched": 0,
    """
    (workspace / "a.py").write_text("print('hello')\n")
    (workspace / "b.py").write_text("print('world')\n")
    (workspace / "c.txt").write_text("ignored\n")

    tool = FileSearchTool()
    res = await tool.execute({"query": "hello", "glob_pattern": "*.py"}, tool_context)
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["files_searched"] == 2
    assert data["total_matches"] == 1

    # When glob matches nothing, files_searched is 0
    res_none = await tool.execute({"query": "hello", "glob_pattern": "*.nonexistent"}, tool_context)
    assert res_none.success is True
    data_none = cast(dict[str, Any], res_none.output)
    assert data_none["files_searched"] == 0
    assert data_none["total_matches"] == 0


@pytest.mark.asyncio
async def test_file_search_empty_query_enumerates_files(
    workspace: Path, tool_context: ToolContext
) -> None:
    """FileSearchTool with empty query enumerates files without content searching.

    Killed by: src/uclone_x/tools/builtin/filesystem.py :: if not params.query:
    Becomes: if params.query:
    """
    (workspace / "docs").mkdir()
    (workspace / "docs" / "guide.md").write_text("# Guide\n")
    (workspace / "docs" / "faq.md").write_text("# FAQ\n")
    (workspace / "main.py").write_text("print('main')\n")

    tool = FileSearchTool()
    res = await tool.execute({"query": "", "path": "docs", "glob_pattern": "*.md"}, tool_context)
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["files_searched"] == 2
    assert data["total_matches"] == 2
    paths = [m["path"] for m in data["matches"]]
    assert "docs/guide.md" in paths
    assert "docs/faq.md" in paths
    assert all(m["line_number"] is None for m in data["matches"])


@pytest.mark.asyncio
async def test_directory_list_tool_basic_and_filtering(
    workspace: Path, tool_context: ToolContext
) -> None:
    """DirectoryListTool lists entries, respects ignore_hidden, glob_pattern, and recursive.

    Killed by: src/uclone_x/tools/builtin/filesystem.py :: name: str = "directory_list"
    Becomes: name: str = "directory_enumeration"
    """
    (workspace / "logs").mkdir()
    (workspace / "logs" / "2026-09-01.log").write_text("log content 1\n")
    (workspace / "logs" / "2026-09-02.log").write_text("log content 2\n")
    (workspace / "logs" / "notes.txt").write_text("notes\n")
    (workspace / "logs" / ".hidden").write_text("hidden\n")
    (workspace / "logs" / "sub").mkdir()
    (workspace / "logs" / "sub" / "deep.log").write_text("deep\n")

    tool = DirectoryListTool()
    assert tool.name == "directory_list"

    # Non-recursive with glob
    res = await tool.execute(
        {"path": "logs", "glob_pattern": "*.log", "recursive": False},
        tool_context,
    )
    assert res.success is True
    data = cast(dict[str, Any], res.output)
    assert data["path"] == "logs"
    assert data["total_entries"] == 2
    names = [e["name"] for e in data["entries"]]
    assert "2026-09-01.log" in names
    assert "2026-09-02.log" in names
    assert ".hidden" not in names
    assert "notes.txt" not in names

    # Recursive
    res_rec = await tool.execute(
        {"path": "logs", "glob_pattern": "*.log", "recursive": True},
        tool_context,
    )
    assert res_rec.success is True
    data_rec = cast(dict[str, Any], res_rec.output)
    rec_paths = [e["path"] for e in data_rec["entries"]]
    assert "logs/2026-09-01.log" in rec_paths
    assert "logs/2026-09-02.log" in rec_paths
    assert "logs/sub/deep.log" in rec_paths


@pytest.mark.asyncio
async def test_directory_list_tool_validation_and_errors(
    workspace: Path, tool_context: ToolContext
) -> None:
    """DirectoryListTool rejects non-existent paths, files, and path traversal."""
    (workspace / "some_file.txt").write_text("content\n")

    tool = DirectoryListTool()

    # Non-existent
    res_miss = await tool.execute({"path": "missing_dir"}, tool_context)
    assert res_miss.success is False
    assert "Directory does not exist" in str(res_miss.error)

    # Path is a file, not a directory
    res_file = await tool.execute({"path": "some_file.txt"}, tool_context)
    assert res_file.success is False
    assert "Path is not a directory" in str(res_file.error)

    # Path traversal outside workspace
    res_trav = await tool.execute({"path": "../.."}, tool_context)
    assert res_trav.success is False
    assert "Path traversal violation" in str(res_trav.error)


@pytest.mark.asyncio
async def test_file_write_directory_target(workspace: Path, tool_context: ToolContext) -> None:
    """FileWriteTool rejects writing to an existing directory."""
    (workspace / "dir").mkdir()
    tool = FileWriteTool()
    res = await tool.execute({"path": "dir", "content": "hello"}, tool_context)
    assert res.success is False
    assert "IsADirectoryError" in str(res.error)


@pytest.mark.asyncio
async def test_file_edit_directory_target(workspace: Path, tool_context: ToolContext) -> None:
    """FileEditTool rejects editing an existing directory."""
    (workspace / "dir2").mkdir()
    tool = FileEditTool()
    res = await tool.execute(
        {"path": "dir2", "target_content": "a", "replacement_content": "b"},
        tool_context,
    )
    assert res.success is False
    assert "IsADirectoryError" in str(res.error)


@pytest.mark.asyncio
async def test_basetool_init_overrides_and_unhandled_exception(tool_context: ToolContext) -> None:
    """BaseTool supports constructor overrides and gracefully captures unexpected exceptions."""

    class CustomTool(BaseTool[DummyParams]):
        def run(self, params: DummyParams, context: ToolContext) -> Any:
            raise RuntimeError("Unexpected boom")

    tool = CustomTool(name="custom_name", description="custom_desc", params_type=DummyParams)
    assert tool.name == "custom_name"
    assert tool.description == "custom_desc"

    res = await tool.execute({"message": "test"}, tool_context)
    assert res.success is False
    assert "RuntimeError: Unexpected boom" in str(res.error)


# ======================================================================================
# 6. Default Registry Integration
# ======================================================================================


def test_default_tool_registry_registration() -> None:
    """Default registry is pre-populated with standard builtin tools."""
    registry = create_default_registry()
    tools = registry.list_tools()
    tool_names = {t.name for t in tools}

    assert "file_read" in tool_names
    assert "file_write" in tool_names
    assert "file_edit" in tool_names
    assert "file_search" in tool_names
    assert "directory_list" in tool_names
    assert "web_fetch" in tool_names
    assert "web_search" in tool_names

    # Direct lookup
    assert registry.get("file_read") is not None
    assert registry.get("file_write") is not None
    assert registry.get("file_edit") is not None
    assert registry.get("file_search") is not None
    assert registry.get("directory_list") is not None
    assert registry.get("web_fetch") is not None
    assert registry.get("web_search") is not None

    # Convenience classmethod
    reg2 = ToolRegistry.with_builtins()
    assert len(reg2.list_tools()) == 14
