"""Comprehensive unit tests for BaseTool and precision filesystem tools suite."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, BinaryIO, Literal, cast
from zoneinfo import ZoneInfo

import pydantic
import pydantic_core
import pytest
from annotated_types import Predicate
from pydantic import BaseModel, Field, FilePath, ImportString, ValidationError, field_validator
from pydantic_core import PydanticCustomError

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
from uclone_x.tools.base import PLAIN_ERROR_PREFIX, describe_invalid_arguments, replace_file


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


#: What a refusal on arguments must not show: the validator's own report, its links and
#: type codes, and the name of the class the arguments are checked against (#1570).
_VALIDATOR_INTERNALS = (
    "pydantic",
    "errors.pydantic.dev",
    "http",
    "validation error",
    "Input should",
    "input_value",
    "type=",
    "int_parsing",
    "greater_than_equal",
    "DummyParams",
    "_VariantParams",
)


def _assert_plain(error: str | None) -> str:
    assert error is not None
    for internal in _VALIDATOR_INTERNALS:
        assert internal not in error, (internal, error)
    return error


@pytest.mark.asyncio
async def test_basetool_parameter_validation_errors(tool_context: ToolContext) -> None:
    """A call refused on its arguments names the argument, says what was expected and that
    nothing was done, in plain words -- not pydantic's report (#1570).

    Killed by: src/uclone_x/tools/base.py :: error=describe_invalid_arguments(
    Becomes: error=str(e) + describe_invalid_arguments(
    Killed by: src/uclone_x/tools/base.py :: ("ge", "at least"),
    Becomes: ("ge_", "at least"),
    Killed by: src/uclone_x/tools/base.py :: "int_parsing": "should be a whole number",
    Becomes: "int_parsin_": "should be a whole number",
    """
    tool = DummyTool()
    tail = "The tool did not run, so nothing was done."

    # Missing required parameter 'message': the arguments the tool takes are listed.
    res_missing = await tool.execute({"count": 5}, tool_context)
    assert res_missing.success is False
    assert res_missing.status is ToolResultStatus.ERROR
    assert _assert_plain(res_missing.error) == (
        "The call to 'dummy_tool' was refused because its arguments did not fit: 'message' "
        f"is missing. {tail} It takes: message (required), count. Call it again with the "
        "arguments corrected."
    )
    assert res_missing.output is None

    # Invalid type for 'count'
    res_invalid = await tool.execute({"message": "test", "count": "not_an_int"}, tool_context)
    assert res_invalid.success is False
    assert _assert_plain(res_invalid.error) == (
        "The call to 'dummy_tool' was refused because its arguments did not fit: 'count' "
        f"should be a whole number. {tail} Call it again with the arguments corrected."
    )

    # Value constraint violation (count < 1): the limit is named.
    res_ge = await tool.execute({"message": "test", "count": 0}, tool_context)
    assert res_ge.success is False
    assert _assert_plain(res_ge.error) == (
        "The call to 'dummy_tool' was refused because its arguments did not fit: 'count' "
        f"should be at least 1. {tail} Call it again with the arguments corrected."
    )


class _BrokenCheckParams(BaseModel):
    value: str = ""

    @field_validator("value")
    @classmethod
    def _broken(cls, value: str) -> str:
        raise TypeError("checker_socket_91 at /srv/checks.py:7")


class _BrokenCheckTool(BaseTool[_BrokenCheckParams]):
    name = "broken_check"
    description = "Its argument check itself fails."

    def run(self, params: _BrokenCheckParams, context: ToolContext) -> dict[str, Any]:
        return {}


@pytest.mark.asyncio
async def test_an_argument_check_that_itself_fails_is_refused_without_its_cause(
    tool_context: ToolContext,
) -> None:
    """A validator raising something pydantic does not turn into a refusal still ends in a
    plain sentence; the exception's class and message go to the log only (#1570).

    Killed by: src/uclone_x/tools/base.py :: logger.warning("%s: its arguments could not be checked", tool_identifier, exc_info=True)
    Becomes: raise
    """
    res = await _BrokenCheckTool().execute({"value": "x"}, tool_context)

    assert res.success is False
    assert res.error == (
        "The arguments of the call to 'broken_check' could not be checked, so the call was "
        "refused. The tool did not run, so nothing was done."
    )


class _VariantParams(BaseModel):
    model_config = {"extra": "forbid"}

    mode: Literal["fast", "slow"]
    size: int | str = 1
    tags: list[str] = Field(default_factory=list, max_length=2)


class _VariantTool(BaseTool[_VariantParams]):
    name = "variant_tool"
    description = "Takes a mode, a size and a few tags."
    not_run_note = "Nothing was changed."

    def run(self, params: _VariantParams, context: ToolContext) -> dict[str, Any]:
        return {"mode": params.mode}


@pytest.mark.asyncio
async def test_a_refusal_names_the_allowed_values_and_every_argument_that_did_not_fit(
    tool_context: ToolContext,
) -> None:
    """Enough for the model to correct its call: the allowed values of a choice, the item
    limit of a list, an argument the tool does not take, and a value that fits no member
    of a union -- whose type names pydantic puts in the error's location -- as one
    argument, not as `size.int` and `size.str` (#1570).

    Killed by: src/uclone_x/tools/base.py :: return f"should be one of {allowed}"
    Becomes: return f"should be one of {ctx}"
    Killed by: src/uclone_x/tools/base.py :: or part.lower() in _UNION_TAGS or part[0].isupper():
    Becomes: or part.lower() in () or part[0].isupper():
    Killed by: src/uclone_x/tools/base.py :: "extra_forbidden": "is not an argument this tool takes",
    Becomes: "extra_forbiddeX": "is not an argument this tool takes",
    Killed by: src/uclone_x/tools/base.py :: return f"should have at most {_plural(ctx['max_length'], 'item', 'items')}"
    Becomes: return "should have at most"
    """
    res = await _VariantTool().execute(
        {"mode": "medium", "size": [3], "tags": ["a", "b", "c"], "colour": "red"}, tool_context
    )

    assert res.success is False
    assert _assert_plain(res.error) == (
        "The call to 'variant_tool' was refused because its arguments did not fit: 'mode' "
        "should be one of 'fast' or 'slow'; 'size' should be a whole number or text; "
        "'tags' should have at most 2 items; 'colour' is not an argument this tool "
        "takes. Nothing was changed. It takes: mode (required), size, tags. Call it again "
        "with the arguments corrected."
    )


class _Sketch(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["sketch"]
    lines: int


class _Photo(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["photo"]
    source: str


class _PictureParams(BaseModel):
    picture: _Sketch | _Photo
    tagged: Annotated[_Sketch | _Photo, Field(discriminator="kind")] | None = None


class _FramedParams(BaseModel):
    framed: Annotated[_Sketch | _Photo, Field(discriminator="kind")]


def _refusal_for(model: type[BaseModel], arguments: dict[str, Any]) -> str:
    try:
        model.model_validate(arguments)
    except ValidationError as error:
        return _assert_plain(
            describe_invalid_arguments("picture_tool", error, model, "Nothing was drawn.")
        )
    raise AssertionError("the arguments were accepted")


def test_a_value_that_fits_no_member_of_a_union_of_models_says_what_each_one_needs() -> None:
    """One line for the argument, with what each form it can take needed -- not the
    members' parts run together as if they were one object (#1602). A union with a
    discriminator names the tags that would do, and its tag is not an argument.

    Killed by: src/uclone_x/tools/base.py :: form = _form_at(loc, names)
    Becomes: form = None
    Killed by: src/uclone_x/tools/base.py :: if " or " not in allowed:
    Becomes: if False:
    Killed by: src/uclone_x/tools/base.py :: self.forms.update(_tag_values(field.annotation, field.discriminator))
    Becomes: pass
    Killed by: src/uclone_x/tools/base.py :: self.forms.update(_tag_values(inner, tag))
    Becomes: pass
    Killed by: src/uclone_x/tools/base.py :: if kind == "union_tag_invalid" and "discriminator" in ctx and "expected_tags" in ctx:
    Becomes: if False:
    """
    tail = "Nothing was drawn. It takes: picture (required), tagged. Call it again with the arguments corrected."

    assert _refusal_for(_PictureParams, {"picture": {"kind": "painting"}}) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: 'picture' "
        "did not fit any of the forms it can take: either 'picture.kind' should be "
        "'sketch' and 'picture.lines' is missing, or 'picture.kind' should be 'photo' and "
        f"'picture.source' is missing. {tail}"
    )
    assert _refusal_for(
        _PictureParams, {"picture": {"kind": "photo", "source": "x"}, "tagged": {"kind": "sketch"}}
    ) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: "
        f"'tagged.lines' is missing. {tail}"
    )
    assert _refusal_for(_FramedParams, {"framed": {"kind": "photo"}}) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: "
        "'framed.source' is missing. Nothing was drawn. It takes: framed (required). Call "
        "it again with the arguments corrected."
    )
    assert _refusal_for(
        _PictureParams,
        {"picture": {"kind": "photo", "source": "x"}, "tagged": {"kind": "secret_tag_77"}},
    ) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: 'tagged' "
        "should have 'kind' set to one of 'sketch', 'photo'. Nothing was drawn. Call it "
        "again with the arguments corrected."
    )


class _AliasedParams(BaseModel):
    target: str = Field(alias="Target-Path")
    Mode: int


def test_an_alias_with_a_capital_or_a_hyphen_is_named() -> None:
    """The argument is named as the tool takes it, even when its alias starts with a
    capital or holds a hyphen -- not "the arguments should be text" (#1602).

    Killed by: src/uclone_x/tools/base.py :: if part in names.arguments:
    Becomes: if part in ():
    """
    assert _refusal_for(_AliasedParams, {"Target-Path": 3, "Mode": "fast"}) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: "
        "'Target-Path' should be text; 'Mode' should be a whole number. Nothing was drawn. "
        "Call it again with the arguments corrected."
    )


class _TakenNameParams(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _free(cls, value: str) -> str:
        if value == "hero":
            raise PydanticCustomError("plain_name_taken", "that name is already taken")
        raise PydanticCustomError("name_odd", "the name {name} is not allowed", {"name": value})


def test_a_custom_error_the_tool_opted_in_keeps_its_own_message() -> None:
    """A `PydanticCustomError` whose type starts with `PLAIN_ERROR_PREFIX` is the tool's own
    sentence, passed on as it is -- not "has a value of the wrong kind". One without the
    prefix keeps the plain fallback, value and all left out (#1602).

    Killed by: src/uclone_x/tools/base.py :: if kind.startswith(PLAIN_ERROR_PREFIX) and error.get("msg"):
    Becomes: if False and error.get("msg"):
    Killed by: src/uclone_x/tools/base.py :: if kind.startswith(PLAIN_ERROR_PREFIX) and error.get("msg"):
    Becomes: if True and error.get("msg"):
    """
    assert _refusal_for(_TakenNameParams, {"name": "hero"}) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: 'name': "
        "that name is already taken. Nothing was drawn. Call it again with the arguments "
        "corrected."
    )
    assert _refusal_for(_TakenNameParams, {"name": "villain_sk_live_1"}) == (
        "The call to 'picture_tool' was refused because its arguments did not fit: 'name' "
        "has a value of the wrong kind. Nothing was drawn. Call it again with the arguments "
        "corrected."
    )
    assert "secret_tag_77" not in _refusal_for(
        _PictureParams,
        {"picture": {"kind": "secret_tag_77"}, "tagged": {"kind": "secret_tag_77"}},
    )


class _SequenceParams(BaseModel):
    value: Sequence[str]


class _DigitsParams(BaseModel):
    value: Annotated[str, Predicate(str.isdigit)]


class _ImportParams(BaseModel):
    value: ImportString[Any]


class _ZoneParams(BaseModel):
    value: ZoneInfo


class _FileParams(BaseModel):
    value: FilePath


@pytest.mark.parametrize(
    ("model", "given"),
    [
        (_SequenceParams, "sk_live_SECRET"),
        (_DigitsParams, "sk_live_SECRET"),
        (_ImportParams, "sk_live_SECRET"),
        (_ZoneParams, "sk-live-SECRET"),
        (_FileParams, "/nowhere/sk_live_SECRET"),
    ],
)
def test_pydantics_own_custom_errors_keep_the_plain_fallback(
    model: type[BaseModel], given: str
) -> None:
    """Pydantic raises `PydanticCustomError` itself, with messages that name its own types
    ("'str' instances are not allowed as a Sequence value") or repeat the value given
    ("invalid timezone: ..."). Those are not the tool's words, so they read as the plain
    fallback and the value is not echoed (#1602 review).

    Killed by: src/uclone_x/tools/base.py :: if kind.startswith(PLAIN_ERROR_PREFIX) and error.get("msg"):
    Becomes: if True and error.get("msg"):
    """
    refusal = _refusal_for(model, {"value": given})

    assert refusal == (
        "The call to 'picture_tool' was refused because its arguments did not fit: 'value' "
        "has a value of the wrong kind. Nothing was drawn. Call it again with the arguments "
        "corrected."
    )
    assert "SECRET" not in refusal


def test_no_custom_error_pydantic_raises_starts_with_the_plain_prefix() -> None:
    """The opt-in holds only while pydantic's own custom error types do not start with the
    prefix. Read from the installed pydantic, so an upgrade that adds one fails here.

    Killed by: src/uclone_x/tools/base.py :: PLAIN_ERROR_PREFIX = "plain_"
    Becomes: PLAIN_ERROR_PREFIX = "pa"
    """
    found: set[str] = set()
    for package in (pydantic, pydantic_core):
        for source in Path(cast(str, package.__file__)).parent.rglob("*.py*"):
            try:
                text = source.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            found |= set(re.findall(r"PydanticCustomError\(\s*['\"]([A-Za-z_]+)", text))

    assert len(found) > 20, found  # the scan saw pydantic's own raises
    assert not [kind for kind in found if kind.startswith(PLAIN_ERROR_PREFIX)]


class _EitherParams(BaseModel):
    count: int | _Sketch = 1
    notes: list[str | _Sketch] = Field(default_factory=list[str | _Sketch])


def test_a_union_of_a_value_and_a_model_reads_as_alternatives() -> None:
    """`int | Sketch` given text fits neither form: one "or", not two `;` lines that read
    as two things to fix (#1602 review). A value that almost fit the model says what each
    form needed instead.

    Killed by: src/uclone_x/tools/base.py :: members.insert(0, {where: problems[where]})
    Becomes: pass
    Killed by: src/uclone_x/tools/base.py :: if where not in unions:
    Becomes: if True:
    Killed by: src/uclone_x/tools/base.py :: if all(set(member) == {where} for member in members):
    Becomes: if False:
    """
    lead = "The call to 'picture_tool' was refused because its arguments did not fit: "
    tail = "Nothing was drawn. Call it again with the arguments corrected."

    assert _refusal_for(_EitherParams, {"count": "many"}) == (
        f"{lead}'count' should be a whole number or an object of named values. {tail}"
    )
    assert _refusal_for(_EitherParams, {"notes": [3]}) == (
        f"{lead}'notes[0]' should be text or an object of named values. {tail}"
    )
    assert _refusal_for(_EitherParams, {"count": {"kind": "sketch"}}) == (
        f"{lead}'count' did not fit any of the forms it can take: either 'count' should be "
        "a whole number, or 'count.lines' is missing. Nothing was drawn. It takes: count, "
        "notes. Call it again with the arguments corrected."
    )


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
    assert len(reg2.list_tools()) == 23


# ======================================================================================
# replace_file: mode and long names (#1589)
# ======================================================================================


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.asyncio
async def test_file_write_over_a_private_file_keeps_it_private(
    workspace: Path, tool_context: ToolContext
) -> None:
    """Replacing a file used to reset `0600` to the umask default.

    Killed by: src/uclone_x/tools/base.py :: _set_mode(descriptor, tmp_file, mode)
    Becomes: pass
    """
    target = workspace / "secret.txt"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o600)
    result = await FileWriteTool().execute(
        {"path": "secret.txt", "content": "new\n", "overwrite": True}, tool_context
    )
    assert result.status == ToolResultStatus.SUCCESS, result.error
    assert target.read_text(encoding="utf-8") == "new\n"
    assert _mode(target) == 0o600


@pytest.mark.asyncio
async def test_file_edit_keeps_the_mode_too(workspace: Path, tool_context: ToolContext) -> None:
    """`file_edit` writes through the same helper, so an executable script stays executable.

    Killed by: src/uclone_x/tools/base.py :: _set_mode(descriptor, tmp_file, mode)
    Becomes: pass
    """
    target = workspace / "run.sh"
    target.write_text("echo old\n", encoding="utf-8")
    target.chmod(0o750)
    result = await FileEditTool().execute(
        {"path": "run.sh", "target_content": "old", "replacement_content": "new"}, tool_context
    )
    assert result.status == ToolResultStatus.SUCCESS, result.error
    assert target.read_text(encoding="utf-8") == "echo new\n"
    assert _mode(target) == 0o750


@pytest.mark.asyncio
async def test_a_new_file_gets_the_umask_default(
    workspace: Path, tool_context: ToolContext
) -> None:
    """Killed by: src/uclone_x/tools/base.py :: descriptor = os.open(tmp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    Becomes: descriptor = os.open(tmp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    """
    previous = os.umask(0o022)
    try:
        result = await FileWriteTool().execute({"path": "new.txt", "content": "x"}, tool_context)
    finally:
        os.umask(previous)
    assert result.status == ToolResultStatus.SUCCESS, result.error
    assert _mode(workspace / "new.txt") == 0o644


@pytest.mark.asyncio
async def test_a_name_near_the_length_limit_is_written_and_overwritten(
    workspace: Path, tool_context: ToolContext
) -> None:
    """The temporary name no longer grows the file's own name past 255 bytes.

    Killed by: src/uclone_x/tools/base.py :: tmp_file = path.parent / f".ucx-{secrets.token_hex(6)}.tmp"
    Becomes: tmp_file = path.parent / f".{path.name}.tmp.{secrets.token_hex(6)}"
    """
    name = "n" * 246 + ".txt"  # 250 bytes: allowed as a name, too long with a suffix
    for content in ("first\n", "second\n"):
        result = await FileWriteTool().execute(
            {"path": name, "content": content, "overwrite": True}, tool_context
        )
        assert result.status == ToolResultStatus.SUCCESS, result.error
    assert (workspace / name).read_text(encoding="utf-8") == "second\n"
    assert sorted(p.name for p in workspace.iterdir()) == [name]  # no temporary file left


def test_the_mode_is_set_before_any_data_is_written(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new contents of a `0600` file never sit in a file others can read (#1589 a).

    The file object that writes the data is made after the mode is set, so its mode at that
    moment is the mode every byte is written under.

    Killed by: src/uclone_x/tools/base.py :: _set_mode(descriptor, tmp_file, mode)
    Becomes: pass
    """
    target = workspace / "secret.txt"
    target.write_text("old\n", encoding="utf-8")
    target.chmod(0o600)
    seen: list[int] = []
    real_fdopen = os.fdopen

    def spy(descriptor: int, mode: str) -> BinaryIO:
        seen.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        return cast(BinaryIO, real_fdopen(descriptor, mode))

    monkeypatch.setattr(os, "fdopen", spy)
    replace_file(target, b"new\n")
    assert seen == [0o600]
    assert target.read_bytes() == b"new\n"


def test_a_failure_to_open_the_file_object_closes_the_descriptor(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/tools/base.py :: os.close(descriptor)
    Becomes: pass
    """
    target = workspace / "notes.txt"
    target.write_text("old\n", encoding="utf-8")
    opened: list[int] = []

    def failing_fdopen(descriptor: int, *args: Any, **kwargs: Any) -> Any:
        opened.append(descriptor)
        raise OSError("no file object")

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    with pytest.raises(OSError, match="no file object"):
        replace_file(target, b"new\n")
    monkeypatch.undo()
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])  # closed
    assert sorted(p.name for p in workspace.iterdir()) == ["notes.txt"]
    assert target.read_text(encoding="utf-8") == "old\n"


@pytest.mark.asyncio
async def test_file_edit_keeps_crlf_line_endings(
    workspace: Path, tool_context: ToolContext
) -> None:
    """A file with CRLF endings used to come back with `\\n` on every line (#1589 c).

    Killed by: src/uclone_x/tools/builtin/filesystem.py :: if crlf:
    Becomes: if False:
    """
    target = workspace / "notes.txt"
    target.write_bytes(b"first\r\nold\r\nlast\r\n")
    result = await FileEditTool().execute(
        {"path": "notes.txt", "target_content": "old", "replacement_content": "new\nextra"},
        tool_context,
    )
    assert result.status == ToolResultStatus.SUCCESS, result.error
    assert target.read_bytes() == b"first\r\nnew\r\nextra\r\nlast\r\n"


@pytest.mark.asyncio
async def test_file_edit_leaves_lf_endings_as_they_are(
    workspace: Path, tool_context: ToolContext
) -> None:
    target = workspace / "notes.txt"
    target.write_bytes(b"first\nold\nlast\n")
    result = await FileEditTool().execute(
        {"path": "notes.txt", "target_content": "old", "replacement_content": "new"},
        tool_context,
    )
    assert result.status == ToolResultStatus.SUCCESS, result.error
    assert target.read_bytes() == b"first\nnew\nlast\n"
