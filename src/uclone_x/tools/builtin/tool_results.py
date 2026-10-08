"""`tool_result_read`: read back a tool result that was too long to keep whole (#1422).

A result above the cap reaches the history as an excerpt naming a handle; the full text is
kept in this conversation's own session log (#1848). This tool returns a window of it. The
window arrives as a new tool result at the end of the conversation -- nothing earlier is
rewritten -- and fits under the same cap, so it is never shortened again.

Read-only, and scoped to the caller's own conversation: the handle resolves only among the
entries of `ToolContext.stored_results`, the session the call runs in, so one conversation
cannot read another's results. It needs no workspace.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.core.tool_results import (
    TOOL_RESULT_READ_TOOL,
    ResultBodies,
    read_tool_result_page,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext


class ToolResultReadParams(BaseModel):
    """Which stored result to read, and from where."""

    model_config = ConfigDict(extra="forbid", strict=True)

    handle: str = Field(
        description=(
            "The name of the stored result, as a shortened tool result shows it: tr_ "
            "followed by 16 letters and digits."
        )
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "The character to start from. A shortened result and each page say which "
            "offset to use next."
        ),
    )
    length: int | None = Field(
        default=None,
        ge=1,
        description=("At most this many characters. Leave it out for as much as fits in one page."),
    )


class ToolResultReadTool(BaseTool[ToolResultReadParams]):
    """Page through a tool result that was stored because it was too long to show."""

    name = TOOL_RESULT_READ_TOOL
    writes_files: ClassVar[bool] = False
    description = (
        "Read more of a tool result that was too long to show in full. A shortened "
        'result begins with "[Stored tool result tr_..." and names the handle and the '
        "offset to read from. Each call returns one page and says where the next page "
        "starts. Only results from this conversation can be read."
    )
    params_type = ToolResultReadParams

    def run(self, params: ToolResultReadParams, context: ToolContext) -> str:
        """Return one page of the stored result as plain text."""
        bodies = context.stored_results
        return read_tool_result_page(
            bodies if isinstance(bodies, ResultBodies) else None,
            params.handle,
            params.offset,
            params.length,
        )
