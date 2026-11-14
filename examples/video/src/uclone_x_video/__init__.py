"""Videos: workspace video editing projects a conversation opens, and who may write them.

This module is the pure part a turn needs: which video project a conversation has open
after its tool calls. The files and the lease are `uclone_x_video.library`, an adapter,
and are not re-exported here so that importing this does not load the filesystem code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

from uclone_x.tools.models import ToolResultStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolContext

__all__ = ["OPEN_VIDEO_KEY", "VideoLifecycleHook", "video_after"]

#: The output key under which a tool declaring `opens_story` names the video project its
#: conversation has open after the call; `None` under it means the project was closed.
OPEN_VIDEO_KEY = "open_video_id"


def video_after(records: Iterable[ToolExecutionRecord], current: str | None) -> str | None:
    """The video project a conversation has open after `records` ran, starting from `current`.

    Only a successful call of a tool that declares `opens_story` moves it, and only by
    naming the project under `OPEN_VIDEO_KEY` (a `None` there closes it). The last such
    call decides.
    """
    project = current
    for record in records:
        if not record.opens_story or record.status is not ToolResultStatus.SUCCESS:
            continue
        output = record.output
        if isinstance(output, Mapping) and OPEN_VIDEO_KEY in output:
            value = cast(Mapping[str, object], output)[OPEN_VIDEO_KEY]
            if value is None or isinstance(value, str):
                project = value
    return project


class VideoLifecycleHook:
    """The turn lifecycle hook that moves a turn's open video project between its steps."""

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        """`context` with the video project `records` left open (`video_after`)."""
        project = video_after(records, context.story_id)
        if project == context.story_id:
            return context
        return context.model_copy(update={"story_id": project})
