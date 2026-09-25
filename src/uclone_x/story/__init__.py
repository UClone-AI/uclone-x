"""Stories: workspace artifacts a conversation opens, and who may write them (#1555).

This module is the pure part a turn needs: which story a conversation has open after its
tool calls. The files and the lease are `uclone_x.story.library`, an adapter, and are not
re-exported here so that importing this does not load the filesystem code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

from uclone_x.tools.models import ToolResultStatus

if TYPE_CHECKING:
    from uclone_x.agent.models import ToolExecutionRecord

__all__ = ["OPEN_STORY_KEY", "story_after"]

#: The output key under which a tool declaring `opens_story` names the story its
#: conversation has open after the call; `None` under it means the story was closed.
OPEN_STORY_KEY = "open_story_id"


def story_after(records: Iterable[ToolExecutionRecord], current: str | None) -> str | None:
    """The story a conversation has open after `records` ran, starting from `current`.

    Only a successful call of a tool that declares `opens_story` moves it, and only by
    naming the story under `OPEN_STORY_KEY` (a `None` there closes it). The last such call
    decides. A tool's name is not the signal: the declaration is.
    """
    story = current
    for record in records:
        if not record.opens_story or record.status is not ToolResultStatus.SUCCESS:
            continue
        output = record.output
        if isinstance(output, Mapping) and OPEN_STORY_KEY in output:
            value = cast(Mapping[str, object], output)[OPEN_STORY_KEY]
            if value is None or isinstance(value, str):
                story = value
    return story
