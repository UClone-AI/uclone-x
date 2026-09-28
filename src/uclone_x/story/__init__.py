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
    from collections.abc import Sequence

    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolContext

__all__ = ["OPEN_STORY_KEY", "StoryLifecycleHook", "story_after"]

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


class StoryLifecycleHook:
    """The turn lifecycle hook that moves a turn's open story between its steps (#1732).

    A `TurnLifecycleHookProtocol`: the agent carries `story_id` but imports nothing from
    here, so a host composing an agent without this hook gets one whose story never
    moves mid-turn. Every UClone-X head composes it in (`agent/clone_builder.py`).
    """

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        """`context` with the story `records` left open (`story_after`)."""
        story = story_after(records, context.story_id)
        if story == context.story_id:
            return context
        return context.model_copy(update={"story_id": story})
