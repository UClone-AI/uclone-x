"""Turn lifecycle and context injection hook for interactive video editing (`uclone_x_video.section_turn`).

Injects the active video project's `TimelineContextBundle` and guided start stage into the
turn context. The hook never reads meaning out of the person's words (#2204): which frames,
which clips, which quality tier, and whether to bake a derived asset or apply a proposal are
decided by the model through explicit tool arguments.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from uclone_x.tools.models import ToolResultStatus
from uclone_x_video import video_after
from uclone_x_video.context import (
    TimelineContextBundle,
    build_timeline_context,
    context_digest_line,
)
from uclone_x_video.digests import bundle_view, digest_of_value
from uclone_x_video.library import VideoError, VideoLibrary
from uclone_x_video.schemas import VideoFileError
from uclone_x_video.work import VideoWork

if TYPE_CHECKING:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolContext

__all__ = [
    "APPLY_IN_VIDEOVIEW_LINE",
    "VIDEO_SECTION_HEADING",
    "VideoTurnAid",
    "VideoTurnHook",
]

logger = logging.getLogger(__name__)

VIDEO_SECTION_HEADING = "[Video timeline context]"

_VIDEO_TOOLS = frozenset(
    {
        "video_library",
        "video_start",
        "video_media",
        "video_search",
        "video_manifest",
        "video_timeline",
        "video_fx",
        "video_context",
        "video_proposal",
        "video_audit",
    }
)

_READ_ONLY_ACTIONS = frozenset({"read", "list", "list_assets", "inspect_lineage", "status"})
_READ_ONLY_TOOLS = frozenset({"video_search", "video_context", "video_audit"})

APPLY_IN_VIDEOVIEW_LINE = {
    "en": "The proposal has not been applied yet. You can approve or reject it in VideoView.",
    "ko": "제안은 아직 적용되지 않았습니다. VideoView에서 승인하거나 거절할 수 있습니다.",
}


def _render_video_section(
    bundle: TimelineContextBundle,
    *,
    start_stage: str | None = None,
    recommended_next: str | None = None,
) -> str:
    payload = bundle_view(bundle)
    digest = digest_of_value(payload)
    lines: list[str] = [
        VIDEO_SECTION_HEADING,
        f"The video project as it is now {context_digest_line(digest)}. It is rebuilt after "
        "every step that changes the project, so it already includes this turn's writes.",
    ]
    if start_stage is not None and start_stage != "completed":
        rec_note = f" (recommended next step: {recommended_next})" if recommended_next else ""
        lines.append(
            f"Guided video editing stage: {start_stage}{rec_note}. "
            "Advance one step at a time with video_start."
        )
    lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _changes_the_project(record: ToolExecutionRecord) -> bool:
    if record.status is not ToolResultStatus.SUCCESS or record.tool_name not in _VIDEO_TOOLS:
        return False
    if record.tool_name in _READ_ONLY_TOOLS:
        return False
    action = dict(record.arguments).get("action")
    return action not in _READ_ONLY_ACTIONS


def _video_section(
    workspace_root: Path, project_id: str, room_id: str | None
) -> tuple[str, str | None]:
    """The `[Video timeline context]` section for the project now, and its guided start stage."""
    library = VideoLibrary(workspace_root)
    work = VideoWork(library, project_id, room_id)
    bundle = build_timeline_context(work)
    from uclone_x_video.start import read_start_state

    start_pair = read_start_state(work)
    start_stage = start_pair[0].stage if start_pair is not None else None
    rec_next = start_pair[0].recommended_next if start_pair is not None else None
    return (
        _render_video_section(bundle, start_stage=start_stage, recommended_next=rec_next),
        start_stage,
    )


@dataclass(frozen=True)
class VideoTurnAid:
    """The turn aid injected when a video project is open in a Director turn."""

    section: str
    project_id: str
    start_stage: str | None = None
    room_id: str | None = None
    workspace_root: Path | None = None

    def section_after_step(self, records: Sequence[ToolExecutionRecord]) -> str | None:
        """The section rebuilt from the project after a step that changed it, else `None`."""
        if self.workspace_root is None or not any(_changes_the_project(r) for r in records):
            return None
        try:
            section, _ = _video_section(self.workspace_root, self.project_id, self.room_id)
        except (VideoError, VideoFileError):
            return None
        return section

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        """Where to approve a proposal this turn tried and failed to apply in chat."""
        refused = any(r.tool_name == "video_proposal" and r.ends_turn for r in records)
        return [APPLY_IN_VIDEOVIEW_LINE["ko" if korean else "en"]] if refused else []


class VideoTurnHook:
    """Turn lifecycle and context aid hook for interactive video editing."""

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        """Advance `context.story_id` when a video tool opens or closes a project."""
        project = video_after(records, context.story_id)
        if project == context.story_id:
            return context
        return context.model_copy(update={"story_id": project})

    def turn_aid(
        self,
        *,
        message: str,
        story_id: str | None,
        room_id: str | None,
        workspace_root: Path | None,
        tool_names: frozenset[str],
    ) -> VideoTurnAid | None:
        """Build the `[Video timeline context]` turn aid when a video project is open.

        `message` is not read: the person's words reach the model, and the model acts on
        them through explicit tool calls.
        """
        del message
        if story_id is None or workspace_root is None or not (tool_names & _VIDEO_TOOLS):
            return None
        try:
            section_text, start_stage = _video_section(workspace_root, story_id, room_id)
            return VideoTurnAid(
                section=section_text,
                project_id=story_id,
                start_stage=start_stage,
                room_id=room_id,
                workspace_root=workspace_root,
            )
        except (VideoError, VideoFileError):
            return None
        except Exception:
            logger.warning(
                "No video section: project %r could not be read for this turn",
                story_id,
                exc_info=True,
            )
            return None
