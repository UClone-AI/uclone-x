"""Turn lifecycle and context injection hook for interactive music composition.

Injects the active song's `ScoreContextBundle` and guided start stage into the turn context.
The hook never reads meaning out of the person's words: which bars, which instruments, and
whether to apply a proposal or change the mix are decided by the model, which states them as
explicit tool arguments (`music_context.selected_region`, `music_orchestra(adjust_mix)`,
`music_proposal(apply)` behind the approval step).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from uclone_x.tools.models import ToolResultStatus
from uclone_x_music import song_after
from uclone_x_music.context import (
    ScoreContextBundle,
    build_score_context,
    context_digest_line,
)
from uclone_x_music.digests import bundle_view, digest_of_value
from uclone_x_music.library import MusicError, SongLibrary
from uclone_x_music.schemas import MusicFileError
from uclone_x_music.work import SongWork

if TYPE_CHECKING:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolContext

__all__ = [
    "APPLY_IN_MUSICVIEW_LINE",
    "MUSIC_SECTION_HEADING",
    "MusicTurnAid",
    "MusicTurnHook",
]

logger = logging.getLogger(__name__)

MUSIC_SECTION_HEADING = "[Music score context]"

_MUSIC_TOOLS = frozenset(
    {
        "music_library",
        "music_start",
        "music_score",
        "music_orchestra",
        "music_track",
        "music_context",
        "music_proposal",
        "music_audit",
    }
)


#: The music tools whose successful calls can change what the score context shows.
_READ_ONLY_ACTIONS = frozenset({"read", "list"})
_READ_ONLY_TOOLS = frozenset({"music_context", "music_audit"})

#: The line a turn ends with when a proposal could not be applied in chat (#2429).
APPLY_IN_MUSICVIEW_LINE = {
    "en": "The proposal has not been applied yet. You can approve or reject it in MusicView.",
    "ko": "제안은 아직 적용되지 않았습니다. MusicView에서 승인하거나 거절할 수 있습니다.",
}


def _render_music_section(
    bundle: ScoreContextBundle,
    *,
    start_stage: str | None = None,
    recommended_next: str | None = None,
) -> str:
    payload = bundle_view(bundle)
    digest = digest_of_value(payload)
    lines: list[str] = [
        MUSIC_SECTION_HEADING,
        f"The song as it is now {context_digest_line(digest)}. It is rebuilt after every step "
        "that changes the song, so it already includes this turn's writes.",
    ]
    if start_stage is not None and start_stage != "completed":
        rec_note = f" (recommended next instrument: {recommended_next})" if recommended_next else ""
        lines.append(
            f"Guided co-composition stage: {start_stage}{rec_note}. "
            "Advance one instrument layer at a time with music_start, then customize its bars "
            'with music_track(action="patch_bars", propose=False) when composing a bespoke '
            "melody or responding to the person's rhythm."
        )
    lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _changes_the_song(record: ToolExecutionRecord) -> bool:
    if record.status is not ToolResultStatus.SUCCESS or record.tool_name not in _MUSIC_TOOLS:
        return False
    if record.tool_name in _READ_ONLY_TOOLS:
        return False
    action = dict(record.arguments).get("action")
    return action not in _READ_ONLY_ACTIONS


def _music_section(
    workspace_root: Path, story_id: str, room_id: str | None
) -> tuple[str, str | None]:
    """The `[Music score context]` section for the song now, and its guided start stage."""
    library = SongLibrary(workspace_root)
    work = SongWork(library, story_id, room_id)
    score_pair = work.read_score()
    default_sec = (
        score_pair[0].sections[0].id
        if score_pair is not None and score_pair[0].sections
        else "sec_verse_01"
    )
    bundle = build_score_context(work, section_id=default_sec)
    from uclone_x_music.start import read_start_state

    start_pair = read_start_state(work)
    start_stage = start_pair[0].stage if start_pair is not None else None
    rec_next = start_pair[0].recommended_next if start_pair is not None else None
    return (
        _render_music_section(bundle, start_stage=start_stage, recommended_next=rec_next),
        start_stage,
    )


@dataclass(frozen=True)
class MusicTurnAid:
    """The turn aid injected when a song is open in a Composer turn."""

    section: str
    song_id: str
    start_stage: str | None = None
    room_id: str | None = None
    workspace_root: Path | None = None

    def section_after_step(self, records: Sequence[ToolExecutionRecord]) -> str | None:
        """The section rebuilt from the song after a step that changed it, else `None`.

        The section is the turn context's, which is built per request at the tail and never
        written to history, so a fresh one changes no earlier message (#2429).
        """
        if self.workspace_root is None or not any(_changes_the_song(r) for r in records):
            return None
        try:
            section, _ = _music_section(self.workspace_root, self.song_id, self.room_id)
        except (MusicError, MusicFileError):
            return None
        return section

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        """Where to approve a proposal this turn tried and failed to apply in chat."""
        refused = any(r.tool_name == "music_proposal" and r.ends_turn for r in records)
        return [APPLY_IN_MUSICVIEW_LINE["ko" if korean else "en"]] if refused else []


class MusicTurnHook:
    """Turn lifecycle and context aid hook for interactive music composition."""

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        """Advance `context.story_id` when a music tool opens or closes a song."""
        song = song_after(records, context.story_id)
        if song == context.story_id:
            return context
        return context.model_copy(update={"story_id": song})

    def turn_aid(
        self,
        *,
        message: str,
        story_id: str | None,
        room_id: str | None,
        workspace_root: Path | None,
        tool_names: frozenset[str],
    ) -> MusicTurnAid | None:
        """Build the `[Music score context]` turn aid when a song is open.

        `message` is not read: the person's words reach the model, and the model acts on them
        through explicit tool calls. Nothing here applies, mutes or selects anything.
        """
        del message
        if story_id is None or workspace_root is None or not (tool_names & _MUSIC_TOOLS):
            return None
        try:
            section_text, start_stage = _music_section(workspace_root, story_id, room_id)
            return MusicTurnAid(
                section=section_text,
                song_id=story_id,
                start_stage=start_stage,
                room_id=room_id,
                workspace_root=workspace_root,
            )
        except (MusicError, MusicFileError):
            return None
        except Exception:
            logger.warning(
                "No music section: song %r could not be read for this turn",
                story_id,
                exc_info=True,
            )
            return None
