"""The next scene's material, put in a Writer turn's context by code (#1808).

The 2026-09-28 Writer eval on qwen3:8b found that the model writes the story's facts
when the `for_scene` bundle is pasted into its context, and misses them when it is left
to call `story_context` itself. So when a
story is open for a turn and the persona may call `story_context`, `SceneTurnHook` puts a
scene's bundle at the tail of the turn's requests, gathered with the person's message as
the request, so `request_conflicts` leads it, with a few rules beside it. The scene is the
one the message names, by id or by its title in 「」, and else the next unwritten one.

The same finding holds for telling the person what was kept: the model did it in 0 of 15
replies. So when the request contradicted the story and the turn saved that scene, the aid
gives the reply a line saying what the story holds against the request and how to change it
(`SceneTurnAid.reply_lines`). The line does not claim the scene kept it: the prose is not
checked here.

An adapter: it reads the story's files through `StoryWork`. The agent reaches it only as a
`TurnAidHookProtocol` among its lifecycle hooks, composed in by `agent/clone_builder.py`.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from uclone_x.llm.compactor import estimate_text_tokens
from uclone_x.story.context import NamedChange, last_and_next, named_changes, scene_context
from uclone_x.story.library import StoryLibrary
from uclone_x.story.schemas import Outline, Scene
from uclone_x.story.work import StoryWork
from uclone_x.tools.models import ToolResultStatus

if TYPE_CHECKING:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolContext

__all__ = ["SCENE_SECTION_TOKENS", "SceneTurnAid", "SceneTurnHook", "scene_section"]

logger = logging.getLogger(__name__)

#: The most the scene section may cost, by `estimate_text_tokens` (a token per four UTF-8
#: bytes). A small local model such as qwen3:8b runs with a 16,384-token window, and the
#: section is sent
#: with every request of the turn, beside the system prompt, the tool schemas and the
#: history. 2,000 is about an eighth of that window. It holds moon-seal's largest bundle
#: less its manifest (ch02.s02, about 1,700 with the request's conflicts), which is the
#: size of what the model would have pulled with `story_context` anyway.
SCENE_SECTION_TOKENS = 2000

#: What is left out, in order, until the section fits: the text of the neighbours first,
#: then the manifest. Then codex entries, from the last (the least bound to the scene: the
#: bundle lists the scene's own characters and places first). `request_conflicts`, the
#: scene itself and `earlier_changes` are never left out; a bundle that does not fit
#: without them adds nothing.
_DROP_FIRST = ("end_of_previous_scene", "previous_scene", "next_scene", "manifest")

_HEADING = "[Scene to write]"

_RULES = (
    "- Give each character the appearance and gender the codex gives them.",
    "- A character who died before this scene does not act or speak in it: they are only "
    "remembered, and the prose says they are dead.",
    "- A thing lost before this scene is not used in it.",
    "- Where the request conflicts with the story (request_conflicts), write the story.",
    "- If the person means a different scene, call story_context with action 'for_scene' and "
    "that scene's scene_id.",
)

_STORY_CONTEXT = "story_context"
_STORY_MANUSCRIPT = "story_manuscript"

_HANGUL = re.compile(r"[가-힣]")


def _render(
    scene_id: str,
    title: str,
    bundle: Mapping[str, Any],
    left_out: Sequence[str],
    *,
    named: bool,
) -> str:
    which = (
        "The scene this message names is"
        if named
        else "The next unwritten scene of the open story is"
    )
    lines = [
        _HEADING,
        f"{which} {scene_id} 「{title}」. Its material is below, as story_context 'for_scene' "
        "gives it, gathered with this message as the request.",
        *_RULES,
    ]
    if left_out:
        lines.append(
            f"Left out to save room: {', '.join(left_out)}. story_context 'for_scene' gives "
            "them in full."
        )
    lines.append(json.dumps(bundle, ensure_ascii=False, default=str))
    return "\n".join(lines)


def scene_section(
    scene_id: str,
    title: str,
    bundle: Mapping[str, Any],
    *,
    cap: int = SCENE_SECTION_TOKENS,
    named: bool = False,
) -> str | None:
    """The section for `bundle`, cut to `cap` tokens, or `None` when it cannot fit.

    What is cut, and in which order, is `_DROP_FIRST`, then codex entries from the last.
    The section names what it left out, so the model knows to ask for it. `named` says
    the message named the scene; else it is the next unwritten one.
    """
    kept = dict(bundle)
    left_out: list[str] = []
    section = _render(scene_id, title, kept, left_out, named=named)
    for key in _DROP_FIRST:
        if estimate_text_tokens(section) <= cap:
            return section
        if kept.pop(key, None) is not None:
            left_out.append(key)
            section = _render(scene_id, title, kept, left_out, named=named)
    entries = list(cast("list[Any]", kept.get("codex") or []))
    dropped = 0
    while estimate_text_tokens(section) > cap and entries:
        entries.pop()
        dropped += 1
        kept["codex"] = entries
        section = _render(
            scene_id, title, kept, [*left_out, f"{dropped} codex entries"], named=named
        )
    return section if estimate_text_tokens(section) <= cap else None


def _josa(word: str, with_final: str, without_final: str) -> str:
    """`word` with the Korean particle its last syllable takes: 은/는, 을/를."""
    last = word[-1:]
    if not _HANGUL.match(last):
        return f"{word}{with_final}({without_final})"
    return word + (with_final if (ord(last) - 0xAC00) % 28 else without_final)


def _named_scene(outline: Outline, message: str) -> Scene | None:
    """The scene `message` names: by its id (``ch02.s04``), else by its title in 「」."""
    scenes = [scene for _, scene in outline.scenes_in_order()]
    for scene in scenes:
        # Not \w: Korean is a word character, and a particle follows an id unspaced.
        if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(scene.id)}(?![A-Za-z0-9_])", message):
            return scene
    for scene in scenes:
        if scene.title and f"「{scene.title}」" in message:
            return scene
    return None


def _title_of(outline: Outline, scene_id: str) -> str:
    found = outline.find(scene_id)
    return found[1].title if found and found[1].title else scene_id


def _kept(change: NamedChange, outline: Outline, *, korean: bool) -> str:
    """What the story holds about one thing the request named: plain words, no ids."""
    where = _title_of(outline, change.at)
    thing = change.entry.name
    if change.kind == "dead":
        return (
            f"{_josa(thing, '은', '는')} 「{where}」 장면에서 죽었습니다"
            if korean
            else f"{thing} died in the scene 「{where}」"
        )
    if change.owner is None:
        return (
            f"{_josa(thing, '은', '는')} 「{where}」 장면에서 잃었습니다"
            if korean
            else f"{thing} was lost in the scene 「{where}」"
        )
    owner = change.owner.name
    return (
        f"{_josa(owner, '은', '는')} 「{where}」 장면에서 {_josa(thing, '을', '를')} 잃었습니다"
        if korean
        else f"{owner} lost {thing} in the scene 「{where}」"
    )


@dataclass(frozen=True)
class SceneTurnAid:
    """One turn's scene section, and what its request contradicted in the story."""

    section: str
    scene_id: str
    outline: Outline
    conflicts: tuple[NamedChange, ...] = ()

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        """The story-facts line, when the request conflicted and the turn saved the scene.

        Saved means a successful `story_manuscript` write of this section's scene. The line
        says what the story holds against the request and how to change it, in the person's
        language. It does not say the scene kept those facts: the prose is not checked here,
        and a live run broke them in 2 of 5 saved scenes (#1808).
        """
        if not self.conflicts or not any(self._saved(record) for record in records):
            return []
        kept = [_kept(change, self.outline, korean=korean) for change in self.conflicts]
        if korean:
            return [
                f"요청이 이야기 설정과 어긋납니다: {', '.join(kept)}. 바꾸고 싶으시면 설정 "
                "변경을 제안받아 승인하시면 됩니다."
            ]
        return [
            f"The request goes against the story as it stands: {'; '.join(kept)}. To change "
            "that, ask for a change to the story's settings to be proposed, and approve it."
        ]

    def _saved(self, record: ToolExecutionRecord) -> bool:
        if record.tool_name != _STORY_MANUSCRIPT or record.status is not ToolResultStatus.SUCCESS:
            return False
        output = record.output
        return (
            record.arguments.get("action") == "write"
            and isinstance(output, Mapping)
            and cast(Mapping[str, object], output).get("scene_id") == self.scene_id
        )


def _meta(work: StoryWork, room_id: str | None) -> dict[str, Any]:
    """The story's line of the bundle, as `story_context` gives it."""
    record = work.record()
    meta: dict[str, Any] = {"story_id": record.story_id, "title": record.title}
    for key in ("genre", "logline", "style_notes"):
        value = getattr(record, key)
        if value:
            meta[key] = value
    meta["writable_here"] = record.lease is not None and record.lease.holder == room_id
    return meta


def build_aid(
    work: StoryWork, *, message: str, room_id: str | None, cap: int = SCENE_SECTION_TOKENS
) -> SceneTurnAid | None:
    """The aid for the scene `message` names, else the next unwritten scene of `work`, or
    `None` when there is none to add.

    Raises what reading the story raises; `SceneTurnHook.turn_aid` logs it.
    """
    current = work.outline()
    if current is None:
        logger.info("No scene section: story %r has no outline yet", work.story_id)
        return None
    outline = current[0]
    named = _named_scene(outline, message)
    scene = named
    if scene is None:
        _, following = last_and_next(outline, work.written_scenes())
        if following is None:
            logger.info("No scene section: story %r has no unwritten scene left", work.story_id)
            return None
        scene = following[1]
    ordered = [s.id for _, s in outline.scenes_in_order()]
    index = ordered.index(scene.id)
    previous = work.manuscript(ordered[index - 1]) if index > 0 else None
    codex = work.codex()
    bundle = scene_context(
        outline,
        scene.id,
        codex,
        previous_text=previous.text if previous else None,
        story=_meta(work, room_id),
        request=message or None,
    )
    section = scene_section(
        scene.id, _title_of(outline, scene.id), bundle, cap=cap, named=named is not None
    )
    if section is None:
        logger.warning(
            "No scene section: scene %r of story %r does not fit in %d tokens even cut down",
            scene.id,
            work.story_id,
            cap,
        )
        return None
    conflicts = (
        tuple(named_changes(outline, scene.id, codex, message))
        if bundle.get("request_conflicts")
        else ()
    )
    return SceneTurnAid(section, scene.id, outline, conflicts)


class SceneTurnHook:
    """The lifecycle hook that gives a Writer turn its scene (`TurnAidHookProtocol`).

    It moves nothing between steps: `after_tool_step` returns the context as it is.
    """

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        """`context`, unchanged."""
        del records
        return context

    def turn_aid(
        self,
        *,
        message: str,
        story_id: str | None,
        room_id: str | None,
        workspace_root: Path | None,
        tool_names: frozenset[str],
    ) -> SceneTurnAid | None:
        """The scene's aid, when a story is open and the turn may call `story_context`.

        Nothing is added, and the turn runs as it would without this, when no story is
        open, the turn is not offered `story_context`, the message names no scene and the
        story has no scene left to write, or reading the story fails. The last is logged.
        """
        if story_id is None or workspace_root is None or _STORY_CONTEXT not in tool_names:
            return None
        try:
            return build_aid(
                StoryWork(StoryLibrary(workspace_root), story_id), message=message, room_id=room_id
            )
        except Exception:
            logger.warning(
                "No scene section: story %r could not be read for this turn",
                story_id,
                exc_info=True,
            )
            return None
