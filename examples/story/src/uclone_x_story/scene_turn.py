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
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from uclone_x.i18n.scripts import sentences
from uclone_x.llm.compactor import estimate_text_tokens
from uclone_x.tools.models import ToolResultStatus
from uclone_x_story.context import NamedChange, last_and_next, named_changes, scene_context
from uclone_x_story.library import StoryLibrary
from uclone_x_story.schemas import Outline, Scene
from uclone_x_story.work import StoryWork

if TYPE_CHECKING:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolContext

__all__ = [
    "OWN_SENTENCES_SHOWN",
    "SCENE_SECTION_TOKENS",
    "SceneTurnAid",
    "SceneTurnHook",
    "StartTurnAid",
    "StoryNamesAid",
    "named_ids",
    "scene_section",
]

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
_STORY_START = "story_start"

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
    #: The story's ids and the names a person reads for them (`named_ids`).
    names: Mapping[str, str] = field(default_factory=dict[str, str])

    def reply_text(self, reply: str) -> str:
        """`reply` with the story's ids put as their names (`named_ids`)."""
        return named_ids(reply, self.names)

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


def named_ids(reply: str, names: Mapping[str, str]) -> str:
    """`reply` with each of the story's ids put as the name a person reads for it (#2300).

    A scene id (`ch01.s02`), a chapter id and the story's id are the tools' handles; the
    model sees them in every bundle and result, and a live reply said "현재 진행 중인
    장면은 **ch01.s02**입니다". Only the ids this story has are replaced, each where it
    stands as a whole token: the edges are ASCII, since a Korean particle follows an id
    unspaced, and a dot ends a token only when no letter or digit follows it (so `ch01` is
    not cut out of `ch01.s022`). Longer ids first, so a scene's id is never cut at its
    chapter's.
    """
    for found in sorted(names, key=len, reverse=True):
        reply = re.sub(
            rf"(?<![A-Za-z0-9_.-]){re.escape(found)}(?![A-Za-z0-9_]|\.[A-Za-z0-9])",
            lambda _match, name=names[found]: name,
            reply,
        )
    return reply


def _story_names(work: StoryWork) -> dict[str, str]:
    """Each id of the story a person could be shown, with the name they read instead."""
    names: dict[str, str] = {}
    record = work.record()
    if record.title:
        names[record.story_id] = record.title
    current = work.outline()
    if current is not None:
        for chapter in current[0].chapters:
            if chapter.title:
                names[chapter.id] = chapter.title
            for scene in chapter.scenes:
                if scene.title:
                    names[scene.id] = scene.title
    return names


@dataclass(frozen=True)
class StoryNamesAid:
    """A turn with a story open and nothing to add but its ids kept from the reply."""

    names: Mapping[str, str]
    section: str = ""

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        """No lines."""
        del records, korean
        return []

    def reply_text(self, reply: str) -> str:
        """`reply` with the story's ids put as their names (`named_ids`)."""
        return named_ids(reply, self.names)


#: How many sentences of its own the Writer's reply may hold, while a start waits, before
#: the reply says they were not saved. Its own: not in what the turn's tools returned and
#: not in the story's saved files (`own_sentences`), so a reply that relays the premises or
#: the plan it was given is not told it wrote them. Counted in sentences, not characters,
#: so the bar is the same in every script (#2303): 400 characters was a short scene in
#: Hangul and about seventy words in English. A live run's unsaved "chapters" were 778 to
#: 1,580 Hangul characters, some twenty to forty sentences (#2300); a note and a question
#: stay well under ten.
OWN_SENTENCES_SHOWN = 10
#: A sentence shorter than this, in letters and digits, is not counted: "네.", "좋아요!".
_MIN_COUNTED = 8


def _key(text: str) -> str:
    """Text as sentences are compared: letters and digits only, case folded."""
    return "".join(ch for ch in unicodedata.normalize("NFC", text).casefold() if ch.isalnum())


def _strings(value: object) -> list[str]:
    """Every string in a tool's output, however deep."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for v in cast(Mapping[object, object], value).values() for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in cast(Sequence[object], value) for s in _strings(v)]
    return []


def own_sentences(reply: str, known: Sequence[str]) -> int:
    """How many sentences of `reply` are the Writer's own: long enough to count, and found
    in none of `known` (what the turn's tools returned and the story saved), compared by
    their letters and digits alone, so markdown, spacing and numbering do not hide a relay.
    """
    seen = "\n".join(_key(text) for text in known)
    return sum(
        1
        for sentence in sentences(reply)
        if len(key := _key(sentence)) >= _MIN_COUNTED and key not in seen
    )


@dataclass
class StartTurnAid:
    """A turn while the story's start waits for the person (#2300).

    At the outline stop of a live run, the turn's context held "[Scene to write]" with the
    first scene's material, and qwen3:8b wrote the chapter in chat instead of passing the
    person's "네" to `story_start`: in all four conversations no chapter was saved. While a
    start waits, the turn gets this section in place of the scene's, saying the person's
    message answers the stop; and a turn that did not move the start says, in the reply,
    that nothing in it was saved.
    """

    section: str
    names: Mapping[str, str] = field(default_factory=dict[str, str])
    #: The story, and its start's record as the turn began: a turn that changed it moved
    #: the start, whatever its calls' statuses say (a skipped call is recorded a success).
    work: StoryWork | None = None
    start_digest: str | None = None
    #: The reply as the Writer gave it (`reply_text`, which the agent calls before the
    #: reply lines), for `own_sentences`.
    reply: str = ""

    def reply_lines(self, records: Sequence[ToolExecutionRecord], *, korean: bool) -> list[str]:
        """What the reply did not save, while the start still waits.

        When no call of the turn moved the start (its record is as the turn found it), the
        line says nothing was saved: a skipped call is recorded a success, so the record
        decides, not the calls. It is said only when the turn tried the story (called a
        story tool) or wrote a scene's worth of its own text (`OWN_SENTENCES_SHOWN`): a
        turn that only asked for a picture saved nothing either, and is not told so. When
        the start moved but the Writer also wrote that much text of its own, the line says
        that text was not saved. A record that cannot be read, or a start no longer
        waiting, gives no line.
        """
        tried = any(record.tool_name.startswith("story_") for record in records)
        if self.work is None:
            return []
        from uclone_x_story.start import START_FILE, start_waiting, stop_words

        try:
            found = self.work.read(START_FILE)
            waiting = start_waiting(self.work)
            saved = _saved_texts(self.work) if waiting is not None else []
        except Exception as error:
            logger.warning(
                "No reply line: the start of %r could not be read: %s", self.work.story_id, error
            )
            return []
        moved = found is None or found.digest != self.start_digest
        given = [text for record in records for text in _strings(record.output)]
        long_text = own_sentences(self.reply, [*given, *saved]) >= OWN_SENTENCES_SHOWN
        if waiting is None or not (long_text or (tried and not moved)):
            return []
        stop = stop_words(waiting)
        if korean:
            said = (
                "제가 직접 쓴 글은 이야기에 저장되지 않았습니다."
                if moved
                else "이번 답으로 이야기에 저장된 것은 없습니다."
            )
            return [f"{said} 이야기는 {stop['ko']}에 대한 답을 기다리고 있습니다."]
        said = (
            "What I wrote in my own words was not saved to the story."
            if moved
            else "Nothing in this reply was saved to the story."
        )
        return [f"{said} It is still waiting for your answer to {stop['en']}."]

    def reply_text(self, reply: str) -> str:
        """`reply` with the story's ids put as their names (`named_ids`), kept for the
        reply lines (`own_sentences`)."""
        named = named_ids(reply, self.names)
        self.reply = named
        return named


def _saved_texts(work: StoryWork) -> list[str]:
    """What the story has saved that a reply may relay: the start's record (its premises
    and plans), the outline, and each written scene's text."""
    from uclone_x_story.start import START_FILE

    found = [work.read(START_FILE), work.read("outline.yaml")]
    found += [work.manuscript(scene_id) for scene_id in work.written_scenes()]
    return [f.text for f in found if f is not None]


def _start_section(stop: str, written: Sequence[str] = ()) -> str:
    """The waiting start's section; `written` are the titles of the scenes already written,
    which the person may ask to change at a chapter's plan (#2303)."""
    lines = [
        "[Story start waiting]",
        f"This story's start is waiting for the person's answer to {stop}.",
        "- If the person is only asking a question or chatting without deciding on the "
        "current stop, answer them in chat and do not call story_start.",
        "- When the person decides, call story_start 'continue' with structured parameters: "
        "'choice' (1–3) when they pick a genre or premise, and for a bare yes at the "
        "premises (the recommended number: a premise is taken only by its number); 'genre' "
        "when they name a genre; 'go_on=true' when they accept what was shown (the "
        "recommended genre, settling the outline, proceeding over pending cast proposals, or "
        "writing the planned chapter); 'changes' "
        "(never 'request') when they reject the premises or ask to change the outline or "
        "scene plan; 'must_have' and 'avoid' for elements to include or keep out; "
        "'rewrite=true' to rewrite continuity-flagged scenes.",
        "- Never call story_outline, story_codex, or story_manuscript while the start "
        "waits: only story_start 'continue' revises the premises, cast, outline, or scene "
        "plan.",
        "- Do not write the outline or a scene in your reply: nothing you write in chat "
        "is saved to the story.",
        "- Do not answer the next stop for the person: after story_start returns, end "
        "your reply and wait for their next message.",
    ]
    if written:
        titles = ", ".join(f"「{title}」" for title in written)
        lines.append(
            f"- Scenes already written: {titles}. A message that asks to change one of them "
            "is not an answer to the plan: pass it to story_start 'continue', word for word, "
            "as 'changes'. story_start writes that scene again and the plan keeps waiting; "
            "do not write the scene yourself."
        )
    return "\n".join(lines)


def _written_titles(work: StoryWork, stage: str) -> list[str]:
    """At a chapter's plan, the titles of the scenes already written, in story order."""
    current = work.outline() if stage == "review_chapter" else None
    if current is None:
        return []
    written = set(work.written_scenes())
    return [
        scene.title or scene.id for _, scene in current[0].scenes_in_order() if scene.id in written
    ]


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
    return SceneTurnAid(section, scene.id, outline, conflicts, _story_names(work))


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
    ) -> SceneTurnAid | StartTurnAid | StoryNamesAid | None:
        """The turn's aid, when a story is open and the turn may call the story's tools.

        While the story's start waits for the person and the turn may call `story_start`,
        the start's aid (`StartTurnAid`), and no scene: the scene's material invited a
        live model to write the chapter in chat (#2300). Otherwise the scene's aid, when
        the turn may call `story_context` and there is a scene to add. Either way, and
        alone when there is nothing else, the story's ids are kept out of the reply
        (`named_ids`). Nothing is added when no story is open, the turn is offered neither
        tool, or reading the story fails; the last is logged.
        """
        offered = tool_names & {_STORY_CONTEXT, _STORY_START}
        if story_id is None or workspace_root is None or not offered:
            return None
        try:
            # story_start loads the story tools; it is reached only when a story is open.
            from uclone_x_story.start import START_FILE, start_waiting, stop_words

            work = StoryWork(StoryLibrary(workspace_root), story_id)
            names = _story_names(work)
            waiting = start_waiting(work) if _STORY_START in tool_names else None
            if waiting is not None:
                stop = stop_words(waiting)
                found = work.read(START_FILE)
                return StartTurnAid(
                    _start_section(stop["en"], _written_titles(work, waiting.stage)),
                    names,
                    work,
                    found.digest if found is not None else None,
                )
            scene = (
                build_aid(work, message=message, room_id=room_id)
                if _STORY_CONTEXT in tool_names
                else None
            )
            return scene or (StoryNamesAid(names) if names else None)
        except Exception:
            logger.warning(
                "No scene section: story %r could not be read for this turn",
                story_id,
                exc_info=True,
            )
            return None
