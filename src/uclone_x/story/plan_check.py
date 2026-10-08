"""A written scene held to its plan and to the settled plot (§1.1 row 6).

The continuity check (`uclone_x.story.continuity`) holds a scene to the codex: who is alive,
what they hold, their gender and family. This module holds it to what was decided before it
was written: the scene's plan in the outline (its cast and its beats, which the person saw
and agreed to at the chapter stop) and the settled plot's twist, which stays hidden until
the chapter the arc gives it.

What code decides, with no model:

- **a planned character who never appears** (`absent_cast`): the scene's plan lists a cast
  member, and the scene's text never names them by name or alias. A Korean name carries its
  particle (`하린은`), so a name is looked for inside words, as `CodexNames.named_in` does;
  a name of several words also counts by any one of them, since a scene calls "마을 장로"
  just "장로".

What one model call reads, and code then decides (`plan_prompt`, `plan_findings`):

- **a planned beat the scene leaves out**: the model says, beat by beat, whether the scene
  carries it out;
- **the twist told early**: asked only of a scene in a chapter before the twist's, and kept
  only with a quote the scene contains (`quote_found`);
- **an unplanned turn**: an event that changes the story (a death, a secret told, someone
  leaving for good) that the plan does not call for, kept only with a quote the scene
  contains, and at most `MAX_UNPLANNED` per scene.

Each finding is a `continuity.Finding`, stored with the continuity check's findings, so the
reply, the rewrite the person may ask for and anything that shows the start's findings read
them the same way. A finding is written for the person in plain Korean with no id and no
internals; it never stops the flow and never rewrites the scene by itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from uclone_x.story.context import CodexIndex
from uclone_x.story.continuity import Finding, reply_object
from uclone_x.story.names import MIN_NAME_CHARACTERS
from uclone_x.story.quotes import folded, quote_found
from uclone_x.story.schemas import Scene

__all__ = [
    "MAX_UNPLANNED",
    "PLAN_MARKER",
    "absent_cast",
    "asks_model",
    "plan_findings",
    "plan_prompt",
]

#: The first line of every plan-check request, so a scripted model can tell it apart.
PLAN_MARKER = "[story plan check]"
#: The most unplanned turns reported for one scene.
MAX_UNPLANNED = 2


def _batchim(word: str) -> bool:
    last = word.strip()[-1:]
    return "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0


def _i_ga(word: str) -> str:
    return f"{word}{'이' if _batchim(word) else '가'}"


def _names(codex: CodexIndex, entry_id: str) -> tuple[str, list[str]] | None:
    """The shown name of a character entry and every name it may be called by."""
    for item in codex.items:
        if item.kind == "characters" and item.entry.id == entry_id:
            entry = item.entry
            return entry.name, [entry.name, *entry.aliases]
    return None


def absent_cast(codex: CodexIndex, scene: Scene, text: str) -> list[Finding]:
    """The characters the scene's plan lists that its text never names, one finding each.

    A planned id the codex does not have (a rejected cast proposal) is not looked for, and
    neither is a character whose every name is shorter than `MIN_NAME_CHARACTERS`.
    """
    haystack = folded(text)
    findings: list[Finding] = []
    for entry_id in scene.characters:
        found = _names(codex, entry_id)
        if found is None:
            continue
        shown, called = found
        words = [w for n in called for w in (n, *n.split())]
        keys = [folded(w) for w in words if len(folded(w)) >= MIN_NAME_CHARACTERS]
        if not keys or any(key in haystack for key in keys):
            continue
        findings.append(
            Finding(
                scene_id=scene.id,
                scene_title=scene.title,
                quote="",
                note=f"‘{scene.title}’ 장면 계획에 있던 {_i_ga(shown)} 쓴 장면에는 나오지 않습니다.",
                kind="plan_cast_absent",
            )
        )
    return findings


def asks_model(scene: Scene, *, chapter: int, twist: str | None, twist_chapter: int | None) -> bool:
    """Whether the scene has anything for the model to read: beats, or a twist to keep."""
    return bool(scene.beats) or _twist_window(chapter, twist, twist_chapter)


def _twist_window(chapter: int, twist: str | None, twist_chapter: int | None) -> bool:
    return bool(twist) and twist_chapter is not None and chapter < twist_chapter


def plan_prompt(
    scene: Scene,
    text: str,
    *,
    chapter: int,
    twist: str | None,
    twist_chapter: int | None,
) -> str:
    """What the reading model is asked: the scene's plan, the twist if it must stay hidden,
    and the scene."""
    beats = "\n".join(f"{n}. {beat}" for n, beat in enumerate(scene.beats, start=1))
    hidden = _twist_window(chapter, twist, twist_chapter)
    twist_part = (
        f"\nThe story's twist, which must stay hidden in this chapter:\n{twist}\n"
        "twist_revealed is true only if the scene tells the reader the twist itself, not "
        "just a hint or a clue. Its quote is copied exactly from the scene.\n"
        if hidden
        else ""
    )
    return f"""{PLAN_MARKER}
You compare one written scene with the plan it was written from. You only report what
the text does; you do not judge the writing.

Scene: {scene.title}. {scene.summary}
Planned beats, in order:
{beats or "(none)"}
{twist_part}
For each planned beat, say whether the scene carries it out (done true or false), with a
short quote copied exactly from the scene when it does.
List as unplanned only events that change the story and that neither the summary nor any
beat calls for: a death, a secret told, someone leaving for good. Each with a quote copied
exactly from the scene. Most scenes have none.

Reply with JSON only:
{{"beats": [{{"beat": 1, "done": true, "quote": "<exact words>"}}], "twist_revealed": false, "twist_quote": "", "unplanned": [{{"quote": "<exact words>"}}]}}

<scene>
{text}
</scene>"""


def _dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [cast("dict[str, Any]", v) for v in cast("list[object]", value) if isinstance(v, dict)]


def _false(value: object) -> bool:
    """Whether a reply said no: `false`, or a word for it. Anything unclear is not a no."""
    if isinstance(value, bool):
        return not value
    return isinstance(value, str) and folded(value) in ("false", "no", "아니오", "아니요")


def _true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and folded(value) in ("true", "yes", "예", "네")


def plan_findings(
    scene: Scene,
    text: str,
    reply: str,
    *,
    chapter: int,
    twist: str | None,
    twist_chapter: int | None,
) -> list[Finding]:
    """What the model's reading of scene `scene` shows against its plan and the twist.

    Raises:
        ValueError: the reply has no JSON object; the scene is then not read, never called
            consistent.
    """
    data = reply_object(reply)
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    found = cast("Mapping[str, Any]", data)
    title = scene.title
    findings: list[Finding] = []
    missed: set[int] = set()
    for read in _dicts(found.get("beats")):
        number = read.get("beat")
        if isinstance(number, bool) or not isinstance(number, int):
            continue
        if 1 <= number <= len(scene.beats) and _false(read.get("done")):
            missed.add(number)
    for number in sorted(missed):
        beat = scene.beats[number - 1]
        findings.append(
            Finding(
                scene_id=scene.id,
                scene_title=title,
                quote="",
                note=f"‘{title}’ 장면 계획의 “{beat}” 부분이 쓴 장면에 보이지 않습니다.",
                kind="plan_beat_missing",
            )
        )
    if _twist_window(chapter, twist, twist_chapter) and _true(found.get("twist_revealed")):
        quote = str(found.get("twist_quote") or "").strip()
        if quote_found(quote, text):
            findings.append(
                Finding(
                    scene_id=scene.id,
                    scene_title=title,
                    quote=quote,
                    note=(
                        f"‘{title}’ 장면의 “{quote}” — 반전은 {twist_chapter}장에서 드러나기로 "
                        f"정했는데, 이 문장이 {chapter}장에서 먼저 드러냅니다."
                    ),
                    kind="twist_early",
                )
            )
    kept: list[str] = []
    for event in _dicts(found.get("unplanned")):
        quote = str(event.get("quote") or "").strip()
        if quote in kept or not quote_found(quote, text):
            continue
        kept.append(quote)
        if len(kept) > MAX_UNPLANNED:
            break
        findings.append(
            Finding(
                scene_id=scene.id,
                scene_title=title,
                quote=quote,
                note=f"‘{title}’ 장면의 “{quote}” — 장면 계획에 없던 큰 사건입니다.",
                kind="plan_unplanned_event",
            )
        )
    return findings
