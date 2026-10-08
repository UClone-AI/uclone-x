"""What the Writer is given before it writes a scene, and where a new conversation picks up (#1556).

`scene_context` is the bundle for one scene: the scene and its neighbours, the end of the
scene before it, and the codex entries it needs, with a **manifest** that says which
entries went in and why, which were left out and why, and what was not applied. Nothing is
cut in silence: an entry over the budget is listed as left out (P6).

`recap` is the start of a new conversation: what the story is, what the last conversations
did, how far the manuscript has got, and the next scene's bundle, so that a conversation
with no memory of the last one can continue from the recap alone.

Progressions (a change to an entry once a scene has ended) are **applied in story-time
order** (`uclone_x.story.timeline`): the state an entry shows is the one the scene starts
from, so a flashback sees the world as it was then. The manifest lists what was applied,
what the scene itself changes, what could not be placed, and which scenes were placed in
time by assumption because they have no `story_time`. Every entry the scene does not list
that a change before the scene touched is also one line under `earlier_changes`, so a death
or a loss reaches the Writer without the whole entry, and plainly (#1613). So is an entry the
scene lists that the budget left out of the bundle.

This module is pure: it is given the files' contents and reads nothing itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from uclone_x.story.schemas import (
    Chapter,
    CharacterEntry,
    CodexEntry,
    CodexKind,
    Outline,
    Scene,
    SessionsFile,
    ThreadEntry,
)
from uclone_x.story.timeline import EntrySnapshot, assumptions, entry_snapshot, place_scenes

__all__ = [
    "MAX_EARLIER_CHANGES",
    "MAX_ENTRIES",
    "RECENT_SESSIONS",
    "TAIL_CHARS",
    "CodexIndex",
    "CodexItem",
    "UnreadableFile",
    "NamedChange",
    "changed_before_and_named",
    "continuity_note",
    "last_and_next",
    "named_changes",
    "recap",
    "request_conflicts",
    "render_entry",
    "scene_context",
    "tail",
]

#: How much of the previous scene's end the Writer is given, in characters.
TAIL_CHARS = 1200
#: How many codex entries one scene's context holds; the rest are listed as left out.
MAX_ENTRIES = 12
#: How many one-line changes to entries outside the bundle one scene's context holds; the
#: latest are kept and the number left out is said.
MAX_EARLIER_CHANGES = 24
#: How many of the latest conversations a recap describes.
RECENT_SESSIONS = 3
#: Said with a character's appearance once a visual change applies: the prose does not change.
_APPEARANCE_NOTE = (
    "appearance is the starting description; visual_tags include the changes up to this "
    "scene and hold where the two disagree."
)
#: A name shorter than this is not looked for in the text: one letter matches everything.
_MIN_NAME = 2


@dataclass(frozen=True)
class CodexItem:
    """A codex entry that loaded, and which kind it is."""

    kind: CodexKind
    entry: CodexEntry


@dataclass(frozen=True)
class UnreadableFile:
    """A story file that did not load, and the reason, naming the field."""

    file: str
    reason: str


@dataclass(frozen=True)
class CodexIndex:
    """Every codex entry that loaded, and every file that did not -- none is skipped."""

    items: tuple[CodexItem, ...] = ()
    unreadable: tuple[UnreadableFile, ...] = ()

    def find(self, entry_id: str, kind: CodexKind | None = None) -> list[CodexItem]:
        """The entries called `entry_id`, of `kind` when given."""
        return [
            item
            for item in self.items
            if item.entry.id == entry_id and (kind is None or item.kind == kind)
        ]


def tail(text: str, limit: int = TAIL_CHARS) -> str:
    """The end of `text`, at most `limit` characters, starting at a word when it is cut."""
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    space = cut.find(" ")
    if 0 <= space < limit // 4:
        cut = cut[space + 1 :]
    return "..." + cut


def render_entry(item: CodexItem, snapshot: EntrySnapshot | None = None) -> dict[str, Any]:
    """A codex entry as the Writer reads it; empty fields are left out.

    With a `snapshot`, the state (and a character's visual tags) are the ones at that
    moment of the story rather than the entry's starting ones.

    A character with a visual block also carries what a prose writer needs of it: its
    `appearance` (the block's prose description) and its `gender`. The tags are written
    for an image model; the rest of the block (seed, style, negative tags) is left out.
    The prose has no progressions, so when a visual change was applied before this
    moment, `appearance_note` says the tags hold where the two disagree.
    """
    entry = item.entry
    out: dict[str, Any] = {"id": entry.id, "kind": item.kind, "name": entry.name}
    if entry.aliases:
        out["aliases"] = list(entry.aliases)
    if entry.profile:
        out["profile"] = entry.profile
    state = snapshot.state if snapshot is not None else dict(entry.state)
    if state:
        out["state"] = state
    relations = snapshot.relations if snapshot is not None else dict(entry.relations)
    if relations:
        out["relations"] = relations
    if snapshot is not None and snapshot.visual_tags is not None:
        out["visual_tags"] = list(snapshot.visual_tags)
    visual = entry.visual if isinstance(entry, CharacterEntry) else None
    if visual is not None:
        if visual.prose:
            out["appearance"] = visual.prose
            if snapshot is not None and any(p["kind"] == "visual" for p in snapshot.applied):
                out["appearance_note"] = _APPEARANCE_NOTE
        if visual.gender is not None:
            out["gender"] = visual.gender
    if entry.notes:
        out["notes"] = entry.notes
    if isinstance(entry, ThreadEntry):
        for key in ("planted_in", "pay_off_by", "paid_off_in"):
            value = getattr(entry, key)
            if value is not None:
                out[key] = value
    if visual is not None:
        out["has_visual"] = True
    return out


def _scene_line(chapter: Chapter, scene: Scene) -> dict[str, Any]:
    line: dict[str, Any] = {"scene_id": scene.id, "chapter_id": chapter.id, "title": scene.title}
    if scene.summary:
        line["summary"] = scene.summary
    return line


def _mentions(entry: CodexEntry, haystack: str) -> bool:
    return any(
        len(name) >= _MIN_NAME and name.casefold() in haystack
        for name in (entry.name, *entry.aliases)
    )


def _value_words(value: Any) -> str:
    if value is None:
        return "(cleared)"
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in cast("list[Any]", value)) + "]"
    return str(value)


def _change_line(item: CodexItem, snapshot: EntrySnapshot) -> str | None:
    """One line: what the story changed in an entry before the scene, and where.

    Only state changes: visual tags are for drawing, and this line is for the prose. Each
    change still in force is given once, in story order, as the values it left (a cleared
    value says so), the scene it came at and its note; a value a later change replaced is
    not repeated. `None` when no state change applies yet.
    """
    parts: list[str] = []
    said: set[str] = set()
    for change in reversed(snapshot.applied):
        if change["kind"] != "state":
            continue
        keys = [
            k for k in change["set"] if k not in said and snapshot.set_at.get(k) == change["at"]
        ]
        if not keys:
            continue
        said.update(keys)
        values = ", ".join(f"{k}: {_value_words(snapshot.state.get(k))}" for k in keys)
        where = f"since {change['at']}" + (f", {change['note']}" if change.get("note") else "")
        parts.append(f"{values} ({where})")
    if not parts:
        return None
    return f"{item.entry.name} ('{item.entry.id}', {item.kind}): " + "; ".join(reversed(parts))


#: How many sentences of the new text a continuity line quotes, and how long each may be.
_QUOTED_SENTENCES = 2
_QUOTE_CHARS = 160
#: Where a sentence ends: after terminal punctuation followed by space, or at a line break.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+|\n+")


_HANGUL = re.compile(r"[\uac00-\ud7a3]")


def continuity_note(scene_id: str, digest: str, text: str, *, rewrite_of_own: bool) -> str:
    """What a saved scene is told after the lines of `changed_before_and_named` (#1613).

    The write is done either way: a memory or a flashback names the dead too, so the lines
    are a notice to check against, never a refusal. The first notice opens with the next
    action -- rewrite the scene -- and its exact arguments. A write that replaces text the
    same conversation wrote is already the revision, so its notice does not ask for
    another: one rewrite per notice, and the Writer cannot be sent round the same lines
    forever.

    The note is in the language of the scene. Measured with qwen3:8b on a Korean
    manuscript (paired seeds, tempt and present, 3 reps each): an English note, whether
    the one below, was followed by a rewrite in 0 of 6
    runs, the same note placed first in the result in 0 of 6, and the Korean note below in
    6 of 6 (design doc section 5.3).
    """
    korean = bool(_HANGUL.search(text))
    if rewrite_of_own:
        if korean:
            return (
                "이 대화가 쓴 원고 위에 고친 장면이 저장되었습니다. 이 줄들 때문에 다시 "
                "고치지는 마세요. 인용한 문장이 아직 죽은 인물을 살아 있는 것처럼, 잃은 "
                "물건을 가진 것처럼 보여 준다면 답장에 그렇게 적고 사람이 정하게 하세요."
            )
        return (
            "The scene was saved over the text this conversation wrote, so it is not to be "
            "rewritten again for these lines. If the quoted words still show the dead alive "
            "or a lost thing in hand, say so in your reply and let the person decide."
        )
    if korean:
        return (
            "아직 끝나지 않았습니다. 이 장면을 고쳐 다시 쓰세요: 위 줄에 인용한 문장이 이 "
            "장면보다 먼저 죽은 인물을 살아 있거나 곁에 있는 것처럼, 잃은 물건을 가진 것처럼 "
            f"보여 줍니다. story_manuscript 'write'를 scene_id '{scene_id}', digest "
            f"'{digest}'로 다시 불러 고친 장면을 저장하세요. 인용한 문장이 회상이거나 죽음과 "
            "상실을 말하는 것이라면 그대로 두어도 되고, 그렇다고 답장에 적으세요."
        )
    return (
        "The scene was saved, and it needs one decision before you reply. Each line above "
        "quotes where the new text names someone who died, or something that was lost, "
        "before this scene happens. If the scene shows them as they were before -- the dead "
        "alive or present, the lost thing in hand -- rewrite the scene so it does not, and "
        f"save it now with story_manuscript 'write', scene_id '{scene_id}' and digest "
        f"'{digest}'. If each is only a memory, a flashback, or says they are dead or it is "
        "gone, keep the scene as it is and say so in your reply."
    )


#: A sentence that remembers rather than shows (#1808): the dead recalled, missed or heard
#: in the mind, the lost thing remembered. A write-time notice fired on 30 of 30 qwen3:8b
#: drafts, the clean ones included, because a memory names the dead as an act does; a
#: notice on every draft is one the Writer learns to pass over. Narrow on purpose: an act
#: in the same sentence as one of these words is read as memory too, and missed.
_RECALL = re.compile(
    r"기억|회상|떠올|그리워|그리운|그립|생전|목소리가\s*(?:머릿속|귓가)|꿈에"
    r"|\b(?:remember\w*|recall\w*|memor(?:y|ies))\b",
    re.IGNORECASE,
)

#: A sentence that says the change happened, or that the act did not (#1808): the dead said
#: to be dead or gone, the lost thing said to be lost or taken, or a use negated ("월광검을
#: 뽑지 않았다"). Such sentences ("예린은 이미 영원히 떠났고", "월광검을 뽑아내지
#: 않았다") keep a scene true to the change, and quoting them asks for a needless rewrite.
#: As narrow as `_RECALL`, and missing the same way: an act in the same sentence as one of
#: these words ("죽은 듯 잠든 예린이 눈을 떴다", "the lost sword in her hand", "그는 월광검을
#: 들고도 휘두르지 않았다", where he holds it) is read as a statement, and missed. A sentence
#: that only describes or asks about the dead without these words is still quoted. Of the
#: "~지 않" forms only "않았" and "않는" count: "~지 않고" is also the everyday "without
#: ~ing", and counting it passed over "주저하지 않고 월광검을 뽑아 들었다".
_STATED_OR_DENIED = re.compile(
    r"죽었|죽은|숨을\s*거[두둔뒀]|떠났|떠난|세상을\s*떠|잃었|잃은|빼앗겼|빼앗긴"
    r"|지\s*않았|지\s*않는|지\s*못"
    r"|\b(?:died|dead|gone|lost|(?:did|does|do|could)(?:\s+not|n['’]t))\b",
    re.IGNORECASE,
)

#: A use of a lost item refused or replaced (#1808), read from right after its name: the
#: next word is one of the item's own use verbs (including 쓰, 사용하, 잡, 꺼내), negated
#: ("월광검을 뽑지 않고", "월광검을 뽑아내지 않았다"), or 아닌 ("월광검이 아닌 칼"). Anchored
#: and limited to the use verbs because "~지 않고" is also "without ~ing": "월광검을 주저하지
#: 않고 뽑았다" is a use and is quoted, and so is "꿈이 아닌 듯, 라온은 월광검을 쥐었다". Never
#: for a character: "예린은 주저하지 않고 칼을 들었다" is the dead acting. Still quoted:
#: a word between the name and the negated verb ("월광검을 끝내 뽑지 않고"), unless
#: `_STATED_OR_DENIED` excuses it.
_DENIED_USE = re.compile(
    r"\S*\s+(?:(?:뽑|쥐|휘두르|들|차|겨누|베|끼|착용하|챙기|쓰|사용하|잡|꺼내)\S*지\s*않|아닌)"
)


def _use_denied(sentence: str, names: Sequence[str]) -> bool:
    """Whether a name in `sentence` (casefolded) is followed by `_DENIED_USE`."""
    return any(
        _DENIED_USE.match(sentence, found.end())
        for name in names
        for found in re.finditer(re.escape(name), sentence)
    )


def _naming_sentences(entry: CodexEntry, text: str, *, lost: bool) -> str | None:
    """The first sentences of `text` that name `entry`, quoted, for a continuity line.

    Sentences that only remember (`_RECALL`), that say the death or loss happened or the act
    did not (`_STATED_OR_DENIED`), or, for a `lost` item, that refuse its use
    (`_use_denied`), are left out, and when every sentence that names the entry is one,
    there is nothing to tell: `None`. A name the sentences do not hold -- split across a
    line -- is still told, with no quote.
    """
    names = [n.casefold() for n in (entry.name, *entry.aliases) if len(n) >= _MIN_NAME]
    found: list[str] = []
    passed_over = False
    for sentence in _SENTENCE_END.split(text):
        sentence = " ".join(sentence.split())
        if not (sentence and any(n in sentence.casefold() for n in names)):
            continue
        if (
            _RECALL.search(sentence)
            or _STATED_OR_DENIED.search(sentence)
            or (lost and _use_denied(sentence.casefold(), names))
        ):
            passed_over = True
            continue
        if len(sentence) > _QUOTE_CHARS:
            sentence = sentence[: _QUOTE_CHARS - 1] + "…"
        found.append(f'"{sentence}"')
        if len(found) == _QUOTED_SENTENCES:
            break
    if passed_over and not found:
        return None
    return "; ".join(found)


#: The `status` value that marks a character as dead, compared ignoring case.
_DEAD = "dead"


def _held(value: Any) -> list[str]:
    """What a `possesses` value names: a list of names or ids, or one."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in cast("list[Any]", value)]
    return []


def _is_item(entry: CodexEntry, value: str) -> bool:
    wanted = value.strip().casefold()
    return any(wanted == name.casefold() for name in (entry.id, entry.name, *entry.aliases))


def _at_scene(outline: Outline, at: str, change: Mapping[str, Any] | None) -> str:
    """`scene <id> 「<title>」`, with the change's note when it has one."""
    found = outline.find(at)
    words = f"scene {at}" + (f" 「{found[1].title}」" if found else "")
    note = change.get("note") if change else None
    return f"{words} ({note})" if note else words


def _change_at(snapshot: EntrySnapshot, key: str) -> Mapping[str, Any] | None:
    at = snapshot.set_at.get(key)
    return next(
        (
            c
            for c in reversed(snapshot.applied)
            if c["kind"] == "state" and c["at"] == at and key in c["set"]
        ),
        None,
    )


@dataclass(frozen=True)
class NamedChange:
    """One thing a text names that the story changed before a scene: a death or a loss.

    `entry` is the dead character, or the lost item; `owner` is who lost the item last and
    `holders` who has it when the scene starts. `at` is the scene the change is placed at.
    """

    kind: str  # "dead" or "lost"
    entry: CodexEntry
    at: str
    change: Mapping[str, Any] | None
    owner: CodexEntry | None = None
    holders: tuple[str, ...] = ()


def named_changes(
    outline: Outline, scene_id: str, codex: CodexIndex, text: str
) -> list[NamedChange]:
    """The deaths and losses before `scene_id` that `text` names, deaths first (#1613).

    Two rules, both decided from the codex and needing no facts from the model:

    - a character the story marked dead (`status: dead`, set by a change at a scene that
      happens before this one) whose name or alias the text contains;
    - an item the text names that someone held earlier and no longer holds when this scene
      starts. Of those who lost it, the one who lost it last is named, with who holds it
      now.

    "Before" is in story time, as for `scene_context`: a flashback set before the death sees
    the character alive and gets nothing. A change placed at this scene itself is the scene's
    own and is not here. A dead entry with no change that made it so (a founder dead from
    the start) is not here either; its entry says so.

    Raises:
        ValueError: the outline has no scene `scene_id` (the caller checks first).
    """
    placements = place_scenes(outline)
    haystack = text.casefold()
    snapshots = {
        (item.kind, item.entry.id): entry_snapshot(
            item.entry, placements, scene_id, through_scene=False
        )
        for item in codex.items
    }
    found: list[NamedChange] = []
    for item in codex.items:
        snapshot = snapshots[(item.kind, item.entry.id)]
        status = snapshot.state.get("status")
        if (
            isinstance(status, str)
            and status.strip().casefold() == _DEAD
            and "status" in snapshot.set_at
            and _mentions(item.entry, haystack)
        ):
            at = snapshot.set_at["status"]
            found.append(NamedChange("dead", item.entry, at, _change_at(snapshot, "status")))
    for thing in codex.items:
        if thing.kind != "items" or not _mentions(thing.entry, haystack):
            continue
        holders: list[str] = []
        lost: list[tuple[tuple[Any, ...], CodexItem, EntrySnapshot]] = []
        for item in codex.items:
            snapshot = snapshots[(item.kind, item.entry.id)]
            if any(_is_item(thing.entry, v) for v in _held(snapshot.state.get("possesses"))):
                holders.append(item.entry.name)
                continue
            at = snapshot.set_at.get("possesses")
            earlier = [_held(item.entry.state.get("possesses"))] + [
                _held(c["set"].get("possesses")) for c in snapshot.applied if c["kind"] == "state"
            ]
            if at is not None and any(_is_item(thing.entry, v) for vs in earlier for v in vs):
                lost.append((placements[at].position, item, snapshot))
        if not lost:
            continue
        _, owner, snapshot = max(lost, key=lambda found: found[0])
        found.append(
            NamedChange(
                "lost",
                thing.entry,
                snapshot.set_at["possesses"],
                _change_at(snapshot, "possesses"),
                owner=owner.entry,
                holders=tuple(holders),
            )
        )
    return found


def changed_before_and_named(
    outline: Outline, scene_id: str, codex: CodexIndex, text: str
) -> list[str]:
    """What the story changed before `scene_id` that `text` names, one plain line each (#1613).

    The findings of `named_changes`, each with the sentences of `text` that name it: what a
    saved scene is told. A finding named only in sentences that remember it -- `기억`,
    `떠올`, `remember` -- or that say it happened or the act did not -- `떠났`, `빼앗긴`,
    `지 않았`, `died` -- is not told (#1808).

    Raises:
        ValueError: the outline has no scene `scene_id` (the caller checks first).
    """
    lines: list[str] = []
    for item in named_changes(outline, scene_id, codex, text):
        quoted = _naming_sentences(item.entry, text, lost=item.kind == "lost")
        if quoted is None:
            continue  # only remembered or stated: the story's own, not a slip
        where = _at_scene(outline, item.at, item.change)
        if item.kind == "dead":
            name = item.entry.name
            lines.append(
                f"{name} has been dead since {where}, which happens before this scene, and "
                f"the new text names {name}: {quoted}"
            )
            continue
        owner = item.owner.name if item.owner else ""
        line = (
            f"{owner} no longer has {item.entry.name} since {where}, which happens "
            f"before this scene, and the new text names {item.entry.name}: {quoted}"
        )
        if item.holders:
            line += f" -- {', '.join(item.holders)} has it now."
        lines.append(line)
    if lines and codex.unreadable:
        lines.append(
            f"{len(codex.unreadable)} of the story's files could not be read, so what they "
            "hold was not checked."
        )
    return lines


def _josa(word: str, with_final: str, without_final: str) -> str:
    """`word` with the Korean particle its last syllable takes: 을/를, 은/는, 이/가."""
    last = word[-1:] if word else ""
    if not _HANGUL.match(last):
        return f"{word}{with_final}({without_final})"
    return word + (with_final if (ord(last) - 0xAC00) % 28 else without_final)


def _scene_words(outline: Outline, at: str, *, korean: bool) -> str:
    """The scene a change is placed at, by its title, for a person: no ids or field names."""
    found = outline.find(at)
    title = found[1].title if found and found[1].title else at
    return f"「{title}」 장면" if korean else f"the scene 「{title}」"


def request_conflicts(
    outline: Outline, scene_id: str, codex: CodexIndex, request: str
) -> tuple[list[str], str | None]:
    """What the person's request names that the story has already ended, and what to do (#1613).

    One line per character the request names who died before this scene, and per item it
    names that its holder lost before this scene -- the detection `named_changes` makes for
    a saved scene, applied to the request instead, so a flashback set before the death gets
    nothing. Each line is a direct instruction to keep the story as written: the dead only as
    memory or grief, the lost thing still lost. The second value is the note that says to
    tell the person, in one line, what was kept and how to change it; `None` with no lines.

    The words are the request's language, which is the scene's: the Writer answers in the
    language the person writes in, and qwen3:8b followed a Korean write notice on a Korean
    scene where it ignored the English one (design doc section 5.3). No codex id, field name
    or file path is written: the Writer repeats these words to the person.
    """
    korean = bool(_HANGUL.search(request))
    lines: list[str] = []
    for change in named_changes(outline, scene_id, codex, request):
        where = _scene_words(outline, change.at, korean=korean)
        note = change.change.get("note") if change.change else None
        why = f"({note})" if note else ""
        thing = change.entry.name
        if change.kind == "dead":
            lines.append(
                f"{thing}: {where}에서 죽었습니다{why}. 이 장면은 그 뒤입니다. "
                f"{_josa(thing, '을', '를')} 살아 있거나 곁에 있는 인물로 쓰지 말고, 기억이나 슬픔으로만 쓰세요."
                if korean
                else f"{thing} died in {where}{' ' + why if why else ''}, before this scene. "
                f"Do not write {thing} alive or present; write {thing} only as a memory or "
                "as grief."
            )
            continue
        owner = change.owner.name if change.owner else ""
        now_ko = f" 지금은 {', '.join(change.holders)}에게 있습니다." if change.holders else ""
        now_en = f" {', '.join(change.holders)} has it now." if change.holders else ""
        lines.append(
            f"{thing}: {_josa(owner, '은', '는')} {where}에서 {_josa(thing, '을', '를')} "
            f"잃었습니다{why}.{now_ko} {_josa(owner, '이', '가')} {_josa(thing, '을', '를')} "
            "쥐거나 쓰는 장면으로 쓰지 말고, 잃은 것으로 두세요."
            if korean
            else f"{thing}: {owner} lost it in {where}{' ' + why if why else ''}, before this "
            f"scene.{now_en} Do not write {owner} holding or using {thing}; keep it lost."
        )
    if not lines:
        return [], None
    note = (
        "요청이 이야기와 어긋나는 곳은 위 줄대로, 이야기대로 쓰세요. 먼저 묻지 말고 장면을 "
        "쓴 뒤, 답장 끝에 한 줄로 무엇을 이야기대로 두었는지와, 바꾸고 싶으면 설정 변경을 "
        "제안받아 승인하면 된다는 것을 사람에게 알리세요."
        if korean
        else "Where the request contradicts the story, write the story as it stands, as the "
        "lines above say. Do not ask first: write the scene, then end your reply with one "
        "line telling the person what you kept, and that to change it they can have a change "
        "to the story's settings proposed and approve it."
    )
    return lines, note


def scene_context(
    outline: Outline,
    scene_id: str,
    codex: CodexIndex,
    *,
    previous_text: str | None,
    story: Mapping[str, Any],
    request: str | None = None,
) -> dict[str, Any]:
    """The bundle the Writer reads before writing `scene_id`, with its manifest.

    Entries go in in this order until `MAX_ENTRIES`: the characters and places the scene
    names, then every `always_include` entry, then every entry whose name or alias the
    scene's title, summary or beats -- or the end of the scene before -- mention.

    Every entry the scene does not list (in `characters` or `places`) that changed before
    the scene -- in story time, as for the entries in the bundle -- is under
    `earlier_changes`, one line each, latest first, up to `MAX_EARLIER_CHANGES`. An entry
    only mentioned or always included has its line too: its full state was not enough for a
    small model to notice a death (#1613). So has an entry the scene lists that is over the
    `MAX_ENTRIES` budget: it is not in the bundle, and without the line nothing of it would be.

    With `request` (the person's request or brief for this scene, verbatim), each character
    it names who died before the scene and each item it names that was lost before the scene
    is one line under `request_conflicts`, the first key of the bundle, with
    `request_conflicts_note` after it (`request_conflicts`). A request that names nothing
    of the kind, or no request, leaves the bundle as it was.

    Raises:
        ValueError: the outline has no scene `scene_id` (the caller checks first).
    """
    ordered = outline.scenes_in_order()
    index = next((i for i, (_, s) in enumerate(ordered) if s.id == scene_id), None)
    if index is None:
        raise ValueError(f"the outline has no scene '{scene_id}'")
    chapter, scene = ordered[index]
    previous_tail = tail(previous_text) if previous_text else None

    candidates: list[tuple[CodexItem, str]] = []
    named_missing: list[dict[str, str]] = []
    named: tuple[tuple[CodexKind, list[str]], ...] = (
        ("characters", scene.characters),
        ("places", scene.places),
    )
    for kind, ids in named:
        for entry_id in ids:
            found = codex.find(entry_id, kind)
            if found:
                candidates.append((found[0], "named in the scene"))
            else:
                named_missing.append({"id": entry_id, "kind": kind})
    for item in codex.items:
        if item.entry.always_include:
            candidates.append((item, "always included"))
    haystack = " ".join([scene.title, scene.summary, *scene.beats, previous_tail or ""]).casefold()
    for item in codex.items:
        if _mentions(item.entry, haystack):
            candidates.append((item, "mentioned in the scene or the end of the scene before"))

    placements = place_scenes(outline)
    snapshots: dict[tuple[str, str], EntrySnapshot] = {}
    seen: set[tuple[str, str]] = set()
    included: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    for item, reason in candidates:
        key = (item.kind, item.entry.id)
        if key in seen:
            continue
        seen.add(key)
        if len(entries) >= MAX_ENTRIES:
            left_out.append(
                {
                    "id": item.entry.id,
                    "kind": item.kind,
                    "reason": f"over the budget of {MAX_ENTRIES} entries ({reason})",
                }
            )
            continue
        snapshot = entry_snapshot(item.entry, placements, scene.id, through_scene=False)
        snapshots[key] = snapshot
        entries.append(render_entry(item, snapshot))
        included.append({"id": item.entry.id, "kind": item.kind, "reason": reason})
    for item in codex.items:
        if (item.kind, item.entry.id) not in seen:
            left_out.append(
                {
                    "id": item.entry.id,
                    "kind": item.kind,
                    "reason": "not named, always-included or mentioned in this scene",
                }
            )

    # An entry the scene does not list can still have changed before it: a teacher who
    # died, a sword that was lost. One line each, latest first, so the Writer does not bring
    # them back (#1613). A change placed at this scene itself is not here: the scene starts
    # before it, as for the entries in the bundle. An entry the scene lists but the budget
    # left out has its line too: nothing else of it is in the bundle.
    listed = {(kind, entry_id) for kind, ids in named for entry_id in ids}
    listed_in_full = {key for key in listed if key in snapshots}
    dated: list[tuple[tuple[Any, ...], str]] = []
    for item in codex.items:
        if (item.kind, item.entry.id) in listed_in_full:
            continue
        outside = entry_snapshot(item.entry, placements, scene.id, through_scene=False)
        line = _change_line(item, outside)
        if line is not None:
            latest = max(placements[at].position for at in outside.set_at.values())
            dated.append((latest, line))
    dated.sort(key=lambda pair: pair[0], reverse=True)
    earlier_changes = [line for _, line in dated[:MAX_EARLIER_CHANGES]]

    applied: list[dict[str, Any]] = []
    in_this_scene: list[dict[str, Any]] = []
    not_placed: list[dict[str, Any]] = []
    timed_scenes: set[str] = {scene.id}
    for (kind, entry_id), snapshot in snapshots.items():
        if snapshot.applied:
            applied.append({"id": entry_id, "kind": kind, "applied": snapshot.applied})
        if snapshot.in_this_scene:
            in_this_scene.append({"id": entry_id, "kind": kind, "changes": snapshot.in_this_scene})
        if snapshot.not_placed:
            not_placed.append({"id": entry_id, "kind": kind, "progressions": snapshot.not_placed})
        timed_scenes.update(p["at"] for p in (*snapshot.applied, *snapshot.in_this_scene))

    manifest: dict[str, Any] = {"included": included, "left_out": left_out}
    if named_missing:
        manifest["named_without_an_entry"] = named_missing
    if applied:
        manifest["progressions_applied"] = applied
    if in_this_scene:
        manifest["changes_in_this_scene"] = in_this_scene
    if not_placed:
        manifest["progressions_not_placed"] = not_placed
    if applied or in_this_scene or not_placed:
        manifest["progressions_note"] = (
            "Each entry's state is the one this scene starts from: the changes placed at "
            "scenes that happen before it in story time are applied. The changes this scene "
            "makes are under changes_in_this_scene. A change placed at a scene that is not in "
            "the outline is not applied; it is under progressions_not_placed."
        )
        placed_by_assumption = [a for a in assumptions(placements) if a["scene_id"] in timed_scenes]
        if placed_by_assumption:
            manifest["story_time_assumptions"] = placed_by_assumption
    if earlier_changes:
        manifest["earlier_changes_note"] = (
            "earlier_changes lists, one line each, what the story changed before this scene "
            "in entries the scene does not list, or lists but that did not fit in codex: a "
            "character marked dead is dead here, and "
            "what someone no longer possesses is not theirs to use."
        )
    if len(dated) > len(earlier_changes):
        manifest["earlier_changes_not_shown"] = len(dated) - len(earlier_changes)
    if codex.unreadable:
        manifest["unreadable_files"] = [
            {"file": u.file, "reason": u.reason} for u in codex.unreadable
        ]

    bundle: dict[str, Any] = {}
    conflicts, conflicts_note = (
        request_conflicts(outline, scene_id, codex, request) if request else ([], None)
    )
    if conflicts:
        bundle["request_conflicts"] = conflicts
        bundle["request_conflicts_note"] = conflicts_note
    bundle |= {
        "story": dict(story),
        "chapter": {"chapter_id": chapter.id, "title": chapter.title},
        "scene": scene.model_dump(mode="json", exclude_defaults=True),
        "previous_scene": _scene_line(*ordered[index - 1]) if index > 0 else None,
        "next_scene": _scene_line(*ordered[index + 1]) if index + 1 < len(ordered) else None,
        "end_of_previous_scene": previous_tail,
        "codex": entries,
        "manifest": manifest,
    }
    if earlier_changes:
        bundle["earlier_changes"] = earlier_changes
    if chapter.act is not None:
        bundle["chapter"]["act"] = chapter.act
    return bundle


def last_and_next(
    outline: Outline, written: Sequence[str]
) -> tuple[tuple[Chapter, Scene] | None, tuple[Chapter, Scene] | None]:
    """The last written scene in reading order, and the first unwritten scene after it.

    With nothing written, the next scene is the first one. With every scene after the last
    written one written too, there is no next scene.
    """
    ordered = outline.scenes_in_order()
    done = set(written)
    last_index = max((i for i, (_, s) in enumerate(ordered) if s.id in done), default=None)
    start = 0 if last_index is None else last_index + 1
    following = next((pair for pair in ordered[start:] if pair[1].id not in done), None)
    return (ordered[last_index] if last_index is not None else None), following


def recap(
    *,
    story: Mapping[str, Any],
    outline: Outline | None,
    written: Sequence[str],
    sessions: SessionsFile,
    last_text: str | None,
    next_context: Mapping[str, Any] | None,
    problems: Sequence[UnreadableFile] = (),
) -> dict[str, Any]:
    """Where the story stands, for a conversation that remembers none of it.

    `last_text` is the text of the last written scene `last_and_next` names, and
    `next_context` the `scene_context` of the next one; the caller reads both.
    """
    recent = list(sessions.sessions[-RECENT_SESSIONS:])
    out: dict[str, Any] = {
        "story": dict(story),
        "recent_sessions": [s.model_dump(mode="json", exclude_defaults=True) for s in recent],
    }
    older = len(sessions.sessions) - len(recent)
    if older:
        out["older_sessions_not_shown"] = older
    if problems:
        out["unreadable_files"] = [{"file": p.file, "reason": p.reason} for p in problems]
    if outline is None:
        out["progress"] = {"scenes_in_outline": 0, "scenes_written": len(written)}
        out["how_to_continue"] = (
            "The story has no outline yet. Start one with story_outline 'init', then add "
            "scenes with 'set_scene'."
        )
        return out
    in_outline = [s.id for _, s in outline.scenes_in_order()]
    progress: dict[str, Any] = {
        "scenes_in_outline": len(in_outline),
        "scenes_written": sum(1 for s in in_outline if s in set(written)),
    }
    stray = sorted(set(written) - set(in_outline))
    if stray:
        progress["written_but_not_in_outline"] = stray
    out["progress"] = progress
    last, following = last_and_next(outline, written)
    if last is not None:
        out["last_written_scene"] = {
            **_scene_line(*last),
            "end_of_text": tail(last_text) if last_text else None,
        }
    if following is None:
        out["next_scene"] = None
        out["how_to_continue"] = (
            "Every scene in the outline after the last written one has text. Add the next "
            "scene with story_outline 'set_scene', or revise a written one."
        )
        return out
    out["next_scene"] = dict(next_context) if next_context is not None else None
    out["how_to_continue"] = (
        f"Write scene '{following[1].id}' next: its context is under next_scene. Save it "
        "with story_manuscript 'write', and give a one-paragraph session_summary of where "
        "the story stands."
    )
    return out
