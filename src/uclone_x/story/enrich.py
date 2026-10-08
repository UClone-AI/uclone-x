"""The codex grown from each written scene: what a scene adds, proposed for a person to approve.

§1.1 row 7 of the novel-writer design: the codex used to grow only when the model chose to
call `story_codex propose`. Here every scene `story_manuscript write` saves is read once more
(one model call, temperature 0) for what it **adds** to the codex as it stands at that scene:

- a character, place or item the codex does not have, which becomes a new-entry proposal as
  `story_codex create` makes one;
- a change to an entry it has -- a death (``status``), an item gained or lost (``gains``,
  ``loses``), a wound or other lasting condition (``condition``) -- which becomes a
  progression proposal at that scene, as `story_codex propose` makes one;
- a family relation between two characters it has (key ``family``, the other in ``of``),
  read as `continuity` reads one (`family_role`), and kept only when the quote names a
  kinship word. It is proposed as a change to the entry's `relations:`;
- a new character's family among those the codex has (``family`` on the new character),
  kept on the same terms and written on both sides as the cast's is: in the new entry's
  `relations:`, and a progression proposal at that scene on the character it has, with the
  inverse role (`INVERSE_ROLE`). A relation that is not kept leaves the new character
  proposed without it, and is counted.

The model proposes; code decides what is kept (§5.3). Dropped, and counted by reason:

- a quote that is not in the scene (`quote_found`), which is what a made-up fact looks like;
- a fact the scene only remembers, denies or imagines (``time``/``polarity``, as the
  continuity reading marks them);
- what the codex already says at that scene, or a pending proposal already proposes;
- what contradicts the codex -- a dead character alive again, a kinship the codex gives
  otherwise, an item lost that was not held. That is the continuity check's to report, never
  an update to propose.

Nothing is applied: each addition is a pending proposal in the story's own store
(`StoryWork.add_proposal`), decided by a person in the story's view (§5.3, §10 Q2).

Every scene the Writer's tools save is read here, through one helper (`grow_scene`): each
scene `story_manuscript write` saves, and each scene `story_start` writes -- a chapter's
scenes and the scenes it rewrites after the continuity check.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import ValidationError

from uclone_x.story.context import CodexIndex
from uclone_x.story.continuity import (
    FAMILY_KEY,
    INVERSE_ROLE,
    POLARITIES,
    TIMES,
    family_role,
    gender_of,
    names_family,
    reply_object,
)
from uclone_x.story.names import CodexNames, id_for_name
from uclone_x.story.proposals import new_entry_clash
from uclone_x.story.quotes import folded, quote_found
from uclone_x.story.schemas import Outline, Proposal
from uclone_x.story.timeline import entry_snapshot, place_scenes

if TYPE_CHECKING:
    from uclone_x.story.work import StoryWork

__all__ = [
    "ENRICH_MARKER",
    "MAX_ADDITIONS",
    "Enrichment",
    "SceneModel",
    "enrich_prompt",
    "enrich_scene",
    "grow_scene",
    "growth_line",
    "growth_note",
]

logger = logging.getLogger(__name__)

#: The first line of every enrichment request, so a scripted model can tell it apart.
ENRICH_MARKER = "[story codex growth]"
#: The most additions one scene proposes: a scene's news, not a whole world.
MAX_ADDITIONS = 8

_NEW_KINDS = ("characters", "places", "items")
_STATUSES = ("alive", "dead")
#: Sentence ends, for finding the sentence a quote sits in: . ! ? and their full-width
#: forms, followed by space, and line breaks.
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


class SceneModel(Protocol):
    """What reading a scene needs from a model: text for a prompt (`start.StoryModel`)."""

    async def complete(
        self, prompt: str, *, system: str, temperature: float, max_tokens: int
    ) -> str: ...


@dataclass
class Enrichment:
    """What one scene added to the codex: the proposals saved, and what was dropped why."""

    scene_id: str
    proposed: list[Proposal] = field(default_factory=list[Proposal])
    #: Reason -> how many of the model's additions were dropped for it.
    dropped: dict[str, int] = field(default_factory=dict[str, int])
    #: The reply could not be read as JSON: nothing was proposed, and it is not "nothing new".
    unread: bool = False

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scene_id": self.scene_id,
            "proposed": [p.id for p in self.proposed],
        }
        if self.dropped:
            out["dropped"] = dict(self.dropped)
        if self.unread:
            out["unread"] = True
        return out


def _entries(codex: CodexIndex, kind: str) -> str:
    lines = [
        f"- {item.entry.id}: {', '.join([item.entry.name, *item.entry.aliases])}"
        for item in codex.items
        if item.kind == kind
    ]
    return "\n".join(lines) or "- (none)"


def enrich_prompt(outline: Outline, codex: CodexIndex, scene_id: str, text: str) -> str:
    """What the reading model is asked: the codex's entries, the keys, and the scene."""
    found = outline.find(scene_id)
    if found is None:
        raise ValueError(f"the outline has no scene '{scene_id}'")
    _, scene = found
    return f"""{ENRICH_MARKER}
You read one scene of a story and list what it ADDS to the story's codex: people, places and
things it brings in that the codex does not have, and lasting changes to those it has. You do
not judge the story; you only report what the text says.

Scene {scene.id}: {scene.title}. {scene.summary}

The codex has these characters:
{_entries(codex, "characters")}
Places:
{_entries(codex, "places")}
Items:
{_entries(codex, "items")}

"new": each named character, place or item the scene brings in that is NOT above. kind is
characters, places or items. profile is one sentence from the scene. gender (female or male)
only for a character the scene shows as one. family, only for a new character the scene
states is family of a character above: {{"<id of that character>": what the new character
is to them, in the words listed under family below}}.

"changes": lasting changes, in the scene's present, to a character above (use the id as
subject). Keys (use no other key):
- status: dead, when the character dies in this scene.
- gains / loses: the id of an item above the character gets or loses.
- condition: a lasting wound or state, in a few words ("왼팔 골절", "blinded").
- family: what the subject is to the character whose id is in "of" (another character
  above): father, mother, son, daughter, brother, sister, husband, wife, grandfather,
  grandmother, grandson or granddaughter.

For every change give a time ("now", or "past" for a remembered or recounted event) and a
polarity ("asserted", "negated", "absent" or "hypothetical"), as a continuity reader would.

quote is copied exactly from the scene, at least a few words, and shows the addition.
Leave out anything the scene does not state. Reply with JSON only:
{{"new": [{{"kind": "characters", "name": "<name>", "profile": "<sentence>", "gender": "female", "family": {{}}, "quote": "<exact words>"}}],
 "changes": [{{"subject": "<id>", "key": "<key>", "value": "<value>", "quote": "<exact words>", "time": "now", "polarity": "asserted"}}]}}
A family change also has "of": "<id>".
Reply {{"new": [], "changes": []}} when the scene adds nothing.

<scene>
{text}
</scene>"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _listed(data: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = data.get(key)
    if not isinstance(raw, list):
        return []
    return [cast("dict[str, Any]", r) for r in cast("list[object]", raw) if isinstance(r, dict)]


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


def _relation_key(other: str) -> str:
    """How `_pending_sets` keys a relation to entry `other`, apart from every state key."""
    return f"relations.{other}"


def _pending_sets(pending: Sequence[Proposal]) -> set[tuple[str, str, str]]:
    """(entry id, key, value) of every state value or relation a pending proposal would set."""
    out: set[tuple[str, str, str]] = set()
    for proposal in pending:
        progression = proposal.change.progression
        if progression is None:
            continue
        for key, value in progression.set.items():
            out.add((proposal.entry_id, key, repr(value)))
        for other, word in progression.relations.items():
            out.add((proposal.entry_id, _relation_key(other), repr(word)))
    return out


class _Reader:
    """One scene's additions checked against the codex as it stands at that scene."""

    def __init__(
        self,
        work: StoryWork,
        *,
        outline: Outline,
        scene_id: str,
        text: str,
        room_id: str,
        agent_id: str | None,
        result: Enrichment,
    ) -> None:
        self.work = work
        self.scene_id = scene_id
        self.text = text
        self.room_id = room_id
        self.agent_id = agent_id
        self.result = result
        self.codex = work.codex()
        self.names = CodexNames(self.codex)
        self.kinds = {item.entry.id: item.kind for item in self.codex.items}
        self.items = {item.entry.id: item for item in self.codex.items}
        self.placements = place_scenes(outline)
        self.pending = [p for p, _ in work.proposals()[0] if p.status == "pending"]
        self.pending_sets = _pending_sets(self.pending)
        self.drafts: list[Proposal] = []

    # -- what is kept -------------------------------------------------------------------

    def quoted(self, raw: Mapping[str, Any]) -> str | None:
        quote = _text(raw, "quote")
        if not quote_found(quote, self.text):
            self.result.drop("quote_not_in_scene")
            return None
        return quote

    def names_subject(self, subject: str, quote: str) -> bool:
        """Whether the quote, or a sentence of the scene that holds it, names the subject.

        qwen3:8b gave "비명이 하나 들렸다" (a scream was heard) as the quote for a death of a
        character it does not name: a real line, about someone else. A pronoun-only quote
        ("she did not rise again") is still kept when its sentence names her.
        """
        item = self.items[subject]
        names = [folded(n) for n in (item.entry.name, *item.entry.aliases) if n]
        spans = [quote, *(s for s in _SENTENCE.split(self.text) if quote_found(quote, s))]
        return any(name in folded(span) for span in spans for name in names)

    def new_entry(self, raw: Mapping[str, Any]) -> None:
        kind = _text(raw, "kind").lower()
        name = _text(raw, "name")
        if kind not in _NEW_KINDS or not name:
            self.result.drop("not_readable")
            return
        quote = self.quoted(raw)
        if quote is None:
            return
        if folded(name) not in folded(self.text):
            # The quote is the scene's, but the name the model gave is not in it.
            self.result.drop("quote_not_in_scene")
            return
        entry_id = id_for_name(name)
        if entry_id is None:
            self.result.drop("not_readable")
            return
        entry: dict[str, Any] = {"id": entry_id, "name": name}
        profile = _text(raw, "profile")
        if profile:
            entry["profile"] = profile
        gender = gender_of(raw.get("gender"))
        family: list[tuple[str, str, str | None]] = []
        if kind == "characters":
            entry["state"] = {"status": "alive"}
            family = self.new_family(raw, entry_id, quote)
            if family:
                entry["relations"] = {other: role for other, role, _ in family}
            if gender is None:
                gender = next((g for _, _, g in family if g), None)
            if gender is not None:
                entry["visual"] = {"gender": gender}
        draft = self.proposal(
            kind=kind, entry_id=entry_id, change={"new_entry": entry}, quote=quote, digest=None
        )
        if draft is None:
            return
        clash = new_entry_clash(
            self.work, draft.kind, draft.new_entry(), pending=[*self.pending, *self.drafts]
        )
        if clash is not None:
            self.result.drop("already_known")
            return
        self.drafts.append(draft)
        for other, role, _ in family:
            loaded = self.work.entry("characters", other)
            if loaded is None:
                self.result.drop("not_an_entry")
                continue
            inverse = INVERSE_ROLE[role]
            back = self.proposal(
                kind="characters",
                entry_id=other,
                change={"progression": {"at": self.scene_id, "relations": {entry_id: inverse}}},
                quote=quote,
                digest=loaded[1],
            )
            if back is not None:
                self.pending_sets.add((other, _relation_key(entry_id), repr(inverse)))
                self.drafts.append(back)

    def new_family(
        self, raw: Mapping[str, Any], entry_id: str, quote: str
    ) -> list[tuple[str, str, str | None]]:
        """A new character's family among the codex's characters, as (other id, role the
        new character has to them, the gender the word gives); each relation not kept is
        counted by reason."""
        given = raw.get(FAMILY_KEY)
        if not isinstance(given, dict):
            return []
        family: list[tuple[str, str, str | None]] = []
        for key, value in cast("dict[object, object]", given).items():
            other = self.names.resolve(str(key))
            if other is None or other == entry_id or self.kinds.get(other) != "characters":
                self.result.drop("not_an_entry")
                continue
            word = family_role(value)
            if word is None or not names_family(quote):
                self.result.drop("not_readable")
                continue
            if any(o == other for o, _, _ in family):
                continue
            family.append((other, word[0], word[1]))
        return family

    def change(self, raw: Mapping[str, Any]) -> None:
        time = _text(raw, "time").lower() or "now"
        polarity = _text(raw, "polarity").lower() or "asserted"
        if (time in TIMES and time != "now") or (polarity in POLARITIES and polarity != "asserted"):
            self.result.drop("not_in_the_present")
            return
        subject = self.names.resolve(_text(raw, "subject"))
        if subject is None or self.kinds.get(subject) != "characters":
            self.result.drop("not_an_entry")
            return
        key = _text(raw, "key").lower().replace(" ", "_")
        value = _text(raw, "value")
        quote = self.quoted(raw)
        if quote is None:
            return
        if not self.names_subject(subject, quote):
            self.result.drop("subject_not_named")
            return
        loaded = self.work.entry("characters", subject)
        if loaded is None:
            self.result.drop("not_an_entry")
            return
        entry, digest = loaded
        before = entry_snapshot(entry, self.placements, self.scene_id, through_scene=False)
        after = entry_snapshot(entry, self.placements, self.scene_id, through_scene=True)
        if key == FAMILY_KEY:
            related = self.relation(subject, _text(raw, "of"), value, quote, after.relations)
            if related is None:
                return
            other, role = related
            pending_key, change = _relation_key(other), {"relations": {other: role}}
            new_value: Any = role
        else:
            wanted = self.wanted(key, value, before.state, after.state)
            if wanted is None:
                return
            state_key, new_value = wanted
            pending_key, change = state_key, {"set": {state_key: new_value}}
        if (subject, pending_key, repr(new_value)) in self.pending_sets:
            self.result.drop("already_pending")
            return
        draft = self.proposal(
            kind="characters",
            entry_id=subject,
            change={"progression": {"at": self.scene_id, **change}},
            quote=quote,
            digest=digest,
        )
        if draft is not None:
            self.pending_sets.add((subject, pending_key, repr(new_value)))
            self.drafts.append(draft)

    def relation(
        self, subject: str, of: str, value: str, quote: str, have: Mapping[str, str]
    ) -> tuple[str, str] | None:
        """The other character and the role to propose for a family change, or `None` with
        the reason recorded."""
        other = self.names.resolve(of)
        found = family_role(value)
        if other is None or other == subject or self.kinds.get(other) != "characters":
            self.result.drop("not_an_entry")
            return None
        if found is None or not names_family(quote):
            self.result.drop("not_readable")
            return None
        role = found[0]
        if have.get(other) == role:
            self.result.drop("already_known")
            return None
        if other in have:
            self.result.drop("contradicts_codex")
            return None
        return other, role

    def wanted(
        self,
        key: str,
        value: str,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> tuple[str, Any] | None:
        """The state key and value to propose, or `None` with the reason recorded."""
        if key == "status":
            status = folded(value)
            if status not in _STATUSES:
                self.result.drop("not_readable")
                return None
            if after.get("status") == status:
                self.result.drop("already_known")
                return None
            if status == "alive":
                # Alive where the codex has them dead: continuity reports it; alive is no news.
                dead = before.get("status") == "dead"
                self.result.drop("contradicts_codex" if dead else "already_known")
                return None
            return "status", "dead"
        if key in ("gains", "loses"):
            item = self.names.resolve(value)
            if item is None or self.kinds.get(item) != "items":
                self.result.drop("not_an_entry")
                return None
            held = [str(v) for v in cast("list[object]", after.get("possesses") or [])]
            had = [str(v) for v in cast("list[object]", before.get("possesses") or [])]
            if key == "gains":
                if item in held:
                    self.result.drop("already_known")
                    return None
                return "possesses", [*held, item]
            if item in held:
                return "possesses", [v for v in held if v != item]
            # Lost here already in the codex, or never held: nothing to add.
            self.result.drop("already_known" if item in had else "contradicts_codex")
            return None
        if key == "condition":
            if not value:
                self.result.drop("not_readable")
                return None
            if folded(str(after.get("condition") or "")) == folded(value):
                self.result.drop("already_known")
                return None
            return "condition", value
        self.result.drop("not_readable")
        return None

    def proposal(
        self,
        *,
        kind: str,
        entry_id: str,
        change: dict[str, Any],
        quote: str,
        digest: str | None,
    ) -> Proposal | None:
        data: dict[str, Any] = {
            "id": "p000",
            "kind": kind,
            "entry_id": entry_id,
            "change": change,
            "evidence": [{"scene_id": self.scene_id, "quote": quote}],
            "proposed_at": _now(),
            "room_id": self.room_id,
            "agent_id": self.agent_id,
            "entry_digest": digest,
        }
        try:
            return Proposal.model_validate(data)
        except (ValidationError, ValueError):
            self.result.drop("not_readable")
            return None


async def enrich_scene(
    work: StoryWork,
    *,
    model: SceneModel,
    scene_id: str,
    text: str,
    room_id: str,
    agent_id: str | None = None,
) -> Enrichment:
    """Read saved scene `scene_id` for what it adds, and save each addition as a proposal.

    Needs the story's lease, as every proposal does (`StoryWork.add_proposal`). Raises
    what reading the story or the model raises; the caller decides what a failure means
    for the write it follows.
    """
    outline, _ = work.require_outline()
    result = Enrichment(scene_id=scene_id)
    reply = await model.complete(
        enrich_prompt(outline, work.codex(), scene_id, text),
        system="You extract facts from fiction. Reply with JSON only.",
        temperature=0.0,
        max_tokens=1500,
    )
    try:
        data = reply_object(reply)
    except ValueError:
        result.unread = True
        return result
    if not isinstance(data, dict):
        result.unread = True
        return result
    found = cast("dict[str, Any]", data)
    reader = _Reader(
        work,
        outline=outline,
        scene_id=scene_id,
        text=text,
        room_id=room_id,
        agent_id=agent_id,
        result=result,
    )
    for raw in _listed(found, "new"):
        reader.new_entry(raw)
    for raw in _listed(found, "changes"):
        reader.change(raw)
    for draft in reader.drafts[:MAX_ADDITIONS]:
        result.proposed.append(work.add_proposal(draft, room_id=room_id))
    for _ in reader.drafts[MAX_ADDITIONS:]:
        result.drop("over_the_limit")
    return result


async def grow_scene(
    work: StoryWork,
    *,
    model: SceneModel,
    scene_id: str,
    text: str,
    room_id: str,
    agent_id: str | None = None,
) -> Enrichment | None:
    """`enrich_scene` for a scene just saved, or `None` when it could not be read.

    The one path every writer of a scene takes (`story_manuscript write`, `story_start`),
    so a failure means the same everywhere: the scene stays saved, nothing is proposed,
    and the failure is logged. The caller says so in its own words, if at all.
    """
    try:
        return await enrich_scene(
            work, model=model, scene_id=scene_id, text=text, room_id=room_id, agent_id=agent_id
        )
    except Exception:
        logger.warning("Codex growth failed for scene %r", scene_id, exc_info=True)
        return None


def growth_line(count: int, *, scenes: bool = False) -> dict[str, str] | None:
    """The line the person reads about what was added, in English and Korean; `None` for 0.

    Plain words, no ids or file names: how many additions wait, and where to decide them.
    `scenes` says it of the scenes a call wrote (`story_start`) rather than of one scene.
    """
    if count <= 0:
        return None
    plural = "addition" if count == 1 else "additions"
    where_en = "The scenes just written brought" if scenes else "This scene brought"
    where_ko = "이번에 쓴 장면에서" if scenes else "이 장면에서"
    return {
        "en": f"{where_en} {count} {plural} to the story's settings (new people, "
        "places, things, relationships or changes). They wait for your approval in the "
        "story's view, under Files.",
        "ko": f"{where_ko} 새로 생긴 인물, 장소, 물건, 관계, 상태 변화 {count}건을 설정집 "
        "보강안으로 올렸습니다. Files의 이야기 보기에서 승인하셔야 설정집에 들어갑니다.",
    }


def growth_note(result: Enrichment) -> dict[str, str] | None:
    """The line the person reads about what the scene added (`growth_line`); `None` when
    nothing was proposed. The agent appends it to the reply by code (`REPLY_NOTE_KEY`).
    """
    return growth_line(len(result.proposed))
