"""One reading of each written scene: its facts, and what it adds to the codex as proposals.

§1.1 rows 6 and 7 of the novel-writer design, and uGraph §3.5 for the story's side. Every
scene the Writer's tools save is read once (one model call, temperature 0; `enrich_prompt`;
one more only when the reading gives a new name, `_confirm_aliases`) for three things in one
reply:

- ``facts``: what the scene states about the characters the story has -- alive or dead
  (``status``), an item held (``possesses``) or given up (``loses``), ``gender``, a family
  relation (``family`` with the other in ``of``), a lasting ``condition`` -- each with the
  quote it rests on, a time and a polarity;
- ``present``: for each character the scene names, whether it has them there and acting
  at its own time, or only mentions, remembers or reports them (#2311);
- ``new``: each character, place or item the scene brings in that the story has not.

What every extraction does with a reply is `uclone_x.knowledge.extraction`'s: the reply is
read past a small model's mistakes, each quote is held to the scene (`quote_found`), each
name resolved to one entity of the story (`EntityResolver`; ambiguous names are dropped),
and a fact the reply repeats is kept once. The story's entities are its codex and the
entries pending proposals add, so a cast still waiting for a person is read like any other.

The same facts then feed both uses (`Enrichment.facts`):

- **proposals** for what is new or changed, written here: a new entry as `story_codex
  create` makes one; a name the scene calls an entry by that the codex does not have
  (`ReadFact.alias`: the model answered with the entry's id and the scene's own words, so
  a cast named in Hangul and prose that writes the names in Latin letters meet without a
  rule in code that knows either script), once a second question confirms it is a name
  and not a pronoun (`_confirm_aliases`), as a name to add to that entry, and from then on
  the scene's quotes name it by that name too; a change at this scene as `story_codex
  propose` makes one -- a death,
  an item gained or lost, a condition, a family relation (kept only when the quote names a
  kinship word, `continuity.names_family`; a new character's family is written on both
  sides, as the cast's is). A change to an entry a pending proposal adds is proposed too,
  quoting the digest that entry's file will have once it is added
  (`proposals.digest_once_added`), so it applies once the entry is approved;
- **the continuity check** (`uclone_x_story.continuity.findings_from`), a query of the
  story's store, which `story_start` runs on the facts without asking the model again.

The model proposes; code decides what is kept (§5.3). Dropped, and counted by reason:

- what `verify_facts` drops: a quote not in the scene (``quote_not_in_text``), what a
  made-up fact looks like; a name no entity has, or two share; a repeated fact;
- a character alive, or holding an item, in a scene that does not have them there and acting
  at its own time (``not_acting_here``): the reading's ``present`` says, per character,
  whether the scene has them acting or only mentions, remembers or reports them (#2311). A
  dead man's legacy, a memory of him alive and what someone says of him are no sign he
  walks again. A death is kept however the scene tells it: a death reported is still one;
- a fact the scene only remembers, denies or imagines (``not_in_the_present``), and a quote
  that names someone else (``subject_not_named``); a name for an entry a person already
  rejected as one (``alias_rejected``), or that the alias question did not confirm
  (``alias_not_confirmed``);
- what the story already says at that scene (``already_known``), or a pending proposal
  already proposes (``already_pending``) -- read from the store's approved-and-proposed
  view, so a proposal is never made twice;
- what contradicts it -- a dead character alive again, a kinship the codex gives otherwise,
  an item someone else holds with no word of a handover, one lost that was not held
  (``contradicts_codex``). That is the continuity check's to report, never an update;
- a gender, which is checked and never proposed (``checked_only``), and a status other than
  the last the scene gives a character (``not_the_last``).

Nothing is applied: each addition is a pending proposal in the story's own store
(`StoryWork.add_proposal`), decided by a person in the story's view (§5.3, §10 Q2).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, get_args

from pydantic import ValidationError

from uclone_x.knowledge.extraction import (
    ACTING,
    AliasCandidate,
    ReadFact,
    alias_prompt,
    confirmed_aliases,
    parse_facts,
    read_presence,
    reply_object,
    salvage_extraction,
    verify_facts,
)
from uclone_x.knowledge.models import Position
from uclone_x.knowledge.quotes import folded, quote_found
from uclone_x_story.codex_store import CodexYamlStore
from uclone_x_story.context import CodexIndex
from uclone_x_story.continuity import (
    FAMILY_KEY,
    INVERSE_ROLE,
    family_role,
    gender_of,
    names_family,
)
from uclone_x_story.names import MIN_NAME_CHARACTERS, CodexNames, id_for_name
from uclone_x_story.proposals import digest_once_added, new_entry_clash
from uclone_x_story.schemas import CodexKind, Outline, Proposal

if TYPE_CHECKING:
    from uclone_x_story.work import StoryWork

__all__ = [
    "ENRICH_MARKER",
    "PRESENCE_MARKER",
    "MAX_ADDITIONS",
    "Enrichment",
    "SceneModel",
    "enrich_prompt",
    "enrich_scene",
    "presence_prompt",
    "grow_scene",
    "growth_line",
    "growth_note",
]

logger = logging.getLogger(__name__)

#: The first line of every scene reading, so a scripted model can tell it apart.
ENRICH_MARKER = "[story codex growth]"
#: The most additions one scene proposes: a scene's news, not a whole world.
MAX_ADDITIONS = 8
#: The reading's token cap: the facts of a scene and its additions in one reply.
MAX_READING_TOKENS = 2000
#: The alias question's token cap: one answer per candidate name.
MAX_ALIAS_TOKENS = 300
#: The first line of every presence question (#2317), so a scripted model can tell it apart.
PRESENCE_MARKER = "[presence check]"
#: The presence question's token cap: one answer and its words per fact asked about.
MAX_PRESENCE_TOKENS = 800
#: The presence question's answer for a holder who has the item with them now.
_HOLDS = "holds"
_WANTS = "wants"
#: The answers to the presence question that say a person is not there and acting (#2342):
#: only these drop a fact. Any other word was not offered, and leaves ``present`` to decide.
_NOT_THERE = frozenset({"apparition", "mentioned", "remembered", "reported"})
_NOT_THERE_HOLDING = _NOT_THERE | {_WANTS}

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
    """One scene's reading: the facts it gave, the proposals saved, and what was dropped."""

    scene_id: str
    proposed: list[Proposal] = field(default_factory=list[Proposal])
    #: Reason -> how many of the model's facts and additions were dropped for it.
    dropped: dict[str, int] = field(default_factory=dict[str, int])
    #: The reply could not be read as JSON: nothing was proposed, and it is not "nothing new".
    unread: bool = False
    #: The scene's facts, each held to the scene and resolved to an entity: what the
    #: continuity check reads (`continuity.findings_from`).
    facts: list[ReadFact] = field(default_factory=list[ReadFact])

    def drop(self, reason: str, count: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + count

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
    """What the reading model is asked: the story's entries, the keys, and the scene."""
    found = outline.find(scene_id)
    if found is None:
        raise ValueError(f"the outline has no scene '{scene_id}'")
    _, scene = found
    return f"""{ENRICH_MARKER}
You read one scene of a story and list the facts it states about the story's characters,
and the people, places and things it brings in that the story does not have yet. You do not
judge the story; you only report what the text says.

Scene {scene.id}: {scene.title}. {scene.summary}

Characters (use the id as subject, also when the scene writes the name differently from
the names listed: in another script, spelling or form):
{_entries(codex, "characters")}
Places:
{_entries(codex, "places")}
Items (use the id as value for possesses and loses):
{_entries(codex, "items")}

"facts": what the scene states about a character above. Keys (use no other key):
- status: alive or dead. A character who acts, speaks or moves in the scene's present is
  alive. A ghost, spirit or corpse of someone is dead. One who dies in the scene is dead.
- possesses: the id of an item above that the character holds, carries, wears or wields.
- loses: the id of an item above that the character gives away, drops or loses here.
- gender: female or male, only where the scene shows it: a word such as 그녀, 딸, 아들,
  어머니, 아버지, 누나 or 오빠 used for the character.
- family: what the subject is to the character whose id is in "of" (another character
  above): father, mother, son, daughter, brother, sister, husband or wife, grandfather,
  grandmother, grandson or granddaughter. "하린의 아버지 도윤" is subject doyun, key
  family, of harin, value father.
- condition: a lasting wound or state, in a few words ("왼팔 골절", "blinded").

For every fact give a time and a polarity:
- time "now": true at this scene's own time. time "past": a remembered, recalled or dreamed
  act, a voice or words remembered, and anything a character says happened (dialogue, words
  in quotation marks).
- polarity "asserted": the text says it is so. "negated": the text says it is NOT so.
  "absent": the person or thing is gone or missing. "hypothetical": only imagined, wished,
  feared or likened ("as if").

"present": each character above that the scene names, with how the scene has them at its
own time:
- "acting": there at the scene's own time, doing, saying or undergoing something (dying
  included).
- "mentioned": only named or spoken of: their name, their things, their legacy, what they
  were or did.
- "remembered": only in a memory, flashback, dream or vision.
- "reported": only in what someone tells, says or writes about them.

"new": each named character, place or item the scene brings in that is NOT above. kind is
characters, places or items. profile is one sentence from the scene. gender (female or male)
only for a character the scene shows as one. family, only for a new character the scene
states is family of a character above: {{"<id of that character>": what the new character
is to them, in the words listed under family above}}.

Rules:
1. Only the characters, items and keys above in "facts". A fact the scene does not state
   is left out.
2. quote is copied exactly from the scene, at least a few words, and shows the fact.
3. Report each character's status once, from the sentence that shows it last.
4. named is the name the quote calls the subject by, copied exactly from the quote, even
   when it is written differently from the names listed above. Leave it empty when the
   quote names them by a pronoun only.
5. object_named is the words of the quote that name the item or person the fact is about
   (the item held or given up, the family member), copied exactly from the quote. Leave
   it empty when the fact is about no other item or person, or the quote names it by a
   pronoun only.

Reply with JSON only:
{{"present": {{"<id>": "<acting, mentioned, remembered or reported>"}},
 "facts": [{{"subject": "<id>", "named": "<name in the quote>", "key": "<key>", "value": "<value>", "object_named": "<words in the quote>", "quote": "<exact words>", "time": "now", "polarity": "asserted"}}],
 "new": [{{"kind": "characters", "name": "<name>", "profile": "<sentence>", "gender": "female", "family": {{}}, "quote": "<exact words>"}}]}}
A family fact also has "of": "<id>".
Reply {{"present": {{}}, "facts": [], "new": []}} when the scene states none.

<scene>
{text}
</scene>"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _listed(data: Mapping[str, Any], key: str) -> list[object]:
    raw = data.get(key)
    return cast("list[object]", raw) if isinstance(raw, list) else []


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


def _key(raw: str) -> str:
    return folded(raw).replace(" ", "_")


def _last_statuses(facts: Sequence[ReadFact]) -> set[int]:
    """The index of each character's status fact the scene gives last; the others are not
    news (one who dies in a scene was alive earlier in it)."""
    last: dict[str, tuple[int, int]] = {}
    for index, read in enumerate(facts):
        if _key(read.fact.key) != "status" or not read.fact.checked:
            continue
        if read.subject not in last or read.at >= last[read.subject][0]:
            last[read.subject] = (read.at, index)
    return {index for _, index in last.values()}


class _Reader:
    """One scene's facts and additions, held to the story as its store has it at that scene."""

    def __init__(
        self,
        work: StoryWork,
        *,
        store: CodexYamlStore,
        scene_id: str,
        text: str,
        room_id: str,
        agent_id: str | None,
        result: Enrichment,
    ) -> None:
        self.work = work
        self.store = store
        self.scene_id = scene_id
        self.text = text
        self.room_id = room_id
        self.agent_id = agent_id
        self.result = result
        self.codex = store.codex()
        self.names = CodexNames(self.codex)
        self.items = {item.entry.id: item for item in self.codex.items}
        self.starts: Position = store.point(scene_id, ended=False)
        self.ends: Position = store.point(scene_id, ended=True)
        decided = [p for p, _ in work.proposals()[0]]
        self.pending = [p for p in decided if p.status == "pending"]
        #: Names a person rejected as an entry's, as (entry, folded name).
        self.rejected = {
            (p.entry_id, folded(name))
            for p in decided
            if p.status == "rejected" and p.change.aliases is not None
            for name in p.change.aliases
        }
        #: What this reading proposes, by (entry, key), seen by its later facts.
        self.staged: dict[tuple[str, str], Any] = {}
        #: The names this reading proposes for each entry, seen by its later facts.
        self.called: dict[str, list[str]] = {}
        #: Items a character gives up in this scene: a handover, not a second holder.
        self.losing: set[tuple[str, str]] = set()
        self.drafts: list[Proposal] = []
        #: The names proposed for entries the story has: one per entry and name at most,
        #: so not held to `MAX_ADDITIONS` (a scene's news, which they make readable).
        self.aliases: list[Proposal] = []

    # -- the store at this scene -------------------------------------------------------

    def values(self, subject: str, key: str, *, ended: bool = True) -> tuple[list[str], bool]:
        """`subject`'s values of `key` at the scene's start or end, and whether any is still
        a proposal; what this reading proposed for them stands in once the scene ends."""
        if ended and (subject, key) in self.staged:
            staged: object = self.staged[(subject, key)]
            values = cast("list[object]", staged) if isinstance(staged, list) else [staged]
            return [str(v) for v in values], True
        rows = self.store.values_at(subject, key, self.ends if ended else self.starts)
        return [str(r.edge.value) for r in rows], any(r.edge.status == "proposed" for r in rows)

    def holders(self, item: str) -> set[str]:
        """Who holds `item` as the scene starts."""
        return {
            r.edge.subject_id
            for r in self.store.rows_at(self.starts, proposed=True)
            if r.edge.predicate == "possesses" and r.edge.value == item
        }

    def contradicts(self, read: ReadFact) -> bool:
        """Whether `read` would contradict the store as the scene starts: alive where the
        character is dead, or holding an item another holds who does not give it up here."""
        key = _key(read.fact.key)
        if key == "status":
            before, _ = self.values(read.subject, "status", ended=False)
            return "dead" in before
        found = self.store.resolve(read.fact.value) if read.fact.value.strip() else None
        item = found.entity_id if found is not None else None
        if item is None or self.store.kind_of(item) != "items":
            return False
        return any(h != read.subject and (h, item) not in self.losing for h in self.holders(item))

    def digest(self, subject: str, kind: str = "characters") -> str | None:
        """The digest a change to `subject` quotes: its file's, or, for an entry a pending
        proposal adds, the one that file will have once added (as proposed: the names
        other pending proposals add to it are not in that file)."""
        if kind not in get_args(CodexKind):
            return None
        loaded = self.work.entry(cast("CodexKind", kind), subject)
        if loaded is not None:
            return loaded[1]
        adding = next(
            (
                p
                for p in self.pending
                if p.change.new_entry is not None and p.entry_id == subject and p.kind == kind
            ),
            None,
        )
        return digest_once_added(adding.new_entry()) if adding is not None else None

    # -- what is kept -------------------------------------------------------------------

    def names_subject(self, subject: str, quote: str) -> bool:
        """Whether the quote, or a sentence of the scene that holds it, names the subject.

        qwen3:8b gave "비명이 하나 들렸다" (a scream was heard) as the quote for a death of a
        character it does not name: a real line, about someone else. A pronoun-only quote
        ("she did not rise again") is still kept when its sentence names her. A name of
        several words is named by its last word too (`장로` for `마을 장로`, `Vane` for
        `Lord Vane`), as prose shortens it after the first mention. A name this reading
        proposes for the subject (`alias`) names it too: the prose may write the cast's
        names in another script (#2217).
        """
        item = self.items[subject]
        called = [
            n for n in (item.entry.name, *item.entry.aliases, *self.called.get(subject, ())) if n
        ]
        last = [n.split()[-1] for n in called if len(n.split()) > 1]
        names = [folded(n) for n in (*called, *last) if len(folded(n)) >= MIN_NAME_CHARACTERS]
        spans = [quote, *(s for s in _SENTENCE.split(self.text) if quote_found(quote, s))]
        return any(name in folded(span) for span in spans for name in names)

    def alias(self, read: ReadFact) -> None:
        """Propose `read.alias` as a name of its subject, once per reading.

        The reading model was shown the entry's id and names and answered with the id and
        this name from the quote (`verify_facts`); a person decides whether it is one.
        """
        name = read.alias
        subject = read.subject
        if name is None:
            return
        if (subject, folded(name)) in self.rejected:
            self.result.drop("alias_rejected")
            return
        if any(folded(n) == folded(name) for n in self.called.get(subject, ())):
            return
        kind = self.store.kind_of(subject) or ""
        digest = self.digest(subject, kind)
        if digest is None:
            self.result.drop("not_an_entry")
            return
        draft = self.proposal(
            kind=kind,
            entry_id=subject,
            change={"aliases": [name]},
            quote=read.fact.quote,
            digest=digest,
        )
        if draft is not None:
            self.called.setdefault(subject, []).append(name)
            self.aliases.append(draft)

    def new_entry(self, raw: Mapping[str, Any]) -> None:
        kind = _text(raw, "kind").lower()
        name = _text(raw, "name")
        if kind not in _NEW_KINDS or not name:
            self.result.drop("not_readable")
            return
        if any(folded(name) == folded(n) for names in self.called.values() for n in names):
            # The scene's name for an entry the story has: proposed as that entry's name.
            self.result.drop("already_known")
            return
        quote = _text(raw, "quote")
        if not quote_found(quote, self.text) or folded(name) not in folded(self.text):
            # Not in the scene: the quote, or the name the model gave.
            self.result.drop("quote_not_in_text")
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
            digest = self.digest(other)
            if digest is None:
                self.result.drop("not_an_entry")
                continue
            inverse = INVERSE_ROLE[role]
            back = self.proposal(
                kind="characters",
                entry_id=other,
                change={"progression": {"at": self.scene_id, "relations": {entry_id: inverse}}},
                quote=quote,
                digest=digest,
            )
            if back is not None:
                self.staged[(other, f"relations.{entry_id}")] = inverse
                self.drafts.append(back)

    def new_family(
        self, raw: Mapping[str, Any], entry_id: str, quote: str
    ) -> list[tuple[str, str, str | None]]:
        """A new character's family among the story's characters, as (other id, role the
        new character has to them, the gender the word gives); each relation not kept is
        counted by reason."""
        given = raw.get(FAMILY_KEY)
        if not isinstance(given, dict):
            return []
        family: list[tuple[str, str, str | None]] = []
        for key, value in cast("dict[object, object]", given).items():
            other = self.names.resolve(str(key))
            if other is None or other == entry_id or self.store.kind_of(other) != "characters":
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

    def change(self, read: ReadFact, *, last: bool) -> None:
        fact = read.fact
        if not fact.checked:
            self.result.drop("not_in_the_present")
            return
        subject = read.subject
        if self.store.kind_of(subject) != "characters":
            self.result.drop("not_an_entry")
            return
        if not self.names_subject(subject, fact.quote):
            self.result.drop("subject_not_named")
            return
        key = _key(fact.key)
        if key == "status" and not last:
            self.result.drop("not_the_last")
            return
        if key == "gender":
            self.result.drop("checked_only" if gender_of(fact.value) else "not_readable")
            return
        if key == FAMILY_KEY:
            related = self.relation(subject, read.of, fact.value, fact.quote)
            if related is None:
                return
            other, role = related
            staged_key, change, new_value = f"relations.{other}", {"relations": {other: role}}, role
        else:
            wanted = self.wanted(subject, key, fact.value)
            if wanted is None:
                return
            state_key, new_value = wanted
            staged_key, change = state_key, {"set": {state_key: new_value}}
        digest = self.digest(subject)
        if digest is None:
            self.result.drop("not_an_entry")
            return
        draft = self.proposal(
            kind="characters",
            entry_id=subject,
            change={"progression": {"at": self.scene_id, **change}},
            quote=fact.quote,
            digest=digest,
        )
        if draft is not None:
            self.staged[(subject, staged_key)] = new_value
            self.drafts.append(draft)

    def known(self, pending: bool) -> None:
        self.result.drop("already_pending" if pending else "already_known")

    def relation(
        self, subject: str, other: str | None, value: str, quote: str
    ) -> tuple[str, str] | None:
        """The other character and the role to propose for a family change, or `None` with
        the reason recorded."""
        found = family_role(value)
        if other is None or other == subject or self.store.kind_of(other) != "characters":
            self.result.drop("not_an_entry")
            return None
        if found is None or not names_family(quote):
            self.result.drop("not_readable")
            return None
        role = found[0]
        if (subject, f"relations.{other}") in self.staged:
            have: str | None = str(self.staged[(subject, f"relations.{other}")])
            pending = True
        else:
            row = self.store.relations_at(subject, self.ends).get(other)
            have = row.edge.predicate if row is not None else None
            pending = row is not None and row.edge.status == "proposed"
        if have == role:
            self.known(pending)
            return None
        if have is not None:
            self.result.drop("contradicts_codex")
            return None
        return other, role

    def wanted(self, subject: str, key: str, value: str) -> tuple[str, Any] | None:
        """The state key and value to propose, or `None` with the reason recorded."""
        if key == "status":
            status = folded(value)
            if status not in _STATUSES:
                self.result.drop("not_readable")
                return None
            after, pending = self.values(subject, "status")
            if status in after:
                self.known(pending)
                return None
            if status == "alive":
                # Alive where the story has them dead: continuity reports it; alive is no news.
                before, _ = self.values(subject, "status", ended=False)
                self.result.drop("contradicts_codex" if "dead" in before else "already_known")
                return None
            return "status", "dead"
        if key in ("possesses", "loses"):
            found = self.store.resolve(value) if value.strip() else None
            item = found.entity_id if found is not None else None
            if item is None or self.store.kind_of(item) != "items":
                self.result.drop("not_an_entry")
                return None
            held, pending = self.values(subject, "possesses")
            if key == "possesses":
                if item in held:
                    self.known(pending)
                    return None
                others = {
                    h for h in self.holders(item) if h != subject and (h, item) not in self.losing
                }
                if others:
                    # Someone else holds it and the scene says no word of a handover:
                    # continuity reports it; nothing to propose.
                    self.result.drop("contradicts_codex")
                    return None
                return "possesses", [*held, item]
            if item in held:
                return "possesses", [v for v in held if v != item]
            # Lost here already, or never held: nothing to add.
            had, _ = self.values(subject, "possesses", ended=False)
            self.result.drop("already_known" if item in had else "contradicts_codex")
            return None
        if key == "condition":
            if not value:
                self.result.drop("not_readable")
                return None
            now, pending = self.values(subject, "condition")
            if any(folded(v) == folded(value) for v in now):
                self.known(pending)
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


def _reply(reply: str) -> tuple[list[object], list[object], object] | None:
    """The reply's ``facts`` and ``new`` lists and its ``present`` map as given, or `None`
    when it cannot be read.

    A reply cut off by its token cap keeps the facts it finished (`salvage_extraction`),
    and has no ``present``.
    """
    try:
        data = reply_object(reply)
    except ValueError:
        data = None
    if isinstance(data, dict):
        found = cast("dict[str, Any]", data)
        return _listed(found, "facts"), _listed(found, "new"), found.get("present")
    try:
        salvaged = salvage_extraction(reply)
    except ValueError:
        return None
    return [vars(f) for f in salvaged], [], None


def _needs_acting(read: ReadFact) -> bool:
    """Whether a fact says the character is there now: alive, or holding an item."""
    key = _key(read.fact.key)
    return key == "possesses" or (key == "status" and folded(read.fact.value) == "alive")


def _by_present(read: ReadFact, present: dict[str, str] | None) -> bool:
    """Whether the reading's ``present`` lets `read` be checked: a character alive or
    holding something only in a scene that has them there and acting (#2311).

    A reading that gave no ``present`` leaves the fact as it is: an unanswered question is
    no answer that the character is absent. One that gave it and left a character out does
    not have them acting.
    """
    if present is None:
        return True  # unanswered: no answer that they are absent
    if not _needs_acting(read):
        return True
    return present.get(read.subject) == ACTING


def presence_prompt(asked: Sequence[tuple[str, str]], quotes: Sequence[str], text: str) -> str:
    """The question asked once more about each fact that would contradict the story
    (#2317): who, what the reading said of them, and the words it read it from."""
    lines = "\n".join(
        f'{number}. {who} -- read as: {what} -- from: "{quote}"'
        for number, ((who, what), quote) in enumerate(zip(asked, quotes, strict=True), start=1)
    )
    return f"""{PRESENCE_MARKER}
<scene>
{text}
</scene>

Each numbered line below is a claim from a reading of the scene above. Check the scene and
answer for each number with one word.

For a line with "read as: alive", how does the scene have that person at the scene's own
time?
- "acting": alive and physically there in the scene now, doing or saying something.
- "apparition": only as a ghost, spirit, corpse or vision of someone dead.
- "mentioned": only named or spoken of (their legacy, their past, their name).
- "remembered": only in a memory, flashback, dream or thought.
- "reported": only in what someone tells or says about the past.

For a line with "read as: holds <item>", does that person have that item in their keeping
at the end of the scene?
- "holds": physically has, carries, wears, receives or keeps the item in the scene now.
- "wants": only wants, seeks, covets, asks for or talks about the item without having it.
- "mentioned", "remembered" or "reported": the person or item is only spoken of, remembered,
  or handed away so they no longer have it.

quote is the exact words from <scene> that include that person's name and show your answer
(keep the line's "from:" quote when it shows your answer).

{lines}

Reply with JSON only, one answer per number: {{"1": {{"answer": "<holds, wants, acting, apparition, mentioned, remembered or reported>", "quote": "<exact words from scene with the person's name>"}}}}"""


async def _ask_presence(
    model: SceneModel, reader: _Reader, contested: Sequence[ReadFact]
) -> list[bool | None] | None:
    """For each fact in `contested`, whether one question about that person alone says the
    scene has them there and acting (holding the item, for a holding), in words that are in
    the scene and name them; `None` for a fact the answer leaves out or answers with a word
    it was not offered (#2342: "alive", or "acting" on a holding line). `None` for all when
    the question fails or its answer cannot be read. Where there is no answer, ``present``
    decides, as without the question."""
    asked: list[tuple[str, str]] = []
    for read in contested:
        who = (
            reader.items[read.subject].entry.name if read.subject in reader.items else read.subject
        )
        if _key(read.fact.key) == "possesses":
            item = _held_item(reader.store, read)
            named = reader.items[item].entry.name if item in reader.items else read.fact.value
            asked.append((who, f"holds {named}"))
        else:
            asked.append((who, "alive"))
    try:
        reply = await model.complete(
            presence_prompt(asked, [r.fact.quote for r in contested], reader.text),
            system="You read fiction closely. Reply with JSON only.",
            temperature=0.0,
            max_tokens=MAX_PRESENCE_TOKENS,
        )
        data = reply_object(reply)
    except Exception as failed:
        logger.warning("The presence question failed; the reading's labels decide", exc_info=failed)
        data = None
    if not isinstance(data, dict):
        return None  # unanswered: present decides
    answers = cast("dict[str, Any]", data)
    out: list[bool | None] = []
    for number, read in enumerate(contested, start=1):
        said = answers.get(str(number))
        # A number left out has no answer word, which is one it was not offered.
        given = cast("dict[str, Any]", said) if isinstance(said, dict) else {}
        answer = str(given.get("answer") or "").strip().lower()
        quote = str(given.get("quote") or "")
        expected = _HOLDS if _key(read.fact.key) == "possesses" else ACTING
        not_there = _NOT_THERE_HOLDING if expected == _HOLDS else _NOT_THERE
        if answer == expected:
            shown = quote_found(quote, reader.text)
            out.append(shown and reader.names_subject(read.subject, quote))
        elif answer in not_there:
            out.append(False)
        else:
            out.append(None)  # no word it was offered: present decides
    return out


async def _confirm_aliases(
    model: SceneModel,
    store: CodexYamlStore,
    facts: Sequence[ReadFact],
    rejected: set[tuple[str, str]],
    result: Enrichment,
) -> list[ReadFact]:
    """`facts` with each candidate alias the model does not confirm as a name taken off.

    One question for every new name the reading gave (`alias_prompt`; uGraph §3.4 stage 4):
    the reading model also answers with a pronoun ("그녀", qwen3:8b 28 times in one run),
    which is no name. A name a person rejected is not asked again. A failed or unreadable
    answer confirms nothing (``alias_not_confirmed``), and the reading goes on.
    """
    entities = {e.id: e for e in store.entities()}
    asked: dict[tuple[str, str], AliasCandidate] = {}
    for read in facts:
        if read.alias is None or read.subject not in entities:
            continue
        key = (read.subject, folded(read.alias))
        if key not in rejected and key not in asked:
            asked[key] = AliasCandidate(entities[read.subject], read.alias, read.fact.quote)
    if not asked:
        return list(facts)
    candidates = list(asked.values())
    try:
        reply = await model.complete(
            alias_prompt(candidates),
            system="You tell names from other words. Reply with JSON only.",
            temperature=0.0,
            max_tokens=MAX_ALIAS_TOKENS,
        )
    except Exception as failed:
        logger.warning("The alias question failed; no name was confirmed", exc_info=failed)
        reply = ""
    yes = confirmed_aliases(reply, len(candidates))
    confirmed = {key for index, key in enumerate(asked) if index in yes}
    if len(confirmed) < len(candidates):
        result.drop("alias_not_confirmed", len(candidates) - len(confirmed))
    return [
        read
        if read.alias is None
        or (read.subject, folded(read.alias)) in confirmed
        or (read.subject, folded(read.alias)) in rejected
        else replace(read, alias=None)
        for read in facts
    ]


def _held_item(store: CodexYamlStore, read: ReadFact) -> str | None:
    """The item a ``possesses`` fact says its subject holds, or `None`."""
    if _key(read.fact.key) != "possesses" or not read.fact.value.strip():
        return None
    item = store.resolve(read.fact.value).entity_id
    return item if item is not None and store.kind_of(item) == "items" else None


def _contested(
    reader: _Reader, store: CodexYamlStore, read: ReadFact, claims: Mapping[str, set[str]]
) -> bool:
    """Whether `read` would contradict the story: the store as the scene starts, or another
    character this reading also has holding the same item."""
    return reader.contradicts(read) or any(
        s != read.subject for s in claims.get(_held_item(store, read) or "", ())
    )


def _item_lost(store: CodexYamlStore, read: ReadFact) -> tuple[str, str] | None:
    """`(holder, item)` for a fact that a character gives up an item now, or `None`."""
    if not read.fact.checked or _key(read.fact.key) != "loses" or not read.fact.value.strip():
        return None
    item = store.resolve(read.fact.value).entity_id
    return (read.subject, item) if item is not None else None


async def enrich_scene(
    work: StoryWork,
    *,
    model: SceneModel,
    scene_id: str,
    text: str,
    room_id: str,
    agent_id: str | None = None,
) -> Enrichment:
    """Read saved scene `scene_id` once: its facts, and each addition saved as a proposal.

    Needs the story's lease, as every proposal does (`StoryWork.add_proposal`). Raises
    what reading the story or the model raises; the caller decides what a failure means
    for the write it follows.
    """
    outline, _ = work.require_outline()
    proposals = [p for p, _ in work.proposals()[0]]
    store = CodexYamlStore.read(work.story_id, outline, work.codex(), proposals)
    result = Enrichment(scene_id=scene_id)
    reply = await model.complete(
        enrich_prompt(outline, store.codex(), scene_id, text),
        system="You extract facts from fiction. Reply with JSON only.",
        temperature=0.0,
        max_tokens=MAX_READING_TOKENS,
    )
    lists = _reply(reply)
    if lists is None:
        result.unread = True
        return result
    raw_facts, raw_new, raw_present = lists
    objects: list[object] = [r for r in raw_facts if isinstance(r, dict)]
    if len(objects) < len(raw_facts):
        result.drop("not_readable", len(raw_facts) - len(objects))
    verified = verify_facts(parse_facts(objects), text, store.resolver())
    for reason, count in verified.dropped.items():
        result.drop(reason, count)
    reader = _Reader(
        work,
        store=store,
        scene_id=scene_id,
        text=text,
        room_id=room_id,
        agent_id=agent_id,
        result=result,
    )
    confirmed = await _confirm_aliases(model, store, verified.kept, reader.rejected, result)
    for read in confirmed:
        reader.alias(read)
    reader.losing = {lost for r in confirmed if (lost := _item_lost(store, r)) is not None}
    present = read_presence(raw_present, store.resolver())
    # A fact that would contradict the story is asked about once more, alone (#2317): the
    # reading's labels for everyone at once were wrong both ways at qwen3:8b.
    # An item two characters are read as holding in this very scene is contested too.
    claims: dict[str, set[str]] = {}
    for r in confirmed:
        if _key(r.fact.key) == "possesses" and (item := _held_item(store, r)) is not None:
            claims.setdefault(item, set()).add(r.subject)
    contested = [r for r in confirmed if _needs_acting(r) and _contested(reader, store, r, claims)]
    answers = await _ask_presence(model, reader, contested) if contested else None
    kept: list[ReadFact] = []
    for read in confirmed:
        at = next((i for i, c in enumerate(contested) if c is read), None)
        said = answers[at] if answers is not None and at is not None else None
        ok = said if said is not None else _by_present(read, present)
        if ok:
            kept.append(read)
        else:
            result.drop("not_acting_here")
    result.facts = kept
    for raw in raw_new:
        if isinstance(raw, dict):
            reader.new_entry(cast("dict[str, Any]", raw))
        else:
            result.drop("not_readable")
    last = _last_statuses(kept)
    # What is given up first, so an item a scene hands over is free for the one it goes to.
    order = sorted(range(len(kept)), key=lambda i: _key(kept[i].fact.key) != "loses")
    for index in order:
        reader.change(kept[index], last=index in last)
    for draft in [*reader.aliases, *reader.drafts[:MAX_ADDITIONS]]:
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
        logger.warning("Scene reading failed for scene %r", scene_id, exc_info=True)
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
