"""Continuity after writing: a scene's facts held to the story as it stands, by a query.

The scene is read once, by the reading that also grows the codex (`uclone_x_story.enrich`,
§1.1 rows 6 and 7): one model call gives the scene's facts, each with the quote it read it
from, a time and a polarity, and `uclone_x.knowledge.extraction` holds each quote to the
scene and each name to one entity. Everything here is deterministic: the facts ``asserted``
about the scene's own ``now`` are checked by `uclone_x_story.audit.audit_scene` against the
story's uGraph store (`CodexYamlStore`) in its **approved-and-proposed** view once the
scene has ended. That view is the codex as it would be with every pending proposal made --
a death an earlier scene wrote, an item a scene gave someone, a cast still waiting for a
person -- so what a scene left behind holds a later scene to it before anyone approves it,
with no state of its own beside the store.

What it catches, with the store and the check's own rules (`identity_axioms`):

- a character ``dead`` from scene k who is ``alive`` at a later scene j: the store has the
  death from k's end (`Alive` and `Dead` are disjoint);
- one item with two holders at one point, in one scene or across scenes: a holder who
  gained it earlier and never lost it, and another who holds it now with no handover
  (``possesses`` is inverse-functional). A handover the scene states (the giver ``loses``
  it) ends the giver's edge, so it is no finding;
- one gender per character, and one relation per pair (`relations:`, read as
  `audit.relation_predicate`), so a son called "딸" disagrees with the codex;
- whatever the story's own rules (`story.yaml` `axioms`) say, as the audit always did.

Only these keys of a character are checked: ``status`` (``alive`` or ``dead``),
``possesses`` (an item entry), ``gender`` (``female`` or ``male``), and ``family`` with
``of`` another character (what the subject is to the other: ``parent``, ``child``,
``sibling``, ``spouse``, ``grandparent``, ``grandchild``; `family_role` reads English and
Korean kinship words, and a gendered one such as "딸" also gives a gender). A character's
``status`` is read once per scene, from the last sentence that states it: one who dies in
the scene was alive earlier in it, and that is a change, not a contradiction.

Each contradiction is written for the person in plain Korean, with the quote
(`Finding.note`), and never with an entry id or a reasoner's words; one that rests on what
an earlier scene wrote names that scene.

Nothing here calls a model or writes a file: `story_start` decides what to do with the
findings (it offers a rewrite; it never rewrites by itself).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from uclone_x.knowledge.extraction import ReadFact
from uclone_x.knowledge.quotes import folded
from uclone_x_story.audit import (
    RELATION_PREFIX,
    SubmittedFact,
    audit_scene,
    relation_predicate,
)
from uclone_x_story.codex_store import CodexYamlStore
from uclone_x_story.context import CodexIndex, CodexItem
from uclone_x_story.schemas import Outline, Proposal, StoryAxiom

__all__ = [
    "FAMILY_KEY",
    "INVERSE_ROLE",
    "Finding",
    "checked_store",
    "family_role",
    "findings_from",
    "gender_of",
    "identity_axioms",
    "names_family",
    "with_gender",
]

_STATUSES = ("alive", "dead")

#: The key a reading model gives a family relation under, with the other character in
#: ``of``. It is read into the codex's `relations:`, never into a state key.
FAMILY_KEY = "family"

#: A kinship word, English or Korean, as the role it names and the gender it gives.
#: The role is what the subject is to the other person; the word's gender is the
#: subject's own ("딸" is a female child). A word not here is not checked.
_FAMILY_WORDS: dict[str, tuple[str, str | None]] = {
    **dict.fromkeys(("parent", "부모"), ("parent", None)),
    **dict.fromkeys(("father", "dad", "아버지", "아빠", "부친"), ("parent", "male")),
    **dict.fromkeys(("mother", "mom", "어머니", "엄마", "모친"), ("parent", "female")),
    **dict.fromkeys(("child", "자녀", "자식"), ("child", None)),
    **dict.fromkeys(("son", "아들"), ("child", "male")),
    **dict.fromkeys(("daughter", "딸"), ("child", "female")),
    **dict.fromkeys(("sibling", "형제", "동생", "남매"), ("sibling", None)),
    **dict.fromkeys(("brother", "형", "오빠", "남동생"), ("sibling", "male")),
    **dict.fromkeys(("sister", "누나", "언니", "여동생", "자매"), ("sibling", "female")),
    **dict.fromkeys(("spouse", "배우자"), ("spouse", None)),
    **dict.fromkeys(("husband", "남편"), ("spouse", "male")),
    **dict.fromkeys(("wife", "아내", "부인"), ("spouse", "female")),
    **dict.fromkeys(("grandparent", "조부모"), ("grandparent", None)),
    **dict.fromkeys(("grandfather", "할아버지"), ("grandparent", "male")),
    **dict.fromkeys(("grandmother", "할머니"), ("grandparent", "female")),
    **dict.fromkeys(("grandchild", "손주"), ("grandchild", None)),
    **dict.fromkeys(("grandson", "손자"), ("grandchild", "male")),
    **dict.fromkeys(("granddaughter", "손녀"), ("grandchild", "female")),
}
#: What the other person is to the subject, for each role.
INVERSE_ROLE = {
    "parent": "child",
    "child": "parent",
    "sibling": "sibling",
    "spouse": "spouse",
    "grandparent": "grandchild",
    "grandchild": "grandparent",
}
_GENDER_WORDS = {
    **dict.fromkeys(("female", "woman", "girl", "f", "여성", "여자"), "female"),
    **dict.fromkeys(("male", "man", "boy", "m", "남성", "남자"), "male"),
}
_ROLE_KO = {
    "parent": "부모",
    "child": "자녀",
    "sibling": "형제자매",
    "spouse": "배우자",
    "grandparent": "조부모",
    "grandchild": "손주",
}
_GENDER_KO = {"female": "여성", "male": "남성"}


def family_role(word: object) -> tuple[str, str | None] | None:
    """The role a kinship word names and the gender it gives; `None` for another word."""
    return _FAMILY_WORDS.get(folded(str(word or "")))


def _names_family(quote: str) -> bool:
    """Whether a quote holds a kinship word at all.

    qwen3:8b read "김철수는 그녀에게 세탁소의 운영 방법을 가르쳐주었고" as the two being
    spouses (2026-09-29), and the check told the person the scene contradicted the codex.
    A kinship the scene states names it, so a family fact whose quote names none is dropped.
    """
    said = quote.casefold()
    return any(word in said for word in _FAMILY_WORDS)


def names_family(quote: str) -> bool:
    """Whether a quote holds a kinship word at all; the public name of `_names_family`."""
    return _names_family(quote)


def gender_of(word: object) -> str | None:
    """``female`` or ``male`` for a gender word, `None` for anything else."""
    return _GENDER_WORDS.get(folded(str(word or "")))


#: A status as the person reads it.
_STATUS_KO = {
    "alive": "살아 있는",
    "dead": "죽은",
    "missing": "실종된",
    "unknown": "생사를 알 수 없는",
}


@dataclass(frozen=True)
class Finding:
    """One contradiction a written scene takes part in, as the person is told it."""

    scene_id: str
    scene_title: str
    quote: str
    #: The plain-Korean line the person reads: the quote and what the codex says.
    note: str
    #: The reasoner's kind of contradiction, for the record; never shown to the person.
    kind: str

    def record(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id,
            "scene_title": self.scene_title,
            "quote": self.quote,
            "note": self.note,
            "kind": self.kind,
        }


def _key(raw: str) -> str:
    return re.sub(r"[\s-]+", "_", folded(raw))


def _kept(store: CodexYamlStore, facts: Sequence[ReadFact]) -> list[SubmittedFact]:
    """The facts worth checking: asserted and now, about a character the store has.

    A character's status is kept once, from the fact whose quote comes last in the scene.
    """
    status: dict[str, tuple[int, SubmittedFact]] = {}
    held: list[SubmittedFact] = []
    for read in facts:
        fact = read.fact
        if not fact.checked or store.kind_of(read.subject) != "characters":
            continue
        subject = read.subject
        key = _key(fact.key)
        if key == "status":
            value = folded(fact.value)
            if value not in _STATUSES:
                continue
            submitted = SubmittedFact(subject, "status", value, fact.quote)
            if subject not in status or read.at >= status[subject][0]:
                status[subject] = (read.at, submitted)
        elif key == "possesses":
            item = store.resolve(fact.value) if fact.value.strip() else None
            if item is not None and item.entity_id and store.kind_of(item.entity_id) == "items":
                held.append(SubmittedFact(subject, "possesses", item.entity_id, fact.quote))
        elif key == "gender":
            gender = gender_of(fact.value)
            if gender is not None:
                held.append(SubmittedFact(subject, "gender", gender, fact.quote))
        elif key == FAMILY_KEY:
            other = read.of
            found = family_role(fact.value)
            if other is None or other == subject or store.kind_of(other) != "characters":
                continue
            if not found or not _names_family(fact.quote):
                continue
            role, said = found
            held.append(SubmittedFact(subject, relation_predicate(other), role, fact.quote))
            if said is not None:
                held.append(SubmittedFact(subject, "gender", said, fact.quote))
    unique = list(dict.fromkeys(held))
    return [s for _, s in status.values()] + unique


def _batchim(word: str) -> bool:
    bare = word.strip().rstrip("’”'\")")
    last = bare[-1:]
    return "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0


def _i_ga(word: str) -> str:
    return f"{word}{'이' if _batchim(word) else '가'}"


def _eun_neun(word: str) -> str:
    return f"{word}{'은' if _batchim(word) else '는'}"


def _eul_reul(word: str) -> str:
    return f"{word}{'을' if _batchim(word) else '를'}"


def _clause(names: dict[str, str], triple: str) -> str | None:
    """A fact such as `harin type Dead`, as the person reads it; `None` when it has no words."""
    parts = triple.split(" ", 2)
    if len(parts) != 3:
        return None
    subject, predicate, obj = parts
    who = names.get(subject)
    if who is None:
        return None
    if predicate == "type" and folded(obj) in _STATUS_KO:
        return f"{_i_ga(who)} {_STATUS_KO[folded(obj)]} 것으로"
    if predicate == "possesses" and obj in names:
        return f"{_i_ga(who)} {_eul_reul('‘' + names[obj] + '’')} 가진 것으로"
    if predicate == "gender" and obj in _GENDER_KO:
        return f"{_i_ga(who)} {_GENDER_KO[obj]}인 것으로"
    other = names.get(predicate.removeprefix(RELATION_PREFIX))
    if predicate.startswith(RELATION_PREFIX) and other and obj in _ROLE_KO:
        return f"{_i_ga(who)} {other}의 {_ROLE_KO[obj]}인 것으로"
    return None


def _from(part: dict[str, Any], where: str) -> bool:
    return any(
        s.get("from") == where for s in cast("list[dict[str, Any]]", part.get("sources", ()))
    )


#: The reasoner's kinds of contradiction this module names in its own words
#: (`ontology.rules.KIND_DISJOINT`, `KIND_INVERSE_FUNCTIONAL`), as `audit_scene` reports them.
KIND_DISJOINT = "disjoint"
KIND_INVERSE_FUNCTIONAL = "inverse_functional"


def _stored(part: dict[str, Any]) -> list[dict[str, Any]]:
    """Where a fact came from in the store (an entry file or a pending proposal)."""
    return [
        s
        for s in cast("list[dict[str, Any]]", part.get("sources", ()))
        if s.get("from") in ("codex", "proposal")
    ]


def _earlier_scene(part: dict[str, Any], scene_id: str) -> str | None:
    """The earlier scene whose change the stored fact rests on, or `None` (it holds from the
    start of the story, or was set by this scene itself)."""
    for source in _stored(part):
        at = source.get("set_by_progression_at")
        if isinstance(at, str) and at != scene_id:
            return at
    return None


def _triple(part: dict[str, Any]) -> list[str]:
    return str(part.get("fact", "")).split(" ", 2)


def _named(
    *,
    kind: str,
    scene_part: dict[str, Any],
    stored_part: dict[str, Any],
    shown: dict[str, str],
    earlier_title: str,
) -> tuple[str, str] | None:
    """The two contradictions a reader names by what an earlier scene wrote: a character
    dead there who moves here, and an item held there that someone else holds here.
    `(kind, what is said)`, or `None` for another contradiction."""
    here, there = _triple(scene_part), _triple(stored_part)
    if len(here) != 3 or len(there) != 3:
        return None
    if (
        kind == KIND_DISJOINT
        and here[1] == there[1] == "type"
        and folded(here[2]) == "alive"
        and folded(there[2]) == "dead"
        and here[0] == there[0]
    ):
        name = _eun_neun(shown.get(here[0], here[0]))
        return (
            "dead_then_acting",
            f"{name} 앞의 ‘{earlier_title}’ 장면에서 죽었는데, 이 장면에서 다시 움직입니다.",
        )
    if (
        kind == KIND_INVERSE_FUNCTIONAL
        and here[1] == there[1] == "possesses"
        and here[2] == there[2]
        and here[0] != there[0]
    ):
        item = _eun_neun("‘" + shown.get(here[2], here[2]) + "’")
        return (
            "holder_changed",
            f"{item} 앞의 ‘{earlier_title}’ 장면에서 {_i_ga(shown.get(there[0], there[0]))} "
            f"가지고 있었는데, 이 장면에서는 건네받았다는 말 없이 "
            f"{_i_ga(shown.get(here[0], here[0]))} 가지고 있습니다.",
        )
    return None


def findings_from(
    *,
    store: CodexYamlStore,
    outline: Outline,
    axioms: Sequence[StoryAxiom],
    axioms_are_defaults: bool,
    scene_id: str,
    text: str,
    facts: Sequence[ReadFact],
) -> list[Finding]:
    """The contradictions scene `scene_id` takes part in, each written for the person.

    `store` is the story read with its pending proposals (`checked_store`), and `facts` the
    scene's reading held to its text (`knowledge.extraction.verify_facts`). The check is a
    query of the store's approved-and-proposed view once the scene has ended.
    """
    submitted = _kept(store, facts)
    if not submitted:
        return []
    audit = audit_scene(
        story_id=store.story_id,
        outline=outline,
        scene_id=scene_id,
        scene_text=text,
        codex=store.codex(),
        facts=submitted,
        axioms=[*axioms, *identity_axioms(submitted, axioms)],
        axioms_are_defaults=axioms_are_defaults,
        store=store,
        proposed=True,
    )
    found = outline.find(scene_id)
    title = found[1].title if found else scene_id
    titles = {s.id: s.title for _, s in outline.scenes_in_order()}
    shown = {item.entry.id: item.entry.name for item in store.codex().items}
    findings: list[Finding] = []
    for contradiction in cast("list[dict[str, Any]]", audit["contradictions"]):
        if not contradiction["involves_this_scene"]:
            continue
        kind = str(contradiction["kind"])
        parts = cast("list[dict[str, Any]]", contradiction["facts"])
        quotes = [
            str(s.get("quote") or "")
            for part in parts
            for s in cast("list[dict[str, Any]]", part.get("sources", ()))
            if s.get("from") == "scene"
        ]
        scene_parts = [p for p in parts if "fact" in p and _from(p, "scene")]
        stored_parts = [p for p in parts if "fact" in p and _stored(p) and not _from(p, "scene")]
        quote = quotes[0] if quotes else ""
        said: str | None = None
        if scene_parts and stored_parts:
            earlier = _earlier_scene(stored_parts[0], scene_id)
            if earlier is not None:
                named = _named(
                    kind=kind,
                    scene_part=scene_parts[0],
                    stored_part=stored_parts[0],
                    shown=shown,
                    earlier_title=titles.get(earlier, earlier),
                )
                if named is not None:
                    kind, said = named
        if said is None:
            said = _said(scene_parts, stored_parts, shown, titles, scene_id)
        findings.append(
            Finding(
                scene_id=scene_id,
                scene_title=title,
                quote=quote,
                note=f"‘{title}’ 장면의 “{quote}” — {said}",
                kind=kind,
            )
        )
    return findings


def _said(
    scene_parts: Sequence[dict[str, Any]],
    stored_parts: Sequence[dict[str, Any]],
    shown: dict[str, str],
    titles: dict[str, str],
    scene_id: str,
) -> str:
    """What a contradiction says, in plain Korean: the scene against an earlier scene,
    against the codex (an entry still waiting for approval included: the chapter was
    written over it as if approved, and the reply says so), or against itself."""
    in_scene = next((c for p in scene_parts if (c := _clause(shown, p["fact"]))), None)
    if in_scene and stored_parts:
        stored = stored_parts[0]
        in_store = _clause(shown, stored["fact"])
        earlier = _earlier_scene(stored, scene_id)
        if in_store and earlier is not None:
            where = titles.get(earlier, earlier)
            return f"이 문장에서는 {in_scene} 나오지만, 앞의 ‘{where}’ 장면에서는 {in_store} 나왔습니다."
        if in_store:
            return f"이 문장에서는 {in_scene} 나오지만, 설정집에는 {in_store} 되어 있습니다."
    both = list(dict.fromkeys(c for p in scene_parts if (c := _clause(shown, p["fact"]))))
    if len(both) >= 2:
        return f"이 장면에서 {both[0]} 나오는데, {both[1]}도 나옵니다."
    return "이 문장이 설정집의 기록과 맞지 않습니다."


def checked_store(
    story_id: str, outline: Outline, codex: CodexIndex, proposals: Iterable[Proposal]
) -> CodexYamlStore:
    """The store a written scene is checked against: the codex and every proposal, with a
    character's gender read from its ``visual`` block where its state has none
    (`with_gender`), for the entries files have and for those pending proposals add."""
    lifted: list[Proposal] = []
    for proposal in proposals:
        new = proposal.change.new_entry
        visual = new.get("visual") if new is not None else None
        gender = visual.get("gender") if isinstance(visual, dict) else None
        state = new.get("state") if new is not None else None
        if (
            new is not None
            and gender in _GENDER_KO
            and not (isinstance(state, dict) and "gender" in state)
        ):
            given = dict(state) if isinstance(state, dict) else {}
            change = proposal.change.model_copy(
                update={"new_entry": {**new, "state": {**given, "gender": gender}}}
            )
            proposal = proposal.model_copy(update={"change": change})
        lifted.append(proposal)
    return CodexYamlStore.read(story_id, outline, with_gender(codex), lifted)


def with_gender(codex: CodexIndex) -> CodexIndex:
    """The codex with each character's `visual.gender` also as its state ``gender``.

    A cast proposed before gender went into the state, or an entry the person wrote by
    hand, has it only in the visual block, which the audit does not read. A state
    ``gender`` already there wins.
    """
    items: list[CodexItem] = []
    for item in codex.items:
        entry = item.entry
        visual = getattr(entry, "visual", None)
        gender = getattr(visual, "gender", None)
        if gender in _GENDER_KO and "gender" not in entry.state:
            entry = entry.model_copy(update={"state": {**entry.state, "gender": gender}})
            item = CodexItem(kind=item.kind, entry=entry)
        items.append(item)
    return CodexIndex(items=tuple(items), unreadable=codex.unreadable)


#: The check's own rules, beside the story's: nobody is alive and dead at once, and one
#: thing has one holder at a time. A story that writes its own rules keeps them; these hold
#: a written scene to what earlier scenes wrote whatever the rules say, as `came_back` and
#: `written_deaths` did before the store had pending proposals (#2207).
_CHECK_RULES = (
    StoryAxiom(kind="disjointWith", subject="Alive", object="Dead", note="Alive or dead."),
    StoryAxiom(kind="inverseFunctional", subject="possesses", note="One holder at a time."),
)


def identity_axioms(
    facts: Sequence[SubmittedFact],
    axioms: Sequence[StoryAxiom] = (),
) -> list[StoryAxiom]:
    """One gender per character, one relation per pair, and the check's own rules
    (`_CHECK_RULES`), each one not already among `axioms`.

    Added to the story's own rules for this check only, like the eval's eye colour: a
    scene that calls a son "딸" disagrees with the codex through `functional gender`. A
    pair's relation is functional for every pair a scene fact names: a contradiction that
    involves the scene needs one, and the store's side of it (approved or proposed) is read
    against that rule.
    """
    keys = {"gender"}
    keys |= {f.predicate for f in facts if f.predicate.startswith(RELATION_PREFIX)}
    rules = [
        StoryAxiom(kind="functional", subject=key, note="One value per character.")
        for key in sorted(keys)
    ]
    given = {(folded(a.kind), folded(a.subject), folded(a.object or "")) for a in axioms}
    return [
        rule
        for rule in (*_CHECK_RULES, *rules)
        if (folded(rule.kind), folded(rule.subject), folded(rule.object or "")) not in given
    ]
