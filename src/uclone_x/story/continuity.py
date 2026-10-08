"""Continuity after writing: a scene's facts read by a model, checked against the codex.

The same path as the writer eval's graph grader (`evals/suites/writer_graph.py`), which is
the primary continuity check: a model reads the scene and **proposes** facts, each with the
quote it read it from, a time and a polarity (`continuity_prompt`, `parse_extraction`);
everything after that is deterministic. Only facts ``asserted`` about the scene's own
``now`` are kept, and they go to `uclone_x.story.audit.audit_scene`, which rejects a quote
the scene does not contain and checks the rest with the ontology reasoner against the codex
as it stands once the scene has ended.

What this module adds for a product check, where no fixture tells the reader what to look
for:

- only these keys: ``status`` (``alive`` or ``dead``) of a character entry, ``possesses``
  of an item entry, ``gender`` (``female`` or ``male``) of a character, and ``family``
  with ``of`` another character's id (what the subject is to the other: ``parent``,
  ``child``, ``sibling``, ``spouse``, ``grandparent``, ``grandchild``; `family_role` reads
  English and Korean kinship words, and a gendered one such as "딸" also gives a gender).
  A family fact is checked as the codex's relation between the two (`relations:`,
  `audit.relation_predicate`). Gender and each pair's relation are one value for this
  check (`identity_axioms`), so a son called "딸" disagrees with the codex. A cast whose
  gender is only in ``visual.gender`` is read as if it were in the state (`with_gender`). A fact about a name the codex does not have, or an item it
  does not list, is dropped rather than checked as a bare string -- a lamp handed from one
  person to another in a scene is not a contradiction;
- a character's ``status`` is read once per scene, from the last sentence that states it: a
  character who dies in the scene was alive earlier in it, and that is a change, not a
  contradiction;
- each contradiction is written for the person in plain Korean, with the quote
  (`Finding.note`), and never with an entry id or a reasoner's words; two owners of one
  item in the same scene are named as that, not as a disagreement with the codex;
- a character an earlier scene left dead who acts again (`came_back`). A death a scene
  writes reaches the codex only as a proposal a person approves, so the codex alone would
  still call that character alive; the caller keeps each scene's last status and passes
  the deaths of earlier scenes.

Nothing here calls a model or writes a file: `story_start` asks the model and decides what to
do with the findings (it offers a rewrite; it never rewrites by itself).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from uclone_x.story.audit import (
    RELATION_PREFIX,
    SubmittedFact,
    audit_scene,
    relation_predicate,
)
from uclone_x.story.context import CodexIndex, CodexItem
from uclone_x.story.names import CodexNames
from uclone_x.story.quotes import folded
from uclone_x.story.schemas import Outline, StoryAxiom

__all__ = [
    "CONTINUITY_MARKER",
    "FAMILY_KEY",
    "INVERSE_ROLE",
    "POLARITIES",
    "TIMES",
    "ExtractedFact",
    "Finding",
    "came_back",
    "codex_with",
    "continuity_prompt",
    "family_role",
    "findings_from",
    "gender_of",
    "identity_axioms",
    "names_family",
    "parse_extraction",
    "reply_object",
    "salvage_extraction",
    "with_gender",
]

#: The first line of every continuity request, so a scripted model can tell it apart.
CONTINUITY_MARKER = "[story continuity check]"
TIMES = ("now", "past")
POLARITIES = ("asserted", "negated", "absent", "hypothetical")
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
class ExtractedFact:
    """One fact a model read from a scene, where it read it, and when it holds."""

    subject: str
    key: str
    value: str
    quote: str
    time: str = "now"
    polarity: str = "asserted"
    #: The other character of a family fact (`FAMILY_KEY`); empty for any other key.
    of: str = ""

    @property
    def checked(self) -> bool:
        """Only an asserted fact about the scene's present is checked against the codex."""
        return self.time == "now" and self.polarity == "asserted"


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


# A stray quotation mark opening an object's key: `{ " "name": ...` or `{ ""subject": ...`.
# qwen3:8b wrote it in 5 of 8 cast replies for a premise that quotes a name (2026-09-29).
STRAY_QUOTE = re.compile(r'([{,]\s*)"\s*(?="[^"\n]+"\s*:)')

_FACT_KEYS = ("subject", "key", "of", "value", "quote", "time", "polarity")
_KEY_ALT = "|".join(_FACT_KEYS)
# One field of a fact whose string value may hold unescaped double quotes -- dialogue
# quoted from a scene (`"quote": "그가 "딸아" 하고 불렀다"`): the value runs to the quote
# mark that is followed by the next field's key or by the object's end.
_FIELD = re.compile(
    rf'"({_KEY_ALT})"\s*:\s*"(.*?)"(?=\s*(?:,\s*"+\s*"?(?:{_KEY_ALT})"\s*:|\}}))', re.DOTALL
)


def _fields_read(body: str) -> dict[str, Any] | None:
    """The facts of a reply read field by field; `None` when no field is there.

    A fact starts again at a key the fact being read already has.
    """
    facts: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for match in _FIELD.finditer(body):
        key, value = match.group(1), match.group(2)
        if key in current:
            facts.append(current)
            current = {}
        try:
            current[key] = str(json.loads(f'"{value}"', strict=False))
        except json.JSONDecodeError:
            current[key] = value
    if current:
        facts.append(current)
    return {"facts": facts} if facts else None


def _json_object(content: str) -> Any:
    """The JSON object in a reply, read past the mistakes a small model makes.

    In order: as it is, with raw control characters inside strings allowed; with stray
    quotation marks before keys taken out (`STRAY_QUOTE`); and field by field, which reads
    a quote holding unescaped double quotes (`_fields_read`). The graph grader left 3 of 6
    story scenes unread on replies of these kinds (2026-09-29).
    """
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end < start:
        raise ValueError("the reply has no JSON object")
    body = content[start : end + 1]
    error = ""
    for attempt in (body, STRAY_QUOTE.sub(r"\1", body)):
        try:
            return json.loads(attempt, strict=False)
        except json.JSONDecodeError as exc:
            error = error or exc.msg
    fields = _fields_read(STRAY_QUOTE.sub(r"\1", body))
    if fields is not None:
        return fields
    raise ValueError(f"the reply is not JSON: {error}")


def names_family(quote: str) -> bool:
    """Whether a quote holds a kinship word at all; the public name of `_names_family`."""
    return _names_family(quote)


def reply_object(content: str) -> Any:
    """The JSON object in a model's reply, read as `parse_extraction` reads it.

    Raises:
        ValueError: the reply has no JSON object that can be read.
    """
    return _json_object(content)


def parse_extraction(content: str) -> list[ExtractedFact]:
    """The facts in a model's reply.

    Raises:
        ValueError: the reply is not a JSON object with a ``facts`` list of objects.
    """
    data = _json_object(content)
    if not isinstance(data, dict):
        raise ValueError("the reply has no 'facts' list")
    listed = cast("dict[str, object]", data).get("facts")
    if not isinstance(listed, list):
        raise ValueError("the reply has no 'facts' list")
    facts: list[ExtractedFact] = []
    for raw in cast("list[object]", listed):
        if not isinstance(raw, dict):
            raise ValueError("a fact in the reply is not an object")
        item = cast("dict[str, object]", raw)
        time = str(item.get("time") or "now").strip().lower()
        polarity = str(item.get("polarity") or "asserted").strip().lower()
        facts.append(
            ExtractedFact(
                subject=str(item.get("subject") or ""),
                key=str(item.get("key") or ""),
                value=str(item.get("value") or ""),
                quote=str(item.get("quote") or ""),
                time=time if time in TIMES else "now",
                polarity=polarity if polarity in POLARITIES else "asserted",
                of=str(item.get("of") or ""),
            )
        )
    return facts


def salvage_extraction(content: str) -> list[ExtractedFact]:
    """The complete, distinct fact objects at the head of a reply cut off by its token cap.

    A model that loops runs to its cap; the facts it finished before the cut are still its
    reading of the scene. The object it was writing when cut is dropped.

    Raises:
        ValueError: the reply has no ``facts`` list, or not one complete fact in it.
    """
    head = re.search(r'"facts"\s*:\s*\[', content)
    if head is None:
        raise ValueError("the cut-off reply has no 'facts' list")
    decoder = json.JSONDecoder()
    at = head.end()
    raw: list[object] = []
    while True:
        at = content.find("{", at)
        if at == -1:
            break
        try:
            obj, at = decoder.raw_decode(content, at)
        except json.JSONDecodeError:
            break
        raw.append(obj)
    facts: list[ExtractedFact] = []
    for fact in parse_extraction(json.dumps({"facts": raw})):
        if fact not in facts:
            facts.append(fact)
    if not facts:
        raise ValueError("the cut-off reply has no complete fact")
    return facts


def _entries(codex: CodexIndex, kind: str) -> str:
    lines = [
        f"- {item.entry.id}: {', '.join([item.entry.name, *item.entry.aliases])}"
        for item in codex.items
        if item.kind == kind
    ]
    return "\n".join(lines) or "- (none)"


def continuity_prompt(outline: Outline, codex: CodexIndex, scene_id: str, text: str) -> str:
    """What the reading model is asked: the rule, the story's own entries, and the scene."""
    found = outline.find(scene_id)
    if found is None:
        raise ValueError(f"the outline has no scene '{scene_id}'")
    _, scene = found
    items = any(item.kind == "items" for item in codex.items)
    item_list = (
        f"Items (use the id as value for possesses):\n{_entries(codex, 'items')}\n" if items else ""
    )
    possesses = (
        "- possesses: the id of an item above that the character holds, carries, wears or wields.\n"
        if items
        else ""
    )
    return f"""{CONTINUITY_MARKER}
You read one scene of a story and list the facts it states, for a continuity check. You do
not judge whether a fact is right; you only report what the text says.

Scene {scene.id}: {scene.title}. {scene.summary}

Characters (use the id as subject):
{_entries(codex, "characters")}
{item_list}
Keys (use no other key):
- status: alive or dead. A character who acts, speaks or moves in the scene's present is
  alive. A ghost, spirit or corpse of someone is dead.
- gender: female or male, only where the scene shows it: a word such as 그녀, 딸, 아들,
  어머니, 아버지, 누나 or 오빠 used for the character.
- family: what the subject is to the character whose id is in "of" (another character
  above): father, mother, son, daughter, brother, sister, husband or wife, grandfather,
  grandmother, grandson or granddaughter. "하린의 아버지 도윤" is subject doyun, key
  family, of harin, value father.
{possesses}
For every fact give a time and a polarity:
- time "now": true at this scene's own time. time "past": a remembered, recalled or dreamed
  act, a voice or words remembered, and anything a character says happened (dialogue, words
  in quotation marks).
- polarity "asserted": the text says it is so. "negated": the text says it is NOT so.
  "absent": the person or thing is gone or missing. "hypothetical": only imagined, wished,
  feared or likened ("as if").

Rules:
1. Only the characters and keys above. A fact the scene does not state is left out.
2. quote is copied exactly from the scene, at least a few words, and shows the fact.
3. Report each character's status once, from the sentence that shows it last.

Reply with JSON only:
{{"facts": [{{"subject": "<id>", "key": "<key>", "value": "<value>", "quote": "<exact words>", "time": "now", "polarity": "asserted"}}]}}
A family fact also has "of": "<id>".
Reply {{"facts": []}} when the scene states none.

<scene>
{text}
</scene>"""


def _key(raw: str) -> str:
    return re.sub(r"[\s-]+", "_", folded(raw))


def _kept(
    names: CodexNames, codex: CodexIndex, text: str, facts: Sequence[ExtractedFact]
) -> list[SubmittedFact]:
    """The facts worth checking: asserted and now, about an entry the codex has.

    A character's status is kept once, from the fact whose quote comes last in the scene.
    """
    kinds = {item.entry.id: item.kind for item in codex.items}
    status: dict[str, tuple[int, SubmittedFact]] = {}
    held: list[SubmittedFact] = []
    for fact in facts:
        if not fact.checked:
            continue
        key = _key(fact.key)
        subject = names.resolve(fact.subject)
        if subject is None or kinds.get(subject) != "characters":
            continue
        if key == "status":
            value = folded(fact.value)
            if value not in _STATUSES:
                continue
            at = text.find(fact.quote.strip())
            submitted = SubmittedFact(subject, "status", value, fact.quote)
            if subject not in status or at >= status[subject][0]:
                status[subject] = (at, submitted)
        elif key == "possesses":
            item = names.resolve(fact.value)
            if item is not None and kinds.get(item) == "items":
                held.append(SubmittedFact(subject, "possesses", item, fact.quote))
        elif key == "gender":
            gender = gender_of(fact.value)
            if gender is not None:
                held.append(SubmittedFact(subject, "gender", gender, fact.quote))
        elif key == FAMILY_KEY:
            other = names.resolve(fact.of)
            found = family_role(fact.value)
            if other is None or other == subject or kinds.get(other) != "characters" or not found:
                continue
            if not _names_family(fact.quote):
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


def findings_from(
    *,
    story_id: str,
    outline: Outline,
    codex: CodexIndex,
    axioms: Sequence[StoryAxiom],
    axioms_are_defaults: bool,
    scene_id: str,
    text: str,
    facts: Sequence[ExtractedFact],
) -> list[Finding]:
    """The contradictions scene `scene_id` takes part in, each written for the person."""
    names = CodexNames(codex)
    submitted = _kept(names, codex, text, facts)
    if not submitted:
        return []
    codex = with_gender(codex)
    audit = audit_scene(
        story_id=story_id,
        outline=outline,
        scene_id=scene_id,
        scene_text=text,
        codex=codex,
        facts=submitted,
        axioms=[*axioms, *identity_axioms(codex, submitted)],
        axioms_are_defaults=axioms_are_defaults,
    )
    found = outline.find(scene_id)
    title = found[1].title if found else scene_id
    shown = {item.entry.id: item.entry.name for item in codex.items}
    findings: list[Finding] = []
    for contradiction in cast("list[dict[str, Any]]", audit["contradictions"]):
        if not contradiction["involves_this_scene"]:
            continue
        parts = cast("list[dict[str, Any]]", contradiction["facts"])
        quotes = [
            str(s.get("quote") or "")
            for part in parts
            for s in cast("list[dict[str, Any]]", part.get("sources", ()))
            if s.get("from") == "scene"
        ]
        in_scene = next(
            (_clause(shown, p["fact"]) for p in parts if "fact" in p and _from(p, "scene")), None
        )
        in_codex = next(
            (
                _clause(shown, p["fact"])
                for p in parts
                if "fact" in p and _from(p, "codex") and not _from(p, "scene")
            ),
            None,
        )
        both = list(
            dict.fromkeys(
                c
                for p in parts
                if "fact" in p and _from(p, "scene")
                for c in [_clause(shown, p["fact"])]
                if c
            )
        )
        quote = quotes[0] if quotes else ""
        if in_scene and in_codex:
            said = f"이 문장에서는 {in_scene} 나오지만, 설정집에는 {in_codex} 되어 있습니다."
        elif len(both) >= 2:
            said = f"이 장면에서 {both[0]} 나오는데, {both[1]}도 나옵니다."
        else:
            said = "이 문장이 설정집의 기록과 맞지 않습니다."
        findings.append(
            Finding(
                scene_id=scene_id,
                scene_title=title,
                quote=quote,
                note=f"‘{title}’ 장면의 “{quote}” — {said}",
                kind=str(contradiction["kind"]),
            )
        )
    return findings


def codex_with(codex: CodexIndex, assumed: Sequence[CodexItem]) -> CodexIndex:
    """The codex with entries the chapter was written over, still pending, taken as in it."""
    have = {(item.kind, item.entry.id) for item in codex.items}
    extra = tuple(i for i in assumed if (i.kind, i.entry.id) not in have)
    return CodexIndex(items=(*codex.items, *extra), unreadable=codex.unreadable)


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


def identity_axioms(codex: CodexIndex, facts: Sequence[SubmittedFact]) -> list[StoryAxiom]:
    """One gender per character, and one relation per pair, as rules for the check.

    Added to the story's own rules for this check only, like the eval's eye colour: a
    scene that calls a son "딸" disagrees with the codex through `functional gender`. A
    pair's relation is every one the codex gives, at the start or by a progression.
    """
    keys = {"gender"}
    keys |= {f.predicate for f in facts if f.predicate.startswith(RELATION_PREFIX)}
    for item in codex.items:
        others = {*item.entry.relations}
        others |= {o for p in item.entry.progressions for o in p.relations}
        keys |= {relation_predicate(other) for other in others}
    return [
        StoryAxiom(kind="functional", subject=key, note="One value per character.")
        for key in sorted(keys)
    ]


def came_back(
    codex: CodexIndex,
    *,
    scene_id: str,
    scene_title: str,
    text: str,
    facts: Sequence[ExtractedFact],
    deaths: dict[str, dict[str, str]],
) -> tuple[list[Finding], dict[str, dict[str, str] | None]]:
    """A character an earlier scene left dead who is alive again in this one.

    `deaths` holds, per character id, the earlier scene (``scene_id``, ``scene_title``) and
    the ``quote`` that left them dead. Only asserted, present status facts about a codex
    character count, in the order of their quotes in the scene. Returned with the findings,
    one per character at most, is each character's status once the scene ends: the death
    record when it leaves them dead, `None` when it leaves them alive.
    """
    names = CodexNames(codex)
    kinds = {item.entry.id: item.kind for item in codex.items}
    shown = {item.entry.id: item.entry.name for item in codex.items}
    read: list[tuple[int, str, str, str]] = []
    for fact in facts:
        if not fact.checked or _key(fact.key) != "status":
            continue
        subject = names.resolve(fact.subject)
        value = folded(fact.value)
        quote = fact.quote.strip()
        if subject is None or kinds.get(subject) != "characters" or value not in _STATUSES:
            continue
        at = text.find(quote)
        if at < 0:
            continue
        read.append((at, subject, value, quote))
    findings: list[Finding] = []
    ends: dict[str, dict[str, str] | None] = {}
    for _, subject, value, quote in sorted(read):
        dead = ends[subject] if subject in ends else deaths.get(subject)
        if value == "alive" and dead is not None and dead["scene_id"] != scene_id:
            name = _eun_neun(shown.get(subject, subject))
            findings.append(
                Finding(
                    scene_id=scene_id,
                    scene_title=scene_title,
                    quote=quote,
                    note=(
                        f"‘{scene_title}’ 장면의 “{quote}” — {name} 앞의 "
                        f"‘{dead['scene_title']}’ 장면에서 죽었는데, 이 장면에서 다시 움직입니다."
                    ),
                    kind="dead_then_acting",
                )
            )
        ends[subject] = (
            {"scene_id": scene_id, "scene_title": scene_title, "quote": quote}
            if value == "dead"
            else None
        )
    return findings, ends
