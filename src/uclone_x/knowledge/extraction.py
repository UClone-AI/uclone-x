"""What a model read from a text, as facts: the reply read, and each fact held to its text.

uGraph §3.5: an extraction's prompt is the vessel's own (what to look for differs), but
what comes back is one shape, and what is done with it before a vessel decides anything
is done once, here:

- **reading the reply** (`reply_object`, `parse_facts`, `parse_extraction`,
  `salvage_extraction`): a JSON object read past the mistakes a small model makes, and
  each fact as `ExtractedFact` -- subject, key, value, the other entity a relation names
  (``of``), the quote, and when and how the text says it (``time``, ``polarity``);
- **holding each fact to its text** (`verify_facts`): the quote must be in the text
  (`quotes.quote_found`), what made-up facts lack; the subject and the other entity must
  each resolve to one entity of the scope (`EntityResolver`, stages 1-3: a name two
  entities share is ambiguous and nothing is guessed, G11); and a fact the reply gives
  twice is kept once. What is dropped is counted by reason, never silently lost;
- **a name the text calls a known entity by** (``named``, `ReadFact.alias`): the reading
  model is given each entity's id with the names it already has, and answers with the id
  and the words the text uses. A form that is none of those names -- the same person in
  another script or spelling -- resolves through the id, so no rule in code has to know
  how one script is written in another, and comes back as an alias candidate;
- **confirming a candidate** (`alias_prompt`, `confirmed_aliases`; stage 4, §3.4): the
  reading model also answers with a pronoun or a description ("she"), which is no name. One
  more question, temperature 0, asks whether each candidate is a name of its entity; a
  candidate not confirmed, or a reply that cannot be read, is not one. Nothing here knows
  which words of a language are pronouns. What a vessel does with a confirmed name is its
  policy: the story proposes it as an alias a person approves.
- **who is there at the text's own time** (`PRESENCES`, `read_presence`): a reply may say,
  per entity, whether the text has it there and acting at its own time, or only mentions,
  remembers or reports it. A fact's own ``time`` asks the same of one quote, and a small
  model answered it "now" for a dead man's legacy and for a memory of him (#2311); asked
  of the person instead, the model says how the scene has them. Which facts need someone
  there is the vessel's policy.

Which facts a vessel keeps after that -- only the present, only asserted, only some keys --
is the vessel's decision, not this module's. Nothing here calls a model or reads a file.

This module is pure.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from uclone_x.knowledge.fold import fold
from uclone_x.knowledge.models import Entity
from uclone_x.knowledge.quotes import quote_found
from uclone_x.knowledge.resolver import EntityResolver, name_key

__all__ = [
    "ACTING",
    "ALIAS_MARKER",
    "POLARITIES",
    "PRESENCES",
    "STRAY_QUOTE",
    "TIMES",
    "AliasCandidate",
    "ExtractedFact",
    "ReadFact",
    "Verified",
    "alias_prompt",
    "confirmed_aliases",
    "parse_extraction",
    "parse_facts",
    "listed_facts",
    "read_presence",
    "reply_object",
    "salvage_extraction",
    "verify_facts",
]

TIMES = ("now", "past")
POLARITIES = ("asserted", "negated", "absent", "hypothetical")
#: How a text has an entity at its own time: there and acting (doing, saying or undergoing
#: something, dying included), or only mentioned, remembered or reported by someone.
PRESENCES = ("acting", "mentioned", "remembered", "reported")
ACTING = PRESENCES[0]


@dataclass(frozen=True)
class ExtractedFact:
    """One fact a model read from a text, where it read it, and when it holds."""

    subject: str
    key: str
    value: str
    quote: str
    time: str = "now"
    polarity: str = "asserted"
    #: The other entity a relation names; empty for any other key.
    of: str = ""
    #: The words the text calls the subject by, when the reply gives them.
    named: str = ""
    #: The words of the quote that name the fact's object -- the item held, the kin --
    #: when the reply gives them (#2249).
    object_named: str = ""

    @property
    def checked(self) -> bool:
        """Whether the text asserts it of its own present: the only facts a check reads."""
        return self.time == "now" and self.polarity == "asserted"


# A stray quotation mark opening an object's key: `{ " "name": ...` or `{ ""subject": ...`.
# qwen3:8b wrote it in 5 of 8 cast replies for a premise that quotes a name (2026-09-29).
STRAY_QUOTE = re.compile(r'([{,]\s*)"\s*(?="[^"\n]+"\s*:)')

_FACT_KEYS = (
    "subject",
    "named",
    "key",
    "of",
    "value",
    "object_named",
    "quote",
    "time",
    "polarity",
)
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


def reply_object(content: str) -> Any:
    """The JSON object in a model's reply, read past the mistakes a small model makes.

    In order: as it is, with raw control characters inside strings allowed; with stray
    quotation marks before keys taken out (`STRAY_QUOTE`); and field by field, which reads
    a quote holding unescaped double quotes (`_fields_read`). The graph grader left 3 of 6
    story scenes unread on replies of these kinds (2026-09-29).

    Raises:
        ValueError: the reply has no JSON object that can be read.
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


def parse_facts(listed: Iterable[object]) -> list[ExtractedFact]:
    """Each fact object of a reply's list, as `ExtractedFact`.

    A time or polarity the reply leaves out, or gives as a word not in `TIMES` or
    `POLARITIES`, is read as the present and asserted.

    Raises:
        ValueError: an element of the list is not an object.
    """
    facts: list[ExtractedFact] = []
    for raw in listed:
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
                named=str(item.get("named") or "").strip(),
                object_named=str(item.get("object_named") or "").strip(),
            )
        )
    return facts


def parse_extraction(content: str) -> list[ExtractedFact]:
    """The facts in a model's reply: its object's ``facts`` list.

    Raises:
        ValueError: the reply is not a JSON object with a ``facts`` list of objects.
    """
    data = reply_object(content)
    if not isinstance(data, dict):
        raise ValueError("the reply has no 'facts' list")
    listed = cast("dict[str, object]", data).get("facts")
    if not isinstance(listed, list):
        raise ValueError("the reply has no 'facts' list")
    return parse_facts(cast("list[object]", listed))


def _salvaged(content: str) -> list[object]:
    """The complete objects at the head of a reply's ``facts`` list, as the model wrote them.

    Raises:
        ValueError: the reply has no ``facts`` list.
    """
    head = re.search(r'"facts"\s*:\s*\[', content)
    if head is None:
        raise ValueError("the cut-off reply has no 'facts' list")
    decoder = json.JSONDecoder(strict=False)
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
    return raw


def _json_reply(content: str) -> Any:
    """The reply's JSON object or array as the model wrote it, or `None` when neither reads.

    Whichever the reply opens first, then the other; each as it is, then with stray
    quotation marks before keys taken out (`STRAY_QUOTE`).
    """
    spans = [("{", "}"), ("[", "]")]
    if -1 < content.find("[") < content.find("{"):
        spans.reverse()
    for opening, closing in spans:
        start, end = content.find(opening), content.rfind(closing)
        if start == -1 or end < start:
            continue
        body = content[start : end + 1]
        for attempt in (body, STRAY_QUOTE.sub(r"\1", body)):
            try:
                return json.loads(attempt, strict=False)
            except json.JSONDecodeError:
                continue
    return None


def listed_facts(content: str) -> list[object]:
    """The fact objects of a reply, each as the model wrote it, with its own fields kept.

    For a vessel whose facts carry fields beyond `ExtractedFact`'s -- the clone's
    ``confidence``, ``durable`` and ``why`` -- which it reads from the object, the shared
    fields going through `parse_facts` and `verify_facts`. Read, in order: the reply's
    JSON (its object's ``facts`` list, or a bare array); for a reply cut off by its token
    cap, the complete objects at the head of its ``facts`` list (`salvage_extraction`);
    and field by field (`reply_object`), where only text fields survive.

    Raises:
        ValueError: no fact list can be read from the reply.
    """
    data = _json_reply(content)
    if isinstance(data, list):
        return cast("list[object]", data)
    if isinstance(data, dict):
        listed = cast("dict[str, object]", data).get("facts")
        if isinstance(listed, list):
            return cast("list[object]", listed)
        raise ValueError("the reply has no 'facts' list")
    try:
        salvaged = _salvaged(content)
    except ValueError:
        salvaged = []
    if salvaged:
        return salvaged
    read = reply_object(content)
    listed = cast("dict[str, object]", read).get("facts") if isinstance(read, dict) else None
    if not isinstance(listed, list):
        raise ValueError("the reply has no 'facts' list")
    return cast("list[object]", listed)


def salvage_extraction(content: str) -> list[ExtractedFact]:
    """The complete, distinct fact objects at the head of a reply cut off by its token cap.

    A model that loops runs to its cap; the facts it finished before the cut are still its
    reading of the text. The object it was writing when cut is dropped.

    Raises:
        ValueError: the reply has no ``facts`` list, or not one complete fact in it.
    """
    facts: list[ExtractedFact] = []
    for fact in parse_facts(_salvaged(content)):
        if fact not in facts:
            facts.append(fact)
    if not facts:
        raise ValueError("the cut-off reply has no complete fact")
    return facts


@dataclass(frozen=True)
class ReadFact:
    """A fact whose quote is in its text and whose names are each one entity."""

    fact: ExtractedFact
    #: The entity the subject names.
    subject: str
    #: The entity ``of`` names; `None` when the fact names no other.
    of: str | None
    #: Where the quote starts in the text, folded (`fold`): the order the text says things.
    at: int
    #: The name the text calls the subject by, in the quote, that is none of the subject's
    #: names yet: a candidate alias (module docstring). `None` when the fact gives no
    #: such name.
    alias: str | None = None


@dataclass
class Verified:
    """The facts `verify_facts` kept, in the reply's order, and what it dropped, by reason.

    Reasons: ``quote_not_in_text`` (no quote, too short, or not in the text),
    ``not_an_entity`` (a name no entity has), ``ambiguous_name`` (a name two entities
    share), ``named_other`` (the words the text calls the subject by are another entity's
    name), ``object_not_in_quote`` (the other entity the fact is about -- the item held,
    the kin -- is not named in its quote), ``duplicate`` (the same fact again).
    """

    kept: list[ReadFact] = field(default_factory=list[ReadFact])
    dropped: dict[str, int] = field(default_factory=dict[str, int])

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _entity(
    resolver: EntityResolver, name: str, *, new_names: bool = False
) -> tuple[str | None, str | None]:
    """The entity `name` names, or the reason it names none.

    With `new_names`, a name no entity has is a new one, kept as written (in NFC, white
    space collapsed); a near match (stage 3) is not taken for it. Without, it is dropped.
    """
    if not fold(name):
        return None, "not_an_entity"
    found = resolver.resolve(name)
    ids = (found.entity_id,) if found.entity_id is not None else found.ambiguous_ids
    if len(ids) > 1:
        return None, "ambiguous_name"  # nothing is guessed: not the first, not any
    if ids:
        return ids[0], None
    if new_names:
        return " ".join(unicodedata.normalize("NFC", name).split()), None
    return None, "not_an_entity"


def _alias(resolver: EntityResolver, fact: ExtractedFact, subject: str) -> tuple[str | None, bool]:
    """The alias candidate `fact.named` is for `subject`, and whether it names another.

    Only a name the quote holds is read: the words were copied from the text, not made up.
    A name that is already the subject's is no candidate; one that is another entity's, or
    that several share without the subject, says the reply's id and its words disagree.
    """
    named = fact.named
    if not fold(named):
        return None, False
    # Whether the words name another entity is read whether or not the quote holds them:
    # a fact about 사라 that says it was read as 에이든 is about someone else either way,
    # and a quote that names neither does not settle it (#2239).
    found = resolver.resolve(named)
    if found.kind == "existing":
        return None, found.entity_id != subject
    if found.kind == "ambiguous":
        return None, subject not in found.ambiguous_ids
    if fold(named) not in fold(fact.quote):
        return None, False  # a new name is only a candidate in the words it was copied from
    return named, False


def _squeezed(text: str) -> str:
    """`text` folded with no white space at all: names compared as characters, so a script
    written without spaces, or a name the prose spaces differently, is not missed."""
    return "".join(fold(text).split())


def _anchored(resolver: EntityResolver, entity_id: str, fact: ExtractedFact) -> bool:
    """Whether the fact's quote names `entity_id`, the other entity it is about (#2238).

    By characters, in any script (#2249): one of the entity's names or aliases is in the
    quote whole; or the words the reading gave for the object (`object_named`) are in the
    quote and in one of those names, and are no other entity's name. Nothing reads word
    order, spacing, endings or which word of a name is its head: a French name is
    noun-first, Japanese and Chinese write no spaces, and Korean runs a one-character word
    on into its ending (`키를`). The reading model, which reads the language, says which
    words name the object; the code checks they are the quote's and the object's.

    A new name (the clone files one as written, #2235) is its own only name. A quote that
    names the object only by a pronoun is dropped -- the price of not guessing.
    """
    said = _squeezed(fact.quote)
    names = resolver.names_of(entity_id) or (entity_id,)
    if any(_squeezed(name) in said for name in names):
        return True
    words = _squeezed(fact.object_named)
    if not words or words not in said:
        return False
    if not name_key(fact.object_named):
        return False  # only brackets or quotation marks: no name to resolve (#2254 review)
    found = resolver.resolve(fact.object_named)
    if found.kind != "new" and found.entity_id != entity_id:
        return False  # another entity's name, or several: "도시 중심가" is a place
    return any(_holds_words(name, words) for name in names)


def _holds_words(name: str, words: str) -> bool:
    return words in _squeezed(name)


def _object_of(resolver: EntityResolver, fact: ExtractedFact, subject: str) -> str | None:
    """The other entity a fact's value names -- an item held, a place -- or `None` when the
    value is a word, not an entity (`dead`, `female`)."""
    if not fold(fact.value):
        return None
    found = resolver.resolve(fact.value)
    if found.kind == "existing" and found.entity_id != subject:
        return found.entity_id
    return None


def verify_facts(
    facts: Sequence[ExtractedFact],
    text: str,
    resolver: EntityResolver,
    *,
    new_names: bool = False,
) -> Verified:
    """`facts` held to `text` and to the entities `resolver` knows (module docstring).

    `new_names` is the vessel's policy for a name no entity has (§3.4 stage 5): kept as a
    new entity (the clone, whose subjects are not listed ahead), or dropped as
    ``not_an_entity`` (the story, whose entities come from its codex and its ``new``
    proposals).
    """
    out = Verified()
    haystack = fold(text)
    seen: dict[tuple[str, str, str, str | None, str, str], int] = {}
    for fact in facts:
        if not quote_found(fact.quote, text):
            out.drop("quote_not_in_text")
            continue
        subject, why = _entity(resolver, fact.subject, new_names=new_names)
        if subject is None:
            out.drop(why or "not_an_entity")
            continue
        alias, other_named = _alias(resolver, fact, subject)
        if other_named:
            out.drop("named_other")
            continue
        other: str | None = None
        if fact.of.strip():
            other, why = _entity(resolver, fact.of, new_names=new_names)
            if other is None:
                out.drop(why or "not_an_entity")
                continue
        # A fact about another entity -- what is held, who is kin -- stands on a quote
        # that names it. On 2026-10-11 qwen3:8b gave "에이든은 손바닥을 탁 치며 말했다"
        # for 에이든 holding '도시 중심가의 키': a real line, and no key in it (#2238).
        objects = [e for e in (other, _object_of(resolver, fact, subject)) if e is not None]
        if any(not _anchored(resolver, entity, fact) for entity in objects):
            out.drop("object_not_in_quote")
            continue
        key = (subject, fold(fact.key), fold(fact.value), other, fact.time, fact.polarity)
        at = haystack.find(fold(fact.quote))
        if key in seen:
            # Kept once, where the text says it last: a status said, changed and said
            # again ends the text as it was said last.
            out.drop("duplicate")
            earlier = out.kept[seen[key]]
            if at > earlier.at:
                out.kept[seen[key]] = ReadFact(fact, subject, other, at, alias or earlier.alias)
            elif earlier.alias is None and alias is not None:
                out.kept[seen[key]] = ReadFact(earlier.fact, subject, other, earlier.at, alias)
            continue
        seen[key] = len(out.kept)
        out.kept.append(ReadFact(fact, subject, other, at, alias))
    return out


def read_presence(raw: object, resolver: EntityResolver) -> dict[str, str] | None:
    """How a reply says each entity is in the text at its own time (`PRESENCES`), by id.

    `raw` is the reply's map of a name or id to one of `PRESENCES`. A name no entity has,
    or two share, and a word not in `PRESENCES`, are left out: nothing is guessed. One
    entity given twice is ``acting`` when either says so -- a person both remembered and
    there is there. `None` when the reply gives no map: the question was not answered,
    which is not the same as an answer that names nobody.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, str] = {}
    for name, word in cast("dict[object, object]", raw).items():
        presence = str(word or "").strip().lower()
        if presence not in PRESENCES:
            continue
        entity, _ = _entity(resolver, str(name))
        if entity is None:
            continue
        if out.get(entity) != ACTING:
            out[entity] = presence
    return out


#: The first line of every alias question, so a scripted model can tell it apart.
ALIAS_MARKER = "[alias check]"


@dataclass(frozen=True)
class AliasCandidate:
    """A name a text calls `entity` by that is none of its names yet, and where it said so."""

    entity: Entity
    name: str
    quote: str


def alias_prompt(candidates: Sequence[AliasCandidate]) -> str:
    """The question that confirms each candidate as a name of its entity (stage 4, §3.4).

    Numbered from 1, in the order given; `confirmed_aliases` reads the reply.
    """
    lines = [
        f'{number}. {", ".join([c.entity.name, *c.entity.aliases])} -- words: "{c.name}" '
        f'-- quote: "{c.quote}"'
        for number, c in enumerate(candidates, start=1)
    ]
    listed = "\n".join(lines)
    return f"""{ALIAS_MARKER}
Each line gives someone or something by the names it has, and words a text used for it in
the quote. Answer true when the words are a name of it: the same name in another script or
spelling, a short form, or a name it also goes by. Answer false when they are a pronoun, a
word for a kind of person or thing, or a description that could fit another.

{listed}

Reply with JSON only, one answer per number: {{"1": true, "2": false}}"""


def confirmed_aliases(reply: str, count: int) -> set[int]:
    """The candidates (0-based, as `alias_prompt` numbered them from 1) the reply confirms.

    Only a ``true`` (or ``"true"``, ``"yes"``) is a yes; a reply that cannot be read
    confirms none, so nothing becomes a name on a reply no one could read.
    """
    try:
        data = reply_object(reply)
    except ValueError:
        return set()
    if not isinstance(data, dict):
        return set()
    answers = cast("dict[str, object]", data)
    out: set[int] = set()
    for index in range(count):
        said = answers.get(str(index + 1))
        if said is True or (isinstance(said, str) and said.strip().lower() in ("true", "yes")):
            out.add(index)
    return out
