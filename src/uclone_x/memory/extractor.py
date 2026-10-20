"""Learning from a conversation after the turn: the clone knowledge graph's writer (#1404).

`KnowledgeExtractor` reads what a clone was shown and did in a committed room turn, asks the
clone's own model once for the durable facts in it, curates them and saves them directly into
the clone's `CrossSessionMemory` (clone-knowledge-graph design §3.3). There is no approval
queue: a person corrects or forgets a saved fact afterwards (owner ruling 2026-09-27).

**Off the reply path.** Nothing here runs inside `execute_turn`. The room queues a
`Lesson` when a turn commits and hands the queue to `run` once the room's cascade has ended,
behind the reply (P3, context-assembly C1). A cascade in which one clone spoke three times is
one lesson and one model call (`submit` coalesces by room and clone).

**What may ground a fact.** The human owner's words (`told`) and a tool result the clone
received (`found`). The clone's own statements and another clone's messages are shown to the
model as context only (design §3.3): a clone repeating what another clone said, or what it
made up, must not turn it into a fact. Each fact carries the words it rests on, and they
must be in a person or tool line; which of the two holds them is its source. A quote only in
a clone's line, or in no line, is dropped before saving.

**Read and verified as every vessel reads (uGraph §3.5, #2235).** Only the prompt is the
clone's. The reply is read by `knowledge.extraction` -- its JSON, a reply cut off by its
token cap salvaged, a malformed one field by field (`listed_facts`, `parse_facts`) -- and
held to the span and to the clone's subjects by `verify_facts`: the quote must be in the
span, a name resolves through an `EntityResolver` over the person (`user`), the clone
(`self`), the workspace (`project`) and every subject the clone already holds, a name two of
them share is dropped and not guessed, and a fact given twice is kept once.

**Names (the clone's §3.4 policy).** A name resolves exactly (folded, an honorific dropped)
to a subject the clone holds, or it is a new subject, filed as written. A near match --
`지현` beside a held `김지현` -- is not merged: the clone has no alias record yet to show a
person or undo a merge with, and learning runs with no one to ask. Merging near names is
left to the clone store's step, with the open design question it carries (uGraph §6 Q2).

**A failure is a plain sentence, never the cause.** `ExtractionOutcome.error` is written for
the person reading the turn's row; the exception, the model's raw text and the fact ids stay
in the log.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, Protocol, cast, runtime_checkable

from uclone_x.core.models import PersonaDefinition
from uclone_x.core.provenance import Provenance, require_provenance
from uclone_x.errors import NoRoomToLearnError
from uclone_x.knowledge.extraction import ReadFact, listed_facts, parse_facts, verify_facts
from uclone_x.knowledge.fold import fold, relation_key
from uclone_x.knowledge.models import Entity
from uclone_x.knowledge.quotes import quote_found
from uclone_x.knowledge.resolver import EntityResolver, name_key
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ModelResponse
from uclone_x.memory.models import (
    PERSON_SUBJECT,
    PERSON_WORDS,
    PROJECT_SUBJECT,
    SELF_SUBJECT,
    SELF_WORDS,
    FactOrigin,
    MemoryFact,
    fold_name,
)

__all__ = [
    "EXTRACTION_FAILED",
    "EXTRACTED_CONFIDENCE_CAP",
    "ExtractionOutcome",
    "KnowledgeExtractor",
    "LearningSeatProtocol",
    "Lesson",
    "SpanLine",
]

logger = logging.getLogger(__name__)

#: What the turn's row says when learning from it failed (design §3.3 step 8). No cause, no id.
EXTRACTION_FAILED: Final = "Could not pick up what to remember from this turn."

#: How the extraction call's system message opens, so a scripted connector can tell it from a
#: turn's own call and answer it (the E2E mocks answer `[]`).
INSTRUCTIONS_OPENING: Final = "You pick out durable facts"

#: An extracted fact is never held as surely as a person's correction (design §3.2).
EXTRACTED_CONFIDENCE_CAP: Final = 0.8
#: Below this, or not marked durable, a candidate is dropped (design §3.3 step 5).
MIN_CONFIDENCE: Final = 0.5
#: Known facts sent with the span, so the model can see what is already held (design §3.3).
MAX_KNOWN_FACTS: Final = 20
#: Candidates kept from one extraction; the rest are dropped and the drop is logged.
MAX_FACTS_PER_LESSON: Final = 12
#: Characters of one span line, and of the whole span, sent to the model. The newest lines
#: are kept, so the message the turn answered is always there.
MAX_LINE_CHARS: Final = 4000
MAX_SPAN_CHARS: Final = 16000

#: Tools whose results are never a clone fact: a work's contents belong to the work (§3.4),
#: and the memory tools' own results would echo facts back as new ones.
EXCLUDED_TOOL_PREFIXES: Final = (
    "story_",
    "character_sheet",
    "record_memory_fact",
    "retract_memory_fact",
    "query_memory_facts",
    "generate_image",
    "comfy_image_gen",
)

#: Who a span line is from. Only `person` and `tool` lines can ground a fact; `self` (the
#: clone's own words) and `other` (another clone's) are context only.
LineKind = Literal["person", "tool", "self", "other"]

_SOURCE_ORIGIN: Final[dict[str, FactOrigin]] = {"person": "told", "tool": "found"}


class FactStoreProtocol(Protocol):
    """The part of a clone's memory the extractor uses; `CrossSessionMemory` is the one.

    A protocol, not the class, because the store is an adapter (it writes the file) and this
    module is kernel: it decides what to save and leaves the writing to the store.
    """

    def refresh(self) -> None: ...

    def list_facts(
        self, *, subject: str | None = None, predicate: str | None = None
    ) -> list[MemoryFact]: ...

    def reinforce_fact(self, fact_id: str, turn_id: str) -> MemoryFact: ...

    def record_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        provenance: Provenance,
        source_session_id: str,
        *,
        confidence: float,
        metadata: dict[str, Any],
        origin: FactOrigin,
        source_room_id: str | None,
        source_turn_id: str | None,
        single_valued: bool,
        replaces: Sequence[str],
    ) -> MemoryFact: ...


@runtime_checkable
class LearningSeatProtocol(Protocol):
    """A seat that can learn: it has a memory, and a model call charged to it (#1404).

    `latest_persona_definition` carries the clone's "Learn from conversations" switch
    (#1840), including a saved edit the seat has not taken yet (#2228); `None` is a seat
    with no persona, which learns.
    """

    @property
    def memory(self) -> FactStoreProtocol | None: ...

    @property
    def latest_persona_definition(self) -> PersonaDefinition | None: ...

    async def invoke_auxiliary_model(self, request: LLMRequest) -> ModelResponse: ...


@dataclass(frozen=True)
class SpanLine:
    """One thing the clone saw or did in the turn, labelled by who it came from."""

    kind: LineKind
    speaker: str
    text: str


@dataclass(frozen=True)
class Lesson:
    """One clone's committed turn (or turns, coalesced) in one room, ready to learn from.

    `generate` is the clone's own model call, budget-checked and charged to the clone
    (`BaseAgent.invoke_auxiliary_model`). `person_names` are the human owner's id, display
    name and aliases, so a fact the model files under the person's name lands under `user`;
    the room leaves out a name another participant shares (#1868, `_person_names`).
    `clone_names` are this clone's id, display name and aliases, so a fact the model files
    under the clone's own name lands under `self` (#2016); the room leaves these out on the
    same rule (`_clone_names`).
    """

    clone_id: str
    clone_name: str
    room_id: str
    session_id: str
    turn_id: str
    lines: tuple[SpanLine, ...]
    memory: FactStoreProtocol
    generate: Callable[[LLMRequest], Awaitable[ModelResponse]]
    person_names: tuple[str, ...] = ()
    clone_names: tuple[str, ...] = ()
    #: Every turn this lesson covers, oldest first; `turn_id` is the last of them.
    turn_ids: tuple[str, ...] = ()
    #: Whether the clone still learns from conversations, read when the lesson is learned
    #: rather than when it was queued, so switching learning off stops a queued lesson too
    #: (#2228).
    learns: Callable[[], bool] = field(default=lambda: True)
    #: The relations the clone's rules declare functional, read when the lesson is learned
    #: (#2332). A new value under one of them replaces the current ones.
    functional: Callable[[], frozenset[str]] = field(default=lambda: frozenset[str]())


@dataclass(frozen=True)
class ExtractionOutcome:
    """What one extraction did, for the turn's row and the `knowledge.updated` event."""

    clone_id: str
    room_id: str
    turn_id: str
    added: tuple[str, ...] = ()
    reinforced: tuple[str, ...] = ()
    superseded: tuple[str, ...] = ()
    error: str | None = None
    #: Not learned, to keep what is left of the budget for the person's next reply
    #: (#1851). Not a failure; the turn's row says it in plain words (#2328).
    skipped: bool = False


@dataclass
class _Candidate:
    subject: str
    relation: str
    value: str
    source: str
    confidence: float
    why: str = ""
    keys: tuple[str, str, str] = field(default=("", "", ""))
    #: The held value the model says this fact replaces, as `_value_key` keys it; empty when
    #: it replaces nothing (#2224).
    replaces: str = ""
    #: The words of the person or tool line the fact rests on, as the model quoted them.
    quote: str = ""
    #: The alias the reply used for `subject`, when `subject` merged an earlier name (uGraph §6 Q2).
    merged_from: str = ""


class ExtractionFailed(Exception):
    """The model's reply could not be read as extracted facts. Internal; never shown."""


class KnowledgeExtractor:
    """Queues lessons per room and clone, and turns each into saved facts."""

    def __init__(self) -> None:
        #: room id -> clone id -> the lesson waiting for that clone in that room.
        self._pending: dict[str, dict[str, Lesson]] = {}

    # -- the queue -------------------------------------------------------------------

    def submit(self, lesson: Lesson) -> None:
        """Queue `lesson`, folding it into one already waiting for the same clone and room.

        Folded, the lines are joined in order and the newest turn is the one the facts and
        the row line name: the extraction reads the whole cascade once, after it ends.
        """
        waiting = self._pending.setdefault(lesson.room_id, {})
        held = waiting.get(lesson.clone_id)
        turns = lesson.turn_ids or (lesson.turn_id,)
        if held is None:
            waiting[lesson.clone_id] = replace(lesson, turn_ids=turns)
            return
        waiting[lesson.clone_id] = replace(
            lesson,
            lines=(*held.lines, *lesson.lines),
            turn_ids=(*held.turn_ids, *turns),
        )

    def take(self, room_id: str) -> Lesson | None:
        """The next lesson waiting in `room_id`, removed from the queue; `None` when none is."""
        waiting = self._pending.get(room_id)
        if not waiting:
            self._pending.pop(room_id, None)
            return None
        clone_id = next(iter(waiting))
        lesson = waiting.pop(clone_id)
        if not waiting:
            del self._pending[room_id]
        return lesson

    def has_pending(self, room_id: str) -> bool:
        """Whether a lesson is waiting in `room_id`."""
        return bool(self._pending.get(room_id))

    # -- one extraction --------------------------------------------------------------

    async def extract(self, lesson: Lesson) -> ExtractionOutcome:
        """Ask the clone's model for the lesson's durable facts, curate them and save them.

        Never raises for a failure to learn (cancellation propagates): a model error, a
        reply that is not facts, and a failed save each come back as an outcome carrying
        `EXTRACTION_FAILED`, with the cause logged. Nothing is retried.
        """
        outcome = ExtractionOutcome(
            clone_id=lesson.clone_id, room_id=lesson.room_id, turn_id=lesson.turn_id
        )
        if not any(line.kind in _SOURCE_ORIGIN for line in lesson.lines):
            # Nothing in the span can ground a fact, so the model is not asked.
            return outcome
        if not lesson.learns():  # switched off since the lesson was queued
            return outcome
        try:
            lesson.memory.refresh()  # see a Forget or Correct made elsewhere since the load
            response = await lesson.generate(self._request(lesson))
            candidates = _read(response.content or "", lesson)
            provenance = require_provenance(response.provenance, "The knowledge extraction reply")
        except NoRoomToLearnError:
            # Skipped, not failed (#1851): what is left of the clone's budget is the person's
            # next reply's. Nothing was sent; the row says learning waited, not that it failed.
            logger.info(
                "Clone %s: learning from turn %s skipped; it would leave less than a reply's "
                "worth of the budget",
                lesson.clone_id,
                lesson.turn_id,
            )
            return replace(outcome, skipped=True)
        except Exception as exc:  # any failure to get facts back is the row's one sentence
            logger.warning(
                "Knowledge extraction for clone %s in room %s (turn %s) failed: %s: %s",
                lesson.clone_id,
                lesson.room_id,
                lesson.turn_id,
                type(exc).__name__,
                exc,
            )
            return replace(outcome, error=EXTRACTION_FAILED)
        if not lesson.learns():  # turned off while the model was asked
            return outcome
        # From here to the last save there is no `await`, so on one event loop two
        # extractions for the same clone never interleave their reads and writes.
        return self._save(lesson, candidates, provenance, outcome)

    def _request(self, lesson: Lesson) -> LLMRequest:
        return LLMRequest(
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content=_instructions(lesson)),
                ChatMessage(role=MessageRole.USER, content=_span_text(lesson)),
            ),
            temperature=0.0,
            max_tokens=1500,
            thinking=False,
            auto_compact=False,
        )

    def _save(
        self,
        lesson: Lesson,
        candidates: list[_Candidate],
        provenance: Provenance,
        outcome: ExtractionOutcome,
    ) -> ExtractionOutcome:
        memory = lesson.memory
        kept = _curate(candidates, lesson)
        added: list[str] = []
        reinforced: list[str] = []
        superseded: list[str] = []
        try:
            functional = lesson.functional()
            memory.refresh()  # the model call awaited; a person may have acted meanwhile
            for candidate in kept:
                active = memory.list_facts(subject=candidate.subject, predicate=candidate.relation)
                same = [f for f in active if _value_key(f.object_value) == candidate.keys[2]]
                if same:
                    fact = memory.reinforce_fact(same[0].fact_id, lesson.turn_id)
                    reinforced.append(fact.fact_id)
                    continue
                replaced = _replaced(memory.list_facts(subject=candidate.subject), candidate)
                if candidate.relation in functional:
                    # One value at a time, as a person declared (#2332): every current value
                    # under the relation is replaced, named or not.
                    named = {f.fact_id for f in replaced}
                    replaced += [
                        f for f in active if f.valid_until is None and f.fact_id not in named
                    ]
                corrected = [f for f in (*active, *replaced) if f.origin == "corrected"]
                if corrected:
                    # A person's correction outranks the model (design §3.3 step 5).
                    logger.info(
                        "Clone %s: extracted %r %r dropped; fact %s was corrected by a person",
                        lesson.clone_id,
                        candidate.subject,
                        candidate.relation,
                        corrected[0].fact_id,
                    )
                    continue
                fact_meta: dict[str, Any] = {
                    "observations": [lesson.turn_id],
                    "grounded_in": candidate.source,
                    "why": candidate.why,
                    # Shown in Remembers' history: the words it was learned from (#2332).
                    "quote": candidate.quote,
                }
                if candidate.merged_from:
                    fact_meta["merged_from"] = candidate.merged_from
                    fact_meta["merged_moved"] = True
                fact = memory.record_fact(
                    subject=candidate.subject,
                    predicate=candidate.relation,
                    object_value=candidate.value,
                    provenance=provenance,
                    source_session_id=lesson.session_id,
                    confidence=min(candidate.confidence, EXTRACTED_CONFIDENCE_CAP),
                    metadata=fact_meta,
                    origin=_SOURCE_ORIGIN[candidate.source],
                    source_room_id=lesson.room_id,
                    source_turn_id=lesson.turn_id,
                    # Only what the model named as replaced is superseded: asked instead whether
                    # a relation holds one value, the model said yes for every one (#2224).
                    single_valued=False,
                    replaces=tuple(f.fact_id for f in replaced),
                )
                added.append(fact.fact_id)
                superseded.extend(f.fact_id for f in replaced)
        except Exception as exc:  # a save that failed part-way still reports what landed
            logger.warning(
                "Knowledge extraction for clone %s in room %s (turn %s) could not save: %s: %s",
                lesson.clone_id,
                lesson.room_id,
                lesson.turn_id,
                type(exc).__name__,
                exc,
            )
            return replace(
                outcome,
                added=tuple(added),
                reinforced=tuple(reinforced),
                superseded=tuple(superseded),
                error=EXTRACTION_FAILED,
            )
        return replace(
            outcome,
            added=tuple(added),
            reinforced=tuple(reinforced),
            superseded=tuple(superseded),
        )


# -- the prompt ------------------------------------------------------------------------


def _instructions(lesson: Lesson) -> str:
    name = lesson.clone_name
    known = _known_facts(lesson)
    known_block = (
        "\n".join(f"- {fact.subject} | {fact.predicate} | {fact.object_value}" for fact in known)
        if known
        else "(none)"
    )
    return (
        f"{INSTRUCTIONS_OPENING} that {name}, an AI clone, should remember from one "
        "exchange. You do not reply to anyone.\n\n"
        "Each line of the exchange starts with a label:\n"
        "- [person: NAME] is what the human wrote.\n"
        f"- [tool: NAME] is a tool result {name} received.\n"
        f"- [{name} (you)] is what {name} itself said. It is context only, never a source: "
        f"{name} agreeing with or repeating something does not make it true.\n"
        "- [other clone: NAME] is another clone's message. It is context only, never a "
        "source.\n"
        "Return only facts that a person line or a tool line states, each with the words of "
        "that line that state it.\n\n"
        "Keep only durable, reusable facts: about the person, their projects and their "
        'preferences, and stable facts that were found. Use the subject "user" for the '
        f'person. Use the subject "{PROJECT_SUBJECT}" for durable working preferences of '
        "this workspace, such as the reply language or a code style. "
        f'Use the subject "{SELF_SUBJECT}" when a person line tells '
        f"{name} about {name} itself: its looks, age, gender, personality, speech style, "
        f'name or likes ("you have red hair", "너는 귀엽고 착해"). Use relations such as '
        '"hair", "eyes", "age", "appearance", "personality" or "speech_style". '
        f"{name}'s own lines never define it. "
        f"Tool results and tool arguments never define it: an image prompt or file description is never a fact about {name}. "
        "Skip greetings, the task of the moment, the contents of a story (its "
        "characters, places, plot and style rules), and anything you are unsure the source "
        "asserted.\n\n"
        f"Facts {name} already holds (subject | relation | value):\n{known_block}\n\n"
        'Answer with a JSON object and nothing else: {"facts": [...]}. Each fact is an '
        'object with the keys "subject", "key" (a short snake_case relation such as '
        '"prefers" or "works_at"), "value", "quote" (the words of the person or tool line '
        'that state it, copied exactly, a few words at least), "confidence" (0 to 1), '
        '"durable" (true or false), "replaces" (only when the line says a fact already held '
        "has stopped being true, as when someone moved, changed jobs or stopped doing "
        "something: the held value it replaces, copied from the list above; otherwise null) "
        'and "why" (a few words). '
        'If nothing is worth keeping, answer {"facts": []}.'
    )


def _span_text(lesson: Lesson) -> str:
    labelled = [
        _label(line, lesson.clone_name) + " " + line.text[:MAX_LINE_CHARS] for line in lesson.lines
    ]
    kept: list[str] = []
    spent = 0
    for line in reversed(labelled):
        if kept and spent + len(line) > MAX_SPAN_CHARS:
            break
        kept.append(line)
        spent += len(line) + 1
    return "\n".join(reversed(kept))


def _label(line: SpanLine, clone_name: str) -> str:
    if line.kind == "person":
        return f"[person: {line.speaker}]"
    if line.kind == "tool":
        return f"[tool: {line.speaker}]"
    if line.kind == "self":
        return f"[{clone_name} (you)]"
    return f"[other clone: {line.speaker}]"


def _known_facts(lesson: Lesson) -> list[MemoryFact]:
    """Up to `MAX_KNOWN_FACTS` active facts about the person, the clone itself, or a subject
    the span names. The clone's own facts are shown so that "no, blue hair" reuses the
    relation "red hair" was filed under, and supersedes it.

    Both sides are folded (`fold_name`), so a subject stored decomposed is found in a span
    that writes it composed, and the reverse (#1899).
    """
    text = fold_name(" ".join(line.text for line in lesson.lines))
    known = [
        fact
        for fact in lesson.memory.list_facts()
        if (subject := fold_name(fact.subject)) in (PERSON_SUBJECT, SELF_SUBJECT) or subject in text
    ]
    for fact in lesson.memory.list_facts():
        if (
            fact not in known
            and isinstance(merged := fact.metadata.get("merged_from"), str)
            and bool(merged_folded := fold_name(merged))
            and merged_folded in text
        ):
            known.append(fact)
    known.sort(key=lambda fact: (fact.created_at, fact.fact_id), reverse=True)
    return known[:MAX_KNOWN_FACTS]


# -- reading the reply -----------------------------------------------------------------


def _read(content: str, lesson: Lesson) -> list[_Candidate]:
    """The reply's facts, read and verified as every vessel's are (`knowledge.extraction`),
    with the clone's own fields. Raises `ExtractionFailed` when no fact list can be read.

    An item that is not an object is skipped, not fatal: one bad item in an otherwise usable
    answer should not lose the rest. What `verify_facts` drops is logged by reason.
    """
    try:
        listed = listed_facts(content)
    except ValueError as exc:
        raise ExtractionFailed(str(exc)) from exc
    items = [cast("dict[str, Any]", item) for item in listed if isinstance(item, dict)]
    # The clone's gates before duplicates fold: of two copies, the one kept passes them
    # (#2240 item 3).
    items = [item for item in items if _passes_gates(item)]
    facts = parse_facts(items)
    # The relation as it is filed, before verifying: "works at" and "works_at" are one fact.
    facts = [replace(f, key=relation_key(f.key)) for f in facts]
    grounds = [line for line in lesson.lines if line.kind in _SOURCE_ORIGIN]
    verified = verify_facts(
        facts, "\n".join(line.text for line in grounds), _resolver(lesson), new_names=True
    )
    if verified.dropped:
        logger.info(
            "Clone %s: extracted facts dropped in verifying them: %s",
            lesson.clone_id,
            verified.dropped,
        )
    fields = {id(fact): item for fact, item in zip(facts, items, strict=True)}
    candidates: list[_Candidate] = []
    for read in verified.kept:
        candidate = _candidate(read, fields[id(read.fact)], grounds)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _passes_gates(item: dict[str, Any]) -> bool:
    """Whether a fact the reply gave is durable and sure enough to keep (§3.3 step 5).

    Read on the reply's own object, before `verify_facts` folds a fact given twice into one:
    folded first, the later copy was kept, and a later copy marked not durable lost a fact
    the earlier one said was durable (#2240 item 3).
    """
    confidence = item.get("confidence")
    if item.get("durable") is not True:
        return False
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return False
    return confidence >= MIN_CONFIDENCE


def _resolver(lesson: Lesson) -> EntityResolver:
    """The clone's subjects (§3.4): the person, the clone, the workspace, and each subject
    it holds a fact about. A held subject that is one of the first three's names is theirs,
    as it would be filed, not a fourth that shares the name."""
    scope = f"clone:{lesson.clone_id}"
    # The person first, as `fact_subject` files a fact for `record_memory_fact`: a name the
    # person and the clone share is the person's, so what the person tells about
    # themselves is never filed as the clone's, nor lost as ambiguous (#2240 item 4).
    person = {name_key(n) for n in (*PERSON_WORDS, *lesson.person_names)}
    clone = tuple(n for n in (*SELF_WORDS, *lesson.clone_names) if name_key(n) not in person)
    named = (
        Entity(
            PERSON_SUBJECT, scope, "person", PERSON_SUBJECT, (*PERSON_WORDS, *lesson.person_names)
        ),
        Entity(SELF_SUBJECT, scope, "self", SELF_SUBJECT, clone),
        Entity(PROJECT_SUBJECT, scope, "project", PROJECT_SUBJECT),
    )
    taken = {name_key(n) for e in named for n in (e.id, e.name, *e.aliases)}
    held_facts = lesson.memory.list_facts()
    held = {fact.subject for fact in held_facts}
    aliases_by_subject: dict[str, list[str]] = {}
    for fact in held_facts:
        merged = fact.metadata.get("merged_from")
        if isinstance(merged, str) and merged.strip() and name_key(merged) not in taken:
            aliases = aliases_by_subject.setdefault(fact.subject, [])
            if merged.strip() not in aliases:
                aliases.append(merged.strip())
    subjects = [
        *(Entity(s, scope, "subject", s) for s in sorted(held) if name_key(s) not in taken),
    ]
    subjects = [
        replace(e, aliases=tuple(aliases_by_subject[e.id])) if e.id in aliases_by_subject else e
        for e in subjects
    ]
    return EntityResolver(scope, (*named, *subjects))


def _source(quote: str, grounds: Sequence[SpanLine]) -> str:
    """The kind of line the quote is in, the person's words before a tool's result.

    `verify_facts` has found the quote in the lines joined, so it is in one of them, or it
    runs across two: then it is the line the quote starts in.
    """
    for line in sorted(grounds, key=lambda line: line.kind != "person"):
        if quote_found(quote, line.text):
            return line.kind
    at = fold(" ".join(line.text for line in grounds)).find(fold(quote))
    for line in grounds:
        size = len(fold(line.text)) + 1
        if at < size:
            return line.kind
        at -= size
    return grounds[-1].kind


def _candidate(
    read: ReadFact, item: dict[str, Any], grounds: Sequence[SpanLine]
) -> _Candidate | None:
    fact = read.fact
    if not fact.checked or not fact.value.strip() or not fact.key:
        return None
    source = _source(fact.quote, grounds)
    confidence = cast("float", item["confidence"])  # `_passes_gates` read it as a number
    why = item.get("why")
    value = " ".join(fact.value.split())
    replaces = item.get("replaces")
    raw_subj = " ".join(fact.subject.split())
    merged_from = (
        raw_subj
        if read.subject not in (PERSON_SUBJECT, SELF_SUBJECT, PROJECT_SUBJECT)
        and name_key(raw_subj) != name_key(read.subject)
        else ""
    )
    return _Candidate(
        subject=read.subject,
        relation=fact.key,
        value=value,
        source=source,
        confidence=max(0.0, min(1.0, float(confidence))),
        why=why.strip()[:200] if isinstance(why, str) else "",
        keys=(read.subject.casefold(), fact.key, _value_key(value)),
        replaces=_value_key(replaces) if isinstance(replaces, str) else "",
        quote=fact.quote,
        merged_from=merged_from,
    )


def _replaced(held: Sequence[MemoryFact], candidate: _Candidate) -> list[MemoryFact]:
    """The held facts about the candidate's subject that it says it replaces (#2224, #2328).

    The named value under the candidate's own relation, or, only while that relation holds
    nothing, under any other: the model may file the new value under another key
    ("learning Rust", then "current_language Zig"), but "moved from Seoul" must not replace
    "likes Seoul". A fact with a stated end is history and is never replaced.

    The model is not asked to name the relation too: asked for `{"key", "value"}`, the
    local model replaced the person's home with their sister's new one (#2328).
    """
    if not candidate.replaces or candidate.replaces == candidate.keys[2]:
        return []
    live = [f for f in held if f.valid_until is None]
    named = [f for f in live if _value_key(f.object_value) == candidate.replaces]
    if any(relation_key(f.predicate) == candidate.relation for f in live):
        return [f for f in named if relation_key(f.predicate) == candidate.relation]
    return named


# -- curation --------------------------------------------------------------------------


def _curate(candidates: Sequence[_Candidate], lesson: Lesson) -> list[_Candidate]:
    """The clone's own gates on verified facts (§3.3 step 5): a fact about the clone itself
    only from the person, and no more than one lesson's share. Durable and sure enough are
    read before verifying (`_passes_gates`)."""
    kept: list[_Candidate] = []
    for candidate in candidates:
        if candidate.subject == SELF_SUBJECT and candidate.source != "person":
            logger.info(
                "Clone %s: self fact dropped; self facts can only be grounded in person speech, not %r",
                lesson.clone_id,
                candidate.source,
            )
            continue
        kept.append(candidate)
    if len(kept) > MAX_FACTS_PER_LESSON:
        logger.info(
            "Clone %s: %d extracted facts over the limit of %d were dropped",
            lesson.clone_id,
            len(kept) - MAX_FACTS_PER_LESSON,
            MAX_FACTS_PER_LESSON,
        )
    return kept[:MAX_FACTS_PER_LESSON]


def _value_key(value: str) -> str:
    return " ".join(value.split()).casefold()
