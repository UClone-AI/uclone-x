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
made up, must not turn it into a fact. A fact the model attributes to either, or to a kind of
line the span does not hold, is dropped before saving.

**A failure is a plain sentence, never the cause.** `ExtractionOutcome.error` is written for
the person reading the turn's row; the exception, the model's raw text and the fact ids stay
in the log.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, Protocol, runtime_checkable

from uclone_x.core.provenance import Provenance, require_provenance
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ModelResponse
from uclone_x.memory.models import (
    PERSON_SUBJECT,
    PROJECT_SUBJECT,
    SELF_SUBJECT,
    FactOrigin,
    MemoryFact,
    fact_subject,
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
        auto_retract_conflicts: bool,
        origin: FactOrigin,
        source_room_id: str | None,
        source_turn_id: str | None,
    ) -> MemoryFact: ...


@runtime_checkable
class LearningSeatProtocol(Protocol):
    """A seat that can learn: it has a memory, and a model call charged to it (#1404)."""

    @property
    def memory(self) -> FactStoreProtocol | None: ...

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


@dataclass
class _Candidate:
    subject: str
    relation: str
    value: str
    source: str
    confidence: float
    why: str = ""
    keys: tuple[str, str, str] = field(default=("", "", ""))


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
        try:
            lesson.memory.refresh()  # see a Forget or Correct made elsewhere since the load
            response = await lesson.generate(self._request(lesson))
            candidates = _parse(response.content or "")
            provenance = require_provenance(response.provenance, "The knowledge extraction reply")
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
        kinds = {line.kind for line in lesson.lines}
        kept = _curate(candidates, kinds, lesson)
        added: list[str] = []
        reinforced: list[str] = []
        superseded: list[str] = []
        try:
            memory.refresh()  # the model call awaited; a person may have acted meanwhile
            for candidate in kept:
                active = memory.list_facts(subject=candidate.subject, predicate=candidate.relation)
                same = [f for f in active if _value_key(f.object_value) == candidate.keys[2]]
                if same:
                    fact = memory.reinforce_fact(same[0].fact_id, lesson.turn_id)
                    reinforced.append(fact.fact_id)
                    continue
                corrected = [f for f in active if f.origin == "corrected"]
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
                fact = memory.record_fact(
                    subject=candidate.subject,
                    predicate=candidate.relation,
                    object_value=candidate.value,
                    provenance=provenance,
                    source_session_id=lesson.session_id,
                    confidence=min(candidate.confidence, EXTRACTED_CONFIDENCE_CAP),
                    metadata={
                        "observations": [lesson.turn_id],
                        "grounded_in": candidate.source,
                        "why": candidate.why,
                    },
                    auto_retract_conflicts=True,
                    origin=_SOURCE_ORIGIN[candidate.source],
                    source_room_id=lesson.room_id,
                    source_turn_id=lesson.turn_id,
                )
                added.append(fact.fact_id)
                superseded.extend(f.fact_id for f in active)
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
        '- [person: NAME] is what the human wrote. Use source "person".\n'
        f'- [tool: NAME] is a tool result {name} received. Use source "tool".\n'
        f"- [{name} (you)] is what {name} itself said. It is context only, never a source: "
        f"{name} agreeing with or repeating something does not make it true.\n"
        "- [other clone: NAME] is another clone's message. It is context only, never a "
        "source.\n"
        "Return only facts that a person line or a tool line states.\n\n"
        "Keep only durable, reusable facts: about the person, their projects and their "
        'preferences, and stable facts that were found. Use the subject "user" for the '
        f'person. Use the subject "{PROJECT_SUBJECT}" for durable working preferences of '
        "this workspace, such as the reply language or a code style. "
        f'Use the subject "{SELF_SUBJECT}", with source "person", when a person line tells '
        f"{name} about {name} itself: its looks, age, gender, personality, speech style, "
        f'name or likes ("you have red hair", "너는 귀엽고 착해"). Use relations such as '
        '"hair", "eyes", "age", "appearance", "personality" or "speech_style". '
        f"{name}'s own lines never define it. "
        f"Tool results and tool arguments never define it: an image prompt or file description is never a fact about {name}. "
        "Skip greetings, the task of the moment, the contents of a story (its "
        "characters, places, plot and style rules), and anything you are unsure the source "
        "asserted.\n\n"
        f"Facts {name} already holds (subject | relation | value):\n{known_block}\n\n"
        "Answer with a JSON array and nothing else. Each item is an object with the keys "
        '"subject", "relation", "value", "source" ("person" or "tool"), '
        '"confidence" (0 to 1), "durable" (true or false) and "why" (a few words). '
        'Use a short snake_case relation such as "prefers" or "works_at". '
        "If nothing is worth keeping, answer []."
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
    known.sort(key=lambda fact: (fact.created_at, fact.fact_id), reverse=True)
    return known[:MAX_KNOWN_FACTS]


# -- reading the reply -----------------------------------------------------------------

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _parse(content: str) -> list[_Candidate]:
    """The candidates in the model's reply. Raises `ExtractionFailed` when it is not a list.

    An item that is not a well-formed candidate is skipped, not fatal: one bad object in an
    otherwise usable answer should not lose the rest.
    """
    text = _FENCE.sub("", content.strip())
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        raise ExtractionFailed("the reply holds no JSON array")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionFailed(f"the reply's array is not JSON ({exc.msg})") from exc
    if not isinstance(data, list):
        raise ExtractionFailed("the reply's JSON is not an array")
    candidates: list[_Candidate] = []
    for item in data:  # pyright: ignore[reportUnknownVariableType]
        candidate = _candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate(item: Any) -> _Candidate | None:
    if not isinstance(item, dict):
        return None
    fields: dict[str, Any] = item  # pyright: ignore[reportUnknownVariableType]
    subject, relation, value = fields.get("subject"), fields.get("relation"), fields.get("value")
    source, confidence = fields.get("source"), fields.get("confidence")
    if not all(isinstance(part, str) and part.strip() for part in (subject, relation, value)):
        return None
    if source not in _SOURCE_ORIGIN:
        return None
    if fields.get("durable") is not True:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return None
    why = fields.get("why")
    return _Candidate(
        subject=str(subject),
        relation=str(relation),
        value=str(value),
        source=str(source),
        confidence=max(0.0, min(1.0, float(confidence))),
        why=why.strip()[:200] if isinstance(why, str) else "",
    )


# -- curation --------------------------------------------------------------------------


def _curate(candidates: Sequence[_Candidate], kinds: set[str], lesson: Lesson) -> list[_Candidate]:
    """Durable gate, grounding check, normalisation and in-reply de-duplication (§3.3 step 5)."""
    kept: list[_Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if candidate.confidence < MIN_CONFIDENCE:
            continue
        if candidate.source not in kinds:
            # The model named a source the span does not hold: a claim with nothing under it.
            logger.info(
                "Clone %s: extracted fact dropped; its source %r is not in the span",
                lesson.clone_id,
                candidate.source,
            )
            continue
        subject = fact_subject(candidate.subject, lesson.person_names, lesson.clone_names)
        if subject == SELF_SUBJECT and candidate.source != "person":
            logger.info(
                "Clone %s: self fact dropped; self facts can only be grounded in person speech, not %r",
                lesson.clone_id,
                candidate.source,
            )
            continue
        relation = "_".join(candidate.relation.casefold().split())
        value = " ".join(candidate.value.split())
        keys = (subject.casefold(), relation, _value_key(value))
        if keys in seen:
            continue
        seen.add(keys)
        kept.append(replace(candidate, subject=subject, relation=relation, value=value, keys=keys))
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
