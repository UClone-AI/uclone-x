"""Relevance-ranked recall: which of a clone's facts a turn's request carries.

Design: the clone knowledge graph design, §3.5, and the clone self and scenes design, §3.3.
The section a turn sends is

0. the facts whose subject is the clone itself (`self`), last observed first, up to
   `SELF_FACTS_LIMIT`, under their own line, which says they override the persona
   description where the two disagree (#2016). They are not counted against
   `max_facts_in_prompt`: that budget is the person's, the project's and the ranked facts',
   and a clone with eight facts about itself must not lose the person's;
1. the facts whose subject is the person (`user`), up to `USER_FACTS_LIMIT`. When there are
   more, the ones the message is about come first, ranked as item 3 ranks, and the rest
   are filled last observed first (#2224);
2. then the facts whose subject is the project the clone works on (`project`), last
   observed first, up to `PROJECT_FACTS_LIMIT`: standing preferences such as "reply in
   Korean" that a short message ("continue") shares no word with (#1857);
3. then the clone's other facts, ranked against the turn's message by `search_facts` --
   embedding similarity when the store has an embedder, lexical overlap when it has none;
4. bounded at the store's `max_facts_in_prompt`, with a line counting what was not selected.

"Last observed" is `updated_at`: a fact seen again is reinforced (`reinforce_fact`), which
moves it, so a fact the person keeps repeating stays in front of one said once since.

It is computed **once per turn**, before the first request, and held on the live session for
every step of that turn (`BaseAgent`'s turn executor). The ranking depends on the message, so
the section changes between turns; it goes in the `[Turn Context]` block at the tail of the
request and never in the system turn, where a per-turn change would discard the cached prefix
(the LLM request layering design, §5, layer 5). Holding it for the turn keeps that tail
the same on every step of the turn (§5.5).

No LLM call is made here (context-assembly C1). An embedder call is made only when the store
was given an embedder. If that call fails, the turn is not failed: the facts are ranked
lexically instead, and the section says so (P6).
"""

from __future__ import annotations

import logging
from typing import Protocol

from uclone_x.knowledge.fold import relation_key
from uclone_x.memory.models import (
    PERSON_SUBJECT,
    PROJECT_SUBJECT,
    SELF_SUBJECT,
    MemoryFact,
    fold_name,
)
from uclone_x.memory.retrieval import FactRanking, rank_facts

logger = logging.getLogger(__name__)

#: The subject the extractor files facts about the person under (design §3.3, step 5).
USER_SUBJECT = PERSON_SUBJECT
#: At most this many facts about the clone itself lead the section, outside
#: `max_facts_in_prompt` (#2016).
SELF_FACTS_LIMIT = 8
#: At most this many facts about the person follow the clone's own (design §3.5, item 1).
USER_FACTS_LIMIT = 5
#: At most this many facts about the project follow the person's (design §3.5, item 2).
PROJECT_FACTS_LIMIT = 3
#: Facts below this confidence are not recalled; the same floor `format_prompt_section` uses.
RECALL_MIN_CONFIDENCE = 0.5

MEMORY_SECTION_HEADER = "[Cross-Session Memory Facts]"
#: Marks the value said last when one relation holds several current values (#2328).
LATEST_MARK = "said last"
RECALL_INTRO = "Facts about the user and the project, then the facts most relevant to this message:"
SELF_INTRO = (
    "Facts about you, the clone, as the person defined you. Where one disagrees with your "
    "persona description, the fact wins:"
)
EMBEDDER_FAILED_METHOD = (
    "lexical overlap (the embedder failed this turn — paraphrases will not match)"
)


class RecallSource(Protocol):
    """What recall reads from a fact store. `CrossSessionMemory` is one.

    Named here rather than imported so this kernel module does not depend on the store,
    which is an adapter (it owns the file on disk).
    """

    @property
    def max_facts_in_prompt(self) -> int: ...

    def list_facts(
        self, include_retracted: bool = ..., *, min_confidence: float = ...
    ) -> list[MemoryFact]: ...

    async def search_facts(
        self, query: str, top_k: int = ..., *, min_confidence: float = ...
    ) -> FactRanking: ...

    def format_prompt_section(self, *, min_confidence: float = ...) -> str: ...


def _last_seen(fact: MemoryFact) -> tuple[str, str, str]:
    """When `fact` was last observed: `updated_at`, which a reinforcement moves (#2224).

    The filing time and the id only break ties, so the order is the same on every call.
    """
    return (fact.updated_at, fact.created_at, fact.fact_id)


def _latest_about(active: list[MemoryFact], subject: str, limit: int) -> list[MemoryFact]:
    """Up to `limit` of the `active` facts whose subject is `subject`, last observed first.

    Compared folded (`fold_name`: NFC, whitespace collapsed, casefolded), as every
    other subject lookup is (#1899).
    """
    wanted = fold_name(subject)
    return sorted(
        (fact for fact in active if fold_name(fact.subject) == wanted),
        key=_last_seen,
        reverse=True,
    )[: max(0, limit)]


def _relevant_first(facts: list[MemoryFact], ranking: FactRanking) -> list[MemoryFact]:
    """`facts` in `ranking`'s order, then the ones it did not rank in their own order."""
    ids = {fact.fact_id for fact in facts}
    ranked = [entry.fact for entry in ranking.ranked if entry.fact.fact_id in ids]
    seen = {fact.fact_id for fact in ranked}
    return [*ranked, *(fact for fact in facts if fact.fact_id not in seen)]


async def _rank(memory: RecallSource, message: str, active: list[MemoryFact]) -> FactRanking:
    """Rank every active fact against `message`, through the store's own `search_facts`.

    The whole ranking is kept: the person's facts are picked from it, then the rest.
    """
    try:
        return await memory.search_facts(
            message, top_k=len(active), min_confidence=RECALL_MIN_CONFIDENCE
        )
    except Exception as exc:  # the turn goes on without the embedder; the section says so
        logger.warning(
            "Memory recall could not use the embedder (%s); ranking lexically.",
            type(exc).__name__,
        )
        ranking = await rank_facts(query=message, facts=active, top_k=len(active))
        return FactRanking(
            ranked=ranking.ranked, method=EMBEDDER_FAILED_METHOD, considered=ranking.considered
        )


def _fact_lines(facts: list[MemoryFact], functional: frozenset[str]) -> list[str]:
    """One line per fact, except that the current values of one subject's relation share
    a line, the last observed first and marked so (#2328). Of a relation the clone's rules
    declare functional, only the value said last is shown (#2332).

    Two current values under one relation are either several that hold at once ("prefers
    jazz", "prefers Korean") or a change the extractor was not told replaced the old one
    ("lives_in Busan", then "lives_in Seoul"). Nothing stored tells the two apart, so both
    are shown, and the mark says which was said last. A fact with a stated end keeps its
    own line.
    """
    groups: dict[tuple[str, str], list[MemoryFact]] = {}
    for fact in facts:
        key = (
            (fold_name(fact.subject), fold_name(fact.predicate))
            if fact.valid_until is None
            else ("", fact.fact_id)
        )
        groups.setdefault(key, []).append(fact)
    lines: list[str] = []
    for group in groups.values():
        if len(group) == 1:
            lines.append(f"- {group[0].summary()}")
            continue
        latest, *earlier = sorted(group, key=_last_seen, reverse=True)
        if relation_key(latest.predicate) in functional:
            lines.append(f"- {latest.summary()}")
            continue
        values = "; ".join(fact.object_value for fact in earlier)
        lines.append(
            f"- {latest.subject}: {latest.predicate} -> {latest.object_value} "
            f"({LATEST_MARK}); {values}"
        )
    return lines


async def recall_prompt_section(
    memory: RecallSource, message: str, functional: frozenset[str] = frozenset()
) -> str:
    """The memory section for a turn whose message is `message` (design §3.5).

    Returns what `format_prompt_section` returns when no fact is active, so an empty store
    and an unreadable one keep their different answers.
    """
    active = memory.list_facts(include_retracted=False, min_confidence=RECALL_MIN_CONFIDENCE)
    if not active:
        return memory.format_prompt_section(min_confidence=RECALL_MIN_CONFIDENCE)

    about_self = _latest_about(active, SELF_SUBJECT, SELF_FACTS_LIMIT)
    about_others = [fact for fact in active if fold_name(fact.subject) != SELF_SUBJECT]
    limit = memory.max_facts_in_prompt
    user_limit = min(USER_FACTS_LIMIT, limit)
    users = _latest_about(about_others, USER_SUBJECT, len(about_others))
    ranking: FactRanking | None = None
    if len(users) > user_limit:
        # More than fit: the ones the message is about first, however long ago they were
        # last seen, then the latest seen (#2224).
        ranking = await _rank(memory, message, active)
        users = _relevant_first(users, ranking)
    about_user = users[: max(0, user_limit)]
    about_project = _latest_about(
        active, PROJECT_SUBJECT, min(PROJECT_FACTS_LIMIT, limit - len(about_user))
    )
    standing = [*about_user, *about_project]
    # Every fact about the clone is kept out of the ranking: the ones past its limit are
    # older ones the newer outrank, and are counted below with the rest not selected.
    chosen = {fact.fact_id for fact in standing}
    chosen.update(fact.fact_id for fact in active if fold_name(fact.subject) == SELF_SUBJECT)
    relevant: list[MemoryFact] = []
    if len(about_others) > len(standing) and len(standing) < limit:
        if ranking is None:
            ranking = await _rank(memory, message, active)
        relevant = [entry.fact for entry in ranking.ranked if entry.fact.fact_id not in chosen]
        relevant = relevant[: limit - len(standing)]
    selected = [*standing, *relevant]

    lines = [MEMORY_SECTION_HEADER]
    if about_self:
        lines.append(SELF_INTRO)
        lines.extend(_fact_lines(about_self, functional))
    if about_others or not about_self:
        lines.append(RECALL_INTRO)
        lines.extend(_fact_lines(selected, functional))
        if not selected:
            lines.append("- (none matched this message)")
    omitted = len(active) - len(about_self) - len(selected)
    if omitted > 0:
        method = f" Ranked by {ranking.method}." if ranking is not None else ""
        lines.append(
            f"({omitted} more facts not selected for this message; query memory to view them."
            f"{method})"
        )
    return "\n".join(lines)
