"""Relevance-ranked recall: which of a clone's facts a turn's request carries.

Design: the clone knowledge graph design, §3.5, and the clone self and scenes design, §3.3.
The section a turn sends is

0. the facts whose subject is the clone itself (`self`), newest first, up to
   `SELF_FACTS_LIMIT`, under their own line, which says they override the persona
   description where the two disagree (#2016). They are not counted against
   `max_facts_in_prompt`: that budget is the person's, the project's and the ranked facts',
   and a clone with eight facts about itself must not lose the person's;
1. the facts whose subject is the person (`user`), newest first, up to `USER_FACTS_LIMIT`;
2. then the facts whose subject is the project the clone works on (`project`), newest
   first, up to `PROJECT_FACTS_LIMIT`: standing preferences such as "reply in Korean" that
   a short message ("continue") shares no word with (#1857);
3. then the clone's other facts, ranked against the turn's message by `search_facts` --
   embedding similarity when the store has an embedder, lexical overlap when it has none;
4. bounded at the store's `max_facts_in_prompt`, with a line counting what was not selected.

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


def _newest_about(active: list[MemoryFact], subject: str, limit: int) -> list[MemoryFact]:
    """Up to `limit` of the `active` facts whose subject is `subject`, newest first.

    Compared folded (`fold_name`: NFC, whitespace collapsed, casefolded), as every
    other subject lookup is (#1899).
    """
    wanted = fold_name(subject)
    return sorted(
        (fact for fact in active if fold_name(fact.subject) == wanted),
        key=lambda fact: (fact.created_at, fact.fact_id),
        reverse=True,
    )[: max(0, limit)]


async def _rank_rest(
    memory: RecallSource,
    message: str,
    active: list[MemoryFact],
    chosen: set[str],
    top_k: int,
) -> FactRanking:
    """Rank the active facts not already `chosen`, through the store's own `search_facts`.

    `search_facts` ranks every active fact, so it is asked for `len(chosen)` more than
    `top_k` and the chosen ones are dropped after: they may rank anywhere.
    """
    try:
        ranking = await memory.search_facts(
            message, top_k=top_k + len(chosen), min_confidence=RECALL_MIN_CONFIDENCE
        )
    except Exception as exc:  # the turn goes on without the embedder; the section says so
        logger.warning(
            "Memory recall could not use the embedder (%s); ranking lexically.",
            type(exc).__name__,
        )
        ranking = await rank_facts(query=message, facts=active, top_k=top_k + len(chosen))
        ranking = FactRanking(
            ranked=ranking.ranked, method=EMBEDDER_FAILED_METHOD, considered=ranking.considered
        )
    kept = tuple(entry for entry in ranking.ranked if entry.fact.fact_id not in chosen)[:top_k]
    return FactRanking(ranked=kept, method=ranking.method, considered=ranking.considered)


async def recall_prompt_section(memory: RecallSource, message: str) -> str:
    """The memory section for a turn whose message is `message` (design §3.5).

    Returns what `format_prompt_section` returns when no fact is active, so an empty store
    and an unreadable one keep their different answers.
    """
    active = memory.list_facts(include_retracted=False, min_confidence=RECALL_MIN_CONFIDENCE)
    if not active:
        return memory.format_prompt_section(min_confidence=RECALL_MIN_CONFIDENCE)

    about_self = _newest_about(active, SELF_SUBJECT, SELF_FACTS_LIMIT)
    about_others = [fact for fact in active if fold_name(fact.subject) != SELF_SUBJECT]
    limit = memory.max_facts_in_prompt
    about_user = _newest_about(about_others, USER_SUBJECT, min(USER_FACTS_LIMIT, limit))
    about_project = _newest_about(
        active, PROJECT_SUBJECT, min(PROJECT_FACTS_LIMIT, limit - len(about_user))
    )
    standing = [*about_user, *about_project]
    chosen = {fact.fact_id for fact in standing}
    ranking: FactRanking | None = None
    # Every fact about the clone is kept out of the ranking: the ones past its limit are
    # older ones the newer outrank, and are counted below with the rest not selected.
    chosen.update(fact.fact_id for fact in active if fold_name(fact.subject) == SELF_SUBJECT)
    if len(about_others) > len(standing) and len(standing) < limit:
        ranking = await _rank_rest(memory, message, active, chosen, top_k=limit - len(standing))
    relevant = [entry.fact for entry in ranking.ranked] if ranking is not None else []
    selected = [*standing, *relevant]

    lines = [MEMORY_SECTION_HEADER]
    if about_self:
        lines.append(SELF_INTRO)
        lines.extend(f"- {fact.summary()}" for fact in about_self)
    if about_others or not about_self:
        lines.append(RECALL_INTRO)
        lines.extend(f"- {fact.summary()}" for fact in selected)
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
