"""Ranked recall over memory facts, and the method statement that keeps it honest."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from uclone_x.core.provenance import Provenance
from uclone_x.errors import EmbeddingError
from uclone_x.memory.retrieval import (
    RANKING_METHOD_LEXICAL,
    RANKING_METHOD_NO_CANDIDATES,
    RANKING_METHOD_NO_ROOM,
    RANKING_METHOD_RECENCY,
    embeddable_text,
)
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import QueryMemoryFactsParams, QueryMemoryFactsTool
from uclone_x.tools.models import ToolContext


class _KeywordEmbedder:
    """A deterministic stand-in: one axis per keyword, so similarity is checkable by hand."""

    KEYWORDS = ("database", "postgres", "deploy", "배포", "데이터")

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    @property
    def model_name(self) -> str:
        return "keyword-axes"

    @property
    def dimensions(self) -> int:
        return len(self.KEYWORDS) + 1

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            lowered = text.lower()
            axes = [1.0 if keyword in lowered else 0.0 for keyword in self.KEYWORDS]
            # A constant final axis keeps every vector non-zero, which the store requires.
            vectors.append((*axes, 0.01))
        return tuple(vectors)


class _BrokenEmbedder(_KeywordEmbedder):
    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        raise EmbeddingError("embedding endpoint unreachable")


def _provenance() -> Provenance:
    return Provenance.primary(provider="agent.test", model="memory")


def _memory(embedder: object | None = None) -> CrossSessionMemory:
    memory = CrossSessionMemory(embedder=embedder)  # pyright: ignore[reportArgumentType]
    memory.record_fact(
        subject="production database",
        predicate="runs_on",
        object_value="postgres 16",
        provenance=_provenance(),
        source_session_id="s1",
    )
    memory.record_fact(
        subject="release process",
        predicate="deploy_window",
        object_value="Tuesdays 09:00 KST",
        provenance=_provenance(),
        source_session_id="s1",
        tags=("deploy",),
    )
    return memory


@pytest.mark.asyncio
async def test_lexical_ranking_says_it_is_lexical() -> None:
    """Without an embedder the ranking is lexical, and the caller is told so.

    Killed by: src/uclone_x/memory/retrieval.py :: method=RANKING_METHOD_LEXICAL,
    Becomes: method="embedding similarity",
    """
    ranking = await _memory().search_facts("what database do we run?", top_k=3)

    assert ranking.method == RANKING_METHOD_LEXICAL
    assert ranking.ranked[0].fact.subject == "production database"
    assert "lexical overlap" in ranking.describe()


@pytest.mark.asyncio
async def test_embedding_ranking_names_the_model_that_produced_it() -> None:
    embedder = _KeywordEmbedder()

    ranking = await _memory(embedder).search_facts("postgres", top_k=1)

    assert ranking.method == "embedding similarity (keyword-axes)"
    assert ranking.ranked[0].fact.subject == "production database"


@pytest.mark.asyncio
async def test_a_second_query_does_not_re_embed_the_corpus() -> None:
    """Fact vectors are cached in the store; only the query is embedded again.

    Killed by: src/uclone_x/memory/retrieval.py :: missing = [fact_id for fact_id in by_id if fact_id not in known]
    Becomes: missing = list(by_id)
    """
    embedder = _KeywordEmbedder()
    memory = _memory(embedder)

    await memory.search_facts("postgres", top_k=1)
    calls_after_first = len(embedder.calls)
    await memory.search_facts("deploy", top_k=1)

    assert len(embedder.calls) == calls_after_first + 1


@pytest.mark.asyncio
async def test_an_embedding_failure_propagates_instead_of_degrading_to_lexical() -> None:
    """A silent degradation would report a transport fault as a differently-ordered list."""
    memory = _memory(_BrokenEmbedder())

    with pytest.raises(EmbeddingError, match="unreachable"):
        await memory.search_facts("postgres", top_k=1)


@pytest.mark.asyncio
async def test_an_empty_query_is_reported_as_recency_not_relevance() -> None:
    """Listing by recency is not a ranking, so it does not claim to be one.

    Killed by: src/uclone_x/memory/retrieval.py :: if not query.strip():
    Becomes: if False:
    """
    ranking = await _memory().search_facts("", top_k=2)

    assert ranking.method == RANKING_METHOD_RECENCY
    assert len(ranking.ranked) == 2


@pytest.mark.asyncio
async def test_a_korean_query_scores_on_character_overlap() -> None:
    """Whitespace tokens alone leave Korean unmatchable; bigrams give overlap a surface.

    Killed by: src/uclone_x/memory/retrieval.py :: if _is_cjk(word):
    Becomes: if False:
    """
    memory = CrossSessionMemory()
    memory.record_fact(
        subject="배포 절차",
        predicate="담당",
        object_value="플랫폼 팀이 배포한다",
        provenance=_provenance(),
        source_session_id="s1",
    )

    ranking = await memory.search_facts("배포는 누가 하나", top_k=1)

    assert ranking.method == RANKING_METHOD_LEXICAL
    assert ranking.ranked and ranking.ranked[0].fact.subject == "배포 절차"


@pytest.mark.asyncio
async def test_structured_filters_still_narrow_the_candidate_set() -> None:
    """A filter has to remove the other facts, not merely report a smaller count.

    `considered == 1` alone is satisfied by a filter that keeps the *wrong* fact, so the
    identity of the survivor is asserted too, against the same store unfiltered.

    Killed by: src/uclone_x/memory/store.py :: if target_subj and fact.subject.strip().lower() != target_subj:
    Becomes: if target_subj and fact.subject.strip().lower() == target_subj:
    """
    memory = _memory()

    unfiltered = await memory.search_facts("deploy", top_k=5)
    filtered = await memory.search_facts("deploy", top_k=5, subject="release process")

    assert unfiltered.considered == 2
    assert filtered.considered == 1
    assert [entry.fact.subject for entry in filtered.ranked] == ["release process"]


@pytest.mark.asyncio
async def test_tags_are_part_of_the_indexed_text() -> None:
    """A fact filed under a tag must be findable by that tag's words.

    Killed by: src/uclone_x/memory/retrieval.py :: parts = [fact.subject, fact.predicate, fact.object_value, *fact.tags]
    Becomes: parts = [fact.subject, fact.predicate, fact.object_value]
    """
    memory = CrossSessionMemory()
    fact = memory.record_fact(
        subject="alpha",
        predicate="beta",
        object_value="gamma",
        provenance=_provenance(),
        source_session_id="s1",
        tags=("kubernetes",),
    )

    assert "kubernetes" in embeddable_text(fact)


@pytest.mark.asyncio
async def test_the_tool_reports_the_ranking_method_on_an_empty_result() -> None:
    """ "No matching facts" is a claim about the matcher as much as about memory.

    Killed by: src/uclone_x/memory/tools.py :: f"No matching memory facts found. {ranking.describe()}"
    Becomes: "No matching memory facts found."
    """
    tool = QueryMemoryFactsTool(_memory())
    context = ToolContext(agent_id="a", session_id="s")

    output = await tool.run(
        QueryMemoryFactsParams(query="quantum chromodynamics", top_k=3), context
    )

    assert output.startswith("No matching memory facts found.")
    assert "lexical overlap" in output


@pytest.mark.asyncio
async def test_the_tool_reports_scores_with_each_hit() -> None:
    tool = QueryMemoryFactsTool(_memory())
    context = ToolContext(agent_id="a", session_id="s")

    output = await tool.run(QueryMemoryFactsParams(query="postgres database", top_k=2), context)

    # Exactly one of the two facts overlaps this query, so the count is determined: an
    # `or` over both possibilities would pass whether or not the non-matching fact was
    # filtered out, which is the one thing the count is here to show.
    assert "score:" in output
    assert "Ranked 1 of 2 candidate facts" in output
    assert "production database" in output
    assert "release process" not in output


@pytest.mark.asyncio
async def test_a_retracted_fact_drops_out_of_the_vector_index() -> None:
    """The index must not outgrow the corpus it claims to describe.

    Killed by: src/uclone_x/memory/store.py :: self._vector_store.remove(fact_id)
    Becomes: pass
    """
    embedder = _KeywordEmbedder()
    memory = _memory(embedder)
    fact = memory.list_facts()[0]
    await memory.search_facts("postgres", top_k=2)

    memory.retract_fact(
        fact_id=fact.fact_id,
        reason="superseded",
        provenance=_provenance(),
        session_id="s2",
    )
    ranking = await memory.search_facts("postgres", top_k=2)

    assert all(entry.fact.fact_id != fact.fact_id for entry in ranking.ranked)
    assert memory.vector_store is not None
    assert fact.fact_id not in memory.vector_store.keys()


@pytest.mark.asyncio
async def test_an_empty_store_does_not_claim_a_recency_ordering() -> None:
    """Nothing ranked is not "ranked by recency", and a query was given.

    Killed by: src/uclone_x/memory/retrieval.py :: return FactRanking(ranked=(), method=RANKING_METHOD_NO_CANDIDATES, considered=considered)
    Becomes: return FactRanking(ranked=(), method=RANKING_METHOD_RECENCY, considered=considered)
    """
    ranking = await CrossSessionMemory().search_facts("anything at all", top_k=3)

    assert ranking.ranked == ()
    assert ranking.method == RANKING_METHOD_NO_CANDIDATES
    assert ranking.method != RANKING_METHOD_RECENCY


@pytest.mark.asyncio
async def test_asking_for_no_results_says_that_is_why_there_are_none() -> None:
    """`top_k=0` returns nothing because nothing was asked for, not because nothing matched.

    Killed by: src/uclone_x/memory/retrieval.py :: return FactRanking(ranked=(), method=RANKING_METHOD_NO_ROOM, considered=considered)
    Becomes: return FactRanking(ranked=(), method=RANKING_METHOD_RECENCY, considered=considered)
    """
    ranking = await _memory().search_facts("postgres database", top_k=0)

    assert ranking.ranked == ()
    assert ranking.method == RANKING_METHOD_NO_ROOM
    assert ranking.considered == 2
