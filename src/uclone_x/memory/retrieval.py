"""Ranked retrieval over cross-session memory facts.

`query_memory_facts` used to take only exact `subject` / `predicate` / `tag` filters, which
means an agent could retrieve a fact only by already knowing how it was filed. That is a
lookup, not a recall. This module adds the missing half: a natural-language query, ranked
results, and — the part that keeps it honest — an explicit statement of *which method*
produced the ranking.

The method matters because there are two, and they are not equivalent. With an embedder
wired, ranking is cosine similarity over embeddings and genuinely semantic. Without one it
is lexical overlap, which cannot match a paraphrase and, on a language that does not
delimit words with spaces, degrades further. A caller that is told "lexical overlap" can
decide what an empty result means; a caller told nothing will read it as "memory holds
nothing relevant", which is a claim about the corpus that lexical overlap never earned.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from uclone_x.llm.protocols import EmbedderProtocol
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.vector_store import BruteForceVectorStore, VectorStoreProtocol

RANKING_METHOD_RECENCY = "recency (no query given)"
RANKING_METHOD_NO_CANDIDATES = "nothing ranked (no candidate facts)"
RANKING_METHOD_NO_ROOM = "nothing ranked (top_k asked for no results)"
RANKING_METHOD_LEXICAL = "lexical overlap (no embedder configured — paraphrases will not match)"

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_LEXICAL_TOKEN = 2
_CJK_NGRAM = 2


@dataclass(frozen=True)
class RankedFact:
    """One retrieved fact and the score that placed it."""

    fact: MemoryFact
    score: float


@dataclass(frozen=True)
class FactRanking:
    """A ranked result set that says how it was ranked.

    `considered` is carried alongside `ranked` so that "3 results" can be read against the
    size of the candidate set: three of three is a different answer from three of four
    hundred, and only the second one is a ranking.
    """

    ranked: tuple[RankedFact, ...]
    method: str
    considered: int

    def describe(self) -> str:
        """One line naming the method and the share of candidates returned."""
        return f"Ranked {len(self.ranked)} of {self.considered} candidate facts by {self.method}."


def embeddable_text(fact: MemoryFact) -> str:
    """Render a fact as the text that is embedded or lexically matched.

    Tags are included: they are how a fact was categorised when it was recorded, and
    dropping them from the indexed text makes a fact unfindable by the very words its
    author chose to file it under.
    """
    parts = [fact.subject, fact.predicate, fact.object_value, *fact.tags]
    return " ".join(part for part in parts if part)


def _tokens(text: str) -> set[str]:
    """Tokenize for lexical overlap, with a character-bigram fallback for CJK.

    Word tokens alone answer nothing for Korean, Japanese or Chinese text, which a
    whitespace/`\\w+` tokenizer returns as a handful of long unsegmented runs that match only
    on an exact repeat. Adding character bigrams for CJK runs gives partial overlap a
    surface to score on. It is still lexical — it matches characters, not meaning — which is
    exactly what `RANKING_METHOD_LEXICAL` tells the caller.
    """
    lowered = text.lower()
    tokens = {word for word in _WORD_RE.findall(lowered) if len(word) >= _MIN_LEXICAL_TOKEN}
    for word in list(tokens):
        if _is_cjk(word):
            tokens.update(
                word[index : index + _CJK_NGRAM] for index in range(len(word) - _CJK_NGRAM + 1)
            )
    return tokens


def _is_cjk(word: str) -> bool:
    return any("぀" <= char <= "ヿ" or "㐀" <= char <= "鿿" or "가" <= char <= "힣" for char in word)


def _lexical_score(query_tokens: set[str], fact: MemoryFact) -> float:
    fact_tokens = _tokens(embeddable_text(fact))
    if not fact_tokens or not query_tokens:
        return 0.0
    overlap = query_tokens & fact_tokens
    return len(overlap) / len(query_tokens)


async def rank_facts(
    query: str,
    facts: Sequence[MemoryFact],
    top_k: int = 5,
    embedder: EmbedderProtocol | None = None,
    vector_store: VectorStoreProtocol | None = None,
) -> FactRanking:
    """Rank `facts` against `query`, naming the method used.

    With `embedder` wired, every candidate is embedded (through `vector_store` when one is
    supplied, so repeat queries do not re-embed) and ranked by cosine similarity. Without
    one, ranking is lexical overlap. With no query at all, the "ranking" is recency, and
    says so rather than implying relevance it did not compute.

    Raises:
        EmbeddingError: If an embedder is wired and the embedding call fails. The failure
            propagates rather than degrading to lexical, because a silent degradation would
            report a transport fault as a differently-ordered list of the same facts.
    """
    considered = len(facts)
    # Two different reasons for an empty answer, and neither is recency: saying "recency
    # (no query given)" here claims an ordering nothing computed, about a query that was
    # given. An unranked result must never be able to read as a ranked one.
    if not facts:
        return FactRanking(ranked=(), method=RANKING_METHOD_NO_CANDIDATES, considered=considered)
    if top_k <= 0:
        return FactRanking(ranked=(), method=RANKING_METHOD_NO_ROOM, considered=considered)

    if not query.strip():
        by_recency = sorted(facts, key=lambda fact: fact.created_at, reverse=True)
        return FactRanking(
            ranked=tuple(RankedFact(fact=fact, score=0.0) for fact in by_recency[:top_k]),
            method=RANKING_METHOD_RECENCY,
            considered=considered,
        )

    if embedder is not None:
        return await _rank_by_embedding(query, facts, top_k, embedder, vector_store, considered)

    query_tokens = _tokens(query)
    scored = [RankedFact(fact=fact, score=_lexical_score(query_tokens, fact)) for fact in facts]
    positive = [entry for entry in scored if entry.score > 0.0]
    positive.sort(key=lambda entry: -entry.score)
    return FactRanking(
        ranked=tuple(positive[:top_k]),
        method=RANKING_METHOD_LEXICAL,
        considered=considered,
    )


async def _rank_by_embedding(
    query: str,
    facts: Sequence[MemoryFact],
    top_k: int,
    embedder: EmbedderProtocol,
    vector_store: VectorStoreProtocol | None,
    considered: int,
) -> FactRanking:
    # `is None`, not `or`: a vector store defines `__len__`, so an empty one is falsy and
    # `vector_store or ...` would silently discard the caller's store on every query until
    # it happened to be non-empty — which it never becomes, because each query gets a fresh
    # one and re-embeds the whole corpus.
    store = (
        BruteForceVectorStore(dimensions=embedder.dimensions, model_name=embedder.model_name)
        if vector_store is None
        else vector_store
    )
    by_id = {fact.fact_id: fact for fact in facts}

    known = set(store.keys())
    missing = [fact_id for fact_id in by_id if fact_id not in known]
    if missing:
        vectors = await embedder.embed([embeddable_text(by_id[fact_id]) for fact_id in missing])
        for fact_id, vector in zip(missing, vectors, strict=True):
            store.upsert(fact_id, vector)

    (query_vector,) = await embedder.embed([query])
    # Ask for more than `top_k`: the store may hold vectors for facts the caller's filters
    # excluded from `facts`, and dropping those after the cut would silently return fewer
    # results than asked for.
    surplus = max(0, len(store) - len(by_id))
    matches = store.search(query_vector, top_k=top_k + surplus)
    ranked = tuple(
        RankedFact(fact=by_id[match.key], score=match.score)
        for match in matches
        if match.key in by_id
    )[:top_k]
    return FactRanking(
        ranked=ranked,
        method=f"embedding similarity ({embedder.model_name})",
        considered=considered,
    )
