# Semantic retrieval: embeddings, vector storage, and ranked recall

This document describes the retrieval seam added to cross-session memory, what its default
composition does, and — as much as anything — what it deliberately refuses to do.

## Why it was added

The subsystem previously had three components with "semantic" in their names and no
embeddings anywhere: `SemanticToolScoper` scored lowercase substrings, `SemanticModelRouter`
matched keywords, and `query_memory_facts` took exact `subject` / `predicate` / `tag`
filters. None of them was wrong as code; the names were wrong, and a name that overstates a
mechanism is how a lexical miss gets read as "nothing relevant exists".

Two of the three are addressed here: `SemanticToolScoper` is now `LexicalToolScoper`, and
`query_memory_facts` now ranks — by embeddings when one is wired, and by lexical overlap
otherwise, saying which. **`SemanticModelRouter` is untouched and still keyword-matching
under its original name.** It routes between model tiers rather than retrieving anything, so
it is out of this seam's scope; naming it here is the point, because a document that listed
three problems and silently fixed two would leave the third looking solved.

Alongside that, `CrossSessionMemory` was a real store that no composed agent ever received:
`compose_agent` had no `memory` field, so `BaseAgent` — which registers the three memory
tools only when it is given a store — never registered them. The extension point existed and
nothing consumed it, which is the P0 failure mode.

## The seam

| Layer | Protocol | Default | Where |
| --- | --- | --- | --- |
| Embedding | `EmbedderProtocol` | `OllamaEmbedder` | `uclone_x/llm/protocols.py`, `uclone_x/llm/connectors/ollama_embedder.py` |
| Vector storage | `VectorStoreProtocol` | `BruteForceVectorStore` | `uclone_x/memory/vector_store.py` |
| Ranking | `rank_facts` | lexical, or embedding when one is wired | `uclone_x/memory/retrieval.py` |

`EmbedderProtocol` sits beside `LLMProviderProtocol` rather than inside the memory package,
because embedding is a provider capability (P5) and the two are independently deployed — an
Anthropic reasoning model with a local embedding endpoint is the ordinary case.

`BruteForceVectorStore` is pure Python on purpose. Core's dependencies are pydantic,
pydantic-settings, pyyaml and httpx; a numpy- or hnswlib-backed index is an adapter, not a
prerequisite of the default path. Exact cosine over a few thousand facts costs microseconds.

The **storage form is fixed even though the search layer is not**: little-endian float32,
via `serialize_vector` / `deserialize_vector`. That is what makes a later swap to a real
index a search-layer change rather than a re-embedding of the corpus.

## What it refuses to do

* **No zero-vector fallback.** The prior art answered a failed embedding call with a zero
  vector of the right width. Every cosine similarity against that is exactly 0.0, so a
  connection error reaches the caller as "nothing in memory matched" — a claim about the
  corpus, manufactured by a transport fault. `EmbeddingError` propagates instead, and
  `BruteForceVectorStore` refuses a zero-magnitude vector at the store boundary as well.
* **No silent degradation to lexical.** If an embedder is wired and its call fails, the
  failure raises. It does not quietly fall back to string matching and return a
  differently-ordered list of the same facts.
* **No unlabelled ranking.** `FactRanking.method` always says which method produced the
  order — `embedding similarity (<model>)`, `lexical overlap (no embedder configured …)`, or
  `recency (no query given)` — and `query_memory_facts` prints it, including on the empty
  result, which is the case where its absence does the damage.
* **No padding or truncation.** A vector whose width is not the embedder's declared
  `dimensions` raises `EmbeddingDimensionError`.

## Choosing the default model

`evals/suites/embedding_retrieval.py` asks twenty-four paraphrase questions about twelve
documents, paired Korean/English, and reports recall@1 and MRR **per language** against a
single threshold. It is a live suite: it needs an embedding endpoint.

```bash
ucx eval run embedding_retrieval --live --model bge-m3
```

The set is bilingual because the model carried by the prior art ships with a source comment
conceding that its Korean scores fall below its own relevance threshold — a measurement
recorded next to the model it disqualifies, and then ignored. `DEFAULT_EMBEDDING_MODEL` names
`bge-m3` as the multilingual candidate; the eval, not the constant, is the evidence, and a
better-scoring model should replace it.

## Configuration

| Variable | Meaning |
| --- | --- |
| `UCLONE_EMBEDDING_MODEL` | Embedding model name (default `bge-m3`) |
| `UCLONE_EMBEDDING_DIMENSIONS` | Declared vector width (default `1024`); a non-integer value raises rather than falling back |
| `UCLONE_AGENTS_DIR` | Root holding one directory per agent, each with its `id` and `memory.json` (default `~/.uclone/agents`) |

No embedder is wired by default, so the default composition ranks lexically and says so.
Passing one to `CrossSessionMemory(embedder=…)` is what turns recall semantic.
