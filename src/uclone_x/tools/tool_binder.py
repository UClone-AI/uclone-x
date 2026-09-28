"""Host binding: grow a session's tools layer from the catalog, once per user message.

The tools layer of a request is a pinned base set (design §5.1). On a small local model
the rest of the tools the agent holds -- the catalog -- stay out of the request until a
user message needs them: before the turn, the host embeds the message, and the top
`BIND_TOP_K` catalog tools scoring at least `BIND_MIN_SCORE` are appended to the set.
The set only grows until compaction, so the request prefix breaks at most once per user
message and never by a tool disappearing.

This replaces the per-turn scoper, which re-chose the advertised tools on every turn. In
the `tool_strategy` eval (#1670) that changed the tools layer on 11 of 19 request pairs,
and every empty reply it caused was a call to a tool it had withheld, which Ollama drops
without an error. Binding came closest to pinning every tool, with a quarter of the
schema tokens.

The same ranking serves `search_tools(query)` (`SearchToolsTool`), the model's route to a
catalog tool binding missed: its hits join the same grow-only set, so a found tool is
declared in `tools` and called directly, with no `load_tool` or `call_tool` step.

This module decides only *which catalog tools a message or query finds*. What is base and what is
catalog, the grow-only session set and its reset at compaction live in the agent's tool invoker
(`ToolInvoker.tools_for_turn`), because they are per session and this object is shared by
every seat of a room.

Binding applies only where a local embedder is reachable (`tool_binder_for`). Everywhere
else, and whenever the embedder fails, the agent pins every tool it holds. There is no
lexical fallback: in #1670 lexical matching was worse than both pinning and binding, and
found no tool for any Korean message.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import PlainRefusalError
from uclone_x.tools.base import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from uclone_x.llm.models import ToolDefinition
    from uclone_x.llm.protocols import EmbedderProtocol
    from uclone_x.tools.models import ToolContext

logger = logging.getLogger(__name__)

#: How many catalog tools one user message may bind.
BIND_TOP_K = 3
#: The lowest cosine similarity that binds a tool. Calibrated in #1670 with bge-m3, where
#: it was what kept the one task needing no tool (top score 0.34) from binding any.
BIND_MIN_SCORE = 0.35

#: Providers whose models run on this machine, where binding pays for itself. vLLM is
#: local too, but the only embedder wired here is Ollama's, so it pins every tool.
LOCAL_BINDING_PROVIDERS = frozenset({"ollama"})


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm > 0.0 else 0.0


def _tool_text(tool: ToolDefinition) -> str:
    """What is embedded for a tool: its name and its description."""
    return f"{tool.name}: {tool.description}"


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ToolBinder:
    """Ranks catalog tools against one user message by embedding similarity.

    Catalog embeddings are cached by the hash of the embedded text, so each tool
    description is embedded once for as long as the binder lives, and a tool whose
    description changes is embedded again rather than scored with a stale vector.

    `bind` returns `None` when the embedder fails. The caller then pins every tool; the
    failure is logged once per binder, by exception type only, since an embedder's error
    text can carry a URL or a server's own words and none of it is the person's business.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        *,
        top_k: int = BIND_TOP_K,
        min_score: float = BIND_MIN_SCORE,
    ) -> None:
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._failure_logged = False
        #: Calls made to the embedder for catalog text; one per batch of new descriptions.
        self.catalog_embed_calls = 0

    async def bind(self, message: str, catalog: Sequence[ToolDefinition]) -> tuple[str, ...] | None:
        """The names `message` binds from `catalog`, best first; `None` if the embedder failed.

        At most `top_k` names, each scoring at least `min_score`. Ties break by name, so
        the same message and catalog always bind the same tools. An empty message or an
        empty catalog binds nothing and costs no embedding call.
        """
        return await self._rank(message, catalog)

    async def search(self, query: str, catalog: Sequence[ToolDefinition]) -> tuple[str, ...] | None:
        """The names `search_tools(query)` finds in `catalog`; `None` if the embedder failed.

        The same ranking, floor and cap as `bind` (design §5.1: "`search_tools` uses the
        same embedder for the tools binding misses"), over a query the model wrote rather
        than the person's message.
        """
        return await self._rank(query, catalog)

    async def _rank(self, text: str, catalog: Sequence[ToolDefinition]) -> tuple[str, ...] | None:
        if not text.strip() or not catalog:
            return ()
        try:
            await self._embed_catalog(catalog)
            (query,) = await self._embedder.embed([text])
        except Exception as exc:  # any embedder failure means pin-all, never a crash
            if not self._failure_logged:
                self._failure_logged = True
                logger.warning(
                    "Tool binding is off: the embedder failed (%s); every tool is pinned",
                    type(exc).__name__,
                )
            return None
        scored = [
            (_cosine(query, self._vectors[_text_key(_tool_text(t))]), t.name) for t in catalog
        ]
        kept = [(score, name) for score, name in scored if score >= self._min_score]
        kept.sort(key=lambda entry: (-entry[0], entry[1]))
        return tuple(name for _, name in kept[: self._top_k])

    async def _embed_catalog(self, catalog: Sequence[ToolDefinition]) -> None:
        missing: dict[str, str] = {}
        for tool in catalog:
            text = _tool_text(tool)
            key = _text_key(text)
            if key not in self._vectors:
                missing[key] = text
        if not missing:
            return
        vectors = await self._embedder.embed(list(missing.values()))
        self.catalog_embed_calls += 1
        if len(vectors) != len(missing):
            raise ValueError("the embedder returned a different number of vectors than texts")
        self._vectors.update(zip(missing.keys(), vectors, strict=True))


#: The name of the catalog search tool (design §5.1).
SEARCH_TOOLS_NAME = "search_tools"

#: The tool result when a search cannot run: the embedder failed, or this agent has no
#: catalog to search. Written for the model and the person alike, so it names no endpoint,
#: exception or internal component.
SEARCH_UNAVAILABLE_MESSAGE = (
    "Tool search is not working right now, so nothing was searched. "
    "Use the tools listed in this request."
)


class SearchToolsParams(BaseModel):
    """Parameters of `search_tools`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    query: str = Field(description="A few words describing the capability you need.")


class SearchToolsTool(BaseTool[SearchToolsParams]):
    """`search_tools(query)`: find catalog tools that host binding did not bind (design §5.1).

    The tool only carries the call. What it searches is the calling agent's catalog and
    what it changes is that agent's session: `search` is bound to the agent, which ranks
    its catalog with its `ToolBinder`, appends the hits to the session's grow-only bound
    set, and answers with each hit's name and one-line description. The hits are declared
    in `tools` from the next request on, so the model calls them directly; no schema is
    returned in the result. It is never registered in the shared registry: only an agent
    whose tools layer binds offers it (`ToolInvoker.tools_for_turn`).
    """

    name: str = SEARCH_TOOLS_NAME
    writes_files: ClassVar[bool] = False
    description: str = (
        "Find more tools by what they do. Found tools become callable from your next step."
    )

    def __init__(self, search: Callable[[str, str], Awaitable[str]]) -> None:
        super().__init__()
        self._search = search

    async def run(self, params: SearchToolsParams, context: ToolContext) -> str:
        return await self._search(params.query, context.session_id)


def search_unavailable() -> PlainRefusalError:
    """The refusal a search that cannot run raises, already in plain words."""
    return PlainRefusalError(SEARCH_UNAVAILABLE_MESSAGE)


def tool_binder_for(
    provider: str,
    base_url: str,
    make_embedder: Callable[[str], EmbedderProtocol],
) -> ToolBinder | None:
    """The binder for a host's provider, or `None` where every tool is pinned.

    Only a local provider with an embedder endpoint binds. Whether the endpoint works is
    learned on the first bind, which falls back to pinning if it does not. The host passes
    `make_embedder` (base URL to embedder), since this kernel module imports no adapter.
    """
    if provider.strip().lower() not in LOCAL_BINDING_PROVIDERS or not base_url.strip():
        return None
    try:
        embedder = make_embedder(base_url.strip())
    except Exception as exc:  # a bad configuration pins every tool, as a failed bind does
        logger.warning("Tool binding is off: no embedder (%s)", type(exc).__name__)
        return None
    return ToolBinder(embedder)


__all__ = [
    "BIND_MIN_SCORE",
    "BIND_TOP_K",
    "LOCAL_BINDING_PROVIDERS",
    "SEARCH_TOOLS_NAME",
    "SEARCH_UNAVAILABLE_MESSAGE",
    "SearchToolsParams",
    "SearchToolsTool",
    "ToolBinder",
    "search_unavailable",
    "tool_binder_for",
]
