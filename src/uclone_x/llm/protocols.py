"""Protocols for LLM providers, token budgeting, and context auto-compaction.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed; structural conformance is enforced statically by the bindings in
`tests/unit/test_protocol_conformance.py` (issue 2026-09-02-035).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from uclone_x.llm.models import (
    BudgetDecision,
    ChatMessage,
    CompactionOutcome,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenBudget,
    TokenUsage,
)


class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider adapters (Gemini, Claude, OpenAI, Local).

    Both entry points take an `LLMRequest`. The previous contract had `generate` taking
    four loose arguments while `LLMRequest` sat unused, and offered two streaming methods
    — one of them `AsyncIterator[str]`, which cannot carry tool calls or `TokenUsage`,
    so a provider could implement only the lossy one and P5's token accounting would
    silently read zero for every streamed turn (issues 2026-09-02-011, -036).
    """

    @property
    def provider_name(self) -> str:
        """Name of the provider service."""
        ...

    async def generate(self, request: LLMRequest) -> ModelResponse:
        """Generate a complete model response.

        Raises:
            UCloneXError: On provider failure. Per P6 a failure must propagate; a
                provider adapter never substitutes a value the model did not produce,
                and any declared failover is attributed in the response's `provenance`.
        """
        ...

    def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream normalized chunks carrying deltas, tool calls and usage.

        Declared `def`, not `async def`: an async generator function is a plain function
        returning an `AsyncIterator`.
        """
        ...


class EmbedderProtocol(Protocol):
    """Protocol for text embedding providers, declared beside `LLMProviderProtocol`.

    Embedding is a provider capability, not a memory-subsystem detail: the same seam that
    keeps a reasoning model swappable (P5) has to keep the embedding model swappable, or
    the first vector store written against a concrete embedder fixes the model for every
    later one. It is a separate protocol rather than methods on `LLMProviderProtocol`
    because the two are independently deployed — an Anthropic reasoning model with a local
    embedding endpoint is the ordinary case, not the exotic one.

    `dimensions` is declared rather than discovered so that a store can be opened, and its
    stored width checked, before any text is embedded.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model, recorded with every vector it produces.

        Vectors from two models are not comparable, so this is provenance (P6): a store
        that cannot say which model wrote a vector cannot tell a stale index from a fresh
        one.
        """
        ...

    @property
    def dimensions(self) -> int:
        """Width of every vector this embedder returns."""
        ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed each text, returning one vector per input, in input order.

        Raises:
            EmbeddingError: On provider failure, unreachable endpoint, or a response that
                cannot be read as vectors. Per P6 the failure propagates: an embedder must
                never answer a failed call with a zero vector, because a zero vector scores
                0.0 against everything and so reports a transport error to the caller as an
                empty search result.
            EmbeddingDimensionError: If a returned vector's width is not `dimensions`.
        """
        ...


@runtime_checkable
class TokenBudgetManagerProtocol(Protocol):
    """Protocol for tracking token consumption and token ceiling enforcement."""

    def check_budget(self, session_id: str, provider: str | None = None) -> BudgetDecision:
        """Report whether a session — optionally for one provider — may proceed.

        Returns a `BudgetDecision` rather than a `bool` so a refusal carries its reason;
        P5 requires the quota to be tracked per provider, so the question has to be
        askable per provider too.
        """
        ...

    def record_usage(self, session_id: str, usage: TokenUsage) -> None:
        """Record token usage, attributed to `usage.provider`.

        A `TokenCountSource.ESTIMATE` figure is booked like a provider count: it is what a
        stream that omitted usage leaves, and discarding it lets a listener switch the
        ceiling off (#916). It does not raise; a ceiling the usage reaches refuses the next
        `check_budget` instead.
        """
        ...

    def enforce_budget(self, session_id: str, provider: str | None = None) -> None:
        """Raise `BudgetExceededError` when the session has spent past its ceiling.

        Declared here because the agent turn loop calls it pre-flight: a ceiling the
        decision path cannot reach is indistinguishable from no ceiling (P6).
        """
        ...

    def get_budget(self, session_id: str) -> TokenBudget | None:
        """Retrieve the active budget snapshot."""
        ...

    def record_compaction(
        self,
        reason: str,
        original_tokens: int,
        compacted_tokens: int,
        kept_turns: int = 4,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a context-window compaction in the budget history.

        Declared here because `BaseAgent.compact_session` is its first production
        caller (issue #183) and the alternative was a `hasattr` probe, which is how the
        abandoned `3a8b7d6` attempt reached it — a caller that guesses at a method's
        existence gets no signature checking and silently records nothing against an
        implementation that lacks it.

        Under Principle 5 and Principle 6, compactions are attributed to `session_id`
        when provided, allowing `get_summary(session_id=...)` to report per-session
        compaction metrics.
        """
        ...

    def get_summary(self, session_id: str | None = None) -> dict[str, Any]:
        """Return comprehensive token budget metrics, per-provider attribution, and compaction history."""
        ...


@runtime_checkable
class ContextCompactorProtocol(Protocol):
    """Protocol for automatic conversation pruning and semantic compaction."""

    keep_recent_turns: int
    """Dialogue turns held outside compaction.

    Declared because it is part of what `should_compact` and `compact` mean — the recent
    window is the boundary the ledger summarizes up to — and because a caller recording
    compaction metrics needs it. Without it on the protocol, a caller has to reach for
    `getattr(compactor, "keep_recent_turns", 4)` and silently invent a default that may
    not be the configured one (issue #183).

    A plain attribute rather than a read-only `@property`: implementations here subclass
    this protocol explicitly, so a property declared at protocol level is *inherited*,
    and `ContextCompactor.__init__` assigning `self.keep_recent_turns` would then raise
    `AttributeError: property has no setter`.
    """

    @property
    def superseded_ledger_count(self) -> int:
        """Cumulative count of compaction ledgers superseded by this compactor instance (#196)."""
        ...

    def estimate_tokens(self, messages: Sequence[ChatMessage]) -> int:
        """Estimate the token cost of a message sequence.

        Declared for the same reason: `should_compact` is defined in terms of this
        estimate, and a caller reporting how much a compaction saved must use the *same*
        estimator the trigger used. A caller that re-implements it — as the abandoned
        `3a8b7d6` attempt did behind a `hasattr` probe — reports savings computed by a
        different algorithm than the one that decided to compact, and the two drift.
        """
        ...

    def should_compact(self, messages: Sequence[ChatMessage], context_limit: int) -> bool:
        """Determine whether the message context exceeds the compaction threshold."""
        ...

    def should_compact_at(self, messages: Sequence[ChatMessage], threshold_tokens: int) -> bool:
        """Determine whether the message context meets or exceeds an absolute token threshold (P5)."""
        ...

    def should_compact_request(self, request: LLMRequest, context_limit: int) -> bool:
        """`should_compact`, counted over a whole request: its messages and its tool schemas.

        The trigger that decides whether a turn fits has to count what the turn will send
        (#1422): the rendered system turn with its sections, the turn context at the tail,
        and the tool definitions, which for a dozen tools can outweigh the conversation.
        """
        ...

    def should_compact_request_at(self, request: LLMRequest, threshold_tokens: int) -> bool:
        """`should_compact_at`, counted over a whole request (see `should_compact_request`)."""
        ...

    async def compact(self, messages: Sequence[ChatMessage]) -> CompactionOutcome:
        """Prune and summarize old turns, preserving recent history and system prompts.

        Returns `CompactionOutcome`, not a bare `tuple[ChatMessage, ...]`. The bare tuple
        left a caller unable to say which producer wrote the ledger it received, because
        the LLM summarization path signals "no LLM ledger" identically whether no
        summarizer is configured or a configured one returned empty content. A compaction
        is a result crossing a component boundary and an LLM-written ledger is a value a
        model produced, so P6 requires it to carry attribution — and the only component
        that knows which path ran is the compactor. See `CompactionOutcome`.
        """
        ...
