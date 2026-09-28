"""The usage gate: check the user's limits before a paid call, record its tokens after.

See the token-gateway design §4.1. The factory wraps every connector whose
`paid` is true, so every caller (agent turns, persona drafts, the room's selector, the
compactor) passes the gate without a change at its call site. A connector that is not paid
is returned untouched.

The wrap is a subclass rather than a wrapper, so the connector keeps its type name, its
attributes and every `isinstance` check a head makes.

Overshoot is accepted (§4.2): the check reads usage already recorded, and there is no
reservation, so calls running at once may each finish past a limit. The next call is the
one refused.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from uclone_x.errors import UsageLimitReachedError
from uclone_x.llm.compactor import estimate_reply_tokens, estimate_request_tokens
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
)
from uclone_x.llm.usage.limits import (
    UsageLimits,
    check,
    limit_reached_message,
    load_limits,
)
from uclone_x.llm.usage.store import (
    SqliteUsageStore,
    UsageEntry,
    UsageStore,
    default_usage_file,
)

__all__ = ["UsageGate", "gate_if_paid", "shared_store"]

logger = logging.getLogger(__name__)

_STORES: dict[Path, SqliteUsageStore] = {}


def shared_store(path: Path | None = None) -> SqliteUsageStore:
    """The process's store for `path` (default: beside `settings.json`), opened once.

    Keyed by path because the session root can differ between callers (tests, a dashboard
    with its own storage directory). Opening prunes old rows, so it is done once per path.
    """
    target = path if path is not None else default_usage_file()
    store = _STORES.get(target)
    if store is None:
        store = SqliteUsageStore(target)
        _STORES[target] = store
    return store


def _now() -> datetime:
    return datetime.now(UTC)


class UsageGate:
    """The check and the record, apart from any connector, so a test can drive it."""

    def __init__(
        self,
        store: UsageStore | None = None,
        limits: Callable[[], UsageLimits] = load_limits,
        clock: Callable[[], datetime] = _now,
        *,
        store_file: Path | None = None,
    ) -> None:
        self._store = store
        self._limits = limits
        self._clock = clock
        self._store_file = store_file

    @classmethod
    def for_storage(cls, settings_file: Path, usage_file: Path) -> UsageGate:
        """A gate that reads `settings_file`'s limits and books into `usage_file`.

        For a head that keeps its own storage directory (the dashboard's `storage_dir`), so
        its paid calls are held to the limits its Usage panel shows, not the session root's.
        """
        return cls(limits=lambda: load_limits(settings_file), store_file=usage_file)

    @property
    def store(self) -> UsageStore:
        # Resolved per call when not injected, so a changed session root is honoured, and
        # nothing is opened until a paid call needs it.
        return self._store if self._store is not None else shared_store(self._store_file)

    def admit(self) -> None:
        """Raise `UsageLimitReachedError` when any window's limit is reached."""
        limits = self._limits()
        if limits == UsageLimits():
            return
        now = self._clock()
        blocking = check(limits, self.store, now).blocking
        if blocking is None:
            return
        again = blocking.available_again_at or now
        raise UsageLimitReachedError(
            limit_reached_message(blocking.window, again, now),
            window=blocking.window.value,
            available_again_at=again,
        )

    def record(self, usage: TokenUsage) -> None:
        """Book one call's tokens. The provider's count, or the estimate it is labelled as."""
        self.store.add(
            UsageEntry(
                at=self._clock(),
                tokens=usage.input_tokens + usage.output_tokens,
                provider=usage.provider,
                model=usage.model,
                count_source=usage.count_source,
            )
        )


def _gate_of(connector: BaseLLMConnector) -> UsageGate:
    return getattr(connector, "_usage_gate", None) or UsageGate()


@cache
def _gated_class(base: type[BaseLLMConnector]) -> type[BaseLLMConnector]:
    """`base`, with every `generate` and `stream` passed through the usage gate."""

    class _UsageGated(base):
        _usage_gated = True
        _usage_gate: UsageGate | None = None

        # `base` is always a concrete connector; pyright sees only the abstract bound.
        async def generate(self, request: LLMRequest) -> ModelResponse:
            gate = _gate_of(self)
            gate.admit()
            response = await super().generate(  # pyright: ignore[reportAbstractUsage]
                request
            )
            try:
                gate.record(response.usage)
            except Exception:
                # A failed write must not throw away a reply the provider already served.
                logger.exception("Could not record paid-model usage for a call")
            return response

        def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
            return _gated_stream(
                self,
                super().stream,  # pyright: ignore[reportAbstractUsage]
                request,
            )

    _UsageGated.__name__ = base.__name__
    _UsageGated.__qualname__ = base.__qualname__
    return _UsageGated


async def _gated_stream(
    connector: BaseLLMConnector,
    inner: Callable[[LLMRequest], AsyncIterator[StreamChunk]],
    request: LLMRequest,
) -> AsyncIterator[StreamChunk]:
    """Admit before the first chunk; book the stream's usage when it ends, however it ends.

    A stream that sent no usage, or was abandoned part way, is booked as an estimate from
    the request and the text received, labelled `ESTIMATE` (§4.2). A stream that failed
    before any chunk is not booked: nothing shows the provider served it (#938).
    """
    gate = _gate_of(connector)
    gate.admit()
    reported: list[TokenUsage] = []
    text: list[str] = []
    model = request.model
    received = False
    try:
        async for chunk in inner(request):
            received = True
            if chunk.usage is not None:
                reported.append(chunk.usage)
            if chunk.delta_content:
                text.append(chunk.delta_content)
            if chunk.model:
                model = chunk.model
            yield chunk
    finally:
        if received:
            _record_stream(gate, connector, request, reported, model, "".join(text))


def _record_stream(
    gate: UsageGate,
    connector: BaseLLMConnector,
    request: LLMRequest,
    reported: list[TokenUsage],
    model: str | None,
    text: str,
) -> None:
    """Book a stream that received at least one chunk: its last reported usage, or an estimate."""
    usage = (
        reported[-1]
        if reported
        else TokenUsage(
            provider=connector.provider_name,
            model=model,
            input_tokens=estimate_request_tokens(request),
            output_tokens=estimate_reply_tokens(text, ()),
            count_source=TokenCountSource.ESTIMATE,
        )
    )
    try:
        gate.record(usage)
    except Exception:  # pragma: no cover - a failed write must not mask the stream's end
        logger.exception("Could not record paid-model usage for a stream")


def gate_if_paid(connector: BaseLLMConnector, gate: UsageGate | None = None) -> BaseLLMConnector:
    """Pass `connector`'s calls through the usage gate when it is paid; else return it as is.

    Idempotent: a connector already gated is not gated twice.
    """
    if not connector.paid or getattr(type(connector), "_usage_gated", False):
        return connector
    connector.__class__ = _gated_class(type(connector))
    connector._usage_gate = gate  # pyright: ignore[reportAttributeAccessIssue]
    return connector
