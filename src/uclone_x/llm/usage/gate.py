"""The usage gate: check the user's limits before paid work, record its tokens after.

See the token-gateway design §4.1. The factory wraps every connector whose
`paid` is true, so every caller (agent turns, persona drafts, the room's selector, the
compactor) passes the gate without a change at its call site. A connector that is not paid
is returned untouched.

**A turn is checked once, when it starts** (owner ruling 2026-10-30, #2431): "한도는 턴을
시작할때 확인하고, 일단 턴이 시작하면 한도를 넘더라도 끝내야 한다" ("check the limit when a turn
starts; once it has started, finish it even past the limit"). Stopping an admitted turn at
step N+1 wasted steps 1..N, left their tools half done, and asked for a resend that cost
more than finishing. `enter_admitted_turn` checks at the turn's start and marks the rest
of it (a context variable, so concurrent turns and the tasks a turn starts each see their
own); inside, every call is booked and none is refused. A call outside any turn (a persona
draft, the room selector, the compactor) is checked per call, as before.

**A runaway turn stops at a step boundary** (owner amendment, same day: "폭주 상한 방지
한도는 두자"). Every call booked inside a turn is added to that turn's own spend, and the
gate stops the turn before its next call once the spend passes `per_10_minutes`
(`usage_turn_ceiling`), raising `TurnCeilingReachedError`; the turn ends as
`usage_ceiling`. It never stops a call midway; the person continues the turn in place
(`admission.py`).

The wrap is a subclass rather than a wrapper, so the connector keeps its type name, its
attributes and every `isinstance` check a head makes.

Overshoot is accepted (§4.2): the check reads usage already recorded, and there is no
reservation, so turns running at once may each finish past a limit. The next turn, or the
next call outside a turn, is the one refused.
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
from uclone_x.llm.usage.admission import (
    ADMITTED as _ADMITTED,
)
from uclone_x.llm.usage.admission import (
    book_turn_spend,
    check_turn_ceiling,
)
from uclone_x.llm.usage.admission import (
    enter_admitted_turn as enter_admitted_turn,
)
from uclone_x.llm.usage.admission import (
    leave_admitted_turn as leave_admitted_turn,
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

__all__ = [
    "UsageGate",
    "enter_admitted_turn",
    "gate_connector",
    "gate_if_paid",
    "leave_admitted_turn",
    "shared_store",
]

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

    def remaining(self) -> int | None:
        """The fewest tokens any limited window has left now; `None` when none is limited.

        For work that must leave room for the person's next reply (#1851): the gate refuses
        a call only once a window is reached, so what is left is what a call may still
        spend before the next one is refused.
        """
        limits = self._limits()
        if limits == UsageLimits():
            return None
        status = check(limits, self.store, self._clock())
        left = [w.limit - w.used for w in status.windows if w.limit is not None]
        if not left:
            return None
        return max(0, min(left))

    def turn_ceiling(self) -> int | None:
        """What one turn may spend before it stops at a step boundary: the 10-minute limit.

        `None` when the person chose No limit for that window: then there is no ceiling.
        """
        return self._limits().per_10_minutes

    def record(self, usage: TokenUsage) -> None:
        """Book one call's tokens. The provider's count, or the estimate it is labelled as.

        Inside an admitted turn the tokens are also that turn's own spend (#2431).
        """
        tokens = usage.input_tokens + usage.output_tokens
        self.store.add(
            UsageEntry(
                at=self._clock(),
                tokens=tokens,
                provider=usage.provider,
                model=usage.model,
                count_source=usage.count_source,
                cached_tokens=usage.cache_read_input_tokens,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        )
        book_turn_spend(tokens)  # the running turn's own spend, for its ceiling (#2431)


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
            if not _ADMITTED.get():  # a call outside any admitted turn (#2431)
                gate.admit()
            else:
                check_turn_ceiling()  # an admitted turn past its own ceiling stops here
            response = await super().generate(  # pyright: ignore[reportAbstractUsage]
                request
            )
            try:
                gate.record(response.usage)
            except Exception:
                # A failed write must not throw away a reply the provider already served.
                logger.exception("Could not record paid-model usage for a call")
            return response

        def usage_admit(self) -> None:
            """The check a turn makes at its start (`admission.enter_admitted_turn`, #2431)."""
            _gate_of(self).admit()

        def usage_turn_ceiling(self) -> int | None:
            """The tokens one turn may spend before it stops at a step boundary (#2431)."""
            return _gate_of(self).turn_ceiling()

        def usage_remaining(self) -> int | None:
            """What the person's usage limit leaves now (`UsageRemainingProtocol`, #1851)."""
            return _gate_of(self).remaining()

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
    if not _ADMITTED.get():  # inside an admitted turn nothing is refused (#2431)
        gate.admit()
    else:
        check_turn_ceiling()  # a turn's stream past its own ceiling is not opened
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
    if not connector.paid:
        return connector
    return gate_connector(connector, gate)


def gate_connector(connector: BaseLLMConnector, gate: UsageGate | None = None) -> BaseLLMConnector:
    """Pass `connector`'s calls through `gate`, paid or not. Idempotent, like `gate_if_paid`.

    For a head that books an unpaid model too (the cluster's ledger, #2466); it hands such a
    connector a gate with no limits, so nothing about a local model is ever refused.
    """
    if getattr(type(connector), "_usage_gated", False):
        return connector
    connector.__class__ = _gated_class(type(connector))
    connector._usage_gate = gate  # pyright: ignore[reportAttributeAccessIssue]
    return connector
