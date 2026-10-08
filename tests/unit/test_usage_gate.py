"""The usage gate: refusing a paid call before it is sent, booking its tokens after.

the token-gateway design §5. A fake paid connector counts provider calls, so
"refused with no network call made" is observed rather than assumed; a fake clock and
`MemoryUsageStore` stand in for time and the file. The factory tests are parameterised over
factory output (G8), not call sites.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from uclone_x.agent.models import AgentLLMConfig
from uclone_x.errors import BudgetExceededError, UsageLimitReachedError
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector, is_local_endpoint
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
)
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.llm.usage.limits import UsageLimits, UsageWindow
from uclone_x.llm.usage.store import MemoryUsageStore, UsageEntry

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
REQUEST = LLMRequest(
    model="fake-model",
    messages=(
        ChatMessage(role=MessageRole.USER, content="Tell me about the weather in Seoul today."),
    ),
)


class FakePaidConnector(BaseLLMConnector):
    """A paid connector whose `generate` and `stream` count calls instead of using a network."""

    def __init__(self, chunks: tuple[StreamChunk, ...] = ()) -> None:
        super().__init__(api_key="k")
        self.calls = 0
        self.chunks = chunks

    @property
    def provider_name(self) -> str:
        return "fakepaid"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content="hi",
            usage=TokenUsage(
                provider="fakepaid", model="served", input_tokens=30, output_tokens=12
            ),
            provenance=None,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        for chunk in self.chunks:
            yield chunk


class FakeLocalConnector(FakePaidConnector):
    @property
    def paid(self) -> bool:
        return False


def _gated(
    connector: BaseLLMConnector, store: MemoryUsageStore, limits: UsageLimits | None = None
) -> BaseLLMConnector:
    chosen = limits if limits is not None else UsageLimits()
    return gate_if_paid(connector, UsageGate(store=store, limits=lambda: chosen, clock=lambda: NOW))


def _usage(tokens_in: int, tokens_out: int) -> TokenUsage:
    return TokenUsage(
        provider="fakepaid", model="served", input_tokens=tokens_in, output_tokens=tokens_out
    )


# --- refusal ------------------------------------------------------------------------------

LIMIT = 987_654


def _full_store() -> MemoryUsageStore:
    """One row, a minute ago, of exactly `LIMIT` tokens: inside every window, at every limit."""
    store = MemoryUsageStore()
    store.add(UsageEntry(at=NOW - timedelta(minutes=1), tokens=LIMIT, provider="fakepaid"))
    return store


def _clock_text(moment: datetime) -> str:
    local = moment.astimezone()
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


@pytest.mark.parametrize("window", list(UsageWindow))
async def test_each_window_refuses_generate_before_any_provider_call(window: UsageWindow) -> None:
    connector = FakePaidConnector()
    store = _full_store()
    gated = _gated(connector, store, UsageLimits(**{window.value: LIMIT}))

    with pytest.raises(UsageLimitReachedError) as caught:
        await gated.generate(REQUEST)

    err = caught.value
    assert connector.calls == 0
    assert len(store.entries_since(NOW - timedelta(days=30))) == 1  # nothing booked
    assert isinstance(err, BudgetExceededError)
    assert err.window == window.value
    lifts = NOW - timedelta(minutes=1) + window.duration
    assert err.available_again_at == lifts

    message = str(err)
    assert f"{window.phrase} limit" in message
    assert _clock_text(lifts) in message
    assert "Settings → Usage" in message
    assert "local model" in message
    # No token figure, window code or class name.
    assert not re.search(r"987[,_ .]?654", message)
    for internal in ("UsageLimitReachedError", "Error", window.value, "token"):
        assert internal not in message


@pytest.mark.parametrize("window", list(UsageWindow))
async def test_each_window_refuses_stream_before_any_provider_call(window: UsageWindow) -> None:
    connector = FakePaidConnector(chunks=(StreamChunk(delta_content="x"),))
    store = _full_store()
    gated = _gated(connector, store, UsageLimits(**{window.value: LIMIT}))

    with pytest.raises(UsageLimitReachedError) as caught:
        async for _ in gated.stream(REQUEST):
            pytest.fail("a refused stream yielded a chunk")

    assert caught.value.window == window.value
    assert connector.calls == 0
    assert len(store.entries_since(NOW - timedelta(days=30))) == 1


async def test_ten_minute_message_leads_with_the_wait() -> None:
    gated = _gated(FakePaidConnector(), _full_store(), UsageLimits(per_10_minutes=LIMIT))

    with pytest.raises(UsageLimitReachedError) as caught:
        await gated.generate(REQUEST)

    # A minute-old row in a 10-minute window lifts in 9 minutes.
    assert "in 9 minutes" in str(caught.value)


async def test_a_call_under_every_limit_is_admitted() -> None:
    connector = FakePaidConnector()
    gated = _gated(connector, _full_store(), UsageLimits(per_10_minutes=LIMIT + 1))

    await gated.generate(REQUEST)

    assert connector.calls == 1


# --- recording ----------------------------------------------------------------------------


def _booked(store: MemoryUsageStore) -> list[UsageEntry]:
    return list(store.entries_since(NOW - timedelta(days=30)))


async def test_generate_records_the_provider_usage() -> None:
    store = MemoryUsageStore()
    await _gated(FakePaidConnector(), store).generate(REQUEST)

    assert _booked(store) == [
        UsageEntry(
            at=NOW,
            tokens=42,
            provider="fakepaid",
            model="served",
            count_source=TokenCountSource.PROVIDER,
        )
    ]


async def test_stream_records_the_last_reported_usage() -> None:
    store = MemoryUsageStore()
    chunks = (
        StreamChunk(delta_content="a", usage=_usage(10, 1)),
        StreamChunk(delta_content="b"),
        StreamChunk(usage=_usage(10, 7)),
    )
    got = [c async for c in _gated(FakePaidConnector(chunks), store).stream(REQUEST)]

    assert got == list(chunks)
    booked = _booked(store)
    assert [(e.tokens, e.count_source) for e in booked] == [(17, TokenCountSource.PROVIDER)]


async def test_stream_without_usage_records_an_estimate() -> None:
    store = MemoryUsageStore()
    chunks = (StreamChunk(delta_content="Sunny and "), StreamChunk(delta_content="warm."))
    async for _ in _gated(FakePaidConnector(chunks), store).stream(REQUEST):
        pass

    [entry] = _booked(store)
    assert entry.count_source is TokenCountSource.ESTIMATE
    assert entry.provider == "fakepaid"
    assert entry.tokens > 0


async def test_abandoned_stream_still_records() -> None:
    store = MemoryUsageStore()
    chunks = (StreamChunk(delta_content="one"), StreamChunk(delta_content="two"))
    stream = _gated(FakePaidConnector(chunks), store).stream(REQUEST)

    await stream.__anext__()
    await stream.aclose()  # type: ignore[attr-defined]

    [entry] = _booked(store)
    assert entry.count_source is TokenCountSource.ESTIMATE
    assert entry.tokens > 0


class FailingBeforeFirstChunk(FakePaidConnector):
    """A provider that refuses the request (a 401, a 429, no connection) before any chunk."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.calls += 1
        raise RuntimeError("connection refused")
        yield StreamChunk()  # pragma: no cover - makes this an async generator


async def test_stream_that_fails_before_any_chunk_books_nothing() -> None:
    # Nothing shows the provider served it, so it does not eat into the user's limit (#938).
    store = MemoryUsageStore()
    with pytest.raises(RuntimeError):
        async for _ in _gated(FailingBeforeFirstChunk(), store).stream(REQUEST):
            pass

    assert _booked(store) == []


class UnwritableStore(MemoryUsageStore):
    def add(self, entry: UsageEntry) -> None:
        raise OSError("disk full")


async def test_generate_keeps_the_reply_when_booking_fails() -> None:
    # The provider already served (and billed) the reply; a failed write must not discard it.
    response = await _gated(FakePaidConnector(), UnwritableStore()).generate(REQUEST)

    assert response.content == "hi"


# --- gate_if_paid -------------------------------------------------------------------------


async def test_gate_if_paid_is_idempotent() -> None:
    store = MemoryUsageStore()
    connector = FakePaidConnector()
    once = _gated(connector, store)
    first_type = type(once)

    twice = _gated(once, store)

    assert twice is connector
    assert type(twice) is first_type
    assert type(twice).__bases__ == (FakePaidConnector,)
    await twice.generate(REQUEST)
    assert len(_booked(store)) == 1  # booked once, not once per wrap


def test_gate_if_paid_leaves_an_unpaid_connector_untouched() -> None:
    connector = FakeLocalConnector()
    assert gate_if_paid(connector) is connector
    assert type(connector) is FakeLocalConnector


# --- factory output (G2, G8) --------------------------------------------------------------

_LOCAL_VLLM_HOSTS = [
    "http://127.0.0.1:8000/v1",
    "http://10.1.2.3:8000/v1",
    "http://192.168.0.20:8000/v1",
    "http://gpu-box:8000/v1",
]


@pytest.mark.parametrize(
    ("provider", "kwargs", "cls"),
    [
        ("openai", {"api_key": "sk-test"}, OpenAIConnector),
        ("anthropic", {"api_key": "sk-ant-test"}, AnthropicConnector),
        ("gemini", {"api_key": "gm-test"}, GeminiConnector),
        ("vllm", {"base_url": "https://llm.example.com/v1"}, None),
    ],
)
def test_factory_gates_paid_connectors(
    provider: str, kwargs: dict[str, Any], cls: type | None
) -> None:
    connector = create_llm_connector(provider=provider, **kwargs)
    ungated_type = type(connector).__bases__[0]

    assert getattr(type(connector), "_usage_gated", False) is True
    assert isinstance(connector, ungated_type)
    assert type(connector).__name__ == ungated_type.__name__
    if cls is not None:
        assert isinstance(connector, cls)
        assert type(connector).__name__ == cls.__name__


@pytest.mark.parametrize(
    ("provider", "kwargs"),
    [
        ("ollama", {"base_url": "http://127.0.0.1:11434"}),
        ("mock", {}),
        *[("vllm", {"base_url": url}) for url in _LOCAL_VLLM_HOSTS],
    ],
)
def test_factory_leaves_local_connectors_ungated(provider: str, kwargs: dict[str, Any]) -> None:
    connector = create_llm_connector(provider=provider, **kwargs)

    assert connector.paid is False
    assert getattr(type(connector), "_usage_gated", False) is False


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://[::1]:8000", True),
        ("http://localhost:11434", True),
        ("http://10.0.0.5", True),
        ("http://172.16.4.4", True),
        ("http://192.168.1.1", True),
        ("http://169.254.1.1", True),
        ("http://gpu-box:8000", True),
        ("http://nas.local", True),
        ("http://box.home.arpa", True),
        ("192.168.1.9:8000", True),
        ("https://api.openai.com/v1", False),
        ("https://llm.example.com", False),
        ("http://8.8.8.8", False),
        ("", False),
        (None, False),
    ],
)
def test_is_local_endpoint(url: str | None, local: bool) -> None:
    assert is_local_endpoint(url) is local


# --- no budget field on clone / persona config (G4) ---------------------------------------

#: `AgentLLMConfig`'s fields on origin/main when the usage limits were added, plus the
#: picture model ref (model-gateway §3.4), which is not a limit.
_AGENT_LLM_CONFIG_FIELDS = {
    "auto_compact",
    "compaction_threshold_tokens",
    "context_limit",
    "fast_model",
    "max_tokens",
    "image_model",
    "model_name",
    "model_tier",
    "temperature",
    "token_budget",
    "top_p",
}


def test_agent_llm_config_gains_no_usage_limit_field() -> None:
    fields = set(AgentLLMConfig.model_fields)
    added = fields - _AGENT_LLM_CONFIG_FIELDS
    assert not [f for f in added if "usage" in f or "limit" in f]
    assert fields == _AGENT_LLM_CONFIG_FIELDS
