"""Each cloud connector reads its provider's model listing into `CatalogEntry`s (#1631).

The listings are recorded shapes of each provider's documented `GET /models` answer, served
by an `httpx.MockTransport`: what is asserted is what the connector made of them -- which
models it kept, which it marked as unable to chat, and how it paged -- and that a refused
listing raises the same typed failure a refused turn does.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from uclone_x.errors import ProviderAuthError, ProviderUnreachableError
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.listing import MAX_LISTING_PAGES
from uclone_x.llm.connectors.openai import OpenAIConnector

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _pages(*pages: dict[str, Any], seen: list[httpx.Request]) -> Handler:
    remaining = list(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=remaining.pop(0) if remaining else pages[-1])

    return handler


_GEMINI_PAGE_1: dict[str, Any] = {
    "models": [
        {
            "name": "models/gemini-2.5-flash",
            "displayName": "Gemini 2.5 Flash",
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 65536,
            "supportedGenerationMethods": ["generateContent", "countTokens"],
        },
        {
            "name": "models/gemini-embedding-001",
            "displayName": "Gemini Embedding 001",
            "inputTokenLimit": 2048,
            "supportedGenerationMethods": ["embedContent"],
        },
    ],
    "nextPageToken": "page-2",
}
_GEMINI_PAGE_2: dict[str, Any] = {
    "models": [
        {
            "name": "models/gemini-2.5-pro",
            "supportedGenerationMethods": ["generateContent"],
        }
    ]
}


@pytest.mark.asyncio
async def test_gemini_reads_every_page_and_marks_models_that_cannot_chat() -> None:
    """Killed by: src/uclone_x/llm/connectors/gemini.py :: params = {"pageSize": "1000", "pageToken": token}
    Becomes: break
    """
    seen: list[httpx.Request] = []
    connector = GeminiConnector(
        api_key="k", http_client=_client(_pages(_GEMINI_PAGE_1, _GEMINI_PAGE_2, seen=seen))
    )

    entries = await connector.list_models()

    assert [(e.id, e.chat_capable) for e in entries] == [
        ("gemini-2.5-flash", True),
        ("gemini-embedding-001", False),
        ("gemini-2.5-pro", True),
    ]
    assert entries[0].display_name == "Gemini 2.5 Flash"
    assert (entries[0].context_window, entries[0].max_output_tokens) == (1048576, 65536)
    assert seen[0].url.path.endswith("/models")
    assert seen[0].headers["x-goog-api-key"] == "k"
    assert seen[1].url.params["pageToken"] == "page-2"


@pytest.mark.asyncio
async def test_a_listing_that_never_stops_paging_is_cut_off() -> None:
    """A provider looping on one page token is not followed forever."""
    seen: list[httpx.Request] = []
    looping = {**_GEMINI_PAGE_2, "nextPageToken": "same"}
    connector = GeminiConnector(api_key="k", http_client=_client(_pages(looping, seen=seen)))

    entries = await connector.list_models()

    assert len(seen) == MAX_LISTING_PAGES
    assert len(entries) == MAX_LISTING_PAGES


_ANTHROPIC_PAGE_1: dict[str, Any] = {
    "data": [
        {
            "type": "model",
            "id": "claude-sonnet-4-5-20250929",
            "display_name": "Claude Sonnet 4.5",
            "created_at": "2025-09-29T00:00:00Z",
            "max_input_tokens": 200000,
            "max_tokens": 64000,
        }
    ],
    "has_more": True,
    "first_id": "claude-sonnet-4-5-20250929",
    "last_id": "claude-sonnet-4-5-20250929",
}
_ANTHROPIC_PAGE_2: dict[str, Any] = {
    "data": [
        {
            "type": "model",
            "id": "claude-3-5-haiku-20241022",
            "display_name": "Claude Haiku 3.5",
            "created_at": "2024-10-22T00:00:00Z",
        }
    ],
    "has_more": False,
    "last_id": "claude-3-5-haiku-20241022",
}


@pytest.mark.asyncio
async def test_anthropic_follows_has_more_and_reads_what_each_model_reports() -> None:
    """Killed by: src/uclone_x/llm/connectors/anthropic.py :: params = {"limit": "1000", "after_id": last_id}
    Becomes: break
    """
    seen: list[httpx.Request] = []
    connector = AnthropicConnector(
        api_key="k",
        http_client=_client(_pages(_ANTHROPIC_PAGE_1, _ANTHROPIC_PAGE_2, seen=seen)),
    )

    entries = await connector.list_models()

    assert [e.id for e in entries] == ["claude-sonnet-4-5-20250929", "claude-3-5-haiku-20241022"]
    assert all(e.chat_capable for e in entries)
    assert entries[0].created_at == datetime(2025, 9, 29, tzinfo=UTC)
    assert (entries[0].context_window, entries[0].max_output_tokens) == (200000, 64000)
    assert entries[1].context_window is None
    assert seen[0].headers["x-api-key"] == "k"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert seen[1].url.params["after_id"] == "claude-sonnet-4-5-20250929"


_OPENAI_LISTING: dict[str, Any] = {
    "object": "list",
    "data": [
        {"id": "gpt-5", "object": "model", "created": 1754524800, "owned_by": "system"},
        {"id": "gpt-5-mini", "object": "model", "created": 1754524800, "owned_by": "system"},
        {"id": "text-embedding-3-large", "object": "model", "created": 1705953180},
        {"id": "tts-1-hd", "object": "model", "created": 1699046015},
        {"id": "whisper-1", "object": "model", "created": 1677532384},
        {"id": "gpt-4o-mini-transcribe", "object": "model", "created": 1742068463},
        {"id": "dall-e-3", "object": "model", "created": 1698785189},
        {"id": "omni-moderation-latest", "object": "model", "created": 1731689265},
        {"id": "davinci-002", "object": "model", "created": 1692634301},
    ],
}


@pytest.mark.asyncio
async def test_openai_keeps_every_listed_model_and_marks_the_ones_that_cannot_chat() -> None:
    """Embedding, speech, image and moderation models are listed but never suggested.

    Killed by: src/uclone_x/llm/connectors/openai.py :: chat_capable=not any(fragment in lowered for fragment in _NOT_CHAT_FRAGMENTS),
    Becomes: chat_capable=True,
    """
    seen: list[httpx.Request] = []
    connector = OpenAIConnector(
        api_key="k", http_client=_client(_pages(_OPENAI_LISTING, seen=seen))
    )

    entries = await connector.list_models()

    assert [e.id for e in entries if e.chat_capable] == ["gpt-5", "gpt-5-mini"]
    assert len(entries) == len(_OPENAI_LISTING["data"])
    assert entries[0].created_at == datetime.fromtimestamp(1754524800, tz=UTC)
    assert seen[0].url.path.endswith("/models")
    assert seen[0].headers["authorization"] == "Bearer k"


@pytest.mark.asyncio
async def test_a_refused_key_on_the_listing_raises_the_same_error_a_turn_would() -> None:
    connector = GeminiConnector(
        api_key="k",
        http_client=_client(
            lambda _request: httpx.Response(
                400, json={"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid."}}
            )
        ),
    )

    with pytest.raises(ProviderAuthError):
        await connector.list_models()


@pytest.mark.asyncio
async def test_an_unreachable_listing_raises_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    connector = OpenAIConnector(api_key="k", http_client=_client(refuse))

    with pytest.raises(ProviderUnreachableError):
        await connector.list_models()
