"""The suggested model is always one the provider listed (#1631).

`model_policy.recommend` orders a listing by family pattern; `catalog.read_catalog` turns a
listing, or the failure to get one, into what Settings shows. The property the whole design
rests on is checked over many generated listings: the suggestion is an id from the listing,
and it is never missing while the listing holds a chat model.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    ProviderAuthError,
    ProviderOutageError,
    ProviderUnreachableError,
)
from uclone_x.llm.catalog import CatalogCache, CatalogEntry, CatalogResult, read_catalog
from uclone_x.llm.model_policy import recommend

_T0 = datetime(2026, 9, 25, tzinfo=UTC)


def _entries(*ids: str, chat: bool = True) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(id=model_id, chat_capable=chat) for model_id in ids)


@pytest.mark.parametrize(
    ("provider", "ids", "expected"),
    [
        # The mid tier wins over a bigger and a smaller model, newest version first.
        (
            "gemini",
            ("gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
            "gemini-2.5-flash",
        ),
        ("google", ("gemini-3-flash", "gemini-2.5-flash"), "gemini-3-flash"),
        # A preview is listed and selectable, but not suggested.
        ("gemini", ("gemini-3-flash-preview", "gemini-2.5-flash"), "gemini-2.5-flash"),
        # With no Flash listed, the next family down; never a model that is not listed.
        ("gemini", ("gemini-2.5-pro", "gemini-2.5-flash-lite"), "gemini-2.5-pro"),
        (
            "anthropic",
            ("claude-opus-4-1-20250805", "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"),
            "claude-sonnet-4-5-20250929",
        ),
        (
            "anthropic",
            ("claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219"),
            "claude-3-7-sonnet-20250219",
        ),
        ("openai", ("gpt-5-mini", "gpt-5", "gpt-4.1", "gpt-5-nano"), "gpt-5"),
        ("openai", ("gpt-4.1-mini", "gpt-4.1", "gpt-4o"), "gpt-4.1"),
        ("openai", ("gpt-5-mini", "gpt-4o"), "gpt-5-mini"),
    ],
)
def test_the_suggestion_is_the_newest_mid_tier_model_the_provider_listed(
    provider: str, ids: tuple[str, ...], expected: str
) -> None:
    """Owner ruling 2026-09-25: the mid-tier model is the default.

    Killed by: src/uclone_x/llm/model_policy.py :: return max(matched, key=lambda pair: (pair[0], _newest_key(pair[1])))[1].id
    Becomes: return matched[0][1].id
    """
    assert recommend(provider, _entries(*ids)) == expected


def test_a_renamed_family_falls_back_to_the_newest_listed_chat_model() -> None:
    """No pattern matches after a rename; the suggestion is still a listed model, not nothing."""
    entries = (
        CatalogEntry(id="nova-small", created_at=_T0 - timedelta(days=30)),
        CatalogEntry(id="nova-large", created_at=_T0),
        CatalogEntry(id="nova-embed", created_at=_T0 + timedelta(days=1), chat_capable=False),
    )

    assert recommend("gemini", entries) == "nova-large"


def test_a_listing_with_no_chat_model_suggests_nothing() -> None:
    assert recommend("openai", _entries("text-embedding-3-large", "tts-1", chat=False)) is None


_ID_PARTS = (
    "gemini-2.5-flash",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-embedding-001",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-1",
    "claude-haiku-4-5",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4o-realtime-preview",
    "tts-1",
    "whisper-1",
    "chat-latest",
    "totally-new-family",
)


@pytest.mark.parametrize("seed", range(300))
def test_the_suggestion_is_always_a_listed_chat_model(seed: int) -> None:
    """The property G1 rests on, over generated listings: never an id the listing lacks.

    And never None while a chat model is listed (G2), whatever the families are called.
    """
    rng = random.Random(seed)
    entries = tuple(
        CatalogEntry(
            id=rng.choice(_ID_PARTS) + rng.choice(("", "", "-x", "-20250101")),
            chat_capable=rng.random() > 0.2,
            created_at=_T0 - timedelta(days=rng.randint(0, 900)) if rng.random() > 0.3 else None,
        )
        for _ in range(rng.randint(0, 8))
    )
    provider = rng.choice(("gemini", "google", "anthropic", "openai", "somewhere-else"))

    suggestion = recommend(provider, entries)

    chat_ids = {entry.id for entry in entries if entry.chat_capable}
    if chat_ids:
        assert suggestion in chat_ids
    else:
        assert suggestion is None


def _lister(result: Sequence[CatalogEntry] | Exception):
    async def lister() -> Sequence[CatalogEntry]:
        if isinstance(result, Exception):
            raise result
        return result

    return lister


async def _read(lister: object) -> CatalogResult:
    return await read_catalog(
        provider="gemini",
        display_provider="Google",
        lister=lister,  # type: ignore[arg-type]
        recommend=recommend,
        now=lambda: _T0,
    )


@pytest.mark.asyncio
async def test_a_live_listing_carries_its_entries_and_a_listed_suggestion() -> None:
    result = await _read(_lister(_entries("gemini-2.5-pro", "gemini-2.5-flash")))

    assert result.status == "live"
    assert [entry.id for entry in result.entries] == ["gemini-2.5-pro", "gemini-2.5-flash"]
    assert result.recommended == "gemini-2.5-flash"
    assert result.fetched_at == _T0
    assert result.detail is None


@pytest.mark.parametrize(
    ("lister_result", "status", "detail"),
    [
        (None, "no_key", "Enter an API key to load the models it can use."),
        (LLMCredentialsNotConfiguredError("x"), "no_key", "Enter an API key"),
        (
            ProviderAuthError(provider="Google", model=""),
            "key_rejected",
            "did not accept the API key",
        ),
        (
            ProviderUnreachableError(provider="Google", model=""),
            "unreachable",
            "Couldn't get an answer",
        ),
        (ProviderOutageError(provider="Google", model=""), "unreachable", "having problems"),
        ((), "no_listing", "Google answered, but not with a model list."),
    ],
)
@pytest.mark.asyncio
async def test_a_listing_that_cannot_be_read_says_why_and_lists_nothing(
    lister_result: object, status: str, detail: str
) -> None:
    """G3: no bundled fallback list; an unavailable listing is shown as unavailable.

    Killed by: src/uclone_x/llm/catalog.py :: return CatalogResult(provider=provider, status="key_rejected", detail=str(exc))
    Becomes: return CatalogResult(provider=provider, status="unreachable", detail=str(exc))
    """
    lister = None if lister_result is None else _lister(lister_result)  # type: ignore[arg-type]

    result = await _read(lister)

    assert result.status == status
    assert result.entries == ()
    assert result.recommended is None
    assert result.detail is not None and detail in result.detail
    for internal in ("{", "404", "Error", "http"):
        assert internal not in result.detail, internal


def test_the_cache_keeps_a_live_listing_until_it_expires() -> None:
    """Killed by: src/uclone_x/llm/catalog.py :: if self._now() - stored_at >= ttl:
    Becomes: if False:
    """
    clock = [_T0]
    cache = CatalogCache(ttl=timedelta(hours=6), now=lambda: clock[0])
    live = CatalogResult(provider="gemini", status="live", fetched_at=_T0)
    key = ("gemini", "", "fp")

    cache.put(key, live)
    clock[0] = _T0 + timedelta(hours=5)
    assert cache.get(key) == live
    clock[0] = _T0 + timedelta(hours=6)
    assert cache.get(key) is None


def test_a_failed_listing_is_kept_for_a_minute_not_six_hours() -> None:
    """An offline machine is not asked on every page load, nor hidden once it is back online.

    Killed by: src/uclone_x/llm/catalog.py :: ttl = self._ttl if result.status == "live" else self._failure_ttl
    Becomes: ttl = self._ttl
    """
    clock = [_T0]
    cache = CatalogCache(now=lambda: clock[0])
    key = ("gemini", "", "fp")
    failed = CatalogResult(provider="gemini", status="unreachable")

    cache.put(key, failed)
    clock[0] = _T0 + timedelta(seconds=30)
    assert cache.get(key) == failed
    clock[0] = _T0 + timedelta(seconds=60)
    assert cache.get(key) is None
