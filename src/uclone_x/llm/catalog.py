"""What a cloud provider says it serves to this key, and the model to suggest from it (#1631).

A model id is a fact about a provider's catalogue on some date, so none is written here. The
provider's own listing is the only source for whether a model exists; `model_policy` only
orders what the listing returned. When the listing cannot be read, the result says why and
carries no entries -- never a bundled fallback list, which would render a guess as a
detection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    ProviderAuthError,
    ProviderFailureError,
    ProviderUnreachableError,
)
from uclone_x.llm.context_window import LISTED_CONTEXT_WINDOWS, ListedContextWindows
from uclone_x.llm.providers import canonical_provider

CatalogStatus = Literal["live", "no_key", "key_rejected", "unreachable", "no_listing"]

#: How long a listing is reused before the provider is asked again. Listings are free on all
#: three providers but count against rate limits; six hours keeps it to a few calls a day.
CATALOG_TTL: Final = timedelta(hours=6)

#: How long a listing that could not be read is reused. Settings is read on every page load,
#: and an offline machine or a refused key would otherwise wait out the listing timeout each
#: time; a minute is short enough that a fixed network or a new key shows at the next open,
#: and Refresh asks again at once.
CATALOG_FAILURE_TTL: Final = timedelta(seconds=60)


class CatalogEntry(BaseModel):
    """One model the provider listed, as its listing described it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="The id a request names, as the provider spells it.")
    display_name: str | None = Field(default=None, description="The provider's own label.")
    context_window: int | None = Field(
        default=None, description="Input tokens the provider reports; None when it is silent."
    )
    max_output_tokens: int | None = Field(default=None)
    chat_capable: bool = Field(
        default=True, description="False for embedding, speech, image and moderation models."
    )
    created_at: datetime | None = Field(
        default=None, description="When the provider says the model was published, if it does."
    )
    accepts_images: bool = Field(
        default=False,
        description="True only when the provider's own listing says the model reads images "
        "(Anthropic's `capabilities.image_input.supported`, #2107). False when the listing "
        "is silent, as OpenAI's and Gemini's are: nothing here guesses from a model's name.",
    )


class ListedImageInput:
    """Which listed models their provider says read images (#2107).

    Filled from the catalogue (`read_catalog`) whenever a listing is read, and by a connector
    that reads its provider's listing to answer `accepts_images`. Like `ListedContextWindows`
    a model is found by its id exactly, never by a prefix. `get` is `None` for a provider
    whose listing was never read here, so a caller can tell "not read" from "read, and no".
    """

    def __init__(self) -> None:
        self._read: set[str] = set()
        self._accepting: set[tuple[str, str]] = set()

    def remember(self, provider: str, entries: Sequence[CatalogEntry]) -> None:
        """Record which of `entries` read images, and that `provider`'s listing was read."""
        key = canonical_provider(provider) or provider.strip().lower()
        self._read.add(key)
        for entry in entries:
            if entry.accepts_images:
                self._accepting.add((key, entry.id.strip()))

    def get(self, provider: str | None, model: str | None) -> bool | None:
        """Whether `model` reads images, or `None` when no listing of `provider` was read."""
        if not provider:
            return None
        key = canonical_provider(provider) or provider.strip().lower()
        if key not in self._read:
            return None
        return bool(model) and (key, (model or "").strip()) in self._accepting


#: The store a turn reads. One per process, like `LISTED_CONTEXT_WINDOWS`.
LISTED_IMAGE_INPUT = ListedImageInput()


class CatalogResult(BaseModel):
    """A provider's listing for one key and endpoint, or the reason there is none."""

    model_config = ConfigDict(frozen=True)

    provider: str
    status: CatalogStatus
    entries: tuple[CatalogEntry, ...] = ()
    recommended: str | None = Field(
        default=None, description="Always the id of an entry, or None when there is none."
    )
    fetched_at: datetime | None = None
    detail: str | None = Field(
        default=None, description="A plain sentence for a status that is not `live`."
    )


#: What a connector's `list_models` does: ask the provider, or raise a `ProviderFailureError`.
Lister = Callable[[], Awaitable[Sequence[CatalogEntry]]]

_NO_KEY_DETAIL: Final = "Enter an API key to load the models it can use."
_NO_LISTING_DETAIL: Final = (
    "{provider} answered, but not with a model list. You can still type a model name."
)


async def read_catalog(
    *,
    provider: str,
    display_provider: str,
    lister: Lister | None,
    recommend: Callable[[str, Sequence[CatalogEntry]], str | None],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    windows: ListedContextWindows | None = None,
    images: ListedImageInput | None = None,
) -> CatalogResult:
    """Ask the provider for its listing, and say plainly why when it cannot be read.

    `lister` is None when there is no key to ask with. `display_provider` is the name the
    user holds the key with ("Google"), for the sentence; `provider` is the settings id.
    The window each listed model reports is remembered in `windows`
    (`LISTED_CONTEXT_WINDOWS` by default), where the compaction trigger and the room's
    context readout look it up (#1978). Whether each reads images is remembered in `images`
    (`LISTED_IMAGE_INPUT` by default), where a turn asks before offering `look` (#2107).
    """
    if lister is None:
        return CatalogResult(provider=provider, status="no_key", detail=_NO_KEY_DETAIL)
    try:
        listed = tuple(await lister())
    except LLMCredentialsNotConfiguredError:
        return CatalogResult(provider=provider, status="no_key", detail=_NO_KEY_DETAIL)
    except ProviderAuthError as exc:
        return CatalogResult(provider=provider, status="key_rejected", detail=str(exc))
    except ProviderUnreachableError as exc:
        return CatalogResult(provider=provider, status="unreachable", detail=str(exc))
    except ProviderFailureError as exc:
        # An outage, a spent quota, or an answer that is not a listing (a proxy without
        # `/models`): the provider cannot say what exists right now, and nothing is guessed.
        return CatalogResult(provider=provider, status="unreachable", detail=str(exc))
    if not listed:
        return CatalogResult(
            provider=provider,
            status="no_listing",
            detail=_NO_LISTING_DETAIL.format(provider=display_provider),
        )
    (windows if windows is not None else LISTED_CONTEXT_WINDOWS).remember(provider, listed)
    (images if images is not None else LISTED_IMAGE_INPUT).remember(provider, listed)
    return CatalogResult(
        provider=provider,
        status="live",
        entries=listed,
        recommended=recommend(provider, listed),
        fetched_at=now(),
    )


def key_fingerprint(api_key: str | None) -> str:
    """A cache key part that tells two keys apart without holding either."""
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


class CatalogCache:
    """Listings kept in memory per (provider, endpoint, key).

    A `live` result is kept for `CATALOG_TTL`; any other for `CATALOG_FAILURE_TTL`, so a
    refused connection is not asked about on every page load, nor hidden for six hours once
    the network is back. Nothing is written to disk, because a stale listing on disk would
    bring back the retired defaults this module replaces.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = CATALOG_TTL,
        failure_ttl: timedelta = CATALOG_FAILURE_TTL,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._ttl = ttl
        self._failure_ttl = failure_ttl
        self._now = now
        self._results: dict[tuple[str, str, str], tuple[datetime, CatalogResult]] = {}

    def get(self, key: tuple[str, str, str]) -> CatalogResult | None:
        """The kept result for `key`, or None when there is none or it has expired."""
        kept = self._results.get(key)
        if kept is None:
            return None
        stored_at, result = kept
        ttl = self._ttl if result.status == "live" else self._failure_ttl
        if self._now() - stored_at >= ttl:
            del self._results[key]
            return None
        return result

    def put(self, key: tuple[str, str, str], result: CatalogResult) -> None:
        """Keep `result` for `key`, timed from now."""
        self._results[key] = (self._now(), result)

    def clear(self) -> None:
        """Forget every listing, for an explicit refresh."""
        self._results.clear()
