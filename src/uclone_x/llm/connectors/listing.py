"""The shared half of each cloud connector's `list_models`: one GET, failures typed (#1631).

A listing request fails the way a turn does -- a revoked key, no network, an outage -- and it
is reported with the same typed errors, so Settings can say which of those it was. What the
listing *contains* is parsed in each connector's own module, where the provider's shape is.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import httpx

from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.failures import failed_request, failed_status, unusable_response

#: A listing names no model; the failure's model field is left empty rather than invented.
_NO_MODEL = ""

#: Pages read before a listing is taken as complete. At a thousand models a page, reaching it
#: means a provider looping on its page token, not a catalogue that large.
MAX_LISTING_PAGES = 10


async def get_listing_page(
    connector: BaseLLMConnector,
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GET one page of a provider's model listing, or raise its `ProviderFailureError`."""
    client = connector._get_client()  # pyright: ignore[reportPrivateUsage]
    should_close = connector._http_client is None  # pyright: ignore[reportPrivateUsage]
    try:
        resp = await client.get(url, headers=headers, params=params, timeout=connector.timeout)
        if resp.status_code != 200:
            raise failed_status(
                provider=provider, model=_NO_MODEL, status_code=resp.status_code, body=resp.text
            )
        data: object = resp.json()
    except httpx.RequestError as exc:
        raise failed_request(provider=provider, model=_NO_MODEL, exc=exc) from exc
    except json.JSONDecodeError as exc:
        raise unusable_response(provider=provider, model=_NO_MODEL, detail=str(exc)) from exc
    finally:
        if should_close:
            await client.aclose()
    if not isinstance(data, dict):
        raise unusable_response(provider=provider, model=_NO_MODEL, detail="listing not an object")
    return cast(dict[str, Any], data)


def listed_items(page: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The dict items under `key`, skipping anything that is not one."""
    raw = page.get(key)
    if not isinstance(raw, list):
        return []
    return [
        cast(dict[str, Any], item) for item in cast(list[object], raw) if isinstance(item, dict)
    ]


def optional_int(value: object) -> int | None:
    """A positive integer the listing reported, or None when it did not report one."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def from_unix(value: object) -> datetime | None:
    """OpenAI's `created`, seconds since the epoch."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return datetime.fromtimestamp(value, tz=UTC)


def from_iso(value: object) -> datetime | None:
    """Anthropic's `created_at`, an RFC 3339 timestamp."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
