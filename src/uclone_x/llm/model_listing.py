"""Where a connection's model list comes from: a cloud provider's listing or a local server.

Moved out of the dashboard (`ui/app.py`) so the Core's model gateway
(`uclone_x.llm.gateway`) and the dashboard read a connection's models one way. Nothing
here keeps a remembered list: a listing that cannot be read is reported as such
(provider-model-catalog G3).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, Final, cast

import httpx

from uclone_x.llm.catalog import (
    CatalogCache,
    CatalogEntry,
    CatalogResult,
    Lister,
    key_fingerprint,
    read_catalog,
)
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.ollama import resolve_ollama_base_url
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.vllm import has_configured_vllm_endpoint, resolve_vllm_base_url
from uclone_x.llm.model_policy import recommend

logger = logging.getLogger(__name__)


def vllm_request_headers(
    api_key: str | None = None, *, env_fallback: bool = True
) -> dict[str, str]:
    """`Authorization` only when a key exists, because `vllm serve --api-key` is optional.

    An empty bearer is not the same as no header: a server started without `--api-key`
    accepts the request either way, but one behind a proxy that reads the header rejects
    `Bearer ` with a 401 that describes a credential nobody configured (#385).

    `env_fallback=False` sends only the key passed in: for an address typed on the form, which
    must not receive the key held for the saved server (#1666).
    """
    env_key = os.getenv("VLLM_API_KEY") if env_fallback else None
    key = (api_key or env_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def vllm_model_ids(payload: object) -> list[str]:
    """The `id`s in an OpenAI-compatible `/v1/models` listing, or an empty list.

    vLLM answers `{"object": "list", "data": [{"id": "<the --model argument>", ...}]}`, and
    that one entry is the model the server was started with — which is why this listing is
    worth showing at all: for vLLM it is not a catalogue of what *could* be run but a
    statement of what *is* running. Parsed defensively for the reason the Ollama branch
    below is: this is another process's JSON, and a gateway in front of it may answer
    something else entirely.
    """
    if not isinstance(payload, dict):
        return []
    raw = cast(dict[str, Any], payload).get("data")
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            value = cast(dict[str, Any], item).get("id")
            if value is not None:
                ids.append(str(value))
    return ids


class LocalKeyRefusedError(Exception):
    """A local server answered 401 or 403: it is running, and it refused the key (#1672).

    Raised by `list_local_models` only when asked to, so the form can say "the server refused
    the key" rather than "no model list came back", which sends the user to look for a
    stopped server that is not stopped.
    """


#: The statuses that mean the server is there and turned the key away.
_KEY_REFUSED_STATUSES = frozenset({401, 403})


#: Ollama's own capability names (`/api/tags` and `/api/show` both carry them since 0.6):
#: a model that can hold a conversation lists `completion`; an embedder lists only
#: `embedding`.
_OLLAMA_CHAT_CAPABILITY: Final = "completion"


def _ollama_capabilities(item: Mapping[str, Any]) -> list[str] | None:
    """The `capabilities` list an Ollama answer carries for one model, or None when silent."""
    raw = item.get("capabilities")
    if not isinstance(raw, list):
        return None
    return [str(c) for c in cast(list[object], raw)]


def _ollama_encoder_only(item: Mapping[str, Any]) -> bool:
    """Whether the model's own family says it cannot chat: a BERT-family encoder.

    The fallback for an Ollama older than its `capabilities` field. It reads the family the
    listing reports (`details.families`: `bert`, `nomic-bert`, `xlm-roberta`, each a BERT), never the
    model's name. Every embedder Ollama's library ships (`bge-m3`, `nomic-embed-text`,
    `all-minilm`, `mxbai-embed-large`, `snowflake-arctic-embed`) is one, and no chat
    model is.
    """
    details = item.get("details")
    if not isinstance(details, Mapping):
        return False
    details_map = cast(Mapping[str, Any], details)
    families: list[object] = []
    raw_families = details_map.get("families")
    if isinstance(raw_families, list):
        families.extend(cast(list[object], raw_families))
    families.append(details_map.get("family"))
    return any(isinstance(f, str) and "bert" in f.lower() for f in families)


async def list_ollama_entries(
    base_url: str | None = None, timeout: float = 3.0
) -> list[CatalogEntry] | None:
    """An Ollama server's installed models, each marked whether it can chat (#2167).

    None when no listing came back, as for `list_local_models`. Whether a model chats is
    Ollama's own answer: the `capabilities` its `/api/tags` entry carries, else the ones
    `/api/show` reports for that model, else (an Ollama too old for either) the family the
    listing names (`_ollama_encoder_only`). Nothing here reads the model's name. A model
    none of the three speaks for is offered, as before: a listing that is silent is not a
    reason to hide a model.
    """
    ollama_url = ((base_url or "").strip() or resolve_ollama_base_url()).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as http_c:
            resp = await http_c.get(f"{ollama_url}/api/tags")
            if resp.status_code != 200:
                return None  # something answered, but not Ollama's listing
            data_obj: object = resp.json()
            items: list[Mapping[str, Any]] = []
            if isinstance(data_obj, dict):
                raw_models = cast(dict[str, Any], data_obj).get("models")
                if isinstance(raw_models, list):
                    for item in cast(list[object], raw_models):
                        if isinstance(item, dict):
                            mapping = cast(dict[str, Any], item)
                            if mapping.get("name") is not None:
                                items.append(mapping)

            async def capabilities_of(item: Mapping[str, Any]) -> list[str] | None:
                listed = _ollama_capabilities(item)
                if listed is not None:
                    return listed
                return await _ollama_show_capabilities(http_c, ollama_url, str(item["name"]))

            answers = await asyncio.gather(*(capabilities_of(item) for item in items))
            return [
                CatalogEntry(
                    id=str(item["name"]),
                    chat_capable=(
                        _OLLAMA_CHAT_CAPABILITY in capabilities
                        if capabilities is not None
                        else not _ollama_encoder_only(item)
                    ),
                )
                for item, capabilities in zip(items, answers, strict=True)
            ]
    except Exception as exc:
        logger.debug("Failed to query Ollama tags from %s: %s", ollama_url, exc)
        return None


async def _ollama_show_capabilities(
    http_c: httpx.AsyncClient, ollama_url: str, name: str
) -> list[str] | None:
    """`/api/show`'s `capabilities` for one model, or None when it does not say."""
    try:
        resp = await http_c.post(f"{ollama_url}/api/show", json={"model": name})
        if resp.status_code != 200:
            return None
        body: object = resp.json()
    except Exception as exc:  # one model's detail failing does not lose the listing
        logger.debug("Ollama /api/show for %s failed: %s", name, exc)
        return None
    return _ollama_capabilities(cast(dict[str, Any], body)) if isinstance(body, dict) else None


async def list_local_models(
    provider: str,
    base_url: str | None = None,
    timeout: float = 3.0,
    *,
    vllm_headers: dict[str, str] | None = None,
    raise_on_refused_key: bool = False,
) -> list[str] | None:
    """The models a local server has installed, or None when no listing came back (#1666).

    None and `[]` are different answers, and Settings says different things for them: no
    listing (nothing answered, or something that is not this server did) names a stopped
    server or a wrong address, where an empty list names an install
    the user has not done yet. `fetch_available_models` folds the two together.

    `vllm_headers` replaces the default ones, which carry `VLLM_API_KEY`; for an address the person typed.
    With `raise_on_refused_key`, a vLLM server's 401 or 403 raises `LocalKeyRefusedError`
    instead of reading as no listing.
    """
    clean_provider = provider.strip().lower()
    if clean_provider == "ollama":
        entries = await list_ollama_entries(base_url, timeout)
        return None if entries is None else [entry.id for entry in entries]
    if clean_provider == "vllm":
        if not has_configured_vllm_endpoint(base_url):
            # An unconfigured endpoint is not an empty inventory. Probing vLLM's documented
            # default port to fill the dropdown would be this module guessing where the
            # operator's server is, and reporting a refused connection as "no models" (P6).
            return None
        vllm_url = resolve_vllm_base_url(base_url)
        listing_url = f"{vllm_url.rstrip('/')}/models"
        refused = False
        try:
            async with httpx.AsyncClient(timeout=timeout) as http_c:
                resp = await http_c.get(
                    listing_url,
                    headers=vllm_request_headers() if vllm_headers is None else vllm_headers,
                )
                if resp.status_code == 200:
                    return vllm_model_ids(resp.json())
                refused = resp.status_code in _KEY_REFUSED_STATUSES
        except Exception as exc:
            logger.debug("Failed to query vLLM models from %s: %s", vllm_url, exc)
            return None
        if refused and raise_on_refused_key:
            raise LocalKeyRefusedError(vllm_url)
        return None
    if clean_provider == "mock":
        return ["mock-gpt-4o", "mock-llm"]
    return None


async def fetch_available_models(
    provider: str,
    base_url: str | None = None,
    timeout: float = 3.0,
) -> list[str]:
    """Enumerate the models a local server has installed (P0/Recognition over Recall).

    The cloud providers are not answered here: their models come from each provider's own
    listing, through `read_provider_catalog` (#1631). The remembered lists this function
    returned for them had retired models in them.
    """
    return await list_local_models(provider, base_url, timeout) or []


#: The providers whose installed models Settings lists from the server itself (#1666).
LOCAL_LISTING_PROVIDERS = frozenset({"ollama", "vllm"})

CatalogConnector = type[OpenAIConnector] | type[AnthropicConnector] | type[GeminiConnector]

#: The cloud providers whose own model listing Settings shows (#1631): the settings id, the
#: connector that reads the listing, and the name the user holds the key with.
CATALOG_PROVIDERS: dict[str, tuple[str, CatalogConnector, str]] = {
    "openai": ("openai", OpenAIConnector, "OpenAI"),
    "anthropic": ("anthropic", AnthropicConnector, "Anthropic"),
    "gemini": ("gemini", GeminiConnector, "Google"),
    "google": ("gemini", GeminiConnector, "Google"),
}

#: A listing takes one request per page; Settings waits for it before the picker fills.
_CATALOG_TIMEOUT_SECONDS = 10.0


async def read_provider_catalog(
    provider: str,
    *,
    base_url: str | None,
    api_key: str | None,
    cache: CatalogCache,
    cache_id: str | None = None,
) -> CatalogResult | None:
    """The provider's own model listing for this key and endpoint, or None for a local server.

    A live listing is reused from `cache` until it expires; anything else is asked again, so
    a fixed key or network shows on the next open (`CatalogCache`). `cache_id` is the
    connection the listing is kept under (model-gateway §3.3: connection id, key
    fingerprint and address); by default the provider's id.
    """
    spec = CATALOG_PROVIDERS.get(provider.strip().lower())
    if spec is None:
        return None
    settings_id, connector_cls, display_provider = spec
    endpoint = (base_url or "").strip() or None
    cache_key = (cache_id or settings_id, endpoint or "", key_fingerprint(api_key))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    lister: Lister | None = None
    if api_key and api_key.strip():
        key = api_key.strip()

        async def list_models() -> list[CatalogEntry]:
            connector = connector_cls(
                api_key=key, base_url=endpoint, timeout=_CATALOG_TIMEOUT_SECONDS
            )
            return await connector.list_models()

        lister = list_models

    result = await read_catalog(
        provider=settings_id,
        display_provider=display_provider,
        lister=lister,
        recommend=recommend,
    )
    cache.put(cache_key, result)
    return result
