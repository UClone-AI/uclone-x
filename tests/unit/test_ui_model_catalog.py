"""Settings shows the models the provider lists for this key, and nothing remembered (#1631).

The connector's `list_models` is replaced by a recorded listing, so what is checked is the
wiring: which key and endpoint the listing is asked with, what `/api/settings` and
`/api/models` return from it, and when a kept listing is asked for again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from uclone_x.errors import ProviderAuthError
from uclone_x.llm.catalog import CatalogCache, CatalogEntry
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.ui import app as app_module
from uclone_x.ui.app import create_ui_app, list_local_models, read_provider_catalog

_LISTING = [
    CatalogEntry(id="gemini-2.5-pro", context_window=1048576),
    CatalogEntry(id="gemini-2.5-flash", context_window=1048576),
    CatalogEntry(id="gemini-embedding-001", chat_capable=False),
]


@pytest.fixture(autouse=True)
def _no_model_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    for name in ("GEMINI_MODEL", "LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class _Listing:
    """A recorded `list_models`, counting the calls and the connectors that made them."""

    def __init__(self, result: list[CatalogEntry] | Exception) -> None:
        self.result = result
        self.calls: list[GeminiConnector] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        listing = self

        async def list_models(self: GeminiConnector) -> list[CatalogEntry]:
            listing.calls.append(self)
            if isinstance(listing.result, Exception):
                raise listing.result
            return listing.result

        monkeypatch.setattr(GeminiConnector, "list_models", list_models)


async def _get(app: Any, path: str) -> dict[str, Any]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get(path)
    assert res.status_code == 200, res.text
    return cast(dict[str, Any], res.json())


@pytest.mark.asyncio
async def test_settings_lists_what_the_provider_listed_and_suggests_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G1/G2: the picker holds the provider's chat models; no remembered id is shown.

    Killed by: src/uclone_x/ui/app.py :: settings["catalog"] = catalog.model_dump(mode="json") if catalog else None
    Becomes: settings["catalog"] = None

    Killed by: src/uclone_x/ui/app.py :: "llm_model": active_model or "",
    Becomes: "llm_model": active_model or "gemini-1.5-pro",
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = create_ui_app(
        static_dir=tmp_path, llm=GeminiConnector(api_key="AIza-test-key"), storage_dir=tmp_path
    )

    data = await _get(app, "/api/settings")

    catalog = data["catalog"]
    assert catalog["status"] == "live"
    assert catalog["recommended"] == "gemini-2.5-flash"
    assert [e["id"] for e in catalog["entries"]] == [e.id for e in _LISTING]
    assert data["available_models"] == ["gemini-2.5-pro", "gemini-2.5-flash"]
    assert data["llm_model"] == ""
    assert listing.calls[0].api_key == "AIza-test-key"


@pytest.mark.asyncio
async def test_a_kept_listing_is_reused_until_refresh_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: catalog_cache.clear()  # the saved provider's "Refresh list"
    Becomes: pass
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = create_ui_app(
        static_dir=tmp_path, llm=GeminiConnector(api_key="AIza-test-key"), storage_dir=tmp_path
    )

    await _get(app, "/api/settings")
    models = await _get(app, "/api/models")
    assert len(listing.calls) == 1
    assert models["catalog"]["recommended"] == "gemini-2.5-flash"

    refreshed = await _get(app, "/api/models?refresh=1")

    assert len(listing.calls) == 2
    assert refreshed["models"] == ["gemini-2.5-pro", "gemini-2.5-flash"]


@pytest.mark.asyncio
async def test_a_refused_key_is_reported_and_no_models_are_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G3: no remembered list fills the picker when the provider refused to answer."""
    _Listing(ProviderAuthError(provider="Google", model="")).install(monkeypatch)
    app = create_ui_app(
        static_dir=tmp_path, llm=GeminiConnector(api_key="AIza-test-key"), storage_dir=tmp_path
    )

    data = await _get(app, "/api/settings")

    assert data["catalog"]["status"] == "key_rejected"
    assert "did not accept the API key" in data["catalog"]["detail"]
    assert data["available_models"] == []


@pytest.mark.asyncio
async def test_with_no_key_the_provider_is_not_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank key reads as no key, and the listing is not requested with it."""
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)

    result = await read_provider_catalog(
        "google", base_url=None, api_key="  ", cache=CatalogCache()
    )

    assert result is not None
    assert (result.provider, result.status) == ("gemini", "no_key")
    assert listing.calls == []


@pytest.mark.asyncio
async def test_a_custom_endpoint_is_where_the_listing_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listing goes where the turns would, and each endpoint keeps its own listing."""
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    cache = CatalogCache()

    await read_provider_catalog(
        "gemini", base_url="https://proxy.invalid/v1beta", api_key="k", cache=cache
    )
    await read_provider_catalog("gemini", base_url=None, api_key="k", cache=cache)

    assert [c.base_url for c in listing.calls] == [
        "https://proxy.invalid/v1beta",
        GeminiConnector(api_key="k").base_url,
    ]


@pytest.mark.asyncio
async def test_a_local_server_keeps_its_installed_model_list(tmp_path: Path) -> None:
    """Ollama, vLLM and the mock have no provider listing; their inventory is unchanged."""
    app = create_ui_app(
        static_dir=tmp_path,
        llm=MockLLMConnector(api_key="sk-abcdef123456", base_url="http://mock-llm.invalid:8000"),
        storage_dir=tmp_path,
    )

    data = await _get(app, "/api/settings")

    assert data["catalog"] is None
    assert "mock-llm" in data["available_models"]


@pytest.mark.asyncio
async def test_a_new_key_reads_the_listing_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listing kept for one key is never shown for another: the new key may see other models.

    Killed by: src/uclone_x/ui/app.py :: cache_key = (settings_id, endpoint or "", key_fingerprint(api_key))
    Becomes: cache_key = (settings_id, endpoint or "", "")
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    cache = CatalogCache()

    await read_provider_catalog("gemini", base_url=None, api_key="first-key", cache=cache)
    await read_provider_catalog("gemini", base_url=None, api_key="first-key", cache=cache)
    await read_provider_catalog("gemini", base_url=None, api_key="second-key", cache=cache)

    assert [c.api_key for c in listing.calls] == ["first-key", "second-key"]


async def _post(
    app: Any, path: str, body: dict[str, Any], headers: dict[str, str] | None = None
) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        return await client.post(path, json=body, headers=headers or {})


@pytest.mark.asyncio
async def test_a_picked_provider_is_listed_with_the_key_typed_on_the_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1657: Gemini picked while Ollama is saved shows Gemini's models before saving."""
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)

    res = await _post(app, "/api/models/catalog", {"provider": "gemini", "api_key": " AIza-typed "})

    assert res.status_code == 200, res.text
    data = res.json()
    assert data["catalog"]["recommended"] == "gemini-2.5-flash"
    assert data["models"] == ["gemini-2.5-pro", "gemini-2.5-flash"]
    assert [c.api_key for c in listing.calls] == ["AIza-typed"]


@pytest.mark.asyncio
async def test_a_picked_provider_is_never_asked_with_another_providers_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI key in use is not sent to Google to preview Gemini; Gemini's own env key is.

    Killed by: src/uclone_x/ui/app.py :: api_key = typed_key or session_mgr.stored_api_key_for(provider, base_url)
    Becomes: api_key = typed_key or session_mgr.stored_api_key_for("openai", base_url)

    Killed by: src/uclone_x/ui/app.py :: saved = api_key_for(settings_data(self._settings_file), canonical)
    Becomes: saved = api_key_for(settings_data(self._settings_file), "openai")
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = create_ui_app(
        static_dir=tmp_path, llm=OpenAIConnector(api_key="sk-openai-in-use"), storage_dir=tmp_path
    )

    saved = await _post(
        app, "/api/settings", {"llm_provider": "openai", "llm_api_key": "sk-saved-for-openai"}
    )
    assert saved.status_code == 200, saved.text
    refused = (await _post(app, "/api/models/catalog", {"provider": "gemini"})).json()

    assert refused["catalog"]["status"] == "no_key"
    assert listing.calls == []

    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env")
    await _post(app, "/api/models/catalog", {"provider": "gemini"})

    assert [c.api_key for c in listing.calls] == ["AIza-from-env"]


@pytest.mark.asyncio
async def test_a_held_key_goes_only_to_the_endpoint_it_was_saved_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An endpoint left on the form from another provider is not sent a held key.

    Clicking Gemini while a vLLM box's address is still in the endpoint field must not send
    `GEMINI_API_KEY` to that box, nor a key saved with one endpoint to another.

    Killed by: src/uclone_x/ui/app.py :: if base_url is not None and not (
    Becomes: if False and not (

    Killed by: src/uclone_x/ui/app.py :: and base_url.rstrip("/") == (self._configured_base_url or "").rstrip("/")
    Becomes: and True

    Killed by: src/uclone_x/ui/app.py :: and same_provider(self._configured_provider, provider)
    Becomes: and True
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-from-env")
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    proxy = "https://proxy.example/v1"

    async def save(provider: str, key: str) -> None:
        body = {"llm_provider": provider, "llm_api_key": key, "llm_base_url": proxy}
        saved = await _post(app, "/api/settings", body)
        assert saved.status_code == 200, saved.text

    async def ask(base_url: str) -> None:
        await _post(app, "/api/models/catalog", {"provider": "gemini", "base_url": base_url})

    await ask("http://gpu-box:8000/v1")
    await save("openai", "sk-saved-for-openai")
    await ask(proxy)  # the endpoint saved with another provider's key
    await save("gemini", "AIza-saved-for-proxy")
    await ask("https://other.example/v1")
    assert listing.calls == []

    await ask(proxy + "/")
    # The environment overrides the key saved in Settings, so its key is the one held.
    assert [c.api_key for c in listing.calls] == ["AIza-from-env"]
    monkeypatch.delenv("GEMINI_API_KEY")
    await ask(proxy)
    assert [c.api_key for c in listing.calls][-1] == "AIza-saved-for-proxy"


@pytest.mark.asyncio
async def test_another_site_cannot_spend_the_key_on_a_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # a page in another tab must not spend the user's key
    Becomes: pass
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)

    res = await _post(
        app,
        "/api/models/catalog",
        {"provider": "gemini", "api_key": "AIza-typed"},
        headers={"Origin": "https://evil.example"},
    )

    assert res.status_code == 403
    assert listing.calls == []


@pytest.mark.asyncio
async def test_a_local_server_that_does_not_answer_is_not_an_empty_one() -> None:
    """#1666: "nothing answered" and "nothing installed" are different answers.

    Settings says "is Ollama running?" for the first and "install a model" for the second;
    folded together, a stopped server read as an install with no models in it.

    Killed by: src/uclone_x/ui/app.py :: return None  # something answered, but not Ollama's listing
    Becomes: return []
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 11435:
            raise httpx.ConnectError("refused", request=request)
        if request.url.port == 8080:
            return httpx.Response(404, text="not Ollama")
        if request.url.host == "empty":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "gemma3:4b"}]})

    async def ask(url: str) -> list[str] | None:
        stub = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("uclone_x.ui.app.httpx.AsyncClient", return_value=stub):
            return await list_local_models("ollama", url)

    assert await ask("http://127.0.0.1:11435") is None
    assert await ask("http://127.0.0.1:8080") is None
    assert await ask("http://empty:11434") == []
    assert await ask("http://localhost:11434") == ["qwen3:8b", "gemma3:4b"]


@pytest.mark.asyncio
async def test_a_picked_local_provider_lists_the_server_at_the_address_on_the_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1666: the Ollama picker is filled from the address typed, before it is saved.

    Killed by: src/uclone_x/ui/app.py :: provider, base_url, vllm_headers=headers, raise_on_refused_key=True
    Becomes: provider, None, vllm_headers=headers, raise_on_refused_key=True

    Killed by: src/uclone_x/ui/app.py :: "reachable": local is not None,
    Becomes: "reachable": True,
    """
    asked: list[tuple[str, str | None]] = []

    async def fake(provider: str, base_url: str | None = None, **_: object) -> list[str] | None:
        asked.append((provider, base_url))
        return ["qwen3:8b"] if base_url == "http://localhost:11434" else None

    monkeypatch.setattr(app_module, "list_local_models", fake)
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)

    live = (
        await _post(
            app, "/api/models/catalog", {"provider": "ollama", "base_url": "http://localhost:11434"}
        )
    ).json()
    dead = (
        await _post(
            app, "/api/models/catalog", {"provider": "ollama", "base_url": "http://127.0.0.1:11435"}
        )
    ).json()

    assert live == {
        "provider": "ollama",
        "models": ["qwen3:8b"],
        "reachable": True,
        "catalog": None,
    }
    assert dead == {"provider": "ollama", "models": [], "reachable": False, "catalog": None}
    assert asked == [("ollama", "http://localhost:11434"), ("ollama", "http://127.0.0.1:11435")]


@pytest.mark.asyncio
async def test_a_typed_vllm_address_is_not_sent_the_held_vllm_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1666: the form's address is fetched with no click, so a held key must not go with it.

    Typing "http://gpu" and pausing would otherwise send `VLLM_API_KEY` to host `gpu`, or to a
    typo'd domain. Only a key typed on the form goes there.

    Killed by: src/uclone_x/ui/app.py :: headers = vllm_request_headers(api_key, env_fallback=False)
    Becomes: headers = vllm_request_headers(api_key)
    """
    monkeypatch.setenv("VLLM_API_KEY", "sk-held-for-the-saved-box")
    sent: list[dict[str, str] | None] = []

    async def fake(
        provider: str, base_url: str | None = None, **kw: dict[str, str] | None
    ) -> list[str] | None:
        sent.append(kw.get("vllm_headers"))
        return ["served-model"]

    monkeypatch.setattr(app_module, "list_local_models", fake)
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)

    await _post(app, "/api/models/catalog", {"provider": "vllm", "base_url": "http://gpu:8000/v1"})
    await _post(
        app,
        "/api/models/catalog",
        {"provider": "vllm", "base_url": "http://gpu:8000/v1", "api_key": "sk-typed"},
    )

    assert sent == [{}, {"Authorization": "Bearer sk-typed"}]


def _refusing_vllm(request: httpx.Request) -> httpx.Response:
    """A vLLM server started with `--api-key`: it lists its model only for that key."""
    if request.headers.get("Authorization") == "Bearer sk-right":
        return httpx.Response(200, json={"object": "list", "data": [{"id": "served-model"}]})
    return httpx.Response(401, json={"error": "Unauthorized"})


@pytest.mark.asyncio
async def test_a_vllm_server_that_refuses_the_key_is_not_reported_as_absent() -> None:
    """#1672: a 401 or 403 means the server is there and turned the key away.

    Read as "no listing", the form said "No model list came back from <address>", which sends
    the user looking for a stopped server that is running.

    Killed by: src/uclone_x/ui/app.py :: refused = resp.status_code in _KEY_REFUSED_STATUSES
    Becomes: refused = False
    """

    async def ask(key: str, *, raising: bool) -> list[str] | None:
        stub = httpx.AsyncClient(transport=httpx.MockTransport(_refusing_vllm))
        with patch("uclone_x.ui.app.httpx.AsyncClient", return_value=stub):
            return await list_local_models(
                "vllm",
                "http://box:8000/v1",
                vllm_headers={"Authorization": f"Bearer {key}"},
                raise_on_refused_key=raising,
            )

    assert await ask("sk-right", raising=True) == ["served-model"]
    assert await ask("sk-wrong", raising=False) is None
    with pytest.raises(app_module.LocalKeyRefusedError):
        await ask("sk-wrong", raising=True)


@pytest.mark.asyncio
async def test_the_form_hears_that_the_vllm_server_refused_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1672: the listing route says `key_refused`, and still counts the server as reachable.

    Killed by: src/uclone_x/ui/app.py :: "key_refused": True,
    Becomes: "key_refused": False,
    """

    async def fake(provider: str, base_url: str | None = None, **_: object) -> list[str] | None:
        raise app_module.LocalKeyRefusedError(base_url)

    monkeypatch.setattr(app_module, "list_local_models", fake)
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)

    res = await _post(
        app, "/api/models/catalog", {"provider": "vllm", "base_url": "http://box:8000/v1"}
    )

    assert res.json() == {
        "provider": "vllm",
        "models": [],
        "reachable": True,
        "key_refused": True,
        "catalog": None,
    }


@pytest.mark.asyncio
async def test_check_connection_sends_vllm_key_only_to_the_endpoint_it_is_held_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1672: Check connection follows `stored_api_key_for`, like the listing does.

    A typed address gets only a typed key. The saved address gets the key held for it, here
    `VLLM_API_KEY`, so an env-only key still reaches the server it belongs to. A refusal is
    reported as one.

    Killed by: src/uclone_x/ui/app.py :: headers=vllm_request_headers(vllm_key, env_fallback=False),
    Becomes: headers=vllm_request_headers(vllm_key),

    Killed by: src/uclone_x/ui/app.py :: vllm_key = eff_key or session_mgr.stored_api_key_for("vllm", eff_base)
    Becomes: vllm_key = eff_key

    Killed by: src/uclone_x/ui/app.py :: "key_refused": resp.status_code in _KEY_REFUSED_STATUSES,
    Becomes: "key_refused": False,
    """
    monkeypatch.setenv("VLLM_API_KEY", "sk-right")
    saved_box = "http://box:8000/v1"
    app = create_ui_app(static_dir=tmp_path, llm=MockLLMConnector(), storage_dir=tmp_path)
    saved = await _post(app, "/api/settings", {"llm_provider": "vllm", "llm_base_url": saved_box})
    assert saved.status_code == 200, saved.text
    sent: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("Authorization"))
        return _refusing_vllm(request)

    async def check(base_url: str) -> dict[str, Any]:
        stub = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        body = {"target": "llm", "llm_provider": "vllm", "llm_base_url": base_url}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            # Only the route's own client is replaced; this one reaches the app.
            with patch("uclone_x.ui.app.httpx.AsyncClient", return_value=stub):
                res = await client.post("/api/settings/test", json=body)
        return cast(dict[str, Any], res.json()["results"]["llm"])

    typed = await check("http://typo-box:8000/v1")
    held = await check(saved_box)

    assert sent == [None, "Bearer sk-right"]
    assert typed["status"] == "error"
    assert typed["key_refused"] is True
    assert held["status"] == "ok"
    assert held["models"] == ["served-model"]
