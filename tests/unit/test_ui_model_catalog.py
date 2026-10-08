"""The model set holds what each connection lists for its key, and nothing remembered (#1631).

The connector's `list_models` is replaced by a recorded listing, so what is checked is the
wiring: which key and endpoint each connection's listing is asked with, what `GET
/api/models` returns from it (model-gateway §3.7.1), and when a kept listing is asked for
again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest

from uclone_x.errors import ProviderAuthError
from uclone_x.llm import gateway as gateway_module
from uclone_x.llm import model_listing
from uclone_x.llm.catalog import CatalogCache, CatalogEntry
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.model_listing import list_local_models, read_provider_catalog
from uclone_x.ui.app import create_ui_app

_LISTING = [
    CatalogEntry(id="gemini-2.5-pro", context_window=1048576),
    CatalogEntry(id="gemini-2.5-flash", context_window=1048576),
    CatalogEntry(id="gemini-embedding-001", chat_capable=False),
]


@pytest.fixture(autouse=True)
def _no_model_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    for name in (
        "GEMINI_MODEL",
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "VLLM_API_KEY",
        "VLLM_BASE_URL",
    ):
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


def _app(tmp_path: Path, settings: dict[str, Any]) -> Any:
    storage = tmp_path / "sessions"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return create_ui_app(static_dir=tmp_path / "static", storage_dir=storage)


_GEMINI = {"connections": [{"id": "gemini", "kind": "gemini", "key": "AIza-test-key"}]}


@pytest.mark.asyncio
async def test_the_model_set_holds_what_the_connection_listed_and_suggests_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G1/G2: the picker holds the connection's chat models under their refs.

    Killed by: src/uclone_x/llm/gateway.py :: entries = [e for e in listing.entries if e.chat_capable]
    Becomes: entries = list(listing.entries)
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = _app(tmp_path, _GEMINI)

    data = await _get(app, "/api/models?capability=chat")

    (group,) = data["groups"]
    assert (group["connection_id"], group["kind"], group["status"]) == (
        "gemini",
        "gemini",
        "connected",
    )
    assert [m["ref"] for m in group["models"]] == [
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
    ]
    assert group["models"][0]["context_window"] == 1048576
    assert data["recommended"] == {"deep": "gemini/gemini-2.5-flash", "fast": None}
    assert data["defaults"] == {"deep": None, "fast": None, "image": "auto"}
    assert listing.calls[0].api_key == "AIza-test-key"


@pytest.mark.asyncio
async def test_a_kept_listing_is_reused_until_refresh_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/llm/gateway.py :: self._catalog.clear()  # every connection is asked again
    Becomes: pass
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = _app(tmp_path, _GEMINI)

    await _get(app, "/api/models")
    await _get(app, "/api/models")
    assert len(listing.calls) == 1

    refreshed = await _get(app, "/api/models?refresh=1")

    assert len(listing.calls) == 2
    assert [m["id"] for m in refreshed["groups"][0]["models"]] == [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]


@pytest.mark.asyncio
async def test_a_refused_key_is_reported_and_no_models_are_offered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G3: no remembered list fills the group when the provider refused to answer."""
    _Listing(ProviderAuthError(provider="Google", model="")).install(monkeypatch)
    app = _app(tmp_path, _GEMINI)

    (group,) = (await _get(app, "/api/models"))["groups"]

    assert group["status"] == "key_rejected"
    assert "did not accept the API key" in group["detail"]
    assert group["models"] == []


@pytest.mark.asyncio
async def test_each_connection_is_asked_with_its_own_key_at_its_own_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S3 as revised: a key goes only to its connection's address, and each keeps its listing.

    Killed by: src/uclone_x/llm/model_listing.py :: cache_key = (cache_id or settings_id, endpoint or "", key_fingerprint(api_key))
    Becomes: cache_key = (settings_id, "", "")
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = _app(
        tmp_path,
        {
            "connections": [
                {"id": "gemini", "kind": "gemini", "key": "AIza-home"},
                {
                    "id": "work",
                    "kind": "gemini",
                    "key": "AIza-work",
                    "base_url": "https://proxy.invalid/v1beta",
                },
            ]
        },
    )

    data = await _get(app, "/api/models")

    sent = sorted((c.api_key, c.base_url) for c in listing.calls)
    assert sent == [
        ("AIza-home", GeminiConnector(api_key="k").base_url),
        ("AIza-work", "https://proxy.invalid/v1beta"),
    ]
    assert [g["connection_id"] for g in data["groups"]] == ["gemini", "work"]


@pytest.mark.asyncio
async def test_a_mock_connection_lists_its_models(tmp_path: Path) -> None:
    app = _app(tmp_path, {"connections": [{"id": "mock", "kind": "mock"}]})

    (group,) = (await _get(app, "/api/models"))["groups"]

    assert "mock/mock-llm" in [m["ref"] for m in group["models"]]


@pytest.mark.asyncio
async def test_another_site_cannot_spend_the_keys_on_a_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/app.py :: _refuse_cross_origin(request)  # it asks every connection with its key
    Becomes: pass
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    app = _app(tmp_path, _GEMINI)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/api/models", headers={"Origin": "https://evil.example"})

    assert res.status_code == 403
    assert listing.calls == []


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
async def test_a_new_key_reads_the_listing_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A listing kept for one key is never shown for another: the new key may see other models.

    Killed by: src/uclone_x/llm/model_listing.py :: cache_key = (cache_id or settings_id, endpoint or "", key_fingerprint(api_key))
    Becomes: cache_key = (cache_id or settings_id, endpoint or "", "")
    """
    listing = _Listing(_LISTING)
    listing.install(monkeypatch)
    cache = CatalogCache()

    await read_provider_catalog("gemini", base_url=None, api_key="first-key", cache=cache)
    await read_provider_catalog("gemini", base_url=None, api_key="first-key", cache=cache)
    await read_provider_catalog("gemini", base_url=None, api_key="second-key", cache=cache)

    assert [c.api_key for c in listing.calls] == ["first-key", "second-key"]


@pytest.mark.asyncio
async def test_a_local_server_that_does_not_answer_is_not_an_empty_one() -> None:
    """#1666: "nothing answered" and "nothing installed" are different answers.

    Settings says "is Ollama running?" for the first and "install a model" for the second;
    folded together, a stopped server read as an install with no models in it.

    Killed by: src/uclone_x/llm/model_listing.py :: return None  # something answered, but not Ollama's listing
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
        with patch("uclone_x.llm.model_listing.httpx.AsyncClient", return_value=stub):
            return await list_local_models("ollama", url)

    assert await ask("http://127.0.0.1:11435") is None
    assert await ask("http://127.0.0.1:8080") is None
    assert await ask("http://empty:11434") == []
    assert await ask("http://localhost:11434") == ["qwen3:8b", "gemma3:4b"]


@pytest.mark.asyncio
async def test_a_vllm_connection_is_sent_its_own_key_and_never_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1666, #1672 under connections: a box is sent the key saved on it, not `VLLM_API_KEY`.

    Killed by: src/uclone_x/llm/gateway.py :: vllm_headers=vllm_request_headers(conn.key, env_fallback=False),
    Becomes: vllm_headers=vllm_request_headers(conn.key),
    """
    monkeypatch.setenv("VLLM_API_KEY", "sk-held-for-another-box")
    sent: dict[str | None, Any] = {}

    async def fake(provider: str, base_url: str | None = None, **kw: Any) -> list[str] | None:
        sent[base_url] = kw.get("vllm_headers")
        return ["served-model"]

    monkeypatch.setattr(gateway_module, "list_local_models", fake)
    app = _app(
        tmp_path,
        {
            "connections": [
                {"id": "vllm", "kind": "vllm", "base_url": "http://gpu:8000/v1", "key": "sk-own"},
                {"id": "open-box", "kind": "vllm", "base_url": "http://open:8000/v1"},
            ]
        },
    )

    await _get(app, "/api/models")

    # `vllm` is the row the variable would override (S4): the saved row with that id is
    # still sent the variable's key, and the other row is sent none.
    assert sent["http://open:8000/v1"] == {}
    assert sent["http://gpu:8000/v1"] == {"Authorization": "Bearer sk-held-for-another-box"}


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

    Killed by: src/uclone_x/llm/model_listing.py :: refused = resp.status_code in _KEY_REFUSED_STATUSES
    Becomes: refused = False
    """

    async def ask(key: str, *, raising: bool) -> list[str] | None:
        stub = httpx.AsyncClient(transport=httpx.MockTransport(_refusing_vllm))
        with patch("uclone_x.llm.model_listing.httpx.AsyncClient", return_value=stub):
            return await list_local_models(
                "vllm",
                "http://box:8000/v1",
                vllm_headers={"Authorization": f"Bearer {key}"},
                raise_on_refused_key=raising,
            )

    assert await ask("sk-right", raising=True) == ["served-model"]
    assert await ask("sk-wrong", raising=False) is None
    with pytest.raises(model_listing.LocalKeyRefusedError):
        await ask("sk-wrong", raising=True)


@pytest.mark.asyncio
async def test_a_box_that_refuses_the_key_is_reported_as_refusing_and_checked_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1672: a refused key is said as one, by the model set and by Check connection.

    Killed by: src/uclone_x/llm/gateway.py :: listing = Listing("key_rejected", _KEY_REFUSED_LOCAL.format(label=label))
    Becomes: listing = Listing("unreachable", None)
    """

    async def fake(provider: str, base_url: str | None = None, **_: object) -> list[str] | None:
        raise model_listing.LocalKeyRefusedError(base_url)

    monkeypatch.setattr(gateway_module, "list_local_models", fake)
    app = _app(
        tmp_path,
        {"connections": [{"id": "box", "kind": "vllm", "base_url": "http://box:8000/v1"}]},
    )

    (group,) = (await _get(app, "/api/models"))["groups"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        checked = (await client.post("/api/connections/box/check")).json()

    assert (group["status"], group["models"]) == ("key_rejected", [])
    assert checked["status"] == "key_rejected"
    assert "did not accept the key" in checked["detail"]


# -- Ollama's embedders are not conversation models (#2167 part 2) --

_TAG_DETAILS_CHAT: dict[str, Any] = {"family": "qwen3", "families": ["qwen3"]}
_TAG_DETAILS_BERT: dict[str, Any] = {"family": "bert", "families": ["bert"]}


def _ollama(tags: list[dict[str, Any]], show: dict[str, list[str]] | None = None) -> Any:
    """A test Ollama: `/api/tags` answers `tags`; `/api/show` answers `show[model]`, or a
    body without `capabilities` (an older server) when the model is not in it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": tags})
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            body: dict[str, Any] = {"details": {}}
            if show is not None and name in show:
                body["capabilities"] = show[name]
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _chat_flags(client: Any) -> dict[str, bool]:
    with patch("uclone_x.llm.model_listing.httpx.AsyncClient", return_value=client):
        entries = await model_listing.list_ollama_entries("http://127.0.0.1:11434")
    assert entries is not None
    return {e.id: e.chat_capable for e in entries}


@pytest.mark.asyncio
async def test_ollama_capabilities_in_the_listing_decide_chat() -> None:
    """Ollama's own `capabilities` (since 0.6, in `/api/tags`) say `embedding` for an embedder.

    Killed by: src/uclone_x/llm/model_listing.py :: _OLLAMA_CHAT_CAPABILITY in capabilities
    Becomes: True
    """
    flags = await _chat_flags(
        _ollama(
            [
                {"name": "qwen3:14b", "capabilities": ["completion", "tools"]},
                {"name": "bge-m3:latest", "capabilities": ["embedding"]},
            ]
        )
    )
    assert flags == {"qwen3:14b": True, "bge-m3:latest": False}


@pytest.mark.asyncio
async def test_ollama_show_answers_when_the_listing_is_silent() -> None:
    """A listing without `capabilities` asks `/api/show` for each model.

    Killed by: src/uclone_x/llm/model_listing.py :: return await _ollama_show_capabilities(http_c, ollama_url, str(item["name"]))
    Becomes: return None
    """
    flags = await _chat_flags(
        _ollama(
            # No family either, so only `/api/show` can tell these two apart.
            [{"name": "qwen3:14b"}, {"name": "nomic-embed-text:latest"}],
            show={"qwen3:14b": ["completion"], "nomic-embed-text:latest": ["embedding"]},
        )
    )
    assert flags == {"qwen3:14b": True, "nomic-embed-text:latest": False}


@pytest.mark.asyncio
async def test_an_old_ollama_falls_back_to_the_family_it_reports() -> None:
    """With neither field, the listing's family decides: a BERT encoder cannot chat.

    The names here are chosen so no name rule could tell them apart.

    Killed by: src/uclone_x/llm/model_listing.py :: else not _ollama_encoder_only(item)
    Becomes: else True
    """
    flags = await _chat_flags(
        _ollama(
            [
                {"name": "model-a:latest", "details": _TAG_DETAILS_BERT},
                {"name": "model-b:latest", "details": {"family": "nomic-bert"}},
                {"name": "model-c:latest", "details": _TAG_DETAILS_CHAT},
                {"name": "model-d:latest"},  # the listing says nothing: offered, as before
            ]
        )
    )
    assert flags == {
        "model-a:latest": False,
        "model-b:latest": False,
        "model-c:latest": True,
        "model-d:latest": True,
    }


@pytest.mark.asyncio
async def test_the_conversation_set_leaves_ollama_embedders_out(tmp_path: Path) -> None:
    """End to end: `GET /api/models?capability=chat` offers no embedding model.

    Killed by: src/uclone_x/llm/gateway.py :: entries = await list_ollama_entries(conn.base_url)
    Becomes: entries = [CatalogEntry(id=m) for m in (await list_local_models(conn.kind, conn.base_url) or [])]
    """
    app = _app(
        tmp_path,
        {"connections": [{"id": "ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434"}]},
    )
    tags = [
        {"name": "qwen3:14b", "capabilities": ["completion", "tools"]},
        {"name": "bge-m3:latest", "capabilities": ["embedding"]},
        {"name": "nomic-embed-text:latest", "capabilities": ["embedding"]},
    ]
    real_client = httpx.AsyncClient

    def client(**kwargs: Any) -> Any:
        # The test's own ASGI client is real; the listing's client talks to the test Ollama.
        return real_client(**kwargs) if "transport" in kwargs else _ollama(tags)

    with patch("uclone_x.llm.model_listing.httpx.AsyncClient", side_effect=client):
        (group,) = (await _get(app, "/api/models?capability=chat"))["groups"]
    assert [m["ref"] for m in group["models"]] == ["ollama/qwen3:14b"]
