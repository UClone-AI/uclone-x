"""Unit tests for the standalone Model Gateway server and unified agent client (#2475)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

import uclone_x.llm.gateway as gateway_mod
from uclone_x.cli.main import app as ucx_app
from uclone_x.core.provenance import Provenance
from uclone_x.core.remote_worker import (
    PortMapping,
    RemoteGPUInfo,
    SSHTunnelManager,
    TunnelSessionStatus,
)
from uclone_x.errors import (
    EmbeddingDimensionError,
    LLMProviderError,
    ProviderQuotaError,
    ProviderUnreachableError,
    TurnCeilingReachedError,
    UsageLimitReachedError,
)
from uclone_x.llm.catalog import CatalogEntry, CatalogResult
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.factory import create_llm_connector, what_outranks_saved_choice
from uclone_x.llm.connectors.gateway_client import (
    GatewayEmbedder,
    GatewayModelClient,
    deserialize_llm_request,
    has_configured_gateway_endpoint,
    resolve_gateway_base_url,
    serialize_llm_request,
)
from uclone_x.llm.gateway import ModelGateway
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    ImagePart,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.protocols import EmbedderProtocol, ImageInputProbe
from uclone_x.llm.usage.admission import enter_admitted_turn, leave_admitted_turn
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.shells.gateway_server import GatewayService, create_gateway_app


class _StubGatewayConnector(BaseLLMConnector, ImageInputProbe):
    """Deterministic in-memory connector for testing GatewayService routing and quotas."""

    def __init__(
        self,
        *,
        provider: str = "openai",
        paid: bool = True,
        supports_images: bool = True,
        fail_with: Exception | None = None,
    ) -> None:
        super().__init__(api_key="sk-test", base_url="http://stub.local")
        self._provider = provider
        self._paid = paid
        self._supports_images = supports_images
        self._fail_with = fail_with
        self.seen_requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def paid(self) -> bool:
        return self._paid

    async def accepts_images(self, model: str | None = None) -> bool:
        return self._supports_images

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.seen_requests.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        model_id = request.model or "gpt-5.4"
        return ModelResponse(
            content=f"ack:{model_id}:{len(request.messages)}",
            model_name=model_id,
            finish_reason=FinishReason.STOP,
            tool_calls=(ToolCallRequest(id="call_1", name="lookup", arguments={"q": "test"}),),
            usage=TokenUsage(
                input_tokens=40,
                output_tokens=60,
                total_tokens=100,
                model=model_id,
                provider=self._provider,
            ),
            provenance=Provenance.primary(provider=self._provider, model=model_id),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.seen_requests.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        model_id = request.model or "gpt-5.4"
        yield StreamChunk(delta_content="hello ", model=model_id)
        yield StreamChunk(
            delta_content="world",
            finish_reason=FinishReason.STOP,
            model=model_id,
            usage=TokenUsage(
                input_tokens=25,
                output_tokens=25,
                total_tokens=50,
                model=model_id,
                provider=self._provider,
            ),
        )


class _StubEmbedder(EmbedderProtocol):
    def __init__(self, *, dims: int = 4, return_dims: int | None = None) -> None:
        self._dims = dims
        self._return_dims = return_dims if return_dims is not None else dims

    @property
    def model_name(self) -> str:
        return "nomic-embed-text"

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if self._return_dims != self._dims:
            raise EmbeddingDimensionError(self.model_name, self._dims, self._return_dims)
        return tuple(tuple(float(i + 1) for i in range(self._dims)) for _ in texts)


def _build_test_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    connector: _StubGatewayConnector | None = None,
    embedder: EmbedderProtocol | None = None,
    api_key: str | None = None,
) -> tuple[GatewayService, httpx.AsyncClient, _StubGatewayConnector]:
    settings_path = tmp_path / "settings.json"
    usage_path = tmp_path / "usage.sqlite3"
    stub = connector or _StubGatewayConnector()
    usage_gate = UsageGate.for_storage(settings_path, usage_path)

    async def _fake_read_provider_catalog(
        provider: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **_: Any,
    ) -> CatalogResult:
        if not api_key:
            return CatalogResult(
                provider=provider,
                status="no_key",
                detail="No API key configured",
            )
        return CatalogResult(
            provider=provider,
            status="live",
            entries=(
                CatalogEntry(
                    id="gpt-5.4",
                    display_name="GPT 5.4",
                    context_window=128000,
                    chat_capable=True,
                    accepts_images=True,
                ),
                CatalogEntry(
                    id="gpt-5.4-mini",
                    display_name="GPT 5.4 Mini",
                    context_window=128000,
                    chat_capable=True,
                    accepts_images=False,
                ),
            ),
            recommended="gpt-5.4",
        )

    monkeypatch.setattr(gateway_mod, "read_provider_catalog", _fake_read_provider_catalog)

    def _connector_factory(**kwargs: Any) -> Any:
        gate = kwargs.get("usage_gate") or usage_gate
        return gate_if_paid(stub, gate)

    mg = ModelGateway(
        settings_path,
        usage_gate=usage_gate,
        connector_factory=_connector_factory,
    )
    svc = GatewayService(
        settings_path=settings_path,
        usage_path=usage_path,
        workspace_dir=tmp_path,
        gateway=mg,
        usage_gate=usage_gate,
        embedder=embedder or _StubEmbedder(dims=4),
    )
    fastapi_app = create_gateway_app(service=svc, api_key=api_key)
    transport = httpx.ASGITransport(app=fastapi_app)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://gateway.test")
    return svc, http_client, stub


def test_serialize_and_deserialize_llm_request_preserves_image_bytes() -> None:
    """`serialize_llm_request` and `deserialize_llm_request` round-trip `ImagePart.data`.

    Killed by: src/uclone_x/llm/connectors/gateway_client.py :: if image.data is not None:
    Becomes:   if image.data is None:
    """
    img = ImagePart.from_bytes(
        b"\x89PNG\r\n\x1a\nfake-image-bytes", "image/png", width=64, height=64
    )
    req = LLMRequest(
        model="openai/gpt-5.4",
        messages=(ChatMessage(role=MessageRole.USER, content="Describe this", images=(img,)),),
    )
    wire = serialize_llm_request(req)
    assert "image_data" in wire
    assert wire["image_data"][img.digest] == img.data

    restored = deserialize_llm_request(wire)
    assert len(restored.messages) == 1
    assert len(restored.messages[0].images) == 1
    assert restored.messages[0].images[0].data == img.data
    assert restored.messages[0].images[0].digest == img.digest


@pytest.mark.asyncio
async def test_gateway_connections_models_and_defaults_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connections, masked keys, model catalog discovery, and default slots work end-to-end.

    Killed by: src/uclone_x/shells/gateway_server.py :: return f"{clean[:3]}{_MASK_BULLETS}{clean[-4:]}"
    Becomes:   return f"{clean[:3]}{_MASK_BULLETS}{clean[:4]}"
    """
    _, http_client, _ = _build_test_gateway(tmp_path, monkeypatch)
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        initial = await client.list_connections()
        assert "kinds" in initial
        assert any(k["kind"] == "openai" for k in initial["kinds"])

        added = await client.add_connection(
            "openai",
            label="Primary OpenAI",
            base_url=None,
            key="sk-proj-1234567890abcdef",
        )
        conn_id = added["id"]
        assert conn_id == "primary-openai"
        assert added["key_set"] is True
        assert added["key_masked"] == "sk-••••••••cdef"
        assert "1234567890" not in str(added)
        assert added["status"] == "connected"
        assert added["model_count"] == 2

        patched = await client.patch_connection(conn_id, label="Team OpenAI")
        assert patched["label"] == "Team OpenAI"

        models = await client.list_models("chat", refresh=True)
        assert len(models["groups"]) >= 1
        openai_group = next(g for g in models["groups"] if g["connection_id"] == conn_id)
        assert any(m["id"] == "gpt-5.4" for m in openai_group["models"])

        updated_defaults = await client.set_defaults(
            deep=f"{conn_id}/gpt-5.4",
            fast=f"{conn_id}/gpt-5.4-mini",
        )
        assert updated_defaults["deep"] == f"{conn_id}/gpt-5.4"
        assert updated_defaults["fast"] == f"{conn_id}/gpt-5.4-mini"

        deps = await client.connection_dependents(conn_id)
        assert "deep" in deps["defaults"]
        assert "fast" in deps["defaults"]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_generate_and_stream_through_unified_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GatewayModelClient.generate` and `.stream` route slots and refs with provenance.

    Killed by: src/uclone_x/shells/gateway_server.py :: if clean == "fast":
    Becomes:   if clean == "none":
    """
    _, http_client, stub = _build_test_gateway(tmp_path, monkeypatch)
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        await client.add_connection(
            "openai", label="OpenAI", base_url=None, key="sk-live-9876543210"
        )
        await client.set_defaults(deep="openai/gpt-5.4", fast="openai/gpt-5.4-mini")

        img = ImagePart.from_bytes(b"\x89PNG\r\n\x1a\npayload", "image/png", width=32, height=32)
        req = LLMRequest(
            model="fast",
            messages=(ChatMessage(role=MessageRole.USER, content="Hello", images=(img,)),),
        )
        res = await client.generate(req)
        assert res.content == "ack:gpt-5.4-mini:1"
        assert res.model_name == "gpt-5.4-mini"
        assert res.provenance is not None
        assert res.provenance.served_by.provider == "openai"
        assert stub.seen_requests[-1].messages[0].images[0].data == img.data

        chunks = [
            chunk
            async for chunk in client.stream(
                LLMRequest(
                    model="openai/gpt-5.4",
                    messages=(ChatMessage(role=MessageRole.USER, content="Stream please"),),
                )
            )
        ]
        assert "".join(c.delta_content or "" for c in chunks) == "hello world"
        assert chunks[-1].finish_reason == FinishReason.STOP
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 50

        assert await client.accepts_images("openai/gpt-5.4") is True
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_openai_compatible_chat_completions_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/v1/chat/completions` supports both standard JSON and SSE streaming modes.

    Killed by: src/uclone_x/shells/gateway_server.py :: stream = bool(payload.get("stream", False))
    Becomes:   stream = bool(payload.get("stream", True))
    """
    _, http_client, _ = _build_test_gateway(tmp_path, monkeypatch)
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        await client.add_connection(
            "openai", label="OpenAI", base_url=None, key="sk-live-9876543210"
        )
        await client.set_defaults(deep="openai/gpt-5.4")

        resp = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5.4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "ack:gpt-5.4:1"
        assert body["usage"]["total_tokens"] == 100

        stream_resp = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5.4",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert stream_resp.status_code == 200
        lines = [line for line in stream_resp.text.splitlines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        assert len(lines) == 3
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_quota_enforcement_and_turn_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rolling-window token limits and turn-level admission work across the gateway boundary.

    Killed by: src/uclone_x/llm/connectors/gateway_client.py :: if _ADMITTED.get() and paid:
    Becomes:   if _ADMITTED.get() and None:
    """
    _, http_client, _ = _build_test_gateway(tmp_path, monkeypatch)
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        await client.add_connection(
            "openai", label="OpenAI", base_url=None, key="sk-live-9876543210"
        )
        await client.set_defaults(deep="openai/gpt-5.4")

        usage_after_set = await client.set_usage_limits(per_10_minutes=80)
        win_10m = next(w for w in usage_after_set["windows"] if w["window"] == "per_10_minutes")
        assert win_10m["limit"] == 80

        ceiling = await client.check_admission()
        assert ceiling == 80
        assert client.usage_turn_ceiling() == 80

        req = LLMRequest(
            model="deep",
            messages=(ChatMessage(role=MessageRole.USER, content="Turn 1"),),
        )
        # Inside an admitted turn with ceiling=80: first call uses 100 tokens, second call hits TurnCeilingReachedError
        token = enter_admitted_turn(client)
        try:
            await client.generate(req)
            with pytest.raises(TurnCeilingReachedError) as exc_info:
                await client.generate(req)
            assert exc_info.value.used == 100
            assert exc_info.value.ceiling == 80
        finally:
            leave_admitted_turn(token)

        # A fresh turn now exceeds the 80-token 10m window (100 > 80)
        with pytest.raises(UsageLimitReachedError) as limit_exc:
            await client.check_admission()
        assert limit_exc.value.window == "per_10_minutes"
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_authentication_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `api_key` is configured on the gateway server, unauthenticated requests are rejected.

    Killed by: src/uclone_x/shells/gateway_server.py :: if not token or not hmac.compare_digest(token, expected_key):
    Becomes:   if not token and not hmac.compare_digest(token, expected_key):
    """
    _, http_client, _ = _build_test_gateway(tmp_path, monkeypatch, api_key="secret-gateway-token")
    unauthed = GatewayModelClient(
        base_url="http://gateway.test", api_key="wrong-token", http_client=http_client
    )
    authed = GatewayModelClient(
        base_url="http://gateway.test", api_key="secret-gateway-token", http_client=http_client
    )
    try:
        with pytest.raises(LLMProviderError):
            await unauthed.list_connections()

        conns = await authed.list_connections()
        assert "connections" in conns
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_embedder_and_error_propagation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GatewayEmbedder` returns vectors and propagates `EmbeddingDimensionError`.

    Killed by: src/uclone_x/llm/connectors/gateway_client.py :: if len(vec) != self._dimensions:
    Becomes:   if len(vec) == self._dimensions:
    """
    _, http_client, _ = _build_test_gateway(tmp_path, monkeypatch, embedder=_StubEmbedder(dims=4))
    embedder = GatewayEmbedder(
        model_name="nomic-embed-text",
        dimensions=4,
        base_url="http://gateway.test",
        http_client=http_client,
    )
    try:
        assert await embedder.embed(()) == ()
        vecs = await embedder.embed(("alpha", "beta"))
        assert len(vecs) == 2
        assert vecs[0] == (1.0, 2.0, 3.0, 4.0)

        bad_embedder = GatewayEmbedder(
            model_name="nomic-embed-text",
            dimensions=8,
            base_url="http://gateway.test",
            http_client=http_client,
        )
        with pytest.raises(EmbeddingDimensionError) as exc_info:
            await bad_embedder.embed(("alpha",))
        assert exc_info.value.expected == 8
        assert exc_info.value.actual == 4
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_structured_upstream_error_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typed upstream `ProviderFailureError` subclasses survive the gateway HTTP wire.

    Killed by: src/uclone_x/llm/connectors/gateway_client.py :: if err_type == "ProviderQuotaError":
    Becomes:   if err_type == "NeverMatchesQuota_":
    """
    failing_stub = _StubGatewayConnector(
        fail_with=ProviderQuotaError(provider="openai", model="gpt-5.4")
    )
    _, http_client, _ = _build_test_gateway(tmp_path, monkeypatch, connector=failing_stub)
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        await client.add_connection(
            "openai", label="OpenAI", base_url=None, key="sk-live-9876543210"
        )
        await client.set_defaults(deep="openai/gpt-5.4")
        with pytest.raises(ProviderQuotaError) as exc_info:
            await client.generate(
                LLMRequest(
                    model="deep",
                    messages=(ChatMessage(role=MessageRole.USER, content="ping"),),
                )
            )
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-5.4"
    finally:
        await http_client.aclose()


def test_factory_constructs_gateway_client_from_provider_or_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`create_llm_connector` builds `GatewayModelClient` when `provider='gateway'` or `UCX_GATEWAY_URL` is set.

    Killed by: src/uclone_x/llm/connectors/factory.py :: if resolved_provider == "gateway":
    Becomes:   if resolved_provider == "_______":
    """
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("UCX_GATEWAY_URL", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)

    explicit = create_llm_connector(provider="gateway", base_url="http://127.0.0.1:9000")
    assert isinstance(explicit, GatewayModelClient)
    assert explicit.base_url == "http://127.0.0.1:9000"

    monkeypatch.setenv("UCX_GATEWAY_URL", "http://127.0.0.1:9001/")
    assert has_configured_gateway_endpoint() is True
    assert resolve_gateway_base_url() == "http://127.0.0.1:9001"
    assert what_outranks_saved_choice() == "UCX_GATEWAY_URL"

    from_env = create_llm_connector()
    assert isinstance(from_env, GatewayModelClient)
    assert from_env.base_url == "http://127.0.0.1:9001"


def test_cli_gateway_help_and_unreachable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ucx gateway --help` and `ucx gateway status` are registered and report unreachable endpoints cleanly.

    Killed by: src/uclone_x/cli/commands/gateway.py :: raise typer.Exit(1) from exc
    Becomes:   raise typer.Exit(0) from exc
    """

    async def _fail_list_connections(self: GatewayModelClient) -> dict[str, Any]:
        raise ProviderUnreachableError(provider="Gateway", model="default")

    monkeypatch.setattr(GatewayModelClient, "list_connections", _fail_list_connections)

    runner = CliRunner()
    help_res = runner.invoke(ucx_app, ["gateway", "--help"])
    assert help_res.exit_code == 0
    assert "serve" in help_res.output
    assert "status" in help_res.output

    status_res = runner.invoke(
        ucx_app,
        ["gateway", "status", "--url", "http://127.0.0.1:8765"],
    )
    assert status_res.exit_code == 1
    assert "Gateway unreachable" in status_res.output


@pytest.mark.asyncio
async def test_gateway_remote_gpu_tunnel_auto_connect_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gateway owns SSH tunneling and auto-reconnects to remote hosts (Dell / Mac mini).

    Killed by: src/uclone_x/shells/gateway_server.py :: if not tunnel_status.connected:
    Becomes:   if tunnel_status.connected:
    """

    class _FakeTunnelManager(SSHTunnelManager):
        def __init__(self) -> None:
            super().__init__()
            self._fake_status = TunnelSessionStatus(host="", connected=False)

        def get_status(self) -> TunnelSessionStatus:
            return self._fake_status

        async def connect(
            self,
            host: str,
            preferred_local_ollama_port: int | None = None,
            preferred_local_comfyui_port: int | None = None,
            auto_start_comfyui: bool = True,
            auto_start_ollama: bool = True,
            timeout: float = 12.0,
        ) -> TunnelSessionStatus:
            ollama_port = preferred_local_ollama_port or 11435
            comfy_port = preferred_local_comfyui_port or 8189
            self._fake_status = TunnelSessionStatus(
                host=host,
                connected=True,
                pid=4242,
                mappings=[
                    PortMapping(service_name="ollama", remote_port=11434, local_port=ollama_port),
                    PortMapping(service_name="comfyui", remote_port=8188, local_port=comfy_port),
                ],
                gpu=RemoteGPUInfo(
                    name="Apple M4 Pro",
                    total_mb=65536,
                    used_mb=0,
                    driver="Apple Metal",
                ),
                ollama_models=["qwen3:32b"],
            )
            return self._fake_status

        async def disconnect(self) -> None:
            self._fake_status = TunnelSessionStatus(
                host="", connected=False, error="No active remote tunnel"
            )

    ssh_config = tmp_path / "ssh_config"
    ssh_config.write_text("Host dell\n  HostName 10.0.0.10\nHost mac-mini\n  HostName 10.0.0.11\n")

    svc, http_client, _ = _build_test_gateway(tmp_path, monkeypatch)
    svc._tunnel_manager = _FakeTunnelManager()  # pyright: ignore[reportPrivateUsage]
    svc._ssh_config_path = ssh_config  # pyright: ignore[reportPrivateUsage]
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        hosts_payload = await client.remote_gpu_hosts()
        assert hosts_payload["hosts"] == ["dell", "mac-mini"]

        conn_res = await client.connect_remote_gpu("mac-mini")
        assert conn_res["status"] == "ok"
        assert conn_res["connected"] is True
        assert conn_res["llm_on_remote"] is True
        assert conn_res["images_on_remote"] is True
        assert conn_res["applied_changes"]["connection"] == "remote-gpu"

        # Now `mac-mini` is remembered first in hosts, and `remote-gpu` connection is registered
        hosts_after = await client.remote_gpu_hosts()
        assert hosts_after["hosts"][0] == "mac-mini"
        conns = await client.list_connections()
        conn_ids = {c["id"] for c in conns["connections"]}
        assert "remote-gpu" in conn_ids

        # Simulate gateway restart: disconnect tunnel in memory, then call `reconnect_remote_gpu()`
        await svc.tunnel_manager.disconnect()
        reconnected = await svc.reconnect_remote_gpu()
        assert reconnected is not None
        assert reconnected["connected"] is True
        assert reconnected["tunnel"]["host"] == "mac-mini"

        # Explicit disconnect removes `remote-gpu` connection and clears session file
        disc_res = await client.disconnect_remote_gpu()
        assert disc_res["connected"] is False
        assert disc_res["restored_settings"].get("connection") == "remote-gpu"
        assert await svc.reconnect_remote_gpu() is None
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_gateway_persists_offline_target_and_heals_on_demand_and_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline startup target persists `remote_gpu_session.json`, heals on-demand for `remote-gpu/...` turns, and auto-recovers unexpected tunnel drops.

    Killed by: src/uclone_x/shells/gateway_server.py :: await svc.ensure_remote_gpu_for_model(llm_request.model)
    Becomes:   pass
    """

    class _ToggleTunnelManager(SSHTunnelManager):
        def __init__(self) -> None:
            super().__init__()
            self.online = False
            self.connect_calls = 0

        async def connect(
            self,
            host: str,
            preferred_local_ollama_port: int | None = None,
            preferred_local_comfyui_port: int | None = None,
            auto_start_comfyui: bool = True,
            auto_start_ollama: bool = True,
            timeout: float = 12.0,
        ) -> TunnelSessionStatus:
            _ = (preferred_local_comfyui_port, auto_start_comfyui, auto_start_ollama, timeout)
            self.connect_calls += 1
            if not self.online:
                self._session = TunnelSessionStatus(
                    host=host,
                    connected=False,
                    error="ssh: connect to host 10.202.1.123 port 22: Operation timed out",
                )
                return self._session
            ollama_port = preferred_local_ollama_port or 11435
            self._session = TunnelSessionStatus(
                host=host,
                connected=True,
                pid=9191,
                mappings=[
                    PortMapping(service_name="ollama", remote_port=11434, local_port=ollama_port),
                ],
                gpu=RemoteGPUInfo(
                    name="NVIDIA GeForce RTX 5070 Ti",
                    total_mb=16303,
                    used_mb=512,
                    driver="570.12",
                ),
                ollama_models=["qwen3:14b"],
            )
            return self._session

        @property
        def is_connected(self) -> bool:
            return bool(self._session is not None and self._session.connected)

        async def simulate_unexpected_drop(self) -> None:
            assert self._session is not None
            self._session.connected = False
            self._session.pid = None
            self._session.error = "Tunnel process exited (code 255): Broken pipe"
            for cb in list(self._disconnect_listeners):
                res = cb()
                if asyncio.iscoroutine(res):
                    await res

    svc, http_client, _ = _build_test_gateway(tmp_path, monkeypatch)
    toggle_mgr = _ToggleTunnelManager()
    toggle_mgr.add_disconnect_listener(
        svc._on_tunnel_disconnected  # pyright: ignore[reportPrivateUsage]
    )
    svc._tunnel_manager = toggle_mgr  # pyright: ignore[reportPrivateUsage]
    svc._drop_recovery_delays = (0.01,)  # pyright: ignore[reportPrivateUsage]
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        # 1. Startup auto-connect to `dell` while `dell` is offline still saves desired session
        init_res = await svc.reconnect_remote_gpu(explicit_host="dell")
        assert init_res is not None
        assert init_res["connected"] is False

        st_offline = await client.remote_gpu_status()
        assert st_offline["connected"] is False
        assert st_offline["host"] == "dell"
        assert st_offline["desired_host"] == "dell"
        assert st_offline["auto_reconnect"] is True
        assert "Operation timed out" in str(st_offline["error"])

        # 2. `dell` comes online; expire cooldown and send chat request to `remote-gpu/qwen3:14b`
        toggle_mgr.online = True
        svc._last_reconnect_mono = 0.0  # pyright: ignore[reportPrivateUsage]
        resp = await client.generate(
            LLMRequest(
                model="remote-gpu/qwen3:14b",
                messages=(ChatMessage(role=MessageRole.USER, content="ping"),),
            )
        )
        assert resp.content is not None
        st_healed = await client.remote_gpu_status()
        assert st_healed["connected"] is True
        assert st_healed["host"] == "dell"

        # 3. Unexpected SSH drop triggers bounded background recovery when `dell` is still online
        calls_before_drop = toggle_mgr.connect_calls
        await toggle_mgr.simulate_unexpected_drop()
        for _ in range(100):
            if svc.tunnel_manager.is_connected and toggle_mgr.connect_calls > calls_before_drop:
                break
            await asyncio.sleep(0.01)
        assert svc.tunnel_manager.is_connected is True

        # 4. Explicit disconnect clears session and stops auto-recovery
        await client.disconnect_remote_gpu()
        st_disc = await client.remote_gpu_status()
        assert st_disc["connected"] is False
        assert st_disc["desired_host"] is None
        assert st_disc["auto_reconnect"] is False
    finally:
        await svc.close()
        await http_client.aclose()


def test_cli_gateway_remote_gpu_status_connect_probe_and_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ucx gateway` status/probe/connect/disconnect expose remote GPU diagnostics and zero-restart controls.

    Killed by: src/uclone_x/cli/commands/gateway.py :: if not bool(res.get("connected")):
    Becomes:   if bool(res.get("connected")):
    """

    async def _ok_conns(self: GatewayModelClient) -> dict[str, Any]:
        _ = self
        return {"connections": []}

    async def _ok_usage(self: GatewayModelClient) -> dict[str, Any]:
        _ = self
        return {"windows": []}

    async def _failed_status(self: GatewayModelClient) -> dict[str, Any]:
        _ = self
        return {
            "host": "dell",
            "desired_host": "dell",
            "connected": False,
            "auto_reconnect": True,
            "reconnecting": False,
            "error": "ssh: connect to host 10.202.1.123 port 22: Operation timed out",
        }

    connect_state: dict[str, Any] = {"succeed": False, "seen_hosts": []}

    async def _connect(
        self: GatewayModelClient,
        host: str,
        *,
        apply_settings: bool = True,
        auto_start_comfyui: bool = True,
        sync_llm: bool = True,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        _ = (self, apply_settings, auto_start_comfyui, sync_llm, timeout)
        connect_state["seen_hosts"].append(host)
        if not connect_state["succeed"]:
            return {
                "status": "error",
                "connected": False,
                "error": "Operation timed out",
            }
        return {
            "status": "ok",
            "connected": True,
            "llm_on_remote": True,
            "images_on_remote": True,
            "tunnel": {
                "host": host,
                "pid": 8888,
                "gpu": {"name": "RTX 5070 Ti", "total_mb": 16303},
                "mappings": [{"service_name": "ollama", "local_port": 11435, "remote_port": 11434}],
            },
        }

    async def _probe_host(
        self: GatewayModelClient, host: str, *, timeout: float = 6.0
    ) -> dict[str, Any]:
        _ = (self, timeout)
        return {
            "host": host,
            "reachable": True,
            "gpu": {"name": "RTX 5070 Ti", "total_mb": 16303, "used_mb": 256, "driver": "570.12"},
            "ports": {"ollama": {"port": 11434, "listening": True}},
            "ollama_models": ["qwen3:14b"],
        }

    async def _disconnect(self: GatewayModelClient) -> dict[str, Any]:
        _ = self
        return {
            "status": "ok",
            "connected": False,
            "restored_settings": {"connection": "remote-gpu"},
        }

    monkeypatch.setattr(GatewayModelClient, "list_connections", _ok_conns)
    monkeypatch.setattr(GatewayModelClient, "get_usage", _ok_usage)
    monkeypatch.setattr(GatewayModelClient, "remote_gpu_status", _failed_status)
    monkeypatch.setattr(GatewayModelClient, "connect_remote_gpu", _connect)
    monkeypatch.setattr(GatewayModelClient, "probe_remote_gpu", _probe_host)
    monkeypatch.setattr(GatewayModelClient, "disconnect_remote_gpu", _disconnect)

    runner = CliRunner()

    st = runner.invoke(ucx_app, ["gateway", "status", "--url", "http://127.0.0.1:8765"])
    assert st.exit_code == 0
    assert "dell" in st.output
    assert "Operation timed out" in st.output

    pr = runner.invoke(ucx_app, ["gateway", "probe", "dell", "--url", "http://127.0.0.1:8765"])
    assert pr.exit_code == 0
    assert "RTX 5070 Ti" in pr.output

    # Omitting host uses `desired_host` ("dell"); failure exits 1
    fail_conn = runner.invoke(ucx_app, ["gateway", "connect", "--url", "http://127.0.0.1:8765"])
    assert fail_conn.exit_code == 1
    assert connect_state["seen_hosts"] == ["dell"]

    # Succeeds when host is reachable
    connect_state["succeed"] = True
    ok_conn = runner.invoke(
        ucx_app, ["gateway", "connect", "dell", "--url", "http://127.0.0.1:8765"]
    )
    assert ok_conn.exit_code == 0
    assert "RTX 5070 Ti" in ok_conn.output

    disc = runner.invoke(ucx_app, ["gateway", "disconnect", "--url", "http://127.0.0.1:8765"])
    assert disc.exit_code == 0


@pytest.mark.asyncio
async def test_gateway_non_blocking_background_reconnect_decouples_api_latency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal chat/status API calls trigger background SSH recovery without blocking on SSH latency.

    Killed by: src/uclone_x/shells/gateway_server.py :: self.trigger_background_reconnect(cooldown_seconds=cooldown_seconds)
    Becomes:   pass
    """

    class _SlowTunnelManager(SSHTunnelManager):
        def __init__(self) -> None:
            super().__init__()
            self.online = False
            self.connect_gate = asyncio.Event()

        async def connect(
            self,
            host: str,
            preferred_local_ollama_port: int | None = None,
            preferred_local_comfyui_port: int | None = None,
            auto_start_comfyui: bool = True,
            auto_start_ollama: bool = True,
            timeout: float = 12.0,
        ) -> TunnelSessionStatus:
            _ = (
                preferred_local_ollama_port,
                preferred_local_comfyui_port,
                auto_start_comfyui,
                auto_start_ollama,
                timeout,
            )
            if not self.online:
                self._session = TunnelSessionStatus(
                    host=host,
                    connected=False,
                    error="ssh: connect to host dell port 22: Connection refused",
                )
                return self._session
            await self.connect_gate.wait()
            self._session = TunnelSessionStatus(
                host=host,
                connected=True,
                pid=9292,
                mappings=[
                    PortMapping(service_name="ollama", remote_port=11434, local_port=11435),
                    PortMapping(service_name="comfyui", remote_port=8188, local_port=8188),
                ],
            )
            return self._session

        @property
        def is_connected(self) -> bool:
            return bool(self._session is not None and self._session.connected)

    svc, http_client, _ = _build_test_gateway(tmp_path, monkeypatch)
    slow_mgr = _SlowTunnelManager()
    svc._tunnel_manager = slow_mgr  # pyright: ignore[reportPrivateUsage]
    svc._drop_recovery_delays = (0.01,)  # pyright: ignore[reportPrivateUsage]
    client = GatewayModelClient(base_url="http://gateway.test", http_client=http_client)
    try:
        await svc.add_connection("openai", label="OpenAI", base_url=None, key="sk-test")
        await svc.reconnect_remote_gpu(explicit_host="dell")
        assert svc.tunnel_manager.is_connected is False

        # Host comes back online, but SSH handshake is gated on `connect_gate`
        slow_mgr.online = True
        svc._last_reconnect_mono = 0.0  # pyright: ignore[reportPrivateUsage]

        # Chat call on local/cloud model returns immediately without waiting for `connect_gate`
        resp = await client.generate(
            LLMRequest(
                model="openai/gpt-5.4",
                messages=(ChatMessage(role=MessageRole.USER, content="draw a cat"),),
            )
        )
        assert resp.content is not None
        assert svc._drop_recovery_task is not None  # pyright: ignore[reportPrivateUsage]
        assert svc.tunnel_manager.is_connected is False

        # Release SSH handshake gate; background task finishes and restores `remote-gpu-pictures`
        slow_mgr.connect_gate.set()
        await svc._drop_recovery_task  # pyright: ignore[reportPrivateUsage]
        assert svc.tunnel_manager.is_connected is True
        assert svc.gateway.connection("remote-gpu-pictures") is not None
    finally:
        slow_mgr.connect_gate.set()
        await svc.close()
        await http_client.aclose()


def test_image_engine_choice_infers_gpu_tunnel_comfy_port_from_remote_gpu_pictures(
    tmp_path: Path,
) -> None:
    """`image_engine_choice` and `probe_image_engines` recognize `remote-gpu-pictures` port as `gpu_server`.

    Killed by: src/uclone_x/llm/connectors/factory.py :: gpu_tunnel_comfy_port = parsed_port
    Becomes:   pass
    Killed by: src/uclone_x/tools/builtin/image_status.py :: gpu_tunnel_comfy_port=choice.gpu_tunnel_comfy_port,
    Becomes:   gpu_tunnel_comfy_port=gpu_tunnel_comfy_port,
    """
    from unittest.mock import AsyncMock, patch

    from uclone_x.llm.connectors.factory import image_engine_choice
    from uclone_x.llm.connectors.saved_choice import update_connection
    from uclone_x.tools.builtin.image import ComfyUIImageEngine, image_where
    from uclone_x.tools.builtin.image_status import probe_image_engines, resolve_image_choice

    settings_path = tmp_path / "settings.json"
    update_connection(
        "remote-gpu-pictures",
        kind="comfyui",
        create=True,
        label="Remote GPU pictures",
        base_url="http://127.0.0.1:8188",
        path=settings_path,
    )
    choice = image_engine_choice(settings_path)
    assert choice.gpu_tunnel_comfy_port == 8188
    assert image_where("comfyui-local", choice.comfyui_base_url, choice.gpu_tunnel_comfy_port) == (
        "gpu_server"
    )
    with patch.object(ComfyUIImageEngine, "is_available", AsyncMock(return_value=True)):
        report = probe_image_engines(settings_path)
    assert report.gpu_tunnel_comfy_port == 8188
    resolved = resolve_image_choice(report)
    assert resolved["create"] is not None
    assert resolved["create"]["where"] == "gpu_server"
