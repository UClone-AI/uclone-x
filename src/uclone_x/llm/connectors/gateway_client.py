"""Unified agent client connector for the standalone Model Gateway server (#2475).

Agents, room selectors, compactors, and embedders talk to the Model Gateway through
`GatewayModelClient` (`LLMProviderProtocol`, `ImageInputProbe`, `UsageRemainingProtocol`)
and `GatewayEmbedder` (`EmbedderProtocol`). Neither imports any upstream provider SDK
nor holds upstream provider credentials.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final, NoReturn, cast

import httpx

from uclone_x.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    LLMCredentialsNotConfiguredError,
    LLMModelNotConfiguredError,
    LLMProviderError,
    LLMRequestRefusedError,
    ModelLacksToolSupportError,
    ModelNotAvailableError,
    PictureModelNotForChatError,
    PlainRefusalError,
    ProviderAuthError,
    ProviderFailureError,
    ProviderFailureKind,
    ProviderOutageError,
    ProviderQuotaError,
    ProviderResponseError,
    ProviderUnreachableError,
    StructuredOutputUnsupportedError,
    TurnCeilingReachedError,
    UsageLimitReachedError,
)
from uclone_x.llm.connectors.base import BaseLLMConnector, named_model
from uclone_x.llm.models import (
    ChatMessage,
    ImagePart,
    LLMRequest,
    ModelResponse,
    StreamChunk,
)
from uclone_x.llm.usage.admission import (
    ADMITTED as _ADMITTED,
)
from uclone_x.llm.usage.admission import (
    book_turn_spend,
    check_turn_ceiling,
)

__all__ = [
    "DEFAULT_GATEWAY_URL",
    "GATEWAY_ENDPOINT_ENV_VARS",
    "GATEWAY_KEY_ENV_VAR",
    "GatewayEmbedder",
    "GatewayModelClient",
    "deserialize_llm_request",
    "has_configured_gateway_endpoint",
    "raise_from_gateway_error",
    "resolve_gateway_base_url",
    "serialize_gateway_error",
    "serialize_llm_request",
]

GATEWAY_ENDPOINT_ENV_VARS: Final[tuple[str, ...]] = ("UCX_GATEWAY_URL", "LLM_GATEWAY_URL")
GATEWAY_KEY_ENV_VAR: Final[str] = "UCX_GATEWAY_KEY"
DEFAULT_GATEWAY_URL: Final[str] = "http://127.0.0.1:8765"

#: Header marking a call made inside an already-admitted agent turn (#2431).
TURN_ADMITTED_HEADER: Final[str] = "X-UCX-Turn-Admitted"


def _clean_env(name: str) -> str | None:
    val = os.getenv(name)
    return val.strip() if val is not None and val.strip() else None


def has_configured_gateway_endpoint(base_url: str | None = None) -> bool:
    """Whether an explicit `base_url` or a gateway endpoint variable is set."""
    if base_url is not None and base_url.strip():
        return True
    return any(_clean_env(var) is not None for var in GATEWAY_ENDPOINT_ENV_VARS)


def resolve_gateway_base_url(base_url: str | None = None) -> str:
    """The gateway server address from `base_url`, environment, or `DEFAULT_GATEWAY_URL`."""
    if base_url is not None and base_url.strip():
        return base_url.strip().rstrip("/")
    for var in GATEWAY_ENDPOINT_ENV_VARS:
        found = _clean_env(var)
        if found is not None:
            return found.rstrip("/")
    return DEFAULT_GATEWAY_URL


def serialize_llm_request(request: LLMRequest) -> dict[str, Any]:
    """Serialize `request` to JSON-compatible dict, preserving `ImagePart.data` across the wire."""
    payload = request.model_dump(mode="json")
    image_data: dict[str, str] = {}
    for message in request.messages:
        for image in message.images:
            if image.data is not None:
                image_data[image.digest] = image.data
    if image_data:
        payload["image_data"] = image_data
    return payload


def deserialize_llm_request(payload: Mapping[str, Any]) -> LLMRequest:
    """Reconstruct `LLMRequest` from `serialize_llm_request`, restoring `ImagePart.data`."""
    raw = dict(payload)
    raw_image_data = raw.pop("image_data", None)
    request = LLMRequest.model_validate_json(json.dumps(raw))
    if not isinstance(raw_image_data, Mapping) or not raw_image_data:
        return request
    image_map = {
        str(k): str(v)
        for k, v in cast(Mapping[object, object], raw_image_data).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    if not image_map:
        return request
    messages: list[ChatMessage] = []
    changed = False
    for msg in request.messages:
        if not msg.images:
            messages.append(msg)
            continue
        restored: list[ImagePart] = []
        for img in msg.images:
            data = image_map.get(img.digest)
            if data is not None and img.data is None:
                restored.append(
                    ImagePart(
                        media_type=img.media_type,
                        digest=img.digest,
                        data=data,
                        width=img.width,
                        height=img.height,
                    )
                )
                changed = True
            else:
                restored.append(img)
        messages.append(msg.model_copy(update={"images": tuple(restored)}))
    return request.model_copy(update={"messages": tuple(messages)}) if changed else request


class _DynamicProviderFailureError(ProviderFailureError):
    """A `ProviderFailureError` reconstructed from a gateway error envelope."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: ProviderFailureKind,
        is_retryable: bool,
        provider: str,
        model: str,
        status_code: int | None = None,
        body_excerpt: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.kind = failure_kind  # type: ignore[misc]
        self.retryable = is_retryable  # type: ignore[misc]
        self.status_code = status_code
        self.body_excerpt = body_excerpt


def serialize_gateway_error(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Convert an exception on the gateway server into `(status_code, error_dict)`."""
    if isinstance(exc, UsageLimitReachedError):
        return 429, {
            "error_type": "UsageLimitReachedError",
            "detail": str(exc),
            "window": exc.window,
            "available_again_at": exc.available_again_at.astimezone(UTC).isoformat(),
        }
    if isinstance(exc, TurnCeilingReachedError):
        return 429, {
            "error_type": "TurnCeilingReachedError",
            "detail": str(exc),
            "used": exc.used,
            "ceiling": exc.ceiling,
        }
    if isinstance(exc, ProviderFailureError):
        status = 503 if exc.retryable else 422
        if exc.kind == ProviderFailureKind.PROVIDER_AUTH:
            status = 401
        elif exc.kind == ProviderFailureKind.PROVIDER_QUOTA:
            status = 429
        elif exc.kind == ProviderFailureKind.MODEL_UNAVAILABLE:
            status = 404
        return status, {
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "failure_kind": exc.kind.value,
            "retryable": exc.retryable,
            "provider": exc.provider,
            "model": exc.model,
            "status_code": exc.status_code,
            "body_excerpt": exc.body_excerpt,
        }
    if isinstance(exc, LLMModelNotConfiguredError):
        return 400, {
            "error_type": "LLMModelNotConfiguredError",
            "detail": str(exc),
            "provider": exc.provider,
        }
    if isinstance(exc, LLMCredentialsNotConfiguredError):
        return 401, {
            "error_type": "LLMCredentialsNotConfiguredError",
            "detail": str(exc),
        }
    if isinstance(exc, StructuredOutputUnsupportedError):
        return 400, {
            "error_type": "StructuredOutputUnsupportedError",
            "detail": str(exc),
            "provider": exc.provider,
        }
    if isinstance(exc, ModelLacksToolSupportError):
        return 400, {
            "error_type": "ModelLacksToolSupportError",
            "detail": str(exc),
            "model": exc.model,
        }
    if isinstance(exc, LLMRequestRefusedError):
        return 400, {
            "error_type": "LLMRequestRefusedError",
            "detail": str(exc),
            "status_code": exc.status_code,
        }
    if isinstance(exc, EmbeddingDimensionError):
        return 422, {
            "error_type": "EmbeddingDimensionError",
            "detail": str(exc),
            "model": exc.model,
            "expected": exc.expected,
            "actual": exc.actual,
        }
    if isinstance(exc, EmbeddingError):
        return 502, {
            "error_type": "EmbeddingError",
            "detail": str(exc),
        }
    if isinstance(exc, PlainRefusalError):
        return 400, {
            "error_type": "PlainRefusalError",
            "detail": str(exc),
            "reason_code": exc.reason_code,
        }
    return 502, {
        "error_type": type(exc).__name__,
        "detail": str(exc),
    }


def _str_field(data: Mapping[str, Any], key: str, fallback: str) -> str:
    raw = data.get(key)
    return raw if isinstance(raw, str) and raw else fallback


def _int_field(data: Mapping[str, Any], key: str, fallback: int) -> int:
    raw = data.get(key)
    return int(raw) if isinstance(raw, int) else fallback


def raise_from_gateway_error(
    status_code: int,
    payload: object,
    *,
    default_provider: str = "gateway",
    default_model: str = "default",
) -> NoReturn:
    """Re-raise the typed `UCloneXError` encoded by `serialize_gateway_error`."""
    if not isinstance(payload, Mapping):
        raise LLMProviderError(f"Gateway error (HTTP {status_code}): {payload}")
    data = cast(Mapping[str, Any], payload)
    err_type = _str_field(data, "error_type", "")
    detail = _str_field(data, "detail", f"Gateway request failed with HTTP {status_code}")
    provider = _str_field(data, "provider", default_provider)
    model = _str_field(data, "model", default_model)

    if err_type == "UsageLimitReachedError":
        raw_at = _str_field(data, "available_again_at", "")
        try:
            again = datetime.fromisoformat(raw_at)
        except ValueError:
            again = datetime.now(UTC)
        raise UsageLimitReachedError(
            detail,
            window=_str_field(data, "window", "per_10_minutes"),
            available_again_at=again,
        )
    if err_type == "TurnCeilingReachedError":
        raise TurnCeilingReachedError(
            _int_field(data, "used", 0),
            _int_field(data, "ceiling", 0),
        )
    if err_type == "ModelNotAvailableError":
        raise ModelNotAvailableError(provider=provider, model=model)
    if err_type == "PictureModelNotForChatError":
        raise PictureModelNotForChatError(provider=provider, model=model)
    if err_type == "ProviderAuthError":
        raise ProviderAuthError(provider=provider, model=model)
    if err_type == "ProviderQuotaError":
        raise ProviderQuotaError(provider=provider, model=model)
    if err_type == "ProviderUnreachableError":
        raise ProviderUnreachableError(provider=provider, model=model)
    if err_type == "ProviderOutageError":
        raise ProviderOutageError(provider=provider, model=model)
    if err_type == "ProviderResponseError":
        err = ProviderResponseError(provider=provider, model=model)
        if isinstance(data.get("status_code"), int):
            err.status_code = int(data["status_code"])
        if isinstance(data.get("body_excerpt"), str):
            err.body_excerpt = str(data["body_excerpt"])
        raise err
    if "failure_kind" in data:
        raw_kind = str(data["failure_kind"])
        try:
            kind = ProviderFailureKind(raw_kind)
        except ValueError:
            kind = ProviderFailureKind.PROVIDER_ERROR
        raise _DynamicProviderFailureError(
            detail,
            failure_kind=kind,
            is_retryable=bool(data.get("retryable", True)),
            provider=provider,
            model=model,
            status_code=int(data["status_code"])
            if isinstance(data.get("status_code"), int)
            else None,
            body_excerpt=str(data["body_excerpt"])
            if isinstance(data.get("body_excerpt"), str)
            else None,
        )
    if err_type == "LLMModelNotConfiguredError":
        raise LLMModelNotConfiguredError(provider)
    if err_type == "LLMCredentialsNotConfiguredError":
        raise LLMCredentialsNotConfiguredError(detail)
    if err_type == "StructuredOutputUnsupportedError":
        raise StructuredOutputUnsupportedError(provider)
    if err_type == "ModelLacksToolSupportError":
        raise ModelLacksToolSupportError(model)
    if err_type == "LLMRequestRefusedError":
        raise LLMRequestRefusedError(
            detail,
            status_code=_int_field(data, "status_code", 400),
        )
    if err_type == "EmbeddingDimensionError":
        raise EmbeddingDimensionError(
            model=model,
            expected=_int_field(data, "expected", 0),
            actual=_int_field(data, "actual", 0),
        )
    if err_type == "EmbeddingError":
        raise EmbeddingError(detail)
    if err_type == "PlainRefusalError":
        reason_code = data.get("reason_code")
        raise PlainRefusalError(
            detail,
            reason_code=str(reason_code) if isinstance(reason_code, str) else None,
        )
    raise LLMProviderError(detail)


class GatewayModelClient(BaseLLMConnector):
    """Single unified model API client speaking to the Model Gateway server (#2475).

    Implements `LLMProviderProtocol`, `ImageInputProbe`, and `UsageRemainingProtocol` over
    a persistent warm HTTP connection pool, plus control-plane methods for connections,
    models, default slots, and usage quotas.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else (_clean_env(GATEWAY_KEY_ENV_VAR) or None)
        resolved_url = resolve_gateway_base_url(base_url)
        super().__init__(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
            http_client=http_client,
        )
        self.base_url: str = resolved_url
        self.default_model: str | None = named_model(model)
        self._external_client = http_client is not None
        self._client: httpx.AsyncClient | None = http_client
        self._image_support_cache: dict[str, bool] = {}
        self._cached_ceiling: int | None = None
        self._ceiling_known: bool = False

    @property
    def provider_name(self) -> str:
        return "gateway"

    @property
    def paid(self) -> bool:
        # The gateway server enforces `UsageGate` centrally on paid upstream connections;
        # leaving `paid=False` here avoids double-booking in the caller's local SQLite file.
        return False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(
                    max_keepalive_connections=32,
                    max_connections=128,
                    keepalive_expiry=60.0,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP pool if owned by this client."""
        if self._client is not None and not self._external_client:
            await self._client.aclose()
            self._client = None

    def _headers(self, *, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if _ADMITTED.get():
            headers[TURN_ADMITTED_HEADER] = "1"
        if extra:
            headers.update(extra)
        return headers

    def _effective_request(self, request: LLMRequest) -> LLMRequest:
        if named_model(request.model) is None and self.default_model is not None:
            return request.model_copy(update={"model": self.default_model})
        return request

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: object = None,
        params: Mapping[str, str] | None = None,
        default_model: str = "default",
    ) -> dict[str, Any]:
        client = self._get_client()
        url = f"{self.base_url}{path}" if self._external_client else path
        try:
            resp = await client.request(
                method,
                url,
                headers=self._headers(),
                json=json_body,
                params=params,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnreachableError(
                provider="Gateway",
                model=default_model,
            ) from exc
        try:
            body: object = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        if resp.status_code >= 400:
            raise_from_gateway_error(
                resp.status_code,
                body,
                default_provider="gateway",
                default_model=default_model,
            )
        if not isinstance(body, Mapping):
            raise LLMProviderError(f"Gateway returned non-object JSON from {path}")
        return dict(cast(Mapping[str, Any], body))

    # -- LLMProviderProtocol --

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if _ADMITTED.get():
            check_turn_ceiling()
        effective = self._effective_request(request)
        target_model = effective.model if effective.model is not None else "default"
        data = await self._request_json(
            "POST",
            "/v1/chat/generate",
            json_body=serialize_llm_request(effective),
            default_model=target_model,
        )
        paid = bool(data.pop("paid", False))
        response = ModelResponse.model_validate_json(json.dumps(data))
        if _ADMITTED.get() and paid:
            book_turn_spend(response.usage.input_tokens + response.usage.output_tokens)
        return response

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        if _ADMITTED.get():
            check_turn_ceiling()
        effective = self._effective_request(request)
        target_model = effective.model if effective.model is not None else "default"
        client = self._get_client()
        url = f"{self.base_url}/v1/chat/stream" if self._external_client else "/v1/chat/stream"
        headers = self._headers(extra={"Accept": "text/event-stream"})
        try:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=serialize_llm_request(effective),
            ) as resp:
                if resp.status_code >= 400:
                    raw_bytes = await resp.aread()
                    try:
                        err_body: object = json.loads(raw_bytes.decode("utf-8"))
                    except ValueError:
                        err_body = {"detail": raw_bytes.decode("utf-8", errors="replace")}
                    raise_from_gateway_error(
                        resp.status_code,
                        err_body,
                        default_provider="gateway",
                        default_model=target_model,
                    )
                paid_header = resp.headers.get("X-UCX-Paid", "").strip() == "1"
                event_kind = "chunk"
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        event_kind = "chunk"
                        continue
                    if line.startswith("event:"):
                        event_kind = line[len("event:") :].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    if event_kind == "error":
                        try:
                            err_payload: object = json.loads(data_str)
                        except ValueError:
                            err_payload = {"detail": data_str}
                        raise_from_gateway_error(
                            502,
                            err_payload,
                            default_provider="gateway",
                            default_model=target_model,
                        )
                    chunk = StreamChunk.model_validate_json(data_str)
                    if _ADMITTED.get() and paid_header and chunk.usage is not None:
                        book_turn_spend(chunk.usage.input_tokens + chunk.usage.output_tokens)
                    yield chunk
        except httpx.HTTPError as exc:
            raise ProviderUnreachableError(
                provider="Gateway",
                model=target_model,
            ) from exc

    # -- ImageInputProbe --

    async def accepts_images(self, model: str | None = None) -> bool:
        chosen = named_model(model) or self.default_model
        target = chosen if chosen is not None else "deep"
        if target in self._image_support_cache:
            return self._image_support_cache[target]
        try:
            res = await self._request_json(
                "GET",
                "/v1/models/accepts-images",
                params={"model": target},
                default_model=target,
            )
            supported = bool(res.get("accepts_images", False))
        except Exception:
            return False
        self._image_support_cache[target] = supported
        return supported

    # -- Usage admission & UsageRemainingProtocol --

    async def check_admission(self) -> int | None:
        """Ask the gateway whether a new turn is admitted and retrieve the turn ceiling."""
        res = await self._request_json("POST", "/v1/usage/admit")
        ceiling = res.get("turn_ceiling")
        self._cached_ceiling = int(ceiling) if isinstance(ceiling, int) else None
        self._ceiling_known = True
        return self._cached_ceiling

    def usage_turn_ceiling(self) -> int | None:
        return self._cached_ceiling if self._ceiling_known else None

    def usage_remaining(self) -> int | None:
        return None

    # -- Control Plane: Connections, Models, Defaults, Usage --

    async def list_connections(self) -> dict[str, Any]:
        """Return connectable server kinds and currently configured connections."""
        return await self._request_json("GET", "/v1/connections")

    async def add_connection(
        self,
        kind: str,
        *,
        label: str | None = None,
        base_url: str | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        """Register a new connection on the gateway and probe its live catalog."""
        return await self._request_json(
            "POST",
            "/v1/connections",
            json_body={
                "kind": kind,
                "label": label,
                "base_url": base_url,
                "key": key,
            },
        )

    async def patch_connection(self, conn_id: str, **changes: Any) -> dict[str, Any]:
        """Update a connection's label, address, or API key on the gateway."""
        return await self._request_json(
            "PATCH",
            f"/v1/connections/{conn_id}",
            json_body=changes,
        )

    async def check_connection(self, conn_id: str) -> dict[str, Any]:
        """Force an immediate catalog probe on `conn_id`."""
        return await self._request_json("POST", f"/v1/connections/{conn_id}/check")

    async def connection_dependents(self, conn_id: str) -> dict[str, Any]:
        """Return clones and default slots that reference `conn_id`."""
        return await self._request_json("GET", f"/v1/connections/{conn_id}/dependents")

    async def remove_connection(self, conn_id: str) -> dict[str, Any]:
        """Remove a saved connection from the gateway."""
        return await self._request_json("DELETE", f"/v1/connections/{conn_id}")

    async def list_models(
        self, capability: str = "chat", *, refresh: bool = False
    ) -> dict[str, Any]:
        """Return the grouped live model catalog, defaults, and recommendations."""
        params: dict[str, str] = {"capability": capability}
        if refresh:
            params["refresh"] = "1"
        return await self._request_json("GET", "/v1/models", params=params)

    async def get_defaults(self) -> dict[str, Any]:
        """Return the active default model slots (`deep`, `fast`, `image`)."""
        return await self._request_json("GET", "/v1/defaults")

    async def set_defaults(self, **changes: str | None) -> dict[str, Any]:
        """Update the saved default model slots (`deep`, `fast`, `image`)."""
        return await self._request_json("PUT", "/v1/defaults", json_body=changes)

    async def get_usage(self) -> dict[str, Any]:
        """Return rolling window token usage, saved limits, and provider breakdown."""
        return await self._request_json("GET", "/v1/usage")

    async def set_usage_limits(
        self,
        *,
        per_10_minutes: int | None = None,
        per_5_hours: int | None = None,
        per_week: int | None = None,
    ) -> dict[str, Any]:
        """Update the user's rolling window token limits on the gateway."""
        return await self._request_json(
            "PUT",
            "/v1/usage/limits",
            json_body={
                "per_10_minutes": per_10_minutes,
                "per_5_hours": per_5_hours,
                "per_week": per_week,
            },
        )

    # -- Control Plane: Remote Machine SSH Tunneling & Auto-Connection (Dell / Mac mini) --

    async def remote_gpu_hosts(self) -> dict[str, Any]:
        """Return configured SSH hosts (`~/.ssh/config`) ordered with the last-connected host first."""
        return await self._request_json("GET", "/v1/remote-gpu/hosts")

    async def remote_gpu_status(self) -> dict[str, Any]:
        """Return active SSH tunnel status, `llm_on_remote`, and `images_on_remote`."""
        return await self._request_json("GET", "/v1/remote-gpu/status")

    async def probe_remote_gpu(self, host: str, *, timeout: float = 6.0) -> dict[str, Any]:
        """Probe an SSH remote host (`dell`, `mac-mini`, etc.) for GPU/Metal and AI services."""
        return await self._request_json(
            "POST",
            "/v1/remote-gpu/probe",
            json_body={"host": host, "timeout": timeout},
        )

    async def connect_remote_gpu(
        self,
        host: str,
        *,
        apply_settings: bool = True,
        auto_start_comfyui: bool = True,
        sync_llm: bool = True,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Establish an SSH tunnel on the gateway to `host` and register its Ollama/ComfyUI connections."""
        return await self._request_json(
            "POST",
            "/v1/remote-gpu/connect",
            json_body={
                "host": host,
                "apply_settings": apply_settings,
                "auto_start_comfyui": auto_start_comfyui,
                "sync_llm": sync_llm,
                "timeout": timeout,
            },
        )

    async def disconnect_remote_gpu(self) -> dict[str, Any]:
        """Disconnect the active SSH tunnel on the gateway and restore previous connections."""
        return await self._request_json("POST", "/v1/remote-gpu/disconnect")


class GatewayEmbedder:
    """`EmbedderProtocol` client routing batch text embeddings through the Model Gateway."""

    def __init__(
        self,
        *,
        model_name: str = "nomic-embed-text",
        dimensions: int = 768,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._base_url = resolve_gateway_base_url(base_url)
        self._api_key = (
            api_key if api_key is not None else (_clean_env(GATEWAY_KEY_ENV_VAR) or None)
        )
        self._timeout = timeout
        self._external_client = http_client is not None
        self._client: httpx.AsyncClient | None = http_client

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._external_client:
            await self._client.aclose()
            self._client = None

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        client = self._get_client()
        url = f"{self._base_url}/v1/embeddings" if self._external_client else "/v1/embeddings"
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = await client.post(
                url,
                headers=headers,
                json={
                    "texts": list(texts),
                    "model": self._model_name,
                    "dimensions": self._dimensions,
                },
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"Gateway embedding endpoint at {self._base_url} is unreachable: {exc}"
            ) from exc
        try:
            body: object = resp.json()
        except ValueError as exc:
            raise EmbeddingError("Gateway returned non-JSON embedding response") from exc
        if resp.status_code >= 400:
            raise_from_gateway_error(
                resp.status_code,
                body,
                default_provider="gateway",
                default_model=self._model_name,
            )
        if not isinstance(body, Mapping):
            raise EmbeddingError("Gateway embedding response is not a JSON object")
        raw_vectors: object = cast(Mapping[str, Any], body).get("vectors")
        if not isinstance(raw_vectors, list):
            raise EmbeddingError(f"Gateway returned invalid vectors for {len(texts)} inputs")
        vec_list = cast(list[object], raw_vectors)
        if len(vec_list) != len(texts):
            raise EmbeddingError(
                f"Gateway returned {len(vec_list)} vectors for {len(texts)} inputs"
            )
        vectors: list[tuple[float, ...]] = []
        for item in vec_list:
            if not isinstance(item, list):
                raise EmbeddingError("Gateway returned a non-list embedding vector")
            vec = tuple(float(cast(int | float, x)) for x in cast(list[object], item))
            if len(vec) != self._dimensions:
                raise EmbeddingDimensionError(self._model_name, self._dimensions, len(vec))
            vectors.append(vec)
        return tuple(vectors)
