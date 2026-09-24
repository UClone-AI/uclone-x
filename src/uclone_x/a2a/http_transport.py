"""A2A remote HTTP REST/SSE wire transport conforming to A2ATransportProtocol and A2ADiscoveryProtocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from uclone_x.a2a.discovery import A2ADiscoveryService
from uclone_x.a2a.models import (
    AgentCard,
    TaskMessage,
    TaskResult,
    WireProtocolType,
)
from uclone_x.a2a.wire import task_message_to_wire, task_result_from_wire_json
from uclone_x.errors import (
    A2AError,
    InvalidAgentResponseError,
    MissingProvenanceError,
    TaskNotFoundError,
    UnsupportedOperationError,
    VersionNotSupportedError,
)


class A2AHttpTransport:
    """Remote REST/SSE transport for distributed A2A communication."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        default_agent_card: AgentCard | None = None,
        discovery_service: A2ADiscoveryService | None = None,
    ) -> None:
        self._http_client = http_client
        self._discovery = discovery_service or A2ADiscoveryService(
            local_agent_card=default_agent_card,
            http_client=http_client,
        )

    @property
    def transport_type(self) -> WireProtocolType:
        """Transport mechanism identifier."""
        return WireProtocolType.REST_SSE

    def get_local_agent_card(self) -> AgentCard:
        """Export local agent capabilities."""
        return self._discovery.get_local_agent_card()

    def set_local_agent_card(self, card: AgentCard) -> None:
        """Update local agent capabilities."""
        self._discovery.set_local_agent_card(card)

    async def fetch_remote_agent_card(self, endpoint_url: str) -> AgentCard:
        """Fetch remote /.well-known/agent-card.json per RFC 8615."""
        return await self._discovery.fetch_remote_agent_card(endpoint_url)

    async def send_task(self, target_endpoint: str, message: TaskMessage) -> TaskResult:
        """Dispatch task to remote agent over HTTP REST."""
        url = target_endpoint.strip()
        if not (
            url.endswith("/message:send") or url.endswith("/tasks/send") or url.endswith("/tasks")
        ):
            url = f"{url.rstrip('/')}/message:send"

        client = self._http_client or httpx.AsyncClient()
        should_close = self._http_client is None
        try:
            resp = await client.post(
                url,
                headers={
                    "A2A-Version": "1.0",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=task_message_to_wire(message),
                timeout=30.0,
            )
            if resp.status_code == 404:
                err_msg = f"Endpoint or task not found at {url}"
                try:
                    raw_err = resp.json()
                    if isinstance(raw_err, dict):
                        raw_dict: dict[str, Any] = cast(dict[str, Any], raw_err)
                        detail = raw_dict.get("detail")
                        if isinstance(detail, dict) and "message" in detail:
                            detail_dict: dict[str, Any] = cast(dict[str, Any], detail)
                            err_msg = str(detail_dict["message"])
                        elif isinstance(detail, str):
                            err_msg = detail
                        elif "message" in raw_dict:
                            err_msg = str(raw_dict["message"])
                except Exception:
                    pass
                raise TaskNotFoundError(err_msg)
            if resp.status_code == 400:
                try:
                    err_obj = resp.json()
                    if isinstance(err_obj, dict):
                        err_dict: dict[str, Any] = cast(dict[str, Any], err_obj)
                        detail_obj = err_dict.get("detail")
                        detail_dict: dict[str, Any] = (
                            cast(dict[str, Any], detail_obj) if isinstance(detail_obj, dict) else {}
                        )
                        err_name = err_dict.get("error") or detail_dict.get("error")
                        err_type = str(err_name or "")
                        if err_type == "VersionNotSupportedError":
                            raise VersionNotSupportedError(
                                f"A2A version not supported by {url}: {resp.text}"
                            )
                except (VersionNotSupportedError, TaskNotFoundError):
                    raise
                except Exception:
                    pass
                raise UnsupportedOperationError(f"Bad request sent to {url}: HTTP 400 {resp.text}")
            if resp.status_code != 200:
                raise InvalidAgentResponseError(
                    f"Remote agent returned error {resp.status_code}: {resp.text}"
                )

            result = task_result_from_wire_json(resp.text)
            if result.provenance is None:
                raise MissingProvenanceError(
                    f"Remote agent response from {url} is missing provenance (P6)"
                )
            return result
        except (A2AError, MissingProvenanceError):
            raise
        except Exception as exc:
            raise InvalidAgentResponseError(
                f"Network or protocol error communicating with {url}: {exc}"
            ) from exc
        finally:
            if should_close:
                await client.aclose()

    def stream_task(
        self,
        target_endpoint: str,
        message: TaskMessage,
    ) -> AsyncIterator[str]:
        """Stream task results via remote SSE stream."""
        url = target_endpoint.strip()
        if not (url.endswith("/message:stream") or url.endswith("/tasks/stream")):
            url = f"{url.rstrip('/')}/message:stream"

        async def _stream_generator() -> AsyncIterator[str]:
            client = self._http_client or httpx.AsyncClient()
            should_close = self._http_client is None
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers={
                        "A2A-Version": "1.0",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    json=task_message_to_wire(message),
                    timeout=30.0,
                ) as resp:
                    if resp.status_code == 404:
                        err_msg = f"Endpoint or task not found at {url}"
                        try:
                            body = await resp.aread()
                            import json

                            raw_err: object = json.loads(body)
                            if isinstance(raw_err, dict):
                                err_dict: dict[str, object] = cast(dict[str, object], raw_err)
                                detail_val = err_dict.get("detail")
                                if isinstance(detail_val, dict):
                                    detail_d: dict[str, object] = cast(
                                        dict[str, object], detail_val
                                    )
                                    if "message" in detail_d:
                                        err_msg = str(detail_d["message"])
                                elif isinstance(detail_val, str):
                                    err_msg = detail_val
                                elif "message" in err_dict:
                                    err_msg = str(err_dict["message"])
                        except Exception:
                            pass
                        raise TaskNotFoundError(err_msg)
                    if resp.status_code != 200:
                        raise InvalidAgentResponseError(
                            f"Remote agent returned streaming error {resp.status_code}"
                        )
                    async for line in resp.aiter_lines():
                        if line:
                            yield line
            except (A2AError, MissingProvenanceError):
                raise
            except Exception as exc:
                raise InvalidAgentResponseError(
                    f"Network error during stream from {url}: {exc}"
                ) from exc
            finally:
                if should_close:
                    await client.aclose()

        return _stream_generator()
