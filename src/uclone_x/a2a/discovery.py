"""A2A discovery service for publishing and discovering Agent Cards per RFC 8615."""

from __future__ import annotations

import httpx

from uclone_x.a2a.models import AgentCard
from uclone_x.errors import (
    InvalidAgentResponseError,
    TaskNotFoundError,
)


class A2ADiscoveryService:
    """Service for discovering and publishing Agent Cards per RFC 8615."""

    def __init__(
        self,
        local_agent_card: AgentCard | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._local_card = local_agent_card or AgentCard(
            name="uclone-x-agent",
            description="UClone-X Default Agent",
            version="1.0.1",
        )
        self._http_client = http_client

    def get_local_agent_card(self) -> AgentCard:
        """Export local agent capabilities."""
        return self._local_card

    def set_local_agent_card(self, card: AgentCard) -> None:
        """Update the local agent card."""
        self._local_card = card

    async def fetch_remote_agent_card(self, endpoint_url: str) -> AgentCard:
        """Fetch /.well-known/agent-card.json from remote peer per RFC 8615."""
        url = endpoint_url.strip()
        if not url.endswith("/.well-known/agent-card.json"):
            url = f"{url.rstrip('/')}/.well-known/agent-card.json"

        client = self._http_client or httpx.AsyncClient()
        should_close = self._http_client is None
        try:
            resp = await client.get(
                url,
                headers={"A2A-Version": "1.0", "Accept": "application/json"},
                timeout=10.0,
            )
            if resp.status_code == 404:
                raise TaskNotFoundError(f"Agent card not found at {url}")
            if resp.status_code != 200:
                raise InvalidAgentResponseError(
                    f"Failed to fetch agent card from {url}: HTTP {resp.status_code}"
                )
            try:
                return AgentCard.model_validate_json(resp.text)
            except Exception as exc:
                raise InvalidAgentResponseError(
                    f"Invalid AgentCard payload from {url}: {exc}"
                ) from exc
        except (TaskNotFoundError, InvalidAgentResponseError):
            raise
        except Exception as exc:
            raise InvalidAgentResponseError(
                f"Network error fetching agent card from {url}: {exc}"
            ) from exc
        finally:
            if should_close:
                await client.aclose()
