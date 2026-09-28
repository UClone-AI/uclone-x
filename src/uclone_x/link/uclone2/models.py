"""Records of a uClone2 link, and the server payloads they are built from.

Two kinds of model live here. The server's payloads (`RemoteClone`, `ConnectResult`,
`LinkSelf`) mirror the contract's schemas and ignore fields they do not know, because the
contract adds optional fields without a protocol bump. `LinkRecord` is ours: what the store
keeps for one link.

The token is a `SecretStr` everywhere it appears, so `repr`, `str` and a default
`model_dump` print `**********` rather than the credential. Only the store and the client
read the raw value, and each says so where it does.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

__all__ = [
    "LinkRecord",
    "LinkSelf",
    "LinkView",
    "ConnectResult",
    "RemoteClone",
    "RemotePersona",
    "mask_token",
]

#: Every linked-runtime token starts with this; the rest is the secret.
TOKEN_PREFIX = "ucl_"


def mask_token(token: SecretStr) -> str:
    """`ucl_…` and the last four characters: enough to tell two links apart, not to use one."""
    raw = token.get_secret_value()
    tail = raw[-4:] if len(raw) > len(TOKEN_PREFIX) + 8 else ""
    return f"{TOKEN_PREFIX}…{tail}"


class _Payload(BaseModel):
    """A server payload: frozen, and tolerant of fields a newer server adds."""

    model_config = ConfigDict(frozen=True, extra="ignore")


class RemotePersona(_Payload):
    """`clone.persona` as uClone2 sends it; either part may be empty."""

    text: str
    identity_matrix: dict[str, object] = Field(default_factory=dict[str, object])


class RemoteClone(_Payload):
    """The uClone2 clone a link speaks for."""

    id: str
    username: str
    display_name: str
    avatar_url: str
    persona: RemotePersona


class ConnectResult(_Payload):
    """What `POST /linked/connect` returns: the only time the server hands out the token."""

    token: SecretStr
    bot_id: str
    ws_url: str
    protocol: Literal[1]
    clone: RemoteClone


class LinkSelf(_Payload):
    """What `GET /linked/self` returns: the link as the server sees it."""

    bot_id: str
    clone: RemoteClone
    clone_updated_at: datetime
    pending: int = Field(ge=0)
    online: bool


class LinkRecord(BaseModel):
    """One link, as the store keeps it.

    `unlink_pending` is the *해제 대기* state: the user asked to unlink, uClone2 could not be
    reached, and the `DELETE` is retried on the next start instead of the record being
    dropped (which would leave the clone linked and offline in uClone2).

    `enabled` goes false when uClone2 says the link is over (a revoked token), so the
    section can say so; the record is not deleted behind the user's back.

    `paused` is the user's own *오프라인으로 전환*: the link stays, no session runs for it,
    and uClone2 shows the clone offline until the user switches it back on. It is kept
    apart from `enabled` because the two read differently to the user and only this one is
    theirs to undo.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    link_id: str
    server_url: str
    bot_id: str
    remote_username: str
    remote_display_name: str = ""
    remote_avatar_url: str = ""
    local_agent_id: str
    token: SecretStr
    ws_url: str
    created_at: datetime
    last_connected_at: datetime | None = None
    #: The server's `clone_updated_at` when the profile was last read; `None` before the
    #: first `GET /linked/self`.
    clone_updated_at: datetime | None = None
    enabled: bool = True
    paused: bool = False
    unlink_pending: bool = False

    def view(self) -> LinkView:
        """This record with the token masked: what any head may show."""
        return LinkView(
            link_id=self.link_id,
            server_url=self.server_url,
            bot_id=self.bot_id,
            remote_username=self.remote_username,
            remote_display_name=self.remote_display_name,
            remote_avatar_url=self.remote_avatar_url,
            local_agent_id=self.local_agent_id,
            token_hint=mask_token(self.token),
            created_at=self.created_at,
            last_connected_at=self.last_connected_at,
            enabled=self.enabled,
            paused=self.paused,
            unlink_pending=self.unlink_pending,
        )


class LinkView(BaseModel):
    """A `LinkRecord` without its credential. It has no token field to leak."""

    model_config = ConfigDict(frozen=True)

    link_id: str
    server_url: str
    bot_id: str
    remote_username: str
    remote_display_name: str
    remote_avatar_url: str
    local_agent_id: str
    token_hint: str
    created_at: datetime
    last_connected_at: datetime | None
    enabled: bool
    paused: bool
    unlink_pending: bool
