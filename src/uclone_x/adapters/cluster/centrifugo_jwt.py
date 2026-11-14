"""Centrifugo channel tokens (HS256), through PyJWT (MIT).

Signing, encoding and signature/expiry verification are PyJWT's. This module only fixes the
claims Centrifugo reads: `sub`, `exp`, `channel` (a subscription token) and `channels` (the
same channel as a server-side subscription when the token is used to open a `uni_sse`
connection).

`jwt` is imported lazily, so `uclone_x` imports without the `cluster` extra installed.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping
from typing import Any, Final

from uclone_x.errors import UCloneXError

_ALLOWED_CHANNEL_PREFIXES: Final[tuple[str, str]] = ("ucx:session:", "ucx:room:")
_ALGORITHM: Final[str] = "HS256"

__all__ = [
    "CentrifugoTokenError",
    "CentrifugoTokenSigner",
]


class CentrifugoTokenError(UCloneXError):
    """Raised when a Centrifugo channel token is malformed, tampered, expired, or mis-scoped."""


def _jwt() -> Any:
    return importlib.import_module("jwt")


class CentrifugoTokenSigner:
    """Mints and verifies short-lived HS256 channel tokens for Centrifugo."""

    def __init__(self, secret_key: str | bytes, ttl_seconds: int = 300) -> None:
        raw_secret = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
        if not raw_secret:
            raise ValueError("secret_key must not be empty")
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds}")
        self._secret = raw_secret
        self._ttl_seconds = ttl_seconds

    @property
    def ttl_seconds(self) -> int:
        """Configured token lifetime in seconds."""
        return self._ttl_seconds

    def mint_channel_token(
        self,
        user_id: str,
        channel: str,
        now_epoch: int | None = None,
    ) -> str:
        """Mint a channel-scoped HS256 JWT for `user_id` and `channel`."""
        clean_user = user_id.strip()
        if not clean_user:
            raise ValueError("user_id must not be blank")
        if not channel.startswith(_ALLOWED_CHANNEL_PREFIXES):
            raise ValueError(
                f"channel must start with 'ucx:session:' or 'ucx:room:', got {channel!r}"
            )
        issued_at = int(time.time()) if now_epoch is None else int(now_epoch)
        claims = {
            "sub": clean_user,
            "channel": channel,
            "channels": [channel],
            "iat": issued_at,
            "exp": issued_at + self._ttl_seconds,
        }
        token: str = _jwt().encode(claims, self._secret, algorithm=_ALGORITHM)
        return token

    def verify_channel_token(
        self,
        token: str,
        expected_channel: str,
        now_epoch: int | None = None,
    ) -> Mapping[str, Any]:
        """Verify signature, expiry and channel scope; `now_epoch` pins the clock for tests."""
        jwt_mod = _jwt()
        try:
            claims: dict[str, Any] = jwt_mod.decode(
                token,
                self._secret,
                algorithms=[_ALGORITHM],
                options={"require": ["exp", "sub", "channel"], "verify_exp": now_epoch is None},
            )
        except jwt_mod.ExpiredSignatureError as exc:
            raise CentrifugoTokenError("Centrifugo channel token has expired") from exc
        except jwt_mod.InvalidSignatureError as exc:
            raise CentrifugoTokenError("Token HMAC-SHA256 signature verification failed") from exc
        except jwt_mod.InvalidTokenError as exc:
            raise CentrifugoTokenError(f"Centrifugo channel token is invalid: {exc}") from exc

        if now_epoch is not None and int(claims["exp"]) <= int(now_epoch):
            raise CentrifugoTokenError("Centrifugo channel token has expired")
        if claims.get("channel") != expected_channel:
            raise CentrifugoTokenError(
                f"Token channel claim {claims.get('channel')!r} does not match expected "
                f"{expected_channel!r}"
            )
        return claims
