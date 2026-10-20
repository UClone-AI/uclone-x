"""One-time codes from a saved 2FA setup key (RFC 6238), for `sign_in` (§3.5 path A).

Standard library only: HMAC-SHA1 over the 30-second counter, the six digits authenticator
apps show. The key is the base32 "setup key" a site shows beside its QR code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import struct
import time
from typing import Final

PERIOD_S: Final = 30
DIGITS: Final = 6


class InvalidSetupKeyError(ValueError):
    """The text is not a 2FA setup key; the message says so without repeating it."""


def normalize_setup_key(key: str) -> str:
    """The key as base32 without spaces, upper case, unpadded; or the refusal."""
    clean = "".join(key.split()).upper().rstrip("=")
    if len(clean) < 8:
        raise InvalidSetupKeyError(
            "That is not a 2FA setup key. Copy the setup key the site shows beside its QR code."
        )
    try:
        base64.b32decode(clean + "=" * (-len(clean) % 8))
    except (binascii.Error, ValueError):
        raise InvalidSetupKeyError(
            "That is not a 2FA setup key: it has characters a setup key never has. Copy the "
            "setup key the site shows beside its QR code."
        ) from None
    return clean


def totp(key: str, *, at: float | None = None, digits: int = DIGITS, period: int = PERIOD_S) -> str:
    """The code for `key` at time `at` (now when omitted)."""
    clean = normalize_setup_key(key)
    secret = base64.b32decode(clean + "=" * (-len(clean) % 8))
    at = time.time() if at is None else at
    counter = int(at // period)
    digest = hmac.new(secret, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(value % 10**digits).zfill(digits)
