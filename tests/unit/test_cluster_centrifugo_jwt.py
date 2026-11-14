"""Unit tests for `CentrifugoTokenSigner` HS256 JWT minting and channel-scoped verification."""

from __future__ import annotations

import pytest

from uclone_x.adapters.cluster.centrifugo_jwt import (
    CentrifugoTokenError,
    CentrifugoTokenSigner,
)

# 32 bytes: PyJWT warns below the HS256 key length RFC 7518 §3.2 recommends.
_SECRET = "test-only-hmac-secret-32-bytes!!"


def test_centrifugo_token_signer_mints_and_rejects_signature_tamper() -> None:
    """`verify_channel_token` validates HS256 signatures and rejects tampered payloads.

    Killed by: src/uclone_x/adapters/cluster/centrifugo_jwt.py :: "verify_exp": now_epoch is None},
    Becomes: "verify_exp": now_epoch is None, "verify_signature": False},
    """
    signer = CentrifugoTokenSigner(_SECRET, ttl_seconds=120)
    token = signer.mint_channel_token("user-42", "ucx:session:sess-1", now_epoch=1000)
    claims = signer.verify_channel_token(token, "ucx:session:sess-1", now_epoch=1050)
    assert claims["sub"] == "user-42"
    assert claims["channel"] == "ucx:session:sess-1"
    assert claims["iat"] == 1000
    assert claims["exp"] == 1120

    # Flip one character in the middle of the signature segment (the last one also carries
    # base64 padding bits, so changing it can fail decoding instead of verification).
    header_b64, payload_b64, sig_b64 = token.split(".")
    mid = len(sig_b64) // 2
    tampered_sig = sig_b64[:mid] + ("A" if sig_b64[mid] != "A" else "B") + sig_b64[mid + 1 :]
    tampered_token = f"{header_b64}.{payload_b64}.{tampered_sig}"
    with pytest.raises(CentrifugoTokenError, match="signature"):
        signer.verify_channel_token(tampered_token, "ucx:session:sess-1", now_epoch=1050)


def test_centrifugo_token_signer_rejects_expired_tokens() -> None:
    """`verify_channel_token` rejects tokens once `now_epoch >= exp`.

    Killed by: src/uclone_x/adapters/cluster/centrifugo_jwt.py :: if now_epoch is not None and int(claims["exp"]) <= int(now_epoch):
    Becomes: if False:
    """
    signer = CentrifugoTokenSigner(_SECRET, ttl_seconds=60)
    token = signer.mint_channel_token("user-42", "ucx:room:room-9", now_epoch=2000)

    # At now_epoch=2059 (before exp=2060), token is valid.
    claims = signer.verify_channel_token(token, "ucx:room:room-9", now_epoch=2059)
    assert claims["exp"] == 2060

    # At now_epoch=2060 (at exp), token is rejected as expired.
    with pytest.raises(CentrifugoTokenError, match="expired"):
        signer.verify_channel_token(token, "ucx:room:room-9", now_epoch=2060)


def test_centrifugo_token_signer_rejects_cross_channel_replay() -> None:
    """A valid token minted for `ucx:session:a` cannot authorize `ucx:session:b`.

    Killed by: src/uclone_x/adapters/cluster/centrifugo_jwt.py :: if claims.get("channel") != expected_channel:
    Becomes: if False:
    """
    signer = CentrifugoTokenSigner(_SECRET, ttl_seconds=300)
    token_a = signer.mint_channel_token("user-42", "ucx:session:a", now_epoch=3000)

    with pytest.raises(CentrifugoTokenError, match="channel"):
        signer.verify_channel_token(token_a, "ucx:session:b", now_epoch=3010)
