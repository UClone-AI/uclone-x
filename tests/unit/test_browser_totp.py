"""One-time codes from a saved 2FA setup key (design `browser-agent.md` §3.5 path A)."""

from __future__ import annotations

import pytest

from uclone_x.browser.totp import InvalidSetupKeyError, normalize_setup_key, totp

# RFC 6238 Appendix B: the SHA-1 secret is the ASCII "12345678901234567890".
RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"


@pytest.mark.parametrize(
    ("at", "code"),
    [
        (59, "94287082"),
        (1111111109, "07081804"),
        (1111111111, "14050471"),
        (1234567890, "89005924"),
        (2000000000, "69279037"),
        (20000000000, "65353130"),
    ],
)
def test_the_rfc_6238_vectors(at: int, code: str) -> None:
    """Killed by: src/uclone_x/browser/totp.py :: counter = int(at // period)
    Becomes: counter = int(at // period) + 1
    """
    assert totp(RFC_SECRET, at=at, digits=8) == code


def test_six_digits_is_the_default_and_keeps_leading_zeros() -> None:
    """Killed by: src/uclone_x/browser/totp.py :: return str(value % 10**digits).zfill(digits)
    Becomes: return str(value % 10**digits)
    """
    assert totp(RFC_SECRET, at=1111111109) == "081804"


def test_a_setup_key_as_people_copy_it_is_accepted() -> None:
    """Spaces, lower case and missing padding are how authenticator pages show the key."""
    spaced = "gezd gnbv gy3t qojq gezd gnbv gy3t qojq"
    assert normalize_setup_key(spaced) == RFC_SECRET
    assert totp(spaced, at=59, digits=8) == "94287082"
    assert normalize_setup_key("JBSWY3DPEHPK3PX") == "JBSWY3DPEHPK3PX"


@pytest.mark.parametrize("key", ["", "   ", "not base32!", "1111"])
def test_a_key_that_is_not_a_setup_key_is_refused_in_plain_words(key: str) -> None:
    with pytest.raises(InvalidSetupKeyError) as refused:
        normalize_setup_key(key)
    message = str(refused.value)
    assert "setup key" in message
    assert key.strip() == "" or key not in message
