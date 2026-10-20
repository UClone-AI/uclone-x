"""Telling that a page wants a sign-in (design `browser-agent.md` §3.5 "Detect")."""

from __future__ import annotations

import pytest

from uclone_x.browser.signin import IDENTITY_HOSTS, insecure_for_secrets, sign_in_signal


def test_a_password_field_asks_for_a_sign_in_and_names_the_site() -> None:
    """Killed by: src/uclone_x/browser/signin.py :: if not (secret_field or status in _REFUSED
    Becomes: if not (status in _REFUSED
    """
    signal = sign_in_signal("https://shop.example.com/login?next=/cart", 200, secret_field=True)

    assert signal is not None
    assert "shop.example.com" in signal
    assert "sign_in" in signal


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_page_asks_for_a_sign_in(status: int) -> None:
    """Killed by: src/uclone_x/browser/signin.py :: _REFUSED: Final = frozenset({401, 403})
    Becomes: _REFUSED: Final = frozenset({401})
    """
    assert sign_in_signal("https://example.com/account", status, secret_field=False)


@pytest.mark.parametrize("host", sorted(IDENTITY_HOSTS))
def test_an_identity_provider_page_asks_for_a_sign_in(host: str) -> None:
    assert sign_in_signal(f"https://{host}/signin", 200, secret_field=False)


def test_a_subdomain_of_an_identity_host_counts_and_a_lookalike_does_not() -> None:
    """Killed by: src/uclone_x/browser/signin.py :: return any(host == h or host.endswith("." + h) for h in IDENTITY_HOSTS)
    Becomes: return any(h in host for h in IDENTITY_HOSTS)
    """
    assert sign_in_signal("https://eu.login.microsoftonline.com/x", 200, secret_field=False)
    assert (
        sign_in_signal("https://accounts.google.com.example.net/", 200, secret_field=False) is None
    )


@pytest.mark.parametrize(
    ("url", "status"),
    [
        ("https://example.com/", 200),
        ("https://example.com/missing", 404),
        ("https://example.com/broken", 500),
        ("about:blank", 0),
    ],
)
def test_an_ordinary_page_asks_for_nothing(url: str, status: int) -> None:
    assert sign_in_signal(url, status, secret_field=False) is None


def test_the_signal_is_plain_copy() -> None:
    signal = sign_in_signal("https://example.com/login", 200, secret_field=True)
    assert signal is not None
    for internal in ("401", "403", "password field", "autocomplete", "CDP", "sign_in_likely"):
        assert internal not in signal


@pytest.mark.parametrize(
    ("url", "insecure"),
    [
        ("http://shop.example.com/login", True),
        ("https://shop.example.com/login", False),
        ("http://127.0.0.1:8080/login", False),
        ("http://localhost:3000/login", False),
        ("http://[::1]/login", False),
        ("about:blank", True),
    ],
)
def test_a_password_goes_only_to_an_encrypted_or_loopback_page(url: str, insecure: bool) -> None:
    """Killed by: src/uclone_x/browser/signin.py :: return parsed.scheme != "https"
    Becomes: return False
    """
    assert insecure_for_secrets(url) is insecure
