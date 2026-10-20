"""Saved sign-ins (design `browser-agent.md` §3.5 path A, step 1): rows in the settings
file's `browser_sign_ins`, matched to a page by site, and shown to a head without the secret.

The functions here are pure over the file's contents; the one settings writer applies them.
"""

from __future__ import annotations

from typing import Any

import pytest

from uclone_x.browser.sign_ins import (
    SIGN_INS_KEY,
    InvalidSignInError,
    public_rows,
    read_sign_ins,
    sign_in_for,
    with_sign_in,
    without_sign_in,
)

SECRET = "sentinel-pw-3kQ9-never-shown"  # noqa: S105
SETUP_KEY = "JBSWY3DPEHPK3PXP"


def _saved(data: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """`data` after saving one sign-in, as the settings writer would merge it."""
    return {**data, **with_sign_in(data, *args, **kwargs)}


def test_a_saved_sign_in_is_read_back_under_its_site() -> None:
    data = _saved({}, "https://Shop.Example.com/login", "kenny@example.com", SECRET)

    assert list(data) == [SIGN_INS_KEY]
    assert [(s.site, s.username, s.password, s.totp_key) for s in read_sign_ins(data)] == [
        ("shop.example.com", "kenny@example.com", SECRET, None)
    ]


def test_saving_the_same_site_and_username_again_replaces_it() -> None:
    data = _saved({}, "shop.example.com", "kenny", "old")
    data = _saved(data, "shop.example.com", "kenny", SECRET, totp_key=SETUP_KEY)
    data = _saved(data, "shop.example.com", "other", "x")

    assert [(s.username, s.password, s.totp_key) for s in read_sign_ins(data)] == [
        ("kenny", SECRET, SETUP_KEY),
        ("other", "x", None),
    ]


def test_an_edit_with_no_password_keeps_the_saved_one() -> None:
    """Killed by: src/uclone_x/browser/sign_ins.py :: password = password or (old.password if old else "")
    Becomes: password = password
    """
    data = _saved({}, "shop.example.com", "kenny", SECRET, totp_key=SETUP_KEY)
    data = _saved(data, "shop.example.com", "kenny", "", totp_key=SETUP_KEY)

    assert read_sign_ins(data)[0].password == SECRET


def test_a_new_sign_in_needs_a_site_a_password_and_a_real_setup_key() -> None:
    with pytest.raises(InvalidSignInError, match="password"):
        with_sign_in({}, "shop.example.com", "kenny", "")
    with pytest.raises(InvalidSignInError, match="setup key"):
        with_sign_in({}, "shop.example.com", "kenny", SECRET, totp_key="not a key!")
    with pytest.raises(InvalidSignInError, match="site"):
        with_sign_in({}, "  ", "kenny", SECRET)
    with pytest.raises(InvalidSignInError, match="username"):
        with_sign_in({}, "shop.example.com", " ", SECRET)


def test_the_refusals_never_repeat_what_was_typed() -> None:
    for call in (
        lambda: with_sign_in({}, "shop.example.com", "kenny", SECRET, totp_key="bad key " + SECRET),
    ):
        with pytest.raises(InvalidSignInError) as refused:
            call()
        assert SECRET not in str(refused.value)


def test_delete_removes_one_sign_in() -> None:
    data = _saved({}, "shop.example.com", "kenny", SECRET)
    data = _saved(data, "shop.example.com", "other", SECRET)

    data = {**data, **without_sign_in(data, "shop.example.com", "kenny")}
    data = {**data, **without_sign_in(data, "nowhere.example", "nobody")}

    assert [s.username for s in read_sign_ins(data)] == ["other"]


def test_a_page_finds_only_the_sign_in_saved_for_its_own_host() -> None:
    """A parent domain is not trusted: on shared hosting (`*.github.io`) it is another
    person's site. `www.` is the one variant that counts as the same host.

    Killed by: src/uclone_x/browser/sign_ins.py :: return next((s for s in saved if s.site in hosts), None)
    Becomes: return next((s for s in saved if any(h.endswith(s.site) for h in hosts)), None)
    """
    data = _saved({}, "example.com", "parent", SECRET)
    data = _saved(data, "shop.example.com", "child", SECRET)
    data = _saved(data, "github.io", "pages", SECRET)
    saved = read_sign_ins(data)

    def user(url: str) -> str | None:
        found = sign_in_for(url, saved)
        return found.username if found else None

    assert user("https://shop.example.com/login") == "child"
    assert user("https://www.example.com/") == "parent"
    assert user("https://example.com/") == "parent"
    assert user("https://accounts.example.com/") is None
    assert user("https://someone-else.github.io/") is None
    assert user("https://example.com.attacker.test/") is None
    assert user("https://notexample.com/") is None


def test_an_edit_with_no_setup_key_keeps_the_saved_one_until_it_is_removed() -> None:
    """Killed by: src/uclone_x/browser/sign_ins.py :: key = old.totp_key if old and not clear_totp_key else None
    Becomes: key = None
    """
    data = _saved({}, "shop.example.com", "kenny", SECRET, totp_key=SETUP_KEY)
    data = _saved(data, "shop.example.com", "kenny", "new-password")
    assert read_sign_ins(data)[0].totp_key == SETUP_KEY
    assert read_sign_ins(data)[0].password == "new-password"

    data = _saved(data, "shop.example.com", "kenny", "", clear_totp_key=True)
    assert read_sign_ins(data)[0].totp_key is None
    assert read_sign_ins(data)[0].password == "new-password"


def test_what_a_head_is_shown_carries_no_secret() -> None:
    saved = read_sign_ins(_saved({}, "shop.example.com", "kenny", SECRET, totp_key=SETUP_KEY))

    assert public_rows(saved) == [
        {"site": "shop.example.com", "username": "kenny", "has_2fa": True}
    ]
    assert SECRET not in repr(saved[0]) and SETUP_KEY not in repr(saved[0])
    assert SECRET not in str(saved[0])


def test_a_row_missing_its_password_is_not_offered_to_sign_in_with() -> None:
    data = {
        SIGN_INS_KEY: [
            {"site": "a.example"},
            "junk",
            {"site": "b.example", "username": "u", "password": "p"},
        ]
    }
    assert [s.site for s in read_sign_ins(data)] == ["b.example"]


def test_on_loopback_the_port_tells_two_local_apps_apart() -> None:
    """Killed by: src/uclone_x/browser/sign_ins.py :: if host in _LOOPBACK and port is not None:
    Becomes: if False:
    """
    data = _saved({}, "http://127.0.0.1:8080/login", "app-one", SECRET)
    saved = read_sign_ins(data)

    assert saved[0].site == "127.0.0.1:8080"
    found = sign_in_for("http://127.0.0.1:8080/account", saved)
    assert found is not None and found.username == "app-one"
    assert sign_in_for("http://127.0.0.1:9090/login", saved) is None
    assert sign_in_for("http://127.0.0.1/login", saved) is None
    # Off loopback the port is not part of the site, as before.
    data = _saved({}, "https://shop.example.com:8443/login", "kenny", SECRET)
    assert read_sign_ins(data)[0].site == "shop.example.com"
