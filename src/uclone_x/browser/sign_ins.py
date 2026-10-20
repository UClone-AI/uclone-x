"""Saved sign-ins (design `browser-agent.md` §3.5 path A, step 1).

U0 saves a site, a username, a password and optionally a 2FA setup key. They live in the
settings file under `browser_sign_ins`, beside the API keys (settings-single-source: no
second store, the OS keychain included). This module is pure over the file's contents: the
one settings writer applies `with_sign_in` and `without_sign_in` under its lock, and the
browser service is handed `read_sign_ins` of what it reads.

Nothing here ever puts a password or a setup key into a message, a `repr` or a row a head
is shown: `public_rows` is what Settings lists.
"""

from __future__ import annotations

import logging
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

from uclone_x.browser.totp import InvalidSetupKeyError, normalize_setup_key

logger = logging.getLogger(__name__)

SIGN_INS_KEY: Final = "browser_sign_ins"


class InvalidSignInError(ValueError):
    """A sign-in that cannot be saved; the message names the field, never its value.

    `code` is what a head words the refusal from: `site_missing`, `username_missing`,
    `password_missing` or `setup_key_invalid`.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SavedSignIn:
    """One saved sign-in. The secrets are left out of `repr`, so no log line can carry them."""

    site: str
    username: str
    password: str = field(repr=False)
    totp_key: str | None = field(default=None, repr=False)


_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def site_key(site_or_url: str) -> str:
    """The host a sign-in is saved under: lower case, from a bare host or a whole address.

    On loopback the port is part of it: every local app shares `127.0.0.1`, and one app's
    saved sign-in is not another's (#2388).
    """
    text = site_or_url.strip()
    if not text:
        return ""
    parsed = urllib.parse.urlsplit(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if host in _LOOPBACK and port is not None:
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    return host


def read_sign_ins(data: Mapping[str, Any]) -> list[SavedSignIn]:
    """The saved sign-ins in the file's contents. A row missing a field is not offered."""
    raw = data.get(SIGN_INS_KEY)
    rows = cast(list[object], raw) if isinstance(raw, list) else []
    saved: list[SavedSignIn] = []
    for row in rows:
        if not isinstance(row, Mapping):
            logger.warning("A saved browser sign-in is not a record; it is not used")
            continue
        entry = cast(Mapping[str, Any], row)
        site, username, password = (entry.get(k) for k in ("site", "username", "password"))
        totp_key = entry.get("totp_key")
        if not (isinstance(site, str) and isinstance(username, str) and isinstance(password, str)):
            logger.warning("A saved browser sign-in is missing a field; it is not used")
            continue
        if not (site and username and password):
            logger.warning("A saved browser sign-in has an empty field; it is not used")
            continue
        saved.append(
            SavedSignIn(site, username, password, totp_key if isinstance(totp_key, str) else None)
        )
    return saved


def _rows(saved: Sequence[SavedSignIn]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in saved:
        row = {"site": s.site, "username": s.username, "password": s.password}
        if s.totp_key:
            row["totp_key"] = s.totp_key
        rows.append(row)
    return rows


def with_sign_in(
    data: Mapping[str, Any],
    site: str,
    username: str,
    password: str,
    *,
    totp_key: str | None = None,
    clear_totp_key: bool = False,
) -> dict[str, Any]:
    """The update that saves one sign-in, replacing the one saved for the same site and user.

    An empty `password` keeps the saved one (an edit that changes only the setup key); a new
    sign-in needs one. An empty `totp_key` keeps the saved key too: an edit that changes the
    password must not drop 2FA. Only `clear_totp_key` removes it.
    """
    host = site_key(site)
    if not host:
        raise InvalidSignInError(
            "Give the site the sign-in is for, such as example.com.", "site_missing"
        )
    user = username.strip()
    if not user:
        raise InvalidSignInError(
            "Give the username or email the site signs in with.", "username_missing"
        )
    saved = read_sign_ins(data)
    old = next((s for s in saved if s.site == host and s.username == user), None)
    password = password or (old.password if old else "")
    if not password:
        raise InvalidSignInError("Give the password for this sign-in.", "password_missing")
    key = old.totp_key if old and not clear_totp_key else None
    if totp_key and totp_key.strip():
        try:
            key = normalize_setup_key(totp_key)
        except InvalidSetupKeyError as exc:
            raise InvalidSignInError(str(exc), "setup_key_invalid") from None
    entry = SavedSignIn(host, user, password, key)
    kept = [entry if s is old else s for s in saved] if old else [*saved, entry]
    return {SIGN_INS_KEY: _rows(kept)}


def without_sign_in(data: Mapping[str, Any], site: str, username: str) -> dict[str, Any]:
    """The update that deletes one sign-in. Deleting one not saved is not an error."""
    host, user = site_key(site), username.strip()
    kept = [s for s in read_sign_ins(data) if not (s.site == host and s.username == user)]
    return {SIGN_INS_KEY: _rows(kept)}


def sign_in_for(url: str, saved: Sequence[SavedSignIn]) -> SavedSignIn | None:
    """The sign-in saved for the page's own host, or `None`.

    Only the host itself counts, and its `www.` twin. A parent domain is not trusted: under
    shared hosting (`*.github.io`, `*.vercel.app`) a subdomain is someone else's site, and
    telling those apart needs the public-suffix list, which this does not carry.
    """
    host = site_key(url)
    if not host:
        return None
    hosts = {host, host.removeprefix("www.") if host.startswith("www.") else "www." + host}
    return next((s for s in saved if s.site in hosts), None)


def public_rows(saved: Sequence[SavedSignIn]) -> list[dict[str, Any]]:
    """What Settings lists: the site, the username and whether 2FA is saved. No secret."""
    return [{"site": s.site, "username": s.username, "has_2fa": bool(s.totp_key)} for s in saved]
