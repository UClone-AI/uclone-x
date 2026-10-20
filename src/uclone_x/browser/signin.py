"""Telling that a page wants a sign-in (design `browser-agent.md` §3.5 "Detect").

The observation of a page carries `sign_in_likely` when the page has a password or
one-time-code field, when the page itself answered 401 or 403, or when it is an identity
provider's sign-in host. The value is a sentence naming the site and the next call, so a
small model needs no rule from the tool description to act on it. That call is `sign_in`
(path A), which hands the person (path B) whatever it cannot finish.
"""

from __future__ import annotations

import urllib.parse
from typing import Final

#: Identity providers' sign-in hosts (§3.5). A subdomain counts; a lookalike does not.
IDENTITY_HOSTS: Final = frozenset(
    {
        "accounts.google.com",
        "nid.naver.com",
        "accounts.kakao.com",
        "login.microsoftonline.com",
        "login.live.com",
        "appleid.apple.com",
    }
)

_REFUSED: Final = frozenset({401, 403})

#: The page's location, title, its own HTTP status and whether a visible field is one
#: §3.5 keeps from the clone. One evaluation, run with every observation.
PAGE_PROBE_JS: Final = """[
  location.href,
  document.title,
  (performance.getEntriesByType('navigation')[0] || {}).responseStatus || 0,
  Array.from(document.querySelectorAll(
    'input[type=password], input[autocomplete~="one-time-code"]'
  )).some(function (e) { return !e.disabled && e.getClientRects().length > 0; })
]"""


def site_of(url: str) -> str:
    """The host a person would recognise, or the address itself when it has none."""
    return urllib.parse.urlsplit(url).hostname or url


def _identity_host(host: str) -> bool:
    return any(host == h or host.endswith("." + h) for h in IDENTITY_HOSTS)


def sign_in_signal(url: str, status: int, *, secret_field: bool) -> str | None:
    """The `sign_in_likely` sentence for a page, or `None` when nothing asks for a sign-in."""
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if not (secret_field or status in _REFUSED or (host and _identity_host(host))):
        return None
    return (
        f"{site_of(url)} is asking for a sign-in. Call sign_in: it uses the person's saved "
        "sign-in, and asks the person to finish whatever it cannot."
    )


# -- path A: the Core fills a saved sign-in (§3.5) ------------------------------------------

#: Finds the field `sign_in` fills for one role, evaluated in the page; the element or null.
#: Autocomplete hints first, then the field's label or name, then where it sits. Only
#: visible, enabled, editable fields count; a `new-password` field (a sign-up form) never does.
_FIELD_JS: Final = """(function (role) {
  const usable = (e) => !e.disabled && !e.readOnly && e.getClientRects().length > 0;
  const inputs = Array.from(document.querySelectorAll('input')).filter(usable);
  const ac = (e) => (e.getAttribute('autocomplete') || '').toLowerCase();
  const said = (e) => [
    (e.labels && e.labels[0] && e.labels[0].textContent) || '',
    e.getAttribute('aria-label') || '', e.placeholder || '', e.name || '', e.id || '',
  ].join(' ');
  const textual = (e) => ['text', 'email', 'tel', ''].includes(e.type);
  const isCode = (e) => /\\bone-time-code\\b/.test(ac(e)) ||
    (textual(e) && /\\b(otp|code|2fa|totp|verification)\\b|인증|코드/i.test(said(e)));
  const passwords = inputs.filter((e) => e.type === 'password' && !/new-password/.test(ac(e)));
  if (role === 'password') {
    return passwords.find((e) => /current-password/.test(ac(e))) || passwords[0] || null;
  }
  if (role === 'code') return inputs.find(isCode) || null;
  const users = inputs.filter((e) => textual(e) && !isCode(e));
  return (
    users.find((e) => /\\b(username|email)\\b/.test(ac(e))) ||
    users.find((e) => e.type === 'email') ||
    users.find((e) => /user|e-?mail|login|아이디|이메일|계정/i.test(said(e))) ||
    (passwords[0] && users.filter((e) =>
      e.compareDocumentPosition(passwords[0]) & Node.DOCUMENT_POSITION_FOLLOWING).pop()) ||
    null
  );
})"""

#: Whether the page shows a CAPTCHA, which is always the person's (§3.5 Known limits).
CAPTCHA_JS: Final = """Boolean(document.querySelector(
  '.g-recaptcha, .h-captcha, .cf-turnstile, [data-sitekey], ' +
  'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], iframe[src*="turnstile"]'))"""


def field_expression(role: str) -> str:
    """The page expression that finds `role`'s field: `user`, `password` or `code`."""
    return f"{_FIELD_JS}({role!r})"


_LOOPBACK: Final = frozenset({"127.0.0.1", "localhost", "::1"})


def insecure_for_secrets(url: str) -> bool:
    """Whether a saved password must not be typed into this page: anything but HTTPS,
    loopback excepted (a local app, and this repository's fixtures)."""
    parsed = urllib.parse.urlsplit(url)
    if (parsed.hostname or "") in _LOOPBACK:
        return False
    return parsed.scheme != "https"


def shown_username(username: str) -> str:
    """The username as the outcome names it: enough to tell accounts apart, not the address."""
    if "@" in username:
        return username.split("@", 1)[0] + "@…"
    return username if len(username) <= 3 else username[:3] + "…"
