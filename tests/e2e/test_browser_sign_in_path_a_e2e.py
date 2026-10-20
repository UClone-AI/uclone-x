"""`sign_in` with a saved sign-in (design `browser-agent.md` §3.5 path A, §6 step 4b).

Real Chromium over R2 against a loopback site whose forms post to a handler that checks
what the Core typed: one-step and two-step forms, a 2FA code from a setup key, a CAPTCHA
and a rejected password. Whatever path A cannot finish goes to the person (path B), which
the test plays by giving the tab back.

The saved password is a sentinel: it must reach the site and nothing the clone reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from tests.support.chromium import chromium_executable
from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.sign_ins import SavedSignIn, site_key
from uclone_x.browser.totp import totp

pytestmark = pytest.mark.e2e

USER = "kenny@example.com"
SECRET = "sentinel-pw-8Zr4-never-in-a-result"  # noqa: S105
SETUP_KEY = "JBSWY3DPEHPK3PXP"
KEY = ("conversation-1", "scout")

_HEAD = '<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head><body>'
_PAGES: dict[str, str] = {
    "/one": """<h1>Sign in</h1><form method="post" action="/one">
<label for="u">Email</label><input id="u" name="u" type="email" autocomplete="username">
<label for="p">Password</label><input id="p" name="p" type="password" autocomplete="current-password">
<button>Sign in</button></form>""",
    "/two": """<h1>Sign in</h1><form method="post" action="/two">
<label for="u">아이디</label><input id="u" name="u">
<button>다음</button></form>""",
    "/two-password": """<h1>Enter your password</h1><form method="post" action="/two-password">
<p>{user}</p><label for="p">비밀번호</label><input id="p" name="p" type="password">
<button>로그인</button></form>""",
    "/2fa": """<h1>Sign in</h1><form method="post" action="/2fa">
<label for="u">Username</label><input id="u" name="u" autocomplete="username">
<label for="p">Password</label><input id="p" name="p" type="password">
<button>Sign in</button></form>""",
    "/2fa-code": """<h1>Two-step verification</h1><form method="post" action="/2fa-code">
<label for="c">Code</label><input id="c" name="c" inputmode="numeric" autocomplete="one-time-code">
<button>Verify</button></form>""",
    "/captcha": """<h1>Sign in</h1><form method="post" action="/captcha">
<label for="u">Username</label><input id="u" name="u" autocomplete="username">
<label for="p">Password</label><input id="p" name="p" type="password">
<div class="g-recaptcha" data-sitekey="fixture"></div>
<button>Sign in</button></form>""",
    "/hop": """<h1>Sign in</h1><form method="post" action="/hop">
<label for="u">Email</label><input id="u" name="u" type="email" autocomplete="username">
<button>Next</button></form>""",
    "/alert": """<h1>Sign in</h1><form method="post" action="/one" onsubmit="alert('Checking…')">
<label for="u">Email</label><input id="u" name="u" type="email" autocomplete="username">
<label for="p">Password</label><input id="p" name="p" type="password">
<button>Sign in</button></form>""",
    "/home": "<h1>Welcome back</h1><p>Your orders</p>",
}


@pytest.fixture(scope="module")
def site() -> Iterator[tuple[str, list[dict[str, str]]]]:
    posted: list[dict[str, str]] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _page(self, path: str, status: int = 200, **fill: str) -> None:
            body = (_HEAD.format(title=path) + _PAGES[path].format(**fill)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if path == "/home":
                self.send_header("Set-Cookie", "session=signed-in; Path=/; HttpOnly")
            self.end_headers()
            self.wfile.write(body)

        def _go(self, location: str) -> None:
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            user = urllib.parse.parse_qs(query).get("u", [""])[0]
            self._page(path, user=user) if path in _PAGES else self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            form = {
                k: v[-1] for k, v in urllib.parse.parse_qs(self.rfile.read(length).decode()).items()
            }
            posted.append({"path": self.path, **form})
            right = form.get("p") == SECRET
            if self.path == "/one":
                self._go("/home") if right and form.get("u") == USER else self._page("/one", 401)
            elif self.path == "/hop":
                # The next step is on another host: the saved password must not follow.
                port = cast(tuple[str, int], self.server.server_address)[1]
                self._go(f"http://localhost:{port}/two-password")
            elif self.path == "/two":
                self._go("/two-password?u=" + urllib.parse.quote(form.get("u", "")))
            elif self.path == "/two-password":
                self._go("/home") if right else self._page("/two-password", 401, user="")
            elif self.path == "/2fa":
                self._go("/2fa-code") if right else self._page("/2fa", 401)
            elif self.path == "/2fa-code":
                codes = {totp(SETUP_KEY, at=__import__("time").time() + d) for d in (-30, 0, 30)}
                self._go("/home") if form.get("c") in codes else self._page("/2fa-code", 401)
            elif self.path == "/captcha":
                self._page("/captcha", 403)
            else:
                self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", posted
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium() -> Path:
    return chromium_executable()


@pytest.fixture
async def link(tmp_path: Path, chromium: Path) -> AsyncIterator[R2Link]:
    started = await R2Link.start(
        tmp_path / "profile", chrome=chromium, extra_args=("--headless=new",)
    )
    yield started
    await started.shutdown()


def _service(link: R2Link, saved: list[SavedSignIn]) -> BrowserService:
    async def _factory() -> R2Link:
        return link

    return BrowserService(_factory, sign_ins=lambda: saved)


def _saved(base: str, totp_key: str | None = None) -> list[SavedSignIn]:
    """The sign-in saved for the fixture site: on loopback its port is part of the site."""
    return [SavedSignIn(site_key(base), USER, SECRET, totp_key)]


async def _give_back_when_asked(service: BrowserService) -> dict[str, Any]:
    """Play the person: wait for the hand-off, then give the tab back. Returns the ask."""
    for _ in range(200):
        asked = service.live_state(KEY[0]).get("asked_user")
        if asked:
            service.give_back(*KEY)
            return dict(asked)
        await asyncio.sleep(0.05)
    raise AssertionError("the clone never asked the person")


def _nothing_carries_the_secret(*surfaces: object) -> None:
    for surface in surfaces:
        text = surface if isinstance(surface, str) else json.dumps(surface, ensure_ascii=False)
        assert SECRET not in text
        assert SETUP_KEY not in text


@pytest.mark.parametrize("page", ["/one", "/two", "/2fa"])
async def test_a_saved_sign_in_signs_in_and_the_secret_reaches_only_the_site(
    link: R2Link,
    site: tuple[str, list[dict[str, str]]],
    page: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    base, posted = site
    service = _service(link, _saved(base, SETUP_KEY))
    notices = service.watch(KEY[0])
    await service.open(KEY, base + page)

    result = await service.sign_in(KEY)

    assert result["url"] == base + "/home"
    assert "sign_in_likely" not in result
    assert result["message"] == f"Signed in to {site_key(base)} as kenny@…"
    assert any(p.get("p") == SECRET for p in posted), "the password never reached the site"
    heard = [notices.get_nowait() for _ in range(notices.qsize())]
    _nothing_carries_the_secret(result, heard, service.live_state(KEY[0]), caplog.text)


async def test_with_no_saved_sign_in_the_person_is_asked(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    """A person who gives the tab back still on the sign-in page is not offered a save.

    Killed by: src/uclone_x/browser/service.py :: and "sign_in_likely" not in snap
    Becomes: and True
    """
    base, _ = site
    service = _service(link, [])
    await service.open(KEY, base + "/one")

    signing_in = asyncio.create_task(service.sign_in(KEY))
    asked = await _give_back_when_asked(service)
    result = await asyncio.wait_for(signing_in, 10)

    assert asked["site"] == "127.0.0.1"
    assert result["message"].startswith("No sign-in is saved for 127.0.0.1")
    assert "save_offer" not in service.live_state(KEY[0])
    assert "gave the tab back" in result["message"]


@pytest.mark.parametrize(
    ("page", "reason"),
    [
        ("/captcha", "CAPTCHA"),
        ("/2fa", "code"),
    ],
)
async def test_what_path_a_cannot_finish_goes_to_the_person(
    link: R2Link,
    site: tuple[str, list[dict[str, str]]],
    page: str,
    reason: str,
) -> None:
    base, _ = site
    service = _service(link, _saved(base))
    await service.open(KEY, base + page)

    signing_in = asyncio.create_task(service.sign_in(KEY))
    asked = await _give_back_when_asked(service)
    result = await asyncio.wait_for(signing_in, 10)

    assert reason in asked["message"]
    assert reason in result["message"]
    _nothing_carries_the_secret(result, asked)


async def test_a_rejected_password_is_said_and_not_tried_again_on_that_page(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    base, posted = site
    service = _service(link, [SavedSignIn(site_key(base), USER, "wrong-" + SECRET)])
    await service.open(KEY, base + "/one")
    before = len(posted)

    first = asyncio.create_task(service.sign_in(KEY))
    await _give_back_when_asked(service)
    result = await asyncio.wait_for(first, 10)
    assert "did not accept the saved sign-in" in result["message"]

    second = asyncio.create_task(service.sign_in(KEY))
    await _give_back_when_asked(service)
    again = await asyncio.wait_for(second, 10)
    assert "already tried" in again["message"]
    assert len(posted) == before + 1, "the saved sign-in was submitted twice on one page"
    _nothing_carries_the_secret(result, again)


async def test_a_site_the_clone_signed_in_to_is_listed_and_signing_out_clears_it(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    """Killed by: src/uclone_x/browser/service.py :: if await self._cookies(self._link, target, host):
    Becomes: if True:
    """
    base, _ = site
    service = _service(link, _saved(base))
    await service.open(KEY, base + "/one")
    assert await service.signed_in_sites() == (True, [])

    await service.sign_in(KEY)
    open_, sites = await service.signed_in_sites()
    assert open_ is True
    assert [s["site"] for s in sites] == ["127.0.0.1"]
    assert sites[0]["last_used"]

    assert await service.sign_out("127.0.0.1") == (True, [])
    target = service.live_target(KEY[0], KEY[1], 1)
    assert target is not None
    left = await link.send(target[1], "Network.getCookies", {"urls": [base + "/"]})
    assert left["cookies"] == []


async def test_after_the_person_signs_in_by_hand_saving_it_is_offered(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    base, _ = site
    service = _service(link, [])
    await service.open(KEY, base + "/one")
    signing_in = asyncio.create_task(service.sign_in(KEY))
    for _ in range(200):
        if service.live_state(KEY[0]).get("asked_user"):
            break
        await asyncio.sleep(0.05)
    assert "save_offer" not in service.live_state(KEY[0])

    # The person signs in in the page, then gives the tab back.
    target = service.live_target(KEY[0], KEY[1], 1)
    assert target is not None
    await link.send(
        target[1],
        "Runtime.evaluate",
        {
            "expression": f"document.getElementById('u').value = {json.dumps(USER)};"
            f"document.getElementById('p').value = {json.dumps(SECRET)};"
            "document.querySelector('form').requestSubmit();"
        },
    )
    for _ in range(100):
        where = await link.send(
            target[1], "Runtime.evaluate", {"expression": "location.href", "returnByValue": True}
        )
        if "/home" in json.dumps(where):
            break
        await asyncio.sleep(0.05)
    service.give_back(*KEY)
    result = await asyncio.wait_for(signing_in, 10)

    assert "sign_in_likely" not in result
    assert service.live_state(KEY[0])["save_offer"] == {"clone": "scout", "site": site_key(base)}
    _nothing_carries_the_secret(result, service.live_state(KEY[0]))

    service.sign_in_saved(site_key(base))
    assert "save_offer" not in service.live_state(KEY[0])


async def test_a_redirect_to_another_host_gets_no_password(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    """The username step on 127.0.0.1 sends the browser to localhost, another host: the
    password saved for 127.0.0.1 is not typed there, and nothing claims a sign-in.

    Killed by: src/uclone_x/browser/service.py :: if sign_in_for(here, [saved]) is None:
    Becomes: if False:
    """
    base, posted = site
    service = _service(link, _saved(base))
    await service.open(KEY, base + "/hop")
    before = len(posted)

    signing_in = asyncio.create_task(service.sign_in(KEY))
    asked = await _give_back_when_asked(service)
    result = await asyncio.wait_for(signing_in, 10)

    assert [p["path"] for p in posted[before:]] == ["/hop"], "only the username step posts"
    assert not any(p.get("p") == SECRET for p in posted[before:]), "the password went on"
    assert "localhost" in asked["message"]
    assert "Signed in" not in result["message"]
    _nothing_carries_the_secret(result, asked)


async def test_a_dialog_that_interrupts_the_fill_leaves_nothing_typing_later(
    link: R2Link, site: tuple[str, list[dict[str, str]]]
) -> None:
    """A page that opens a dialog mid-sign-in: the fill is abandoned, not left pending to
    type into whatever page comes next, and the person is asked."""
    base, _ = site
    service = _service(link, _saved(base))
    await service.open(KEY, base + "/alert")

    signing_in = asyncio.create_task(service.sign_in(KEY))
    asked = await _give_back_when_asked(service)
    await asyncio.wait_for(signing_in, 10)

    tab = service.live_tab(KEY[0], KEY[1], 1)
    assert tab is not None and tab.pending is None
    assert "dialog" in asked["message"]
