"""`BrowserService` over R2 against a real Chromium and a fixture site on loopback.

Design `browser-agent.md` §3.10: tests drive a real browser only through `R2Link`. They
launch the Chromium Playwright already installed for the e2e suite, headless so the gate
opens no window, with the same plain command line the product uses for Chrome.
"""

from __future__ import annotations

import base64
import threading
from collections.abc import AsyncIterator, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import sync_playwright

from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.chrome import free_port
from uclone_x.browser.vision import MAX_EDGE, jpeg_size
from uclone_x.errors import PlainRefusalError

pytestmark = pytest.mark.e2e

_FORM = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>양식</title></head><body>
<header><h1>여행 찾기</h1></header>
<main>
<form>
<label for="q">검색어</label><input id="q" name="q">
<label for="r">지역</label><select id="r"><option>서울</option><option>부산</option></select>
<button type="button" id="go">보내기</button>
</form>
<ul>{items}</ul>
</main></body></html>"""
_ITEMS = "".join(f'<li><a href="/next.html?i={i}">Item {i}</a></li>' for i in range(1, 31))

# Content that exists only after a script fetches it: `read` must wait for the network.
_SCRIPTED = """<!doctype html><html><head><meta charset="utf-8"><title>스크립트</title></head><body><div id="out"></div>
<script>
fetch('/data.txt').then(r => r.text()).then(t => {
  document.getElementById('out').innerHTML = '<p>' + t + '</p>';
});
</script></body></html>"""

_NEXT = """<!doctype html><html><head><meta charset="utf-8"><title>다음</title></head><body>
<a href="/form.html">돌아가기</a></body></html>"""

# A button the picture shows at a known place, for the coordinate click (step 5).
_POINTER = """<!doctype html><html><head><meta charset="utf-8"><title>그림</title>
<style>body{margin:0} #b{position:absolute;left:100px;top:200px;width:120px;height:40px}</style>
</head><body><button id="b" onclick="document.body.insertAdjacentHTML('beforeend','<h2>눌렸다</h2>')">누르기</button>
</body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("site")
    (root / "form.html").write_text(_FORM.replace("{items}", _ITEMS), encoding="utf-8")
    (root / "scripted.html").write_text(_SCRIPTED, encoding="utf-8")
    (root / "next.html").write_text(_NEXT, encoding="utf-8")
    (root / "pointer.html").write_text(_POINTER, encoding="utf-8")
    (root / "data.txt").write_text("스크립트가 불러온 문장", encoding="utf-8")

    class _Quiet(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def chromium() -> Path:
    with sync_playwright() as p:
        return Path(p.chromium.executable_path)


@pytest.fixture
async def link(tmp_path: Path, chromium: Path) -> AsyncIterator[R2Link]:
    started = await R2Link.start(
        tmp_path / "profile", chrome=chromium, extra_args=("--headless=new",)
    )
    yield started
    await started.shutdown()


@pytest.fixture
async def service(link: R2Link) -> BrowserService:
    async def _factory() -> R2Link:
        return link

    return BrowserService(_factory)


KEY = ("conversation-1", "scout")


async def test_open_returns_a_snapshot_with_refs_on_controls(
    service: BrowserService, site: str
) -> None:
    result = await service.open(KEY, f"{site}/form.html")

    snapshot = result["snapshot"]
    assert result["title"] == "양식"
    assert 'heading "여행 찾기" [level=1]' in snapshot
    assert 'textbox "검색어" [ref=' in snapshot
    assert 'combobox "지역" [ref=' in snapshot
    assert 'button "보내기" [ref=' in snapshot
    # Thirty links in the list: ten shown, the rest collapsed into one line.
    assert 'link "Item 10"' in snapshot
    assert 'link "Item 11"' not in snapshot
    assert "…and 20 more" in snapshot


async def test_read_waits_for_content_a_script_fetched(service: BrowserService, site: str) -> None:
    await service.open(KEY, f"{site}/scripted.html")

    result = await service.read(KEY, 20000)

    assert "스크립트가 불러온 문장" in result["content"]


async def test_find_reaches_elements_the_snapshot_collapsed(
    service: BrowserService, site: str
) -> None:
    await service.open(KEY, f"{site}/form.html")

    result = await service.find(KEY, "Item 25")

    assert 'link "Item 25" [ref=' in result["matches"]


async def test_refs_hold_within_a_page_and_are_reissued_on_navigation(
    service: BrowserService, site: str
) -> None:
    first = (await service.open(KEY, f"{site}/form.html"))["snapshot"]
    again = (await service.snapshot(KEY))["snapshot"]
    assert first == again

    other = (await service.open(KEY, f"{site}/next.html"))["snapshot"]
    assert 'link "돌아가기" [ref=e1]' in other


async def test_each_clone_in_a_conversation_has_its_own_tab(
    service: BrowserService, site: str
) -> None:
    await service.open(("conversation-1", "scout"), f"{site}/form.html")
    await service.open(("conversation-1", "writer"), f"{site}/next.html")

    scout = await service.snapshot(("conversation-1", "scout"))

    assert scout["title"] == "양식"


async def test_a_tab_the_person_closed_is_reopened(
    service: BrowserService, link: R2Link, site: str
) -> None:
    await service.open(KEY, f"{site}/form.html")
    for tab in list(link._sessions):  # pyright: ignore[reportPrivateUsage]
        await link.close_tab(tab)

    result = await service.open(KEY, f"{site}/next.html")

    assert result["title"] == "다음"


async def test_snapshot_before_any_page_says_to_open_one(service: BrowserService) -> None:
    with pytest.raises(PlainRefusalError, match="Open one first"):
        await service.snapshot(KEY)


async def test_an_unreachable_site_is_refused_in_plain_words(
    service: BrowserService, site: str
) -> None:
    # A port nothing listens on. (Low ports such as 9 are refused by Chrome as unsafe.)
    closed = f"http://127.0.0.1:{free_port()}/"

    with pytest.raises(PlainRefusalError) as refused:
        await service.open(KEY, closed)

    assert "refused the connection" in str(refused.value)
    assert "net::" not in str(refused.value)


async def test_a_restarted_core_reattaches_to_the_open_chrome(
    link: R2Link, tmp_path: Path, chromium: Path, site: str
) -> None:
    again = await R2Link.start(tmp_path / "profile", chrome=chromium)
    try:
        assert again._process is None  # pyright: ignore[reportPrivateUsage]
        tab = await again.open_tab(f"{site}/next.html")
        assert tab
    finally:
        await again.close()


async def test_look_returns_a_jpeg_and_a_point_on_it_clicks_the_page(
    service: BrowserService, site: str
) -> None:
    await service.open(KEY, f"{site}/pointer.html")
    where, image = await service.look(KEY)

    assert image.media_type == "image/jpeg" and image.data is not None
    assert image.width is not None and image.height is not None
    assert max(image.width, image.height) <= MAX_EDGE
    assert jpeg_size(base64.b64decode(image.data)) == (image.width, image.height)
    assert where["title"] == "그림"

    # The viewport is under MAX_EDGE at density 1, so a picture pixel is a CSS pixel.
    result = await service.click_at(KEY, 160, 220)
    assert "눌렸다" in str(result["changes"])

    with pytest.raises(PlainRefusalError, match="action=look"):
        await service.click_at(KEY, 160, 220)
