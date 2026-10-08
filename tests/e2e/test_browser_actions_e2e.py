"""Page actions (design `browser-agent.md` §6 step 2) over R2 against a real Chromium.

Every page is a fixture served on loopback; nothing reaches the network. Each action is
checked by what it returns — the change it caused, or the plain refusal a clone reads.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import AsyncIterator, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import sync_playwright

from uclone_x.browser import BrowserService, R2Link
from uclone_x.browser.actions import SECRET_FIELD_REFUSAL, STALE_REF
from uclone_x.browser.service import LOAD_TIMEOUT_S
from uclone_x.errors import PlainRefusalError

pytestmark = pytest.mark.e2e

_ACTIONS = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>동작</title>
<style>#tall { height: 3000px; }</style></head><body>
<main>
<button id="add" onclick="var p=document.createElement('p');p.textContent='추가된 줄';document.body.appendChild(p)">추가</button>
<a href="/next.html">다음 쪽</a>
<label for="q">검색어</label>
<input id="q" onkeydown="if(event.key==='Enter'){document.getElementById('said').textContent='보냄: '+this.value}">
<p id="said"></p>
<label for="pw">비밀번호</label><input id="pw" type="password">
<label for="otp">인증번호</label><input id="otp" autocomplete="one-time-code">
<label for="r">지역</label><select id="r"><option>서울</option><option>부산</option></select>
<button id="city" aria-haspopup="listbox" aria-expanded="false"
  onclick="document.getElementById('cities').hidden=false;this.setAttribute('aria-expanded','true')">도시 고르기</button>
<ul id="cities" role="listbox" hidden>
  <li role="option" onclick="document.getElementById('city').textContent='도시: '+this.textContent;this.parentNode.hidden=true">대구</li>
  <li role="option" onclick="document.getElementById('city').textContent='도시: '+this.textContent;this.parentNode.hidden=true">광주</li>
</ul>
<input type="checkbox" id="agree"><label for="agree">약관 동의</label>
<button onclick="alert('저장했습니다')">알림</button>
<button onclick="document.getElementById('answer').textContent=confirm('지울까요?')?'지움':'그대로'">확인</button>
<button onclick="document.getElementById('answer').textContent='이름: '+prompt('이름은?')">이름</button>
<p id="answer"></p>
<a href="/next.html" target="_blank">새 창</a>
<a href="/report.txt" download="report.txt">보고서 받기</a>
<label for="file">첨부</label><input type="file" id="file">
<input type="file" id="hidden-file" style="display:none"
  onchange="document.getElementById('picked').textContent='고름: '+this.files[0].name">
<button onclick="document.getElementById('hidden-file').click()">파일 올리기</button>
<p id="picked"></p>
<button onclick="setTimeout(function(){var p=document.createElement('p');p.textContent='늦게 온 소식';document.body.appendChild(p)},500)">나중에</button>
<div id="tall"></div>
<p>맨 아래</p>
</main></body></html>"""

_NEXT = """<!doctype html><html><head><meta charset="utf-8"><title>다음</title></head><body>
<a href="/actions.html">돌아가기</a></body></html>"""


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    root = tmp_path_factory.mktemp("site")
    (root / "actions.html").write_text(_ACTIONS, encoding="utf-8")
    (root / "next.html").write_text(_NEXT, encoding="utf-8")
    (root / "report.txt").write_text("보고서 본문", encoding="utf-8")

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
async def service(link: R2Link, tmp_path: Path) -> BrowserService:
    async def _factory() -> R2Link:
        return link

    return BrowserService(_factory, downloads_dir=tmp_path / "downloads")


KEY = ("conversation-1", "scout")
_INTERNALS = ("CDP", "Error", "backendNode", "objectId", "Runtime.", "autocomplete", "net::")


async def _ref(service: BrowserService, role: str, name: str) -> str:
    """The ref of the element `find` lists as `role "name"`."""
    matches = (await service.find(KEY, name))["matches"]
    found = re.search(rf'{role} "{re.escape(name)}"[^\n]*\[ref=(e\d+)\]', matches)
    assert found, matches
    return found.group(1)


async def _open(service: BrowserService, site: str) -> None:
    await service.open(KEY, f"{site}/actions.html")


def _plain(message: str) -> None:
    for internal in _INTERNALS:
        assert internal not in message


async def test_click_returns_what_changed_on_the_page(service: BrowserService, site: str) -> None:
    await _open(service, site)

    result = await service.click(KEY, await _ref(service, "button", "추가"))

    added = [line for line in result["changes"].splitlines() if line.startswith("+ ")]
    assert any("추가된 줄" in line for line in added), result["changes"]
    assert "snapshot" not in result


async def test_a_click_that_navigates_returns_a_fresh_snapshot(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)

    result = await service.click(KEY, await _ref(service, "link", "다음 쪽"))

    assert result["title"] == "다음"
    assert 'link "돌아가기" [ref=e1]' in result["snapshot"]


async def test_type_clears_appends_submits_and_keeps_korean(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)
    field = await _ref(service, "textbox", "검색어")

    await service.type_text(KEY, field, "서울 맛집")
    await service.type_text(KEY, field, "부산")
    result = await service.type_text(KEY, field, " 여행", append=True, submit=True)

    assert "보냄: 부산 여행" in result["changes"]


@pytest.mark.parametrize("label", ["비밀번호", "인증번호"])
async def test_type_refuses_password_and_one_time_code_fields(
    service: BrowserService, site: str, label: str
) -> None:
    await _open(service, site)
    field = await _ref(service, "textbox", label)

    with pytest.raises(PlainRefusalError) as refused:
        await service.type_text(KEY, field, "hunter2")

    assert str(refused.value) == SECRET_FIELD_REFUSAL
    assert "Use sign_in, or ask the person to take over" in str(refused.value)
    _plain(str(refused.value))


async def test_a_ref_from_a_page_left_behind_is_refused_as_stale(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)
    field = await _ref(service, "textbox", "검색어")
    await service.click(KEY, await _ref(service, "link", "다음 쪽"))

    with pytest.raises(PlainRefusalError) as refused:
        await service.type_text(KEY, "e999", "x")
    assert str(refused.value) == STALE_REF
    with pytest.raises(PlainRefusalError):
        await service.type_text(KEY, field, "x")
    _plain(str(refused.value))


async def test_select_picks_a_native_option_and_refuses_a_missing_one(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)
    region = await _ref(service, "combobox", "지역")

    result = await service.select(KEY, region, ["부산"])
    assert "부산" in result["changes"]

    with pytest.raises(PlainRefusalError) as refused:
        await service.select(KEY, region, ["제주"])
    assert "제주" in str(refused.value)
    _plain(str(refused.value))


async def test_select_opens_a_custom_list_and_clicks_the_option(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)

    result = await service.select(KEY, await _ref(service, "button", "도시 고르기"), ["광주"])

    assert "도시: 광주" in result["changes"]


async def test_check_turns_a_checkbox_on_and_off_only_when_needed(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)
    box = await _ref(service, "checkbox", "약관 동의")

    on = await service.check(KEY, box)
    assert "checked" in on["changes"]
    again = await service.check(KEY, box)
    assert again["note"] == "It was already checked."
    off = await service.check(KEY, box, on=False)
    assert "~ checkbox" in off["changes"]


async def test_press_sends_a_key_to_the_focused_field(service: BrowserService, site: str) -> None:
    await _open(service, site)
    await service.type_text(KEY, await _ref(service, "textbox", "검색어"), "제주")

    result = await service.press(KEY, "Enter")

    assert "보냄: 제주" in result["changes"]


async def test_scroll_says_where_the_page_now_is(service: BrowserService, site: str) -> None:
    await _open(service, site)

    down = await service.scroll(KEY, None, "down", 10)
    assert down["note"] == "Now at the bottom."
    up = await service.scroll(KEY, None, "up", 10)
    assert up["note"] == "Now at the top."


async def test_upload_fills_a_file_field_and_answers_a_file_picker(
    service: BrowserService, site: str, tmp_path: Path
) -> None:
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4")
    await _open(service, site)

    direct = await service.upload(KEY, await _ref(service, "button", "첨부"), [cv])
    assert direct["note"] == "Chose cv.pdf."

    picked = await service.upload(KEY, await _ref(service, "button", "파일 올리기"), [cv])
    assert picked["note"] == "Chose cv.pdf."
    assert "고름: cv.pdf" in picked["changes"]


async def test_upload_on_an_element_without_a_picker_is_refused_plainly(
    service: BrowserService, site: str, tmp_path: Path
) -> None:
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4")
    await _open(service, site)

    with pytest.raises(PlainRefusalError) as refused:
        await service.upload(KEY, await _ref(service, "button", "추가"), [cv])

    assert "did not open a file picker" in str(refused.value)
    _plain(str(refused.value))


async def test_wait_for_text_an_element_or_a_time(service: BrowserService, site: str) -> None:
    await _open(service, site)
    await service.click(KEY, await _ref(service, "button", "나중에"))

    shown = await service.wait(KEY, text="늦게 온 소식")
    assert shown["note"] == '"늦게 온 소식" is on the page.'
    ready = await service.wait(KEY, ref=await _ref(service, "button", "추가"))
    assert ready["note"] == "The element is visible and ready."
    paused = await service.wait(KEY, ms=100)
    assert paused["changes"] == "Nothing on the page changed."


async def test_an_alert_is_reported_and_answered_with_enter(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)

    opened = await service.click(KEY, await _ref(service, "button", "알림"))
    assert '"저장했습니다"' in opened["dialog"]

    with pytest.raises(PlainRefusalError) as frozen:
        await service.snapshot(KEY)
    assert "저장했습니다" in str(frozen.value)

    answered = await service.press(KEY, "Enter")
    assert "dialog" not in answered


async def test_confirm_and_prompt_are_answered_by_escape_and_type(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)

    await service.click(KEY, await _ref(service, "button", "확인"))
    declined = await service.press(KEY, "Escape")
    assert "그대로" in declined["changes"]

    asked = await service.click(KEY, await _ref(service, "button", "이름"))
    assert "이름은?" in asked["dialog"]
    named = await service.type_text(KEY, None, "민지")
    assert "이름: 민지" in named["changes"]


async def test_a_pop_up_becomes_the_current_tab_and_tabs_switch_and_close(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)

    popped = await service.click(KEY, await _ref(service, "link", "새 창"))
    assert "new_tab" in popped
    assert popped["title"] == "다음"
    assert "2. " in popped["tabs"] and "(current)" in popped["tabs"]

    first = await service.tab(KEY, index=1)
    assert first["title"] == "동작"

    closed = await service.tab(KEY, index=2, close=True)
    assert closed["title"] == "동작"
    assert "tabs" not in closed

    with pytest.raises(PlainRefusalError) as missing:
        await service.tab(KEY, index=5)
    assert "There is no tab 5" in str(missing.value)


async def test_a_download_lands_in_the_downloads_folder(
    service: BrowserService, site: str, tmp_path: Path
) -> None:
    await _open(service, site)

    result = await service.click(KEY, await _ref(service, "link", "보고서 받기"))

    assert "report.txt" in result["downloads"]
    saved = list((tmp_path / "downloads").iterdir())
    assert [p.read_text(encoding="utf-8") for p in saved] == ["보고서 본문"]


async def test_back_forward_and_reload_move_through_the_tab_history(
    service: BrowserService, site: str
) -> None:
    await _open(service, site)
    await service.click(KEY, await _ref(service, "link", "다음 쪽"))

    started = time.monotonic()
    back = await service.back(KEY)
    assert back["title"] == "동작"
    # A page from the back-forward cache fires no load event; the wait must not hit its cap.
    assert time.monotonic() - started < LOAD_TIMEOUT_S / 2
    forward = await service.forward(KEY)
    assert forward["title"] == "다음"
    reloaded = await service.reload(KEY)
    assert reloaded["title"] == "다음"

    with pytest.raises(PlainRefusalError) as refused:
        await service.forward(KEY)
    assert str(refused.value) == "There is no later page in this tab."
