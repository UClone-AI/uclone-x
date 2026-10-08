"""`look` and the coordinate `click` over a fake link (design `browser-agent.md` §3.2, step 5).

The fake answers the CDP commands the path sends and records them; a real Chromium drives
the same path in `tests/e2e/test_browser_service_e2e.py`.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from uclone_x.browser.cdp import CdpEvent
from uclone_x.browser.service import BrowserService, TabKey
from uclone_x.browser.tool import NO_IMAGES_NOTE, BrowserParams, BrowserTool
from uclone_x.browser.vision import MAX_EDGE, jpeg_size
from uclone_x.errors import PlainRefusalError
from uclone_x.tools.models import ToolContext, ToolResult

KEY: TabKey = ("conversation-1", "scout")


def _jpeg(width: int, height: int) -> bytes:
    """The smallest byte string `jpeg_size` reads: SOI, an APP0 segment, SOF0, EOI."""
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    sof += b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


class _FakeLink:
    """A tab whose viewport is `css` CSS pixels at `density`, scrolled to `scroll`."""

    def __init__(self, *, css: tuple[int, int], density: float = 1.0) -> None:
        self.css = css
        self.density = density
        self.scroll = (0.0, 0.0)
        self.loader = "L1"
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def open_tab(self, url: str) -> str:
        return "T1"

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        args = dict(params or {})
        self.sent.append((method, args))
        width, height = self.css
        if method == "Page.getLayoutMetrics":
            return {
                "cssVisualViewport": {
                    "clientWidth": width,
                    "clientHeight": height,
                    "pageX": self.scroll[0],
                    "pageY": self.scroll[1],
                },
                "visualViewport": {
                    "clientWidth": width * self.density,
                    "clientHeight": height * self.density,
                },
            }
        if method == "Page.captureScreenshot":
            # Chrome's output is clip size x scale x density.
            clip = args["clip"]
            out_w = round(clip["width"] * clip["scale"] * self.density)
            out_h = round(clip["height"] * clip["scale"] * self.density)
            return {"data": base64.b64encode(_jpeg(out_w, out_h)).decode("ascii")}
        if method == "Page.getFrameTree":
            return {"frameTree": {"frame": {"loaderId": self.loader}}}
        if method == "Runtime.evaluate":
            return {"result": {"value": ["https://example.com/", "Example"]}}
        if method == "Accessibility.getFullAXTree":
            return {"nodes": []}
        return {}

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        return asyncio.Queue()

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        return None

    async def close_tab(self, tab: str) -> None:
        return None

    async def close(self) -> None:
        return None


async def _service(link: _FakeLink) -> BrowserService:
    async def factory() -> Any:
        return link

    service = BrowserService(factory)
    await service.open(KEY, "https://example.com/")
    return service


def _clicks(link: _FakeLink) -> list[dict[str, Any]]:
    return [args for method, args in link.sent if method == "Input.dispatchMouseEvent"]


def test_the_size_is_read_from_the_jpeg_frame_header() -> None:
    assert jpeg_size(_jpeg(640, 400)) == (640, 400)
    assert jpeg_size(b"\x89PNG\r\n") is None
    assert jpeg_size(b"\xff\xd8\xff\xd9") is None


@pytest.mark.parametrize(("css", "density"), [((1600, 1000), 2.0), ((800, 600), 1.0)])
async def test_a_picture_is_at_most_max_edge_and_a_point_maps_back_to_css_pixels(
    css: tuple[int, int], density: float
) -> None:
    """Killed by: src/uclone_x/browser/vision.py :: return x / view.scale, y / view.scale
    Becomes: return x, y
    """
    link = _FakeLink(css=css, density=density)
    service = await _service(link)
    where, image = await service.look(KEY)
    assert image.width is not None and image.height is not None
    assert max(image.width, image.height) <= MAX_EDGE
    assert where["width"] == image.width
    expected_scale = min(1.0, MAX_EDGE / max(css))
    await service.click_at(KEY, image.width // 2, image.height // 2)
    press = next(c for c in _clicks(link) if c["type"] == "mousePressed")
    assert press["x"] == pytest.approx((image.width // 2) / expected_scale, abs=1)
    assert press["y"] == pytest.approx((image.height // 2) / expected_scale, abs=1)
    assert press["button"] == "left" and press["clickCount"] == 1


async def test_click_at_supports_double_right_and_hover() -> None:
    """Killed by: src/uclone_x/browser/service.py :: button = "right" if right else "left"  # coordinate button
    Becomes: button = "left"  # coordinate button
    Killed by: src/uclone_x/browser/service.py :: clicks = 2 if double else 1  # coordinate clicks
    Becomes: clicks = 1  # coordinate clicks
    """
    link = _FakeLink(css=(800, 600))
    service = await _service(link)
    await service.look(KEY)
    await service.click_at(KEY, 10, 10, right=True, double=True)
    presses = [c for c in _clicks(link) if c["type"] == "mousePressed"]
    assert len(presses) == 2
    assert all(p["button"] == "right" for p in presses)
    assert [p["clickCount"] for p in presses] == [1, 2]

    link2 = _FakeLink(css=(800, 600))
    service2 = await _service(link2)
    await service2.look(KEY)
    await service2.click_at(KEY, 10, 10, hover=True)
    assert not any(c["type"] == "mousePressed" for c in _clicks(link2))
    assert any(c["type"] == "mouseMoved" for c in _clicks(link2))


async def test_a_point_needs_a_picture_and_a_page_that_has_not_moved() -> None:
    """Killed by: src/uclone_x/browser/vision.py :: if loader != view.loader or moved:
    Becomes: if False:
    """
    link = _FakeLink(css=(800, 600))
    service = await _service(link)
    with pytest.raises(PlainRefusalError, match="action=look"):
        await service.click_at(KEY, 10, 10)
    await service.look(KEY)
    link.scroll = (0.0, 300.0)
    with pytest.raises(PlainRefusalError, match="moved or changed"):
        await service.click_at(KEY, 10, 10)
    link.scroll = (0.0, 0.0)
    await service.look(KEY)
    with pytest.raises(PlainRefusalError, match="outside the picture"):
        await service.click_at(KEY, 800, 10)
    assert _clicks(link) == []


async def test_a_picture_serves_one_click() -> None:
    """Killed by: src/uclone_x/browser/service.py :: tab.view = None
    Becomes: pass
    """
    link = _FakeLink(css=(800, 600))
    service = await _service(link)
    await service.look(KEY)
    await service.click_at(KEY, 10, 10)
    with pytest.raises(PlainRefusalError, match="action=look"):
        await service.click_at(KEY, 10, 10)


async def test_look_sends_a_picture_only_to_a_model_that_reads_images() -> None:
    """Killed by: src/uclone_x/browser/tool.py :: if not context.accepts_images:
    Becomes: if False:
    """
    link = _FakeLink(css=(800, 600))
    tool = BrowserTool(service=await _service(link))

    seeing = ToolContext(agent_id="scout", session_id="conversation-1", accepts_images=True)
    result = await tool.execute({"action": "look"}, seeing)
    assert isinstance(result, ToolResult) and result.success
    (image,) = result.images
    assert image.media_type == "image/jpeg" and image.data is not None
    assert isinstance(result.output, dict)
    assert image.data not in str(result.output)

    blind = ToolContext(agent_id="scout", session_id="conversation-1")
    result = await tool.execute({"action": "look"}, blind)
    assert result.success and result.images == ()
    assert isinstance(result.output, dict)
    assert result.output["note"] == NO_IMAGES_NOTE
    assert "snapshot" in result.output


def test_a_coordinate_click_needs_both_coordinates() -> None:
    with pytest.raises(ValidationError, match="click needs a ref, or x and y"):
        BrowserParams(action="click", x=3)
    with pytest.raises(ValidationError):
        BrowserParams(action="click", x=-1, y=0)
    assert BrowserParams(action="click", x=0, y=0).x == 0


async def test_a_refusal_reaches_the_model_in_plain_words() -> None:
    link = _FakeLink(css=(800, 600))
    tool = BrowserTool(service=await _service(link))
    context = ToolContext(agent_id="scout", session_id="conversation-1", accepts_images=True)
    result = await tool.execute({"action": "click", "x": 5, "y": 5}, context)
    assert result.success is False
    assert result.error == (
        "Take a picture of the page with action=look before clicking a point on it."
    )
