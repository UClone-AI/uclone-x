"""`look` and the coordinate `click`: the browser's screenshot path (design §3.2, §6 step 5).

A screenshot is of the tab's viewport, as JPEG, no wider or taller than `MAX_EDGE`
pixels. The model names a point by the image's own pixels, and `View` maps it back to the
page's CSS pixels -- the factor depends on the window's size and the screen's pixel
density, which the model cannot know. The factor is read from the JPEG the browser
returned, not assumed from the scale asked for.

A view goes stale when the page scrolls or navigates: a point picked on it would land on
something else, so the coordinate click refuses and asks for a fresh `look`.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Final, cast

from uclone_x.browser.link import BrowserLink
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.models import ImagePart

__all__ = ["JPEG_QUALITY", "MAX_EDGE", "View", "capture", "jpeg_size", "page_point"]

#: Longest side of a screenshot, in pixels. Around the size providers resize a picture to
#: anyway (Anthropic's guidance is ~1.15 MP), so more pixels would cost bytes the model
#: never sees.
MAX_EDGE: Final = 1280
JPEG_QUALITY: Final = 70


@dataclass(frozen=True)
class View:
    """What a screenshot showed, so a point on it can be put back on the page."""

    width: int
    """The image's pixels across."""
    height: int
    scale: float
    """Image pixels per CSS pixel."""
    page_x: float
    """Where the viewport was scrolled to when the picture was taken, in CSS pixels."""
    page_y: float
    loader: str
    """The page load the picture is of (`Page.getFrameTree` loaderId)."""


async def _viewport(link: BrowserLink, target: str) -> dict[str, float]:
    metrics = await link.send(target, "Page.getLayoutMetrics")
    css = metrics.get("cssVisualViewport")
    device = metrics.get("visualViewport")
    found: dict[str, float] = {}
    for name, source in (("css", css), ("device", device)):
        if isinstance(source, dict):
            values = cast(dict[str, Any], source)
            for field in ("clientWidth", "clientHeight", "pageX", "pageY"):
                value = values.get(field)
                if isinstance(value, int | float):
                    found[f"{name}.{field}"] = float(value)
    return found


async def capture(link: BrowserLink, target: str, loader: str) -> tuple[ImagePart, View]:
    """A JPEG of the tab's viewport, at most `MAX_EDGE` on its longer side, and its view.

    Raises:
        PlainRefusalError: The page shows nothing to take a picture of, or the browser
            returned no picture.
    """
    port = await _viewport(link, target)
    width = port.get("css.clientWidth", 0.0)
    height = port.get("css.clientHeight", 0.0)
    if width <= 0 or height <= 0:
        raise PlainRefusalError(
            "The page has no visible area to take a picture of.", reason_code="no_view"
        )
    # Device pixels per CSS pixel, so a high-density screen does not double the picture.
    density = port.get("device.clientWidth", width) / width or 1.0
    fit = min(1.0, MAX_EDGE / max(width, height))
    page_x = port.get("css.pageX", 0.0)
    page_y = port.get("css.pageY", 0.0)
    shot = await link.send(
        target,
        "Page.captureScreenshot",
        {
            "format": "jpeg",
            "quality": JPEG_QUALITY,
            "clip": {
                "x": page_x,
                "y": page_y,
                "width": width,
                "height": height,
                "scale": fit / density,
            },
        },
    )
    data = shot.get("data")
    try:
        raw = base64.b64decode(data, validate=True) if isinstance(data, str) else b""
    except (binascii.Error, ValueError):
        raw = b""
    size = jpeg_size(raw)
    if size is None:
        raise PlainRefusalError(
            "The browser did not return a picture of the page.", reason_code="no_screenshot"
        )
    image_w, image_h = size
    view = View(
        width=image_w,
        height=image_h,
        scale=image_w / width,
        page_x=page_x,
        page_y=page_y,
        loader=loader,
    )
    return ImagePart.from_bytes(raw, "image/jpeg", width=image_w, height=image_h), view


async def page_point(
    link: BrowserLink, target: str, view: View | None, loader: str, x: int, y: int
) -> tuple[float, float]:
    """The CSS viewport point that image pixel (`x`, `y`) of the last `look` shows.

    Raises:
        PlainRefusalError: No picture was taken, the page moved or changed since, or the
            point is outside the picture.
    """
    if view is None:
        raise PlainRefusalError(
            "Take a picture of the page with action=look before clicking a point on it.",
            reason_code="no_look",
        )
    port = await _viewport(link, target)
    moved = (
        abs(port.get("css.pageX", 0.0) - view.page_x) >= 1
        or abs(port.get("css.pageY", 0.0) - view.page_y) >= 1
    )
    if loader != view.loader or moved:
        raise PlainRefusalError(
            "The page has moved or changed since the last picture, so that point may now "
            "be something else. Take a new picture with action=look first.",
            reason_code="stale_look",
        )
    if not (0 <= x < view.width and 0 <= y < view.height):
        raise PlainRefusalError(
            f"That point is outside the picture, which is {view.width} by {view.height} pixels.",
            reason_code="outside_look",
        )
    return x / view.scale, y / view.scale


#: JPEG start-of-frame markers, which carry the image's size: C0-CF except DHT (C4),
#: JPG (C8) and DAC (CC).
_SOF: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def jpeg_size(raw: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's frame header, or `None` when `raw` is not one."""
    if raw[:2] != b"\xff\xd8":
        return None
    at = 2
    while at + 4 <= len(raw):
        if raw[at] != 0xFF:
            return None
        marker = raw[at + 1]
        if marker == 0xFF:  # fill byte
            at += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # no length
            at += 2
            continue
        length = int.from_bytes(raw[at + 2 : at + 4], "big")
        if marker in _SOF:
            if at + 9 > len(raw):
                return None
            height = int.from_bytes(raw[at + 5 : at + 7], "big")
            width = int.from_bytes(raw[at + 7 : at + 9], "big")
            return (width, height) if width and height else None
        if marker in (0xD9, 0xDA) or length < 2:  # end, or scan data before any frame
            return None
        at += 2 + length
    return None
