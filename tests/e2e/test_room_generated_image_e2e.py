"""A generated picture is on screen in the conversation, in a browser (#1208's loss).

The head could draw a generated image once. `MediaCard` did it, from the `generate_image` tool
result, and it lost its only caller when `PlaygroundTab` was retired (`281eefe8`, #1208) -- the
same removal `MessageBody` records for markdown, found the same way: by looking at what a reader
saw rather than at what the suite asserted. `MediaCard`'s own unit test never stopped passing,
so a green gate said nothing about whether any surface still rendered a picture. None did.

What the room gives the head is prose. `RoomMessage` carries no tool results by design -- *"tool
calls, tool results, intermediate assistant turns"* are *"Deliberately not here"*
(`src/uclone_x/room/models.py`) -- so the only evidence of the file is the link the reply writes
about it. `IMAGE_REPLY` below is that reply, verbatim from the room document on 2026-09-21
(`room_74d41ac2f9c9.json`, seq 17): the image tool had written 600KB to disk and the
conversation showed one line of blue text.

End-to-end rather than in the component test alone, because the component test cannot fail the
way this did. It renders an `<img>` against jsdom, which fetches nothing: an address the server
does not serve looks identical to one it does. `naturalWidth` is asserted here for exactly that
-- it is above zero only when the bytes arrived -- and the file is written into the server's own
workspace so the fetch is the real route, `/api/artifacts/content` included.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest
from playwright.async_api import Page, async_playwright

from tests.e2e.conftest import UIServerFactory, running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector

pytestmark = pytest.mark.e2e

#: The reply an artist turn produced on 2026-09-21, unedited. A markdown *link*, which is what a
#: model writes when asked for a picture -- not the `![]()` a renderer would find convenient.
#: Substituted rather than invented: a hand-written `![]()` passes against the renderer that
#: shipped this defect, so a test using one would have proved nothing.
IMAGE_REPLY = (
    "Here's your latest scene: a lone knight on a stormy battlefield.\n\n"
    "Image generated: [Artistic 16:9]  \n"
    "🔗 [View Image](/api/artifacts/content?path=artifacts/images/img_3008971970_sess_r.png)  \n\n"
    "Need adjustments to the lighting, composition, or details?"
)

ARTIFACT_PATH = "artifacts/images/img_3008971970_sess_r.png"

#: The reply an Ollama / local model turn produced, verbatim format from room_6d1560573f65.json.
#: Wrapped in <image>...</image> tags, which in CommonMark opens an HTML block and suppresses
#: markdown parsing unless normalized.
TAG_WRAPPED_IMAGE_REPLY = (
    "<image>\n"
    f"![Bikini Girl Art](/api/artifacts/content?path={ARTIFACT_PATH})\n"
    "</image>  \n"
    "16:9 artistic-style illustration created with Danbooru tags. "
    f"The image is saved as `{ARTIFACT_PATH}`."
)

IMAGE_WIDTH = 8


def _png_bytes(width: int, height: int) -> bytes:
    """Assemble a valid greyscale PNG of `width` x `height`.

    Built rather than pasted as base64. A literal was tried first and it was corrupt -- both
    CRCs wrong and the IDAT undeflatable -- and the server served its 74 bytes with a 200 and
    `image/png` regardless, because serving a file does not involve decoding it. Only the
    browser noticed, as `naturalWidth: 0`, which is the same state a 404 produces and exactly
    what the test below is here to tell apart. Bytes with a checksum over them cannot be
    written from memory, so they are computed.
    """

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    # Bit depth 8, colour type 0 (greyscale), no interlace. Each row carries a leading filter
    # byte, which is what makes the raw size height * (1 + width) rather than height * width.
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def image_workspace_ui_server(tmp_path: Path) -> Iterator[UIServerFactory]:
    """A UI server whose workspace holds the artifact the scripted reply links to.

    `scripted_reply_ui_server` leaves `workspace_dir` at the app's default, the process's cwd,
    so an artifact request under it would read the developer's own checkout -- present on this
    machine and absent in CI, which is the class of difference
    `running_ui`'s docstring records for `eval_reports_dir`. The file is written here instead,
    and the server is pointed at the directory holding it.
    """
    artifact = tmp_path / ARTIFACT_PATH
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(_png_bytes(IMAGE_WIDTH, IMAGE_WIDTH))

    with ExitStack() as stack:

        def _start(reply: str) -> str:
            llm = MockLLMConnector(default_model="mock-gpt-4o", default_response=reply)
            return stack.enter_context(
                running_ui(storage_dir=tmp_path / "state", llm=llm, workspace_dir=tmp_path)
            )

        yield _start


async def _open_room_and_say(page: Page, base_url: str, words: str) -> None:
    """Seat a conversation, open it, send one message and wait for the reply to land."""
    response = await page.request.post(
        f"{base_url}/api/rooms", data={"title": "Studio", "agent_ids": ["scout"]}
    )
    assert response.ok, await response.text()
    room_id = str((await response.json())["room_id"])

    await page.set_viewport_size({"width": 1280, "height": 900})
    await page.goto(base_url, wait_until="commit")
    await page.click(f"[data-testid='conversation-{room_id}']", timeout=15000)
    await page.wait_for_selector("[data-testid='room-conversation']", timeout=15000)

    await page.fill("[data-testid='room-composer']", words)
    await page.click("[data-testid='send-message']")
    # Rows 1 and 2 are the joins, 3 is this message and 4 is the reply to it.
    await page.wait_for_selector("[data-testid='row-4']", timeout=20000)


@pytest.mark.asyncio
async def test_a_reply_that_links_a_generated_image_shows_the_image(
    image_workspace_ui_server: UIServerFactory,
) -> None:
    """The picture is drawn in the row, decoded from the server's own bytes.

    Three assertions for three ways this has been wrong: the `<img>` exists at all (it did not,
    from #1208 until now), it points at the artifact the reply named (a renderer that invented
    an address would satisfy the first), and the browser decoded what came back (jsdom cannot
    tell a served file from a 404, so the component test cannot make this one).

    The prose is asserted too. Upgrading the link must not eat the paragraph around it: the
    YouTube card this follows replaces its link, and a reader losing the description of the
    picture to gain the picture is a different regression, not a fix.

    Mutation-checked by hand rather than as a kill declaration, because the lethality ratchet
    runs a browser test against the committed bundle without rebuilding it (the convention
    `test_room_markdown_e2e.py` states). Rebuilt with `npm run build`, this fails when
    `frontend/src/components/RichText.tsx`'s `const artifact = findArtifactImage(href);`
    becomes `const artifact = findArtifactImage(null);` -- the state that shipped. Measured:
    the sibling case below still passes under it. (`= null` is the shorter way to say it and
    does not compile; `tsc` narrows the binding to `never` and the build fails before the
    browser is reached, which is not the same evidence.)
    """
    base_url = image_workspace_ui_server(IMAGE_REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, "Draw me a knight")

            row = page.locator("[data-testid='row-body-4']")
            image = row.locator("[data-testid='inline-artifact-image']")
            await image.wait_for(timeout=10000)

            src = await image.get_attribute("src")
            assert src is not None and ARTIFACT_PATH in src, (
                f"the image does not address the artifact the reply named: {src!r}"
            )

            decoded = await image.evaluate(
                "(el) => el.complete ? el.naturalWidth : -1",
            )
            assert decoded == IMAGE_WIDTH, (
                f"the picture did not load from the server (naturalWidth {decoded}); "
                "an <img> on screen is not an image on screen"
            )

            text = await row.text_content() or ""
            assert "Need adjustments" in text, "the reply's prose was lost with the link"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_reply_with_image_tag_shows_the_image(
    image_workspace_ui_server: UIServerFactory,
) -> None:
    """A reply wrapped in <image>...</image> tags draws the picture rather than raw tags."""
    base_url = image_workspace_ui_server(TAG_WRAPPED_IMAGE_REPLY)
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, "Draw me an illustration")

            row = page.locator("[data-testid='row-body-4']")
            image = row.locator("[data-testid='inline-artifact-image']")
            await image.wait_for(timeout=10000)

            src = await image.get_attribute("src")
            assert src is not None and ARTIFACT_PATH in src, (
                f"the image does not address the artifact: {src!r}"
            )

            decoded = await image.evaluate(
                "(el) => el.complete ? el.naturalWidth : -1",
            )
            assert decoded == IMAGE_WIDTH, (
                f"the picture did not load from the server (naturalWidth {decoded})"
            )

            text = await row.text_content() or ""
            assert "<image>" not in text and "</image>" not in text, (
                f"raw <image> tags leaked into row text: {text!r}"
            )
            assert "artistic-style illustration" in text, "the reply's prose was lost"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_turn_that_sends_no_text_says_so_rather_than_drawing_an_empty_bubble(
    image_workspace_ui_server: UIServerFactory,
) -> None:
    """A silent turn states what happened (P6).

    The same room recorded three of these on 2026-09-21 -- `completed: true`, `error: null`,
    `content: ""` -- while the image tool was writing files. Each drew a bubble with nothing in
    it, which a reader has no way to tell from the head failing to render.

    The row's *text* is asserted, not the testid alone: an element that exists and says nothing
    is the defect, and a testid on an empty `<div>` satisfies a presence check.

    Mutation-checked by hand, as the case above. Rebuilt, this fails and the case above passes
    when `frontend/src/components/rooms/RoomConversation.tsx`'s
    `) : message.content.trim() === '' ? (` becomes `) : false ? (`.
    """
    base_url = image_workspace_ui_server("")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await _open_room_and_say(page, base_url, "Draw me a knight")

            row = page.locator("[data-testid='row-body-4']")
            said = (await row.text_content() or "").strip()
            assert "without sending any text" in said or "sent no text" in said, (
                f"a silent turn rendered as {said!r}"
            )
        finally:
            await browser.close()
