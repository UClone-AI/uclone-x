"""`BrowserService`: the Core's browser, tabs per (conversation, clone) (design §3.1, §3.3).

One service per Core process. It opens its link on the first call, so a conversation that
never browses starts no Chrome (§5). Every method takes a `TabKey` and works on that key's
current tab, opening one when `open` is the first call. A page that opens a pop-up gives
the key a second tab, which becomes current; `tab` lists, switches and closes them.

Each action returns an observation (§3.2): the URL and title, then a diff of the snapshot
when the page stayed or a fresh snapshot when it navigated, and any dialog, new tab or
download that appeared.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import urllib.parse
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from uclone_x.browser.actions import (
    ACTIONABLE_JS,
    ACTIVE_FIELD,
    CARET_TO_END_JS,
    ELEMENT_SCROLL_JS,
    ELEMENT_SCROLL_POSITION_JS,
    FIELD_INFO_JS,
    IS_FOCUSED_JS,
    PAGE_SCROLL_POSITION,
    POPUP_NOT_FOLLOWED,
    SECRET_FIELD_REFUSAL,
    SELECT_ALL_JS,
    SELECT_OPTIONS_JS,
    STALE_REF,
    dialog_note,
    is_secret_field,
    key_events,
    parse_chord,
    pick_option,
    scroll_delta,
    wait_for_text_expression,
)
from uclone_x.browser.cdp import CdpClosedError, CdpError, CdpEvent
from uclone_x.browser.link import BrowserLink
from uclone_x.browser.snapshot import Entry, RefTable, build_entries, diff, find, render
from uclone_x.browser.vision import View, capture, page_point
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.models import ImagePart
from uclone_x.tools.builtin.web import html_to_markdown

logger = logging.getLogger(__name__)

TabKey = tuple[str, str]
"""(conversation id, clone id): whose tabs they are."""

Notice = dict[str, Any]
"""What the dock's Browser tab is told about a conversation's browser (design §3.4, §3.6):
`{"type": "state"}` (re-read `live_state`), `{"type": "step", ...}` when a call ends, and
`{"type": "overlay", ...}` when an action is about to touch an element."""

#: Refusals that say the browser itself is unavailable, as the codes the Browser tab words.
_PROBLEMS = {"no_chrome": "chrome_missing", "browser_closed": "browser_closed"}

LOAD_TIMEOUT_S = 10.0
NETWORK_IDLE_CAP_S = 3.0
QUIET_S = 0.5
DOWNLOAD_WAIT_S = 10.0
ACTIONABLE_TIMEOUT_MS = 5000
WAIT_CAP_MS = 10_000
POPUP_LOOKUP_S = 2.0
DEFAULT_DOWNLOADS_DIR = Path.home() / ".uclone" / "browser" / "downloads"

Direction = Literal["up", "down", "left", "right"]

_NAVIGATION_ERRORS = {
    "net::ERR_NAME_NOT_RESOLVED": "no site answers at that address",
    "net::ERR_CONNECTION_REFUSED": "the site refused the connection",
    "net::ERR_CONNECTION_TIMED_OUT": "the site did not answer in time",
    "net::ERR_INTERNET_DISCONNECTED": "this computer is not connected to the internet",
    "net::ERR_CERT_AUTHORITY_INVALID": "the site's security certificate is not trusted",
}
_NOT_ACTIONABLE = {
    "detached": STALE_REF,
    "hidden": "That element is not visible on the page right now.",
    "disabled": "That element is disabled right now, so it can't be used yet.",
    "moving": "That element kept moving, so it could not be used. Wait for the page to "
    "finish, then try again.",
    "covered": "Something on the page covers that element, perhaps a banner or a dialog. "
    "Close it first, then try again.",
}
_BROWSER_FAILED = (
    "The browser could not do that on this page. If the clone's tab was closed, "
    "open the page again with action=open."
)
_BROWSER_CLOSED = "The browser window was closed. Open the page again with action=open."


@dataclass
class _Tab:
    target: str
    events: asyncio.Queue[CdpEvent]
    refs: RefTable = field(default_factory=RefTable)
    loader: str = ""
    frame: str = ""
    entries: list[Entry] = field(default_factory=lambda: [])
    url: str = ""
    title: str = ""
    dialog: dict[str, Any] | None = None  # an open JavaScript dialog
    pending: asyncio.Future[Any] | None = None  # input the dialog froze mid-delivery
    controller: str = "clone"  # "user" while U0 has taken over in the dock, else clone id
    controller_resumed: asyncio.Event = field(default_factory=asyncio.Event)
    visited_while_user: list[tuple[str, str]] = field(
        default_factory=lambda: cast(list[tuple[str, str]], [])
    )
    hand_over_pending: bool = False
    asked_user: dict[str, str] | None = None
    view: View | None = None
    """What the last `look` showed, so a coordinate click can be put back on the page."""

    def __post_init__(self) -> None:
        if self.controller != "user":
            self.controller_resumed.set()


@dataclass
class _Window:
    """One key's tabs, and which of them its calls act on."""

    tabs: list[_Tab]
    active: int = 0

    @property
    def tab(self) -> _Tab:
        return self.tabs[self.active]


BrowserTab = _Tab
BrowserWindow = _Window


class BrowserService:
    """Tabs, observations and actions over one `BrowserLink`."""

    def __init__(
        self,
        link_factory: Callable[[], Awaitable[BrowserLink]],
        *,
        downloads_dir: Path = DEFAULT_DOWNLOADS_DIR,
    ) -> None:
        self._link_factory = link_factory
        self._link: BrowserLink | None = None
        self._link_lock = asyncio.Lock()
        self._windows: dict[TabKey, _Window] = {}
        self._downloads_dir = downloads_dir
        self._overlay_listeners: list[Callable[[str, dict[str, Any], str], Any]] = []
        self._screencast_subscribers: dict[str, int] = {}
        # The live view (step 3): who is watching each conversation, which call each key
        # is making, the step it will report, and why the browser is unavailable, if it is.
        self._watchers: dict[str, list[asyncio.Queue[Notice]]] = {}
        self._acting: dict[TabKey, str] = {}
        self._steps: dict[TabKey, Notice] = {}
        self._problems: dict[str, str] = {}

    @property
    def link(self) -> BrowserLink | None:
        """The active CDP link to Chrome, or None if not opened yet."""
        return self._link

    def acquire_screencast(self, target: str) -> int:
        """Record one more subscriber for screencast frames on `target`."""
        count = self._screencast_subscribers.get(target, 0) + 1
        self._screencast_subscribers[target] = count
        return count

    def release_screencast(self, target: str) -> int:
        """Release one subscriber for screencast frames on `target`; return remaining count."""
        count = self._screencast_subscribers.get(target, 0)
        if count <= 1:
            self._screencast_subscribers.pop(target, None)
            return 0
        count -= 1
        self._screencast_subscribers[target] = count
        return count

    def get_room_window(self, room_id: str) -> BrowserWindow | None:
        """The window for `room_id` if any clone has opened one."""
        for (r_id, _), window in self._windows.items():
            if r_id == room_id:
                return window
        return None

    def register_overlay_listener(
        self, callback: Callable[[str, dict[str, Any], str], Any]
    ) -> Callable[[], None]:
        """Register a callback `(room_id, box, label)` invoked before element actions."""
        self._overlay_listeners.append(callback)

        def unregister() -> None:
            if callback in self._overlay_listeners:
                self._overlay_listeners.remove(callback)

        return unregister

    async def broadcast_overlay(
        self,
        room_id: str,
        box: dict[str, Any],
        label: str,
        *,
        tab: _Tab | None = None,
        clone: str = "",
    ) -> None:
        """Broadcast an action overlay to registered listeners and the live view's watchers."""
        if tab is not None:
            step = self._steps.get((room_id, clone), {})
            self._notify(
                room_id,
                {
                    "type": "overlay",
                    "clone": clone,
                    "target": tab.target,
                    "box": box,
                    "label": label,
                    "action": str(step.get("action", "")),
                },
            )
        for listener in list(self._overlay_listeners):
            try:
                res = listener(room_id, box, label)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logger.warning("overlay listener failed for %s", room_id, exc_info=True)

    def _element_name(self, tab: _Tab, ref: str) -> str:
        for entry in tab.entries:
            if entry.ref == ref:
                return entry.name or entry.role
        return ""

    # -- the live view (step 3) ------------------------------------------------------

    def watch(self, conversation: str) -> asyncio.Queue[Notice]:
        """A queue of the conversation's notices, for the dock's Browser tab."""
        queue: asyncio.Queue[Notice] = asyncio.Queue()
        self._watchers.setdefault(conversation, []).append(queue)
        return queue

    def unwatch(self, conversation: str, queue: asyncio.Queue[Notice]) -> None:
        """Stop telling `queue` about the conversation."""
        queues = self._watchers.get(conversation, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._watchers.pop(conversation, None)

    def _notify(self, conversation: str, notice: Notice) -> None:
        for queue in self._watchers.get(conversation, []):
            queue.put_nowait(notice)

    @contextlib.asynccontextmanager
    async def acting(self, key: TabKey, action: str) -> AsyncGenerator[Notice]:
        """Mark one browser call as in progress for the live view, and report its step.

        Yields the step notice, which the call fills in (the element it touched is set by
        `_element`); it is sent when the call ends, with `ok` saying whether it worked.
        """
        conversation, clone = key
        step: Notice = {"type": "step", "clone": clone, "action": action, "ok": True}
        self._acting[key] = action
        self._steps[key] = step
        self._notify(conversation, {"type": "state"})
        try:
            yield step
            self._problems.pop(conversation, None)
        except PlainRefusalError as refused:
            step["ok"] = False
            problem = _PROBLEMS.get(refused.reason_code or "")
            if problem is not None:
                self._problems[conversation] = problem
            raise
        except BaseException:
            step["ok"] = False
            raise
        finally:
            self._acting.pop(key, None)
            self._steps.pop(key, None)
            self._notify(conversation, step)
            self._notify(conversation, {"type": "state"})

    def live_state(self, conversation: str) -> Notice:
        """The conversation's tabs, who is acting, and any problem: codes, never copy."""
        tabs: list[dict[str, Any]] = []
        for (owner, clone), window in self._windows.items():
            if owner != conversation:
                continue
            for number, tab in enumerate(window.tabs, start=1):
                tabs.append(
                    {
                        "clone": clone,
                        "index": number,
                        "url": tab.url,
                        "title": tab.title,
                        "current": number == window.active + 1,
                        "controller": "user" if tab.controller == "user" else "clone",
                    }
                )
        acting = next(
            (
                {"clone": clone, "action": action}
                for (owner, clone), action in self._acting.items()
                if owner == conversation
            ),
            None,
        )
        asked_user: dict[str, Any] | None = None
        for (owner, _), window in self._windows.items():
            if owner == conversation:
                for tab in window.tabs:
                    if tab.asked_user:
                        asked_user = tab.asked_user
                        break
        state: dict[str, Any] = {
            "type": "state",
            "tabs": tabs,
            "acting": acting,
            "problem": self._problems.get(conversation),
        }
        if asked_user is not None:
            state["asked_user"] = asked_user
        return state

    async def wait_for_control(self, key: TabKey) -> None:
        """Wait until control returns to the clone if U0 has taken over the active tab."""
        window = self._windows.get(key)
        if window is None or not window.tabs:
            return
        tab = window.tab
        while tab.controller == "user":
            await tab.controller_resumed.wait()

    def _resolve_tab(
        self, conversation: str, clone: str | None = None, index: int | None = None
    ) -> _Tab | None:
        if clone is not None and index is not None:
            return self.live_tab(conversation, clone, index)
        for (r_id, cl), win in self._windows.items():
            if r_id == conversation and win.tabs:
                if clone is None or cl == clone:
                    return win.tab
        return None

    def take_over(
        self, conversation: str, clone: str | None = None, index: int | None = None
    ) -> None:
        """Give control of the conversation's active or specified tab to U0."""
        tab = self._resolve_tab(conversation, clone, index)
        if tab is not None:
            tab.controller = "user"
            tab.controller_resumed.clear()
            tab.asked_user = None
            self._notify(conversation, {"type": "state"})

    def give_back(
        self, conversation: str, clone: str | None = None, index: int | None = None
    ) -> None:
        """Return control of the conversation's active or specified tab to the clone."""
        tab = self._resolve_tab(conversation, clone, index)
        if tab is not None:
            target_clone = clone
            if target_clone is None or target_clone == "user":
                key = self._key_of(tab)
                target_clone = key[1] if key and key[1] != "user" else "clone"
            tab.controller = target_clone
            tab.hand_over_pending = True
            tab.asked_user = None
            tab.controller_resumed.set()
            self._notify(conversation, {"type": "state"})

    def hand_to(self, conversation: str, target_clone: str, index: int | None = None) -> None:
        """Transfer tab control to another clone."""
        current_key: TabKey | None = None
        for (r_id, cl), win in list(self._windows.items()):
            if r_id == conversation and win.tabs:
                current_key = (r_id, cl)
                break
        if current_key is not None:
            window = self._windows[current_key]
            tab = window.tab
            tab.controller = target_clone
            tab.hand_over_pending = True
            tab.asked_user = None
            tab.controller_resumed.set()
            if current_key != (conversation, target_clone):
                window.tabs.remove(tab)
                if not window.tabs:
                    self._windows.pop(current_key)
                else:
                    window.active = max(0, min(window.active, len(window.tabs) - 1))
                if (conversation, target_clone) in self._windows:
                    target_win = self._windows[(conversation, target_clone)]
                    target_win.tabs.append(tab)
                    target_win.active = len(target_win.tabs) - 1
                else:
                    self._windows[(conversation, target_clone)] = _Window(tabs=[tab], active=0)
            self._notify(conversation, {"type": "state"})

    async def ask_user(
        self, key: TabKey, kind: str = "sign_in", message: str | None = None
    ) -> dict[str, Any]:
        """Ask U0 to take over the tab (e.g. for sign-in), and wait for give_back."""
        conversation, clone = key
        await self._ensure_link()
        _, _, tab = self._existing(key)
        import urllib.parse

        domain = urllib.parse.urlparse(tab.url).netloc or tab.url
        tab.controller = "user"
        tab.controller_resumed.clear()
        tab.asked_user = {
            "clone": clone,
            "kind": kind,
            "site": domain,
            "message": message or "",
        }
        self._notify(conversation, {"type": "state"})
        await tab.controller_resumed.wait()
        tab.asked_user = None
        snap = await self.snapshot(key)
        return {
            "status": "ok",
            "message": f"Signed in — now at {tab.url}",
            **snap,
        }

    async def open_user_tab(self, conversation: str, url: str = "about:blank") -> _Tab:
        """Open a new tab owned by U0 (design §3.4)."""
        link = await self._ensure_link()
        key: TabKey = (conversation, "user")
        window = self._windows.get(key)
        if window is None:
            tab = await self._new_tab(link)
            tab.controller = "user"
            tab.controller_resumed.clear()
            window = _Window(tabs=[tab], active=0)
            self._windows[key] = window
        else:
            tab = await self._new_tab(link)
            tab.controller = "user"
            tab.controller_resumed.clear()
            window.tabs.append(tab)
            window.active = len(window.tabs) - 1
        if url and url != "about:blank":
            if not (
                url.startswith("http://") or url.startswith("https://") or url.startswith("about:")
            ):
                url = f"https://{url}"
            await self._navigate(link, tab, url)
        self._notify(conversation, {"type": "state"})
        return tab

    async def navigate_user_tab(self, conversation: str, url: str) -> None:
        """Navigate the active tab if it is in user control."""
        if not url:
            return
        link = await self._ensure_link()
        for (r_id, _), window in self._windows.items():
            if r_id == conversation and window.tabs:
                tab = window.tab
                if tab.controller == "user":
                    if not (
                        url.startswith("http://")
                        or url.startswith("https://")
                        or url.startswith("about:")
                    ):
                        url = f"https://{url}"
                    await self._navigate(link, tab, url)
                    if not tab.visited_while_user or tab.visited_while_user[-1][0] != tab.url:
                        tab.visited_while_user.append((tab.url, tab.title or tab.url))
                    self._notify(conversation, {"type": "state"})
                    return

    def live_tab(self, conversation: str, clone: str, index: int) -> _Tab | None:
        """One of the conversation's tabs, by clone and number from 1."""
        window = self._windows.get((conversation, clone))
        if window is None or not 1 <= index <= len(window.tabs):
            return None
        return window.tabs[index - 1]

    def live_target(
        self, conversation: str, clone: str, index: int
    ) -> tuple[BrowserLink, str] | None:
        """The link and target of one of the conversation's tabs, to screencast it."""
        tab = self.live_tab(conversation, clone, index)
        if tab is None or self._link is None:
            return None
        return self._link, tab.target

    def _key_of(self, tab: _Tab) -> TabKey | None:
        for key, window in self._windows.items():
            if any(t is tab for t in window.tabs):
                return key
        return None

    # -- step 1: observations --------------------------------------------------------

    async def open(self, key: TabKey, url: str) -> dict[str, Any]:
        """Load `url` in the key's tab, opening the tab if needed, and return a snapshot."""
        _check_url(url)
        try:
            return await self._guarded(self._open(key, url))
        except PlainRefusalError as refused:
            if refused.reason_code != "browser_closed":
                raise
            # Chrome quit since the last call: start it again once.
            return await self._guarded(self._open(key, url))

    async def snapshot(self, key: TabKey) -> dict[str, Any]:
        """The accessibility snapshot of the key's page."""

        async def run() -> dict[str, Any]:
            link, _, tab = self._existing(key)
            await self._track_page_load(link, tab)
            tab.entries = await self._entries(link, tab)
            return _consume_hand_over(
                tab, {**await self._location(link, tab), "snapshot": render(tab.entries)}
            )

        return await self._guarded(run())

    async def read(self, key: TabKey, max_length: int) -> dict[str, Any]:
        """The page's content as Markdown, from the live DOM after scripts ran."""

        async def run() -> dict[str, Any]:
            link, _, tab = self._existing(key)
            html = await self._evaluate(link, tab, "document.documentElement.outerHTML")
            content = html_to_markdown(html if isinstance(html, str) else "")
            return _consume_hand_over(
                tab,
                {
                    **await self._location(link, tab),
                    "content": content[:max_length],
                    "truncated": len(content) > max_length,
                },
            )

        return await self._guarded(run())

    async def find(self, key: TabKey, query: str) -> dict[str, Any]:
        """Elements anywhere on the page whose text matches `query`, with their refs."""

        async def run() -> dict[str, Any]:
            link, _, tab = self._existing(key)
            await self._track_page_load(link, tab)
            hits = find(
                build_entries(await self._ax_nodes(link, tab), tab.refs, collapse=False), query
            )
            found = (
                "\n".join(e.line() for e in hits)
                if hits
                else f'Nothing on the page matches "{query}".'
            )
            return {**await self._location(link, tab), "matches": found}

        return await self._guarded(run())

    async def look(self, key: TabKey) -> tuple[dict[str, Any], ImagePart]:
        """A picture of the key's viewport, and where the page is (design §3.2, step 5)."""
        taken: list[ImagePart] = []

        async def run() -> dict[str, Any]:
            link, _, tab = self._existing(key)
            await self._track_page_load(link, tab)
            image, tab.view = await capture(link, tab.target, tab.loader)
            taken.append(image)
            return {
                **await self._location(link, tab),
                "width": image.width,
                "height": image.height,
            }

        where = await self._guarded(run())
        return where, taken[0]

    async def click_at(
        self,
        key: TabKey,
        x: int,
        y: int,
        *,
        double: bool = False,
        right: bool = False,
        hover: bool = False,
    ) -> dict[str, Any]:
        """Click the point (`x`, `y`) of the last `look`'s picture, in its pixels.

        For what only the picture shows -- a canvas, an image-only button. Returns the
        observation any action returns.
        """

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            loader = await self._read_frame(link, tab)
            css_x, css_y = await page_point(link, tab.target, tab.view, loader, x, y)
            # The picture showed the page before the click; a second click needs a new one.
            tab.view = None
            button = "right" if right else "left"  # coordinate button
            clicks = 2 if double else 1  # coordinate clicks
            seen = await self._perform(
                tab,
                lambda: self._mouse(
                    link, tab, css_x, css_y, button=button, clicks=clicks, hover=hover
                ),
            )
            return await self._observe(link, window, seen)

        return await self._guarded(run())

    async def close(self) -> None:
        """Drop the link. Chrome and its tabs stay open for U0."""
        link, self._link = self._link, None
        self._windows.clear()
        if link is not None:
            await link.close()

    # -- step 2: actions -------------------------------------------------------------

    async def click(
        self,
        key: TabKey,
        ref: str,
        *,
        double: bool = False,
        right: bool = False,
        hover: bool = False,
    ) -> dict[str, Any]:
        """Click (or double-click, right-click, hover over) the element `ref` names."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            backend, element = await self._element(link, tab, ref)
            (x, y), box = await self._point(link, tab, element, backend)
            name = self._element_name(tab, ref)
            await self.broadcast_overlay(key[0], box, name, tab=tab, clone=key[1])
            button = "right" if right else "left"
            seen = await self._perform(
                tab,
                lambda: self._mouse(
                    link, tab, x, y, button=button, clicks=2 if double else 1, hover=hover
                ),
            )
            return await self._observe(link, window, seen)

        return await self._guarded(run())

    async def type_text(
        self,
        key: TabKey,
        ref: str | None,
        text: str,
        *,
        submit: bool = False,
        append: bool = False,
    ) -> dict[str, Any]:
        """Type `text` into a field, clearing it first unless `append`; or answer a prompt."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key, allow_dialog=True)
            if tab.dialog is not None:
                if tab.dialog.get("type") == "prompt":
                    return await self._answer_dialog(link, window, tab, True, text)
                raise PlainRefusalError(dialog_note(tab.dialog), reason_code="dialog_open")
            if not ref:
                raise PlainRefusalError(
                    "type needs the ref of the field to type into.", reason_code="missing_ref"
                )
            backend, element = await self._element(link, tab, ref)
            info = await self._field(link, tab, element)
            if is_secret_field(str(info.get("type", "")), str(info.get("autocomplete", ""))):
                raise PlainRefusalError(SECRET_FIELD_REFUSAL, reason_code="secret_field")
            if not info.get("editable"):
                raise PlainRefusalError(
                    "That element is not a text field, so it can't be typed into. "
                    "Use click, select or check instead.",
                    reason_code="not_a_text_field",
                )
            if info.get("disabled"):
                raise PlainRefusalError(_NOT_ACTIONABLE["disabled"], reason_code="disabled")
            if info.get("readonly"):
                raise PlainRefusalError(
                    "That field is read-only, so it can't be typed into.", reason_code="readonly"
                )
            (x, y), box = await self._point(link, tab, element, backend)
            name = self._element_name(tab, ref)
            await self.broadcast_overlay(key[0], box, name, tab=tab, clone=key[1])

            async def deliver() -> None:
                await self._mouse(link, tab, x, y)
                if not await self._call(link, tab, element, IS_FOCUSED_JS):
                    await link.send(tab.target, "DOM.focus", {"backendNodeId": backend})
                if append:
                    if not await self._call(link, tab, element, CARET_TO_END_JS):
                        await self._keys(link, tab, "End")
                else:
                    await self._call(link, tab, element, SELECT_ALL_JS)
                    await self._keys(link, tab, "Backspace")
                if text:
                    # Inserted as text, not key events, so Korean and other IME text arrives.
                    await link.send(tab.target, "Input.insertText", {"text": text})
                if submit:
                    await self._keys(link, tab, "Enter")

            seen = await self._perform(tab, deliver)
            return await self._observe(link, window, seen)

        return await self._guarded(run())

    async def select(self, key: TabKey, ref: str, options: Sequence[str]) -> dict[str, Any]:
        """Choose `options` in a list: a native `<select>`, or a custom one opened by a click."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            backend, element = await self._element(link, tab, ref)
            info = await self._field(link, tab, element)
            if info.get("select"):
                if info.get("disabled"):
                    raise PlainRefusalError(_NOT_ACTIONABLE["disabled"], reason_code="disabled")
                outcome: dict[str, Any] = {}

                async def choose() -> None:
                    value = await self._call(link, tab, element, SELECT_OPTIONS_JS, list(options))
                    outcome.update(cast(dict[str, Any], value) if isinstance(value, dict) else {})

                seen = await self._perform(tab, choose)
                _refuse_unmatched(outcome)
                return await self._observe(link, window, seen)
            return await self._select_custom(link, window, tab, element, backend, options)

        return await self._guarded(run())

    async def check(self, key: TabKey, ref: str, *, on: bool = True) -> dict[str, Any]:
        """Turn a checkbox, radio button or switch on or off, clicking only if it differs."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            backend, element = await self._element(link, tab, ref)
            info = await self._field(link, tab, element)
            state = info.get("checked")
            if not isinstance(state, bool):
                raise PlainRefusalError(
                    "That element is not a checkbox, radio button or switch. Use click instead.",
                    reason_code="not_checkable",
                )
            if info.get("radio") and not on:
                raise PlainRefusalError(
                    "A radio button can't be turned off on its own; check another option "
                    "in its group instead.",
                    reason_code="radio_off",
                )
            word = "checked" if on else "unchecked"
            if state == on:
                return await self._observe(link, window, Settled(), note=f"It was already {word}.")
            (x, y), box = await self._point(link, tab, element, backend)
            name = self._element_name(tab, ref)
            await self.broadcast_overlay(key[0], box, name, tab=tab, clone=key[1])
            seen = await self._perform(tab, lambda: self._mouse(link, tab, x, y))
            note = None
            if tab.dialog is None and not seen.navigated:
                after = await self._field(link, tab, element)
                if after.get("checked") != on:
                    note = f"Clicking it did not leave it {word}; the page may not allow that now."
            return await self._observe(link, window, seen, note=note)

        return await self._guarded(run())

    async def press(self, key: TabKey, chord: str) -> dict[str, Any]:
        """Press a key or chord in the page; Enter or Escape answers an open dialog."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key, allow_dialog=True)
            press = parse_chord(chord)
            if tab.dialog is not None:
                if press.key == "Enter" and not press.modifiers:
                    return await self._answer_dialog(link, window, tab, True)
                if press.key == "Escape" and not press.modifiers:
                    return await self._answer_dialog(link, window, tab, False)
                raise PlainRefusalError(dialog_note(tab.dialog), reason_code="dialog_open")
            if press.text and press.text != "\r":
                active = await self._evaluate(link, tab, ACTIVE_FIELD)
                fields = cast(dict[str, Any], active) if isinstance(active, dict) else {}
                if is_secret_field(
                    str(fields.get("type", "")), str(fields.get("autocomplete", ""))
                ):
                    raise PlainRefusalError(SECRET_FIELD_REFUSAL, reason_code="secret_field")
            seen = await self._perform(tab, lambda: self._send_keys(link, tab, key_events(press)))
            return await self._observe(link, window, seen)

        return await self._guarded(run())

    async def scroll(
        self, key: TabKey, ref: str | None, direction: Direction, amount: float
    ) -> dict[str, Any]:
        """Scroll the page, or the scrollable element `ref` names, by `amount` screens."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            element: str | None = None
            if ref:
                _, element = await self._element(link, tab, ref)
                box = await self._call(link, tab, element, ELEMENT_SCROLL_JS)
                area = cast(dict[str, float], box) if isinstance(box, dict) else {}
                x, y = area.get("x", 1.0), area.get("y", 1.0)
                width, height = area.get("width", 0.0), area.get("height", 0.0)
            else:
                metrics = await link.send(tab.target, "Page.getLayoutMetrics")
                view = cast(dict[str, float], metrics.get("cssLayoutViewport", {}))
                width, height = view.get("clientWidth", 800.0), view.get("clientHeight", 600.0)
                x, y = width / 2, height / 2
            dx, dy = scroll_delta(direction, amount, width, height)
            wheel = {"type": "mouseWheel", "x": x, "y": y, "deltaX": dx, "deltaY": dy}
            seen = await self._perform(
                tab, lambda: link.send(tab.target, "Input.dispatchMouseEvent", wheel)
            )
            note = None
            if tab.dialog is None and not seen.navigated:
                note = await self._scroll_note(link, tab, element, direction)
            return await self._observe(link, window, seen, note=note)

        return await self._guarded(run())

    async def upload(self, key: TabKey, ref: str, files: Sequence[Path]) -> dict[str, Any]:
        """Answer the page's file chooser with `files` (already resolved in the workspace)."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            backend, element = await self._element(link, tab, ref)
            info = await self._field(link, tab, element)
            paths = [str(p) for p in files]
            names = ", ".join(p.name for p in files)
            name = self._element_name(tab, ref)
            if info.get("file"):
                seen = await self._perform(
                    tab,
                    lambda: link.send(
                        tab.target,
                        "DOM.setFileInputFiles",
                        {"files": paths, "backendNodeId": backend},
                    ),
                )
                return await self._observe(link, window, seen, note=f"Chose {names}.")
            (x, y), box = await self._point(link, tab, element, backend)
            await self.broadcast_overlay(key[0], box, name, tab=tab, clone=key[1])
            await link.send(tab.target, "Page.setInterceptFileChooserDialog", {"enabled": True})
            try:
                seen = await self._perform(tab, lambda: self._mouse(link, tab, x, y))
            finally:
                if tab.dialog is None:
                    await link.send(
                        tab.target, "Page.setInterceptFileChooserDialog", {"enabled": False}
                    )
            chooser = seen.chooser
            if chooser is None:
                if tab.dialog is not None or seen.navigated:
                    return await self._observe(link, window, seen)
                raise PlainRefusalError(
                    "Clicking that element did not open a file picker. Use the page's upload "
                    "button or file field.",
                    reason_code="no_file_chooser",
                )
            chosen = await self._perform(
                tab,
                lambda: link.send(
                    tab.target,
                    "DOM.setFileInputFiles",
                    {"files": paths, "backendNodeId": chooser},
                ),
            )
            return await self._observe(link, window, chosen, note=f"Chose {names}.")

        return await self._guarded(run())

    async def wait(
        self,
        key: TabKey,
        *,
        text: str | None = None,
        ref: str | None = None,
        ms: int | None = None,
    ) -> dict[str, Any]:
        """Wait for text to show, an element to be ready, or a fixed time, up to 10 s."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            seconds = WAIT_CAP_MS // 1000
            if text:
                wanted = text
                found: list[bool] = []

                async def watch() -> None:
                    result = await link.send(
                        tab.target,
                        "Runtime.evaluate",
                        {
                            "expression": wait_for_text_expression(wanted, WAIT_CAP_MS),
                            "awaitPromise": True,
                            "returnByValue": True,
                        },
                    )
                    found.append(_value_of(result) is True)

                seen = await self._perform(tab, watch)
                note = (
                    f'"{text}" is on the page.'
                    if found and found[0]
                    else f'"{text}" did not appear on the page within {seconds} seconds.'
                )
                return await self._observe(link, window, seen, note=note)
            if ref:
                _, element = await self._element(link, tab, ref)
                ready: list[str] = []

                async def watch_element() -> None:
                    value = await self._call(
                        link, tab, element, ACTIONABLE_JS, WAIT_CAP_MS, False, await_promise=True
                    )
                    outcome = cast(dict[str, Any], value) if isinstance(value, dict) else {}
                    ready.append(str(outcome.get("reason", "hidden")))

                seen = await self._perform(tab, watch_element)
                reason = ready[0] if ready else "hidden"
                note = (
                    "The element is visible and ready."
                    if not reason
                    else f"After {seconds} seconds: {_NOT_ACTIONABLE.get(reason, STALE_REF)}"
                )
                return await self._observe(link, window, seen, note=note)
            delay = min(ms or 0, WAIT_CAP_MS) / 1000
            seen = await self._perform(tab, lambda: asyncio.sleep(delay))
            return await self._observe(link, window, seen)

        return await self._guarded(run())

    async def back(self, key: TabKey) -> dict[str, Any]:
        """Go to the previous page in the tab's history."""
        return await self._guarded(self._history(key, -1))

    async def forward(self, key: TabKey) -> dict[str, Any]:
        """Go to the next page in the tab's history."""
        return await self._guarded(self._history(key, 1))

    async def reload(self, key: TabKey) -> dict[str, Any]:
        """Load the current page again."""

        async def run() -> dict[str, Any]:
            link, window, tab = self._existing(key)
            seen = await self._perform(tab, lambda: link.send(tab.target, "Page.reload"))
            return await self._observe(link, window, seen)

        return await self._guarded(run())

    async def tab(
        self, key: TabKey, *, index: int | None = None, close: bool = False
    ) -> dict[str, Any]:
        """List the clone's tabs; with `index`, switch to one; with `close`, close one."""

        async def run() -> dict[str, Any]:
            link, window, _ = self._existing(key, allow_dialog=True)
            count = len(window.tabs)
            if index is not None and not 1 <= index <= count:
                raise PlainRefusalError(
                    f"There is no tab {index}; this clone has {_tabs_word(count)} open.",
                    reason_code="no_such_tab",
                )
            chosen = index - 1 if index is not None else window.active
            if close:
                closing = window.tabs.pop(chosen)
                link.unsubscribe(closing.target, closing.events)
                with contextlib.suppress(CdpError, KeyError):
                    await link.close_tab(closing.target)
                if not window.tabs:
                    del self._windows[key]
                    return {"tabs": "No tabs are open now. Open a page with action=open."}
                if chosen < window.active or window.active >= len(window.tabs):
                    window.active -= 1
            elif index is not None:
                window.active = chosen
                with contextlib.suppress(CdpError):
                    await link.send(window.tab.target, "Page.bringToFront")
            return await self._show(link, window)

        return await self._guarded(run())

    # -- tabs ------------------------------------------------------------------------

    async def _open(self, key: TabKey, url: str) -> dict[str, Any]:
        link = await self._ensure_link()
        window = self._windows.get(key)
        if window is None:
            conversation, clone = key
            user_key: TabKey = (conversation, "user")
            if user_key in self._windows:
                window = self._windows.pop(user_key)
                for tab in window.tabs:
                    tab.controller = clone
                    tab.hand_over_pending = True
                    tab.controller_resumed.set()
                self._windows[key] = window
            else:
                tab = await self._new_tab(link)
                tab.controller = clone
                window = _Window([tab])
                self._windows[key] = window
            self._notify(key[0], {"type": "state"})  # the live view can show it loading
        tab = window.tab
        if tab.dialog is not None:
            # A dialog freezes the page; leaving it means dismissing it first.
            await self._dismiss_dialog(link, tab)
        try:
            seen = await self._navigate(link, tab, url)
        except (CdpError, KeyError):
            # U0 closed the tab: open a fresh one for this clone.
            tab = await self._new_tab(link)
            window.tabs[window.active] = tab
            seen = await self._navigate(link, tab, url)
        # A new page: the observation reads its frame and gives it fresh refs.
        tab.loader = ""
        tab.entries = []
        return await self._observe(link, window, seen, fresh=True)

    async def _new_tab(self, link: BrowserLink) -> _Tab:
        return await self._attach(link, await link.open_tab("about:blank"))

    async def _attach(self, link: BrowserLink, target: str) -> _Tab:
        tab = _Tab(target=target, events=link.subscribe(target))
        await link.send(target, "Page.enable")
        await link.send(target, "Network.enable")
        self._downloads_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(CdpError):
            # Browser-wide in Chrome; a link that cannot set it (R1) leaves Chrome's own.
            await link.send(
                target,
                "Browser.setDownloadBehavior",
                {
                    "behavior": "allow",
                    "downloadPath": str(self._downloads_dir),
                    "eventsEnabled": True,
                },
            )
        return tab

    def _existing(
        self, key: TabKey, *, allow_dialog: bool = False
    ) -> tuple[BrowserLink, _Window, _Tab]:
        window = self._windows.get(key)
        if window is None and self._link is not None:
            conversation, clone = key
            user_key: TabKey = (conversation, "user")
            if user_key in self._windows:
                window = self._windows.pop(user_key)
                for tab in window.tabs:
                    tab.controller = clone
                    tab.hand_over_pending = True
                    tab.controller_resumed.set()
                self._windows[key] = window
                self._notify(conversation, {"type": "state"})
        if window is None or self._link is None:
            raise PlainRefusalError(
                "No page is open for this clone yet. Open one first with action=open.",
                reason_code="no_page",
            )
        tab = window.tab
        _absorb(tab, key[0], self)
        if tab.dialog is not None and not allow_dialog:
            raise PlainRefusalError(dialog_note(tab.dialog), reason_code="dialog_open")
        return self._link, window, tab

    async def _ensure_link(self) -> BrowserLink:
        async with self._link_lock:
            if self._link is None:
                self._link = await self._link_factory()
            return self._link

    def _forget_link(self) -> None:
        self._link = None
        self._windows.clear()

    async def _guarded(self, work: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
        """Run one call, turning the browser's own failures into plain words."""
        try:
            return await work
        except CdpClosedError as exc:
            self._forget_link()
            raise PlainRefusalError(_BROWSER_CLOSED, reason_code="browser_closed") from exc
        except (CdpError, KeyError) as exc:
            raise PlainRefusalError(_BROWSER_FAILED, reason_code="browser_failed") from exc

    async def _adopt_popups(
        self, link: BrowserLink, window: _Window, opener: _Tab
    ) -> tuple[bool, str | None]:
        """Take the tabs `opener` just opened into the clone's window.

        Returns whether any was taken, and a note when the link cannot follow them (the
        extension attaches only to tabs it opened): the action itself still happened.
        """
        known = {t.target for w in self._windows.values() for t in w.tabs}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + POPUP_LOOKUP_S
        fresh: list[str] = []
        while True:
            try:
                listed = await link.send(opener.target, "Target.getTargets")
            except CdpError:
                return False, POPUP_NOT_FOLLOWED
            infos = cast(list[dict[str, Any]], listed.get("targetInfos", []))
            fresh = [
                str(i["targetId"])
                for i in infos
                if i.get("type") == "page"
                and i.get("openerId") == opener.target
                and i.get("targetId") not in known
            ]
            if fresh or loop.time() >= deadline:
                break
            # The new target is announced just after the page asks for it.
            await asyncio.sleep(0.1)
        taken = 0
        for target in fresh:
            try:
                adopted = await link.adopt_tab(opener.target, target)
            except PlainRefusalError as refused:
                if refused.reason_code != "popup_not_followed":
                    raise
                return taken > 0, str(refused)
            taken += 1
            tab = await self._attach(link, adopted)
            await self._loaded(link, tab)
            await self._read_frame(link, tab)
            window.tabs.append(tab)
            window.active = len(window.tabs) - 1
            owner = self._key_of(tab)
            if owner is not None:
                self._notify(owner[0], {"type": "state"})
        return bool(fresh), None

    async def _loaded(self, link: BrowserLink, tab: _Tab) -> None:
        """Wait for a pop-up's load event (it may have fired already), then the network."""
        with contextlib.suppress(CdpError):
            await link.send(
                tab.target,
                "Runtime.evaluate",
                {
                    "expression": "document.readyState === 'complete' || new Promise((r) => "
                    "addEventListener('load', () => r(true), {once: true}))",
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
        seen = await settle(tab.events, expect_load=False, main_frame=tab.frame)
        if seen.dialog is not None:
            tab.dialog = seen.dialog

    # -- acting ----------------------------------------------------------------------

    async def _perform(self, tab: _Tab, deliver: Callable[[], Awaitable[Any]]) -> Settled:
        """Deliver an action's input and wait for the page to settle around it.

        A JavaScript dialog freezes the page and the input that opened it, so the input
        runs as a task the settle-wait races: the dialog is reported at once and the frozen
        input finishes when the clone answers it.
        """
        key = self._key_of(tab)
        _absorb(tab, key[0] if key else None, self)
        task = asyncio.ensure_future(deliver())
        seen = await settle(tab.events, expect_load=False, main_frame=tab.frame, busy=task)
        if seen.dialog is not None:
            tab.dialog = seen.dialog
            if not task.done():
                tab.pending = task
                return seen
        elif not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), LOAD_TIMEOUT_S)
            except TimeoutError:
                tab.pending = task
                return seen
        task.result()
        return seen

    async def _answer_dialog(
        self,
        link: BrowserLink,
        window: _Window,
        tab: _Tab,
        accept: bool,
        text: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"accept": accept}
        if text is not None:
            params["promptText"] = text
        await link.send(tab.target, "Page.handleJavaScriptDialog", params)
        tab.dialog = None
        pending, tab.pending = tab.pending, None
        seen = await settle(tab.events, expect_load=False, main_frame=tab.frame, busy=pending)
        if seen.dialog is not None:
            tab.dialog = seen.dialog
            tab.pending = pending if pending is not None and not pending.done() else None
        elif pending is not None and pending.done() and not pending.cancelled():
            pending.exception()  # the frozen input's own failure is not this call's
        return await self._observe(link, window, seen)

    async def _dismiss_dialog(self, link: BrowserLink, tab: _Tab) -> None:
        with contextlib.suppress(CdpError):
            await link.send(tab.target, "Page.handleJavaScriptDialog", {"accept": False})
        tab.dialog = None
        tab.pending = None

    async def _select_custom(
        self,
        link: BrowserLink,
        window: _Window,
        tab: _Tab,
        element: str,
        backend: int,
        options: Sequence[str],
    ) -> dict[str, Any]:
        """A dropdown built from buttons and roles: open it, then click each option."""
        (x, y), _ = await self._point(link, tab, element, backend)
        seen = await self._perform(tab, lambda: self._mouse(link, tab, x, y))
        for wanted in options:
            if tab.dialog is not None or seen.navigated:
                break
            entries = build_entries(await self._ax_nodes(link, tab), tab.refs, collapse=False)
            option = pick_option(entries, wanted)
            if option is None or option.ref is None:
                shown = [e.name for e in entries if e.role in {"option", "menuitem"} and e.name]
                offer = f" It offers: {_listing(shown)}." if shown else ""
                raise PlainRefusalError(
                    f'Opening that list showed no option "{wanted}".{offer}',
                    reason_code="no_such_option",
                )
            o_backend, o_element = await self._element(link, tab, option.ref)
            (ox, oy), _ = await self._point(link, tab, o_element, o_backend)
            picked = await self._perform(tab, functools.partial(self._mouse, link, tab, ox, oy))
            seen = _merge(seen, picked)
        return await self._observe(link, window, seen)

    async def _history(self, key: TabKey, step: int) -> dict[str, Any]:
        link, window, tab = self._existing(key)
        history = await link.send(tab.target, "Page.getNavigationHistory")
        entries = cast(list[dict[str, Any]], history.get("entries", []))
        index = int(history.get("currentIndex", 0)) + step
        if not 0 <= index < len(entries):
            word = "earlier" if step < 0 else "later"
            raise PlainRefusalError(
                f"There is no {word} page in this tab.", reason_code="no_history"
            )
        entry_id = entries[index].get("id")
        seen = await self._perform(
            tab,
            lambda: link.send(tab.target, "Page.navigateToHistoryEntry", {"entryId": entry_id}),
        )
        return await self._observe(link, window, seen)

    async def _element(self, link: BrowserLink, tab: _Tab, ref: str) -> tuple[int, str]:
        """The DOM node a ref names and a handle to it, or the stale-ref refusal."""
        await self._track_page_load(link, tab)
        backend = tab.refs.node_for(ref.strip())
        key = self._key_of(tab)
        step = self._steps.get(key) if key is not None else None
        if step is not None and backend is not None:
            step.setdefault("element", tab.refs.name_for(ref.strip()))
        stale = PlainRefusalError(STALE_REF, reason_code="stale_ref")
        if backend is None:
            raise stale
        try:
            resolved = await link.send(tab.target, "DOM.resolveNode", {"backendNodeId": backend})
        except CdpError as exc:
            raise stale from exc
        handle = cast(dict[str, Any], resolved.get("object", {})).get("objectId")
        if not isinstance(handle, str):
            raise stale
        return backend, handle

    async def _point(
        self, link: BrowserLink, tab: _Tab, element: str, backend: int
    ) -> tuple[tuple[float, float], dict[str, float]]:
        """Scroll the element into view and wait until it can take a click; its centre and box."""
        with contextlib.suppress(CdpError):
            await link.send(tab.target, "DOM.scrollIntoViewIfNeeded", {"backendNodeId": backend})
        value = await self._call(
            link, tab, element, ACTIONABLE_JS, ACTIONABLE_TIMEOUT_MS, True, await_promise=True
        )
        outcome = cast(dict[str, Any], value) if isinstance(value, dict) else {"reason": "detached"}
        reason = str(outcome.get("reason", ""))
        if reason:
            raise PlainRefusalError(
                _NOT_ACTIONABLE.get(reason, STALE_REF), reason_code=f"element_{reason}"
            )
        cx, cy = float(outcome["x"]), float(outcome["y"])
        raw_box: Any = outcome.get("box")
        box: dict[str, float] = {"x": cx, "y": cy, "width": 0.0, "height": 0.0}
        if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:  # pyright: ignore[reportUnknownArgumentType]
            try:
                box = {
                    "x": float(raw_box[0]),  # pyright: ignore[reportUnknownArgumentType]
                    "y": float(raw_box[1]),  # pyright: ignore[reportUnknownArgumentType]
                    "width": float(raw_box[2]),  # pyright: ignore[reportUnknownArgumentType]
                    "height": float(raw_box[3]),  # pyright: ignore[reportUnknownArgumentType]
                }
            except (ValueError, TypeError):
                pass
        return (cx, cy), box

    async def _field(self, link: BrowserLink, tab: _Tab, element: str) -> dict[str, Any]:
        value = await self._call(link, tab, element, FIELD_INFO_JS)
        return cast(dict[str, Any], value) if isinstance(value, dict) else {}

    async def _call(
        self,
        link: BrowserLink,
        tab: _Tab,
        element: str,
        function: str,
        *args: object,
        await_promise: bool = False,
    ) -> object:
        """Run a page-side function with the element as `this`; its JSON result."""
        try:
            result = await link.send(
                tab.target,
                "Runtime.callFunctionOn",
                {
                    "functionDeclaration": function,
                    "objectId": element,
                    "arguments": [{"value": a} for a in args],
                    "returnByValue": True,
                    "awaitPromise": await_promise,
                },
            )
        except CdpError as exc:
            # The handle died with its node: the page re-rendered or navigated.
            raise PlainRefusalError(STALE_REF, reason_code="stale_ref") from exc
        if "exceptionDetails" in result:
            raise PlainRefusalError(_BROWSER_FAILED, reason_code="browser_failed")
        return _value_of(result)

    async def _mouse(
        self,
        link: BrowserLink,
        tab: _Tab,
        x: float,
        y: float,
        *,
        button: str = "left",
        clicks: int = 1,
        hover: bool = False,
    ) -> None:
        move = {"type": "mouseMoved", "x": x, "y": y}
        await link.send(tab.target, "Input.dispatchMouseEvent", move)
        if hover:
            return
        for count in range(1, clicks + 1):
            for kind in ("mousePressed", "mouseReleased"):
                await link.send(
                    tab.target,
                    "Input.dispatchMouseEvent",
                    {"type": kind, "x": x, "y": y, "button": button, "clickCount": count},
                )

    async def _keys(self, link: BrowserLink, tab: _Tab, chord: str) -> None:
        await self._send_keys(link, tab, key_events(parse_chord(chord)))

    async def _send_keys(
        self, link: BrowserLink, tab: _Tab, events: Sequence[dict[str, Any]]
    ) -> None:
        for event in events:
            await link.send(tab.target, "Input.dispatchKeyEvent", event)

    async def _scroll_note(
        self, link: BrowserLink, tab: _Tab, element: str | None, direction: Direction
    ) -> str:
        if direction in {"left", "right"}:
            return f"Scrolled {direction}."
        if element is not None:
            where = await self._call(link, tab, element, ELEMENT_SCROLL_POSITION_JS)
            thing = "That element"
        else:
            where = await self._evaluate(link, tab, PAGE_SCROLL_POSITION)
            thing = "The page"
        if not isinstance(where, int) or where < 0:
            return f"{thing} has nothing to scroll."
        if where == 0:
            return "Now at the top."
        if where >= 100:
            return "Now at the bottom."
        return f"Now {where}% of the way down."

    # -- observing -------------------------------------------------------------------

    async def _observe(
        self,
        link: BrowserLink,
        window: _Window,
        seen: Settled,
        *,
        note: str | None = None,
        fresh: bool = False,
    ) -> dict[str, Any]:
        """The observation an action returns (§3.2)."""
        if window.tab.hand_over_pending:
            fresh = True
        opener = window.tab
        popped, popup_note = (
            await self._adopt_popups(link, window, opener)
            if seen.windows > 0 and opener.dialog is None
            else (False, None)
        )
        if popup_note:
            note = f"{note} {popup_note}" if note else popup_note
        tab = window.tab
        result: dict[str, Any]
        if tab.dialog is not None:
            # The page is frozen: no script and no accessibility tree until it is answered.
            result = {"url": tab.url, "title": tab.title, "dialog": dialog_note(tab.dialog)}
        else:
            loader = await self._read_frame(link, tab)
            before = tab.entries
            if loader != tab.loader or popped:
                tab.loader = loader
                tab.refs = RefTable()
                before = []
            tab.entries = await self._entries(link, tab)
            result = await self._location(link, tab)
            if before and not fresh:
                result["changes"] = diff(before, tab.entries)
            else:
                result["snapshot"] = render(tab.entries)
        if popped:
            result["new_tab"] = "The page opened a new tab, which is now the current tab."
        if seen.downloads:
            result["downloads"] = "\n".join(
                _download_line(d, self._downloads_dir) for d in seen.downloads.values()
            )
        if len(window.tabs) > 1:
            result["tabs"] = await self._tab_lines(link, window)
        if note:
            result["note"] = note
        return _consume_hand_over(tab, result)

    async def _show(self, link: BrowserLink, window: _Window) -> dict[str, Any]:
        """The current tab as a fresh snapshot, with the tab list."""
        tab = window.tab
        key = self._key_of(tab)
        _absorb(tab, key[0] if key else None, self)
        if tab.dialog is None:
            await self._track_page_load(link, tab)
        return await self._observe(link, window, Settled(), fresh=True)

    async def _tab_lines(self, link: BrowserLink, window: _Window) -> str:
        titles: dict[str, tuple[str, str]] = {}
        if window.tab.dialog is None:
            with contextlib.suppress(CdpError):
                listed = await link.send(window.tab.target, "Target.getTargets")
                for info in cast(list[dict[str, Any]], listed.get("targetInfos", [])):
                    titles[str(info.get("targetId"))] = (
                        str(info.get("title", "")),
                        str(info.get("url", "")),
                    )
        lines: list[str] = []
        for number, tab in enumerate(window.tabs, start=1):
            title, url = titles.get(tab.target, (tab.title, tab.url))
            current = " (current)" if number == window.active + 1 else ""
            lines.append(f"{number}. {title or url} — {url}{current}")
        return "\n".join(lines)

    # -- page state ------------------------------------------------------------------

    async def _navigate(self, link: BrowserLink, tab: _Tab, url: str) -> Settled:
        _absorb(tab)
        result = await link.send(tab.target, "Page.navigate", {"url": url})
        error = result.get("errorText")
        if isinstance(error, str) and error:
            reason = _NAVIGATION_ERRORS.get(error, "the page could not be loaded")
            raise PlainRefusalError(f"{url} did not open: {reason}.", reason_code="page_not_loaded")
        seen = await settle(tab.events, expect_load=True)
        seen.navigated = True
        if seen.dialog is not None:
            tab.dialog = seen.dialog
        return seen

    async def _track_page_load(self, link: BrowserLink, tab: _Tab) -> None:
        """Reissue refs when the page navigated since the last call (a link U0 clicked)."""
        loader = await self._read_frame(link, tab)
        if loader != tab.loader:
            tab.loader = loader
            tab.refs = RefTable()
            tab.entries = []

    async def _read_frame(self, link: BrowserLink, tab: _Tab) -> str:
        """The main frame's current loader id; also records the frame's id."""
        tree = await link.send(tab.target, "Page.getFrameTree")
        frame = cast(
            dict[str, Any], cast(dict[str, Any], tree.get("frameTree", {})).get("frame", {})
        )
        tab.frame = str(frame.get("id", tab.frame))
        return str(frame.get("loaderId", ""))

    async def _ax_nodes(self, link: BrowserLink, tab: _Tab) -> list[dict[str, Any]]:
        result = await link.send(tab.target, "Accessibility.getFullAXTree")
        nodes = result.get("nodes")
        return cast(list[dict[str, Any]], nodes) if isinstance(nodes, list) else []

    async def _entries(self, link: BrowserLink, tab: _Tab) -> list[Entry]:
        return build_entries(await self._ax_nodes(link, tab), tab.refs)

    async def observe_change(self, key: TabKey) -> str:
        """What changed since the last observation, for actions that stay on the page."""
        link, _, tab = self._existing(key)
        before = tab.entries
        await self._track_page_load(link, tab)
        tab.entries = await self._entries(link, tab)
        if not before:
            return render(tab.entries)
        return diff(before, tab.entries)

    async def _evaluate(self, link: BrowserLink, tab: _Tab, expression: str) -> object:
        result = await link.send(
            tab.target, "Runtime.evaluate", {"expression": expression, "returnByValue": True}
        )
        return _value_of(result)

    async def _location(self, link: BrowserLink, tab: _Tab) -> dict[str, Any]:
        value = await self._evaluate(link, tab, "[location.href, document.title]")
        if isinstance(value, list) and len(cast(list[object], value)) == 2:
            href, title = cast(list[object], value)
            tab.url, tab.title = str(href), str(title)
            return {"url": tab.url, "title": tab.title}
        return {"url": tab.url, "title": tab.title}


def _consume_hand_over(tab: _Tab, result: dict[str, Any]) -> dict[str, Any]:
    if not tab.hand_over_pending:
        return result
    tab.hand_over_pending = False
    notes: list[str] = []
    if tab.visited_while_user:
        visited_lines = [
            f'- "{title}" ({url})' if title and title != url else f"- {url}"
            for url, title in tab.visited_while_user
        ]
        notes.append("Pages visited while you were away:\n" + "\n".join(visited_lines))
        tab.visited_while_user.clear()
    else:
        notes.append("Control was handed back to you.")
    result["hand_over_note"] = "\n".join(notes)
    return result


def _absorb(
    tab: _Tab,
    conversation: str | None = None,
    service: BrowserService | None = None,
) -> None:
    """Catch up on what happened in the tab between calls: a dialog, user detach, or navigation."""
    while True:
        try:
            event = tab.events.get_nowait()
        except asyncio.QueueEmpty:
            break
        if event.method == "Page.javascriptDialogOpening":
            params = event.params
            tab.dialog = {k: params[k] for k in ("type", "message", "defaultPrompt") if k in params}
        elif event.method == "Page.javascriptDialogClosed":
            tab.dialog = None
        elif event.method == "UClone.detached":
            if event.params.get("reason") == "canceled_by_user":
                tab.controller = "user"
                tab.controller_resumed.clear()
                if conversation and service:
                    service._notify(conversation, {"type": "state"})  # pyright: ignore[reportPrivateUsage]
        elif event.method == "Page.frameNavigated":
            frame = event.params.get("frame")
            if isinstance(frame, dict) and "parentId" not in frame:
                url = str(cast(dict[str, Any], frame).get("url") or "")
                if url:
                    tab.url = url
                    if tab.controller == "user":
                        if not tab.visited_while_user or tab.visited_while_user[-1][0] != url:
                            tab.visited_while_user.append((url, tab.title or url))
        elif event.method == "Page.navigatedWithinDocument":
            url = str(event.params.get("url") or "")
            if url:
                tab.url = url
                if tab.controller == "user":
                    if not tab.visited_while_user or tab.visited_while_user[-1][0] != url:
                        tab.visited_while_user.append((url, tab.title or url))
    if tab.pending is not None and tab.pending.done():
        if not tab.pending.cancelled():
            tab.pending.exception()  # retrieved, so asyncio does not log it as lost
        tab.pending = None


def _value_of(result: dict[str, Any]) -> object:
    remote = result.get("result")
    return cast(dict[str, Any], remote).get("value") if isinstance(remote, dict) else None


def _refuse_unmatched(outcome: dict[str, Any]) -> None:
    missing = cast(list[str], outcome.get("missing") or [])
    if missing:
        offered = cast(list[str], outcome.get("options") or [])
        raise PlainRefusalError(
            f'That list has no option "{missing[0]}". It offers: {_listing(offered)}.',
            reason_code="no_such_option",
        )
    if outcome.get("multiple") is False:
        raise PlainRefusalError(
            "That list takes one choice; pick a single option.", reason_code="single_choice"
        )


def _listing(names: Sequence[str], limit: int = 15) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f", and {len(names) - limit} more" if len(names) > limit else "")


def _tabs_word(count: int) -> str:
    return "1 tab" if count == 1 else f"{count} tabs"


def _merge(first: Settled, second: Settled) -> Settled:
    return Settled(
        navigated=first.navigated or second.navigated,
        dialog=second.dialog,
        windows=first.windows + second.windows,
        downloads={**first.downloads, **second.downloads},
        chooser=second.chooser,
    )


def _download_line(item: dict[str, str], folder: Path) -> str:
    name = item.get("name", "")
    if item.get("state") == "completed":
        return f'Downloaded "{name}" to {item.get("path") or folder / name}'
    if item.get("state") == "canceled":
        return f'The download of "{name}" was canceled.'
    return f'Still downloading "{name}" into {folder}'


def _check_url(url: str) -> None:
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise PlainRefusalError(
            f"The browser opens web addresses that start with http:// or https://, not {url!r}.",
            reason_code="unsupported_url",
        )


@dataclass
class Settled:
    """What the page did while it settled: the parts of an observation besides the diff."""

    navigated: bool = False
    dialog: dict[str, Any] | None = None
    windows: int = 0
    downloads: dict[str, dict[str, str]] = field(default_factory=lambda: {})
    chooser: int | None = None  # the file input whose chooser a click opened


async def settle(
    events: asyncio.Queue[CdpEvent],
    *,
    expect_load: bool,
    load_timeout: float = LOAD_TIMEOUT_S,
    idle_cap: float = NETWORK_IDLE_CAP_S,
    quiet: float = QUIET_S,
    main_frame: str | None = None,
    busy: asyncio.Future[Any] | None = None,
    download_wait: float = DOWNLOAD_WAIT_S,
    clock: Callable[[], float] | None = None,
) -> Settled:
    """Wait for the page to settle: its load event, then the network quiet (§3.2).

    Driven by the tab's own events, not a sleep loop: it returns once `quiet` seconds pass
    with no request in flight, or when a cap runs out (a slow page still gets observed).

    After an action, `busy` is the action's own input still being delivered: the quiet
    period starts once it returns. A main-frame load starting (`main_frame`) means the
    action navigated, so the wait goes back to the load event. A JavaScript dialog ends the
    wait at once, because the page and the action's input are frozen until it is answered.
    A download that began is waited for, up to `download_wait`.
    """
    loop = asyncio.get_running_loop()
    time_fn = clock or loop.time
    seen = Settled()
    start = time_fn()
    load_deadline = start + load_timeout
    busy_deadline = start + load_timeout
    download_deadline = start
    loaded = not expect_load
    idle_deadline: float | None = None
    inflight: set[str] = set()
    quiet_since = start
    while True:
        now = time_fn()
        if busy is not None and (busy.done() or now >= busy_deadline):
            busy = None
            quiet_since = now
        downloading = any(d["state"] == "in progress" for d in seen.downloads.values())
        if not loaded:
            if now >= load_deadline:
                loaded = True
                quiet_since = now
                continue
            wait = load_deadline - now
        elif busy is not None:
            wait = busy_deadline - now
        else:
            if idle_deadline is None:
                idle_deadline = now + idle_cap
            if (not inflight and now - quiet_since >= quiet) or now >= idle_deadline:
                if not downloading or now >= download_deadline:
                    return seen
                wait = download_deadline - now
            else:
                wait = (
                    idle_deadline - now
                    if inflight
                    else min(quiet - (now - quiet_since), idle_deadline - now)
                )
        event = await _next_event(events, busy, max(wait, 0.001))
        if event is None:
            continue
        params = event.params
        request = params.get("requestId")
        method = event.method
        if method == "Page.loadEventFired":
            loaded = True
            quiet_since = time_fn()  # loadEventFired
        elif method == "Page.frameStoppedLoading":
            # A page restored from the back-forward cache fires no load event.
            if main_frame and params.get("frameId") == main_frame:
                loaded = True
            quiet_since = time_fn()  # frameStoppedLoading
        elif method == "Network.requestWillBeSent" and isinstance(request, str):
            inflight.add(request)
        elif method in {"Network.loadingFinished", "Network.loadingFailed"}:
            inflight.discard(str(request))
            if not inflight:
                quiet_since = time_fn()
        elif method == "Page.javascriptDialogOpening":
            seen.dialog = {
                k: params[k] for k in ("type", "message", "defaultPrompt") if k in params
            }
            return seen
        elif (
            method == "Page.frameStartedLoading"
            and main_frame
            and params.get("frameId") == main_frame
        ):
            seen.navigated = True
            loaded = False
            load_deadline = time_fn() + load_timeout
            idle_deadline = None
        elif method == "Page.frameNavigated" and main_frame:
            frame = params.get("frame")
            if isinstance(frame, dict) and "parentId" not in frame:
                seen.navigated = True
        elif method == "Page.windowOpen":
            seen.windows += 1
        elif method == "Page.fileChooserOpened":
            node = params.get("backendNodeId")
            seen.chooser = node if isinstance(node, int) else None
        elif method == "Browser.downloadWillBegin":
            if not seen.downloads:
                download_deadline = loop.time() + download_wait
            name = str(params.get("suggestedFilename", ""))
            seen.downloads[str(params.get("guid"))] = {"name": name, "state": "in progress"}
        elif method == "Browser.downloadProgress":
            item = seen.downloads.get(str(params.get("guid")))
            state = params.get("state")
            if item is not None and state in {"completed", "canceled"}:
                item["state"] = str(state)
                path = params.get("filePath")
                if isinstance(path, str) and path:
                    item["path"] = path


async def _next_event(
    events: asyncio.Queue[CdpEvent], busy: asyncio.Future[Any] | None, wait: float
) -> CdpEvent | None:
    """The tab's next event, or `None` when `wait` runs out or `busy` finishes first."""
    if busy is None:
        try:
            return await asyncio.wait_for(events.get(), wait)
        except TimeoutError:
            return None
    getting = asyncio.ensure_future(events.get())
    await asyncio.wait({getting, busy}, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
    if getting.done():
        return getting.result()
    getting.cancel()
    return None
