"""R2: a Chrome the Core launches, headed, with no automation switches (design §3.7).

The launch is plain on purpose. `--enable-automation` shows the "controlled by automated
test software" bar and sets `navigator.webdriver`, and so does a debugging port of 0; the
step 0 spike measured both. So the Core picks a free port itself and passes nothing but a
profile directory, the port, and the first-run switches.

The launched Chrome is U0's window. Closing the link does not close it, and a Core that
restarts while it is open reattaches: the port is recorded in the profile directory, and
Chrome's `/json/version` on that port names the browser endpoint. (Chrome's own
`DevToolsActivePort` file is not used: headless Chromium, which the tests run, never
writes it.)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from uclone_x.browser.cdp import CdpConnection, CdpEvent
from uclone_x.errors import PlainRefusalError

CHROME_PATH_ENV_VAR = "UCLONE_CHROME"
DEFAULT_PROFILE_DIR = Path.home() / ".uclone" / "browser" / "chrome"
LAUNCH_TIMEOUT_S = 20.0
PORT_FILE = "uclone-debugging-port"

_MAC_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
_LINUX_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
_WINDOWS_SUFFIX = ("Google", "Chrome", "Application", "chrome.exe")


def find_chrome(env: Mapping[str, str] | None = None) -> Path | None:
    """The installed Google Chrome, or `None`. `UCLONE_CHROME` names another binary."""
    environ = os.environ if env is None else env
    override = environ.get(CHROME_PATH_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    candidates: list[Path] = []
    if sys.platform == "darwin":
        candidates = [Path(p).expanduser() for p in _MAC_CANDIDATES]
    elif sys.platform == "win32":
        for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = environ.get(var)
            if base:
                candidates.append(Path(base).joinpath(*_WINDOWS_SUFFIX))
    else:
        import shutil

        for name in _LINUX_NAMES:
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
    return next((c for c in candidates if c.is_file()), None)


def free_port() -> int:
    """A free TCP port on 127.0.0.1. Never 0: port 0 sets `navigator.webdriver`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_args(
    chrome: Path,
    profile_dir: Path,
    port: int,
    *,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """The R2 command line: a profile, an explicit loopback port, and nothing that automates."""
    if port <= 0:
        raise ValueError("R2 needs an explicit debugging port; port 0 sets navigator.webdriver")
    return [
        str(chrome),
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={port}",
        "--remote-debugging-address=127.0.0.1",
        "--no-first-run",
        "--no-default-browser-check",
        *extra_args,
        "about:blank",
    ]


def recorded_port(profile_dir: Path) -> int | None:
    """The debugging port the last launch recorded for this profile, or `None`."""
    try:
        text = (profile_dir / PORT_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() and int(text) > 0 else None


async def browser_endpoint(port: int) -> str | None:
    """The `ws://` browser endpoint Chrome serves on `port`, or `None` if nothing answers."""
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            response = await client.get(f"http://127.0.0.1:{port}/json/version")
        endpoint: object = response.json().get("webSocketDebuggerUrl")
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    return endpoint if isinstance(endpoint, str) and endpoint.startswith("ws://") else None


class R2Link:
    """A `BrowserLink` to a Chrome the Core launched (or reattached to) with its own profile."""

    def __init__(self, conn: CdpConnection, process: subprocess.Popen[bytes] | None) -> None:
        self._conn = conn
        self._process = process
        self._sessions: dict[str, str] = {}

    @classmethod
    async def start(
        cls,
        profile_dir: Path = DEFAULT_PROFILE_DIR,
        *,
        chrome: Path | None = None,
        extra_args: Sequence[str] = (),
        timeout: float = LAUNCH_TIMEOUT_S,
    ) -> R2Link:
        """Reattach to this profile's running Chrome, or launch one."""
        port = recorded_port(profile_dir)
        existing = await browser_endpoint(port) if port is not None else None
        if existing is not None:
            with contextlib.suppress(OSError):
                return cls(await CdpConnection.connect(existing), None)
        binary = chrome or find_chrome()
        if binary is None:
            raise PlainRefusalError(
                "Google Chrome is not installed on this computer, so the browser cannot "
                "open. Install Chrome and try again.",
                reason_code="no_chrome",
            )
        profile_dir.mkdir(parents=True, exist_ok=True)
        port = free_port()
        (profile_dir / PORT_FILE).write_text(str(port), encoding="utf-8")
        args = launch_args(binary, profile_dir, port, extra_args=extra_args)
        # A session of its own: the window outlives the Core, as U0's window should.
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        endpoint = await _wait_for_endpoint(port, process, timeout)
        return cls(await CdpConnection.connect(endpoint), process)

    async def open_tab(self, url: str) -> str:
        """Open a tab and attach a flattened session to it."""
        created = await self._conn.send("Target.createTarget", {"url": url})
        target = str(created["targetId"])
        attached = await self._conn.send(
            "Target.attachToTarget", {"targetId": target, "flatten": True}
        )
        self._sessions[target] = str(attached["sessionId"])
        return target

    async def adopt_tab(self, opener: str, tab: str) -> str:
        """Attach a flattened session to a tab the page in `opener` opened (a pop-up)."""
        attached = await self._conn.send(
            "Target.attachToTarget", {"targetId": tab, "flatten": True}
        )
        self._sessions[tab] = str(attached["sessionId"])
        return tab

    async def send(
        self, tab: str, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a command to the tab's session."""
        return await self._conn.send(method, params, session_id=self._session(tab))

    def subscribe(self, tab: str, *, frames: bool = False) -> asyncio.Queue[CdpEvent]:
        """Events of the tab's session."""
        return self._conn.subscribe(self._session(tab), frames=frames)

    def unsubscribe(self, tab: str, queue: asyncio.Queue[CdpEvent]) -> None:
        """Stop delivering the tab's events to `queue`."""
        self._conn.unsubscribe(self._sessions.get(tab), queue)

    async def close_tab(self, tab: str) -> None:
        """Close the tab."""
        self._sessions.pop(tab, None)
        await self._conn.send("Target.closeTarget", {"targetId": tab})

    async def close(self) -> None:
        """Drop the connection. Chrome stays open."""
        await self._conn.close()

    async def shutdown(self) -> None:
        """Quit the Chrome this link launched. For tests; the product never quits U0's window."""
        await self.close()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                await asyncio.to_thread(self._process.wait, 10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                await asyncio.to_thread(self._process.wait)

    def _session(self, tab: str) -> str:
        session = self._sessions.get(tab)
        if session is None:
            raise KeyError(f"no open tab {tab!r}")
        return session


async def _wait_for_endpoint(port: int, process: subprocess.Popen[bytes], timeout: float) -> str:
    """Wait for the launched Chrome to answer on its debugging port.

    A bounded probe of a process starting up: Chrome sends no event before its port is
    open, so there is nothing to await instead.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        endpoint = await browser_endpoint(port)
        if endpoint is not None:
            return endpoint
        if process.poll() is not None:
            break
        await asyncio.sleep(0.1)
    raise PlainRefusalError(
        "Chrome did not start for the browser. If a UClone-X Chrome window is already open, "
        "close it and try again.",
        reason_code="chrome_did_not_start",
    )
