"""Fixed answers for `_diagnose_vite`, so the dev-mode launcher tests do not poll a socket (#1082).

`start_ui_server(dev=True)` calls `_diagnose_vite(host, vite_port)` whenever the Vite `Popen`
returned something, and under test that something is a `MagicMock`, never `None`. The helper
then polls `http://<host>:<port>/` to a ten-second deadline (`server.py`), so every test that
reaches the dev branch paid ten seconds *and* let whatever happened to be listening on the
developer's port 5173 choose which of the launcher's two branches ran: this project's Vite
answered with `VITE_IDENTITY_MARKER` and gave the HMR-active branch, another project's dev
server gave the foreign-app branch, and an idle port gave the deadline. No assertion noticed,
so all three passed and the machine-dependence stayed invisible.

`answering_as` replaces `httpx.get` rather than `_diagnose_vite` itself. Stubbing the helper
would be shorter and would also delete the poll, but it would delete the diagnosis with it:
the launcher tests would then assert a branch chosen by their own stub's return value, which
is the assertion-independent-of-the-outcome shape the helper was written to remove (P6). With
the transport fixed instead, the real `_diagnose_vite` runs, decides on the body it is given,
and returns on the first response — no deadline, no socket, and the branch is a consequence of
production code rather than of the test.

`one_line` exists because the launcher reports its branch through `rich`, which hard-wraps to
the console width; a phrase asserted on can otherwise be split by a newline that depends on
the terminal.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx


def answering_as(body: str) -> Callable[..., httpx.Response]:
    """An `httpx.get` stand-in that serves `body` with a 200, without touching the network."""

    def fake_get(url: str, timeout: float = 2.0) -> httpx.Response:
        return httpx.Response(200, text=body)

    return fake_get


#: A page served by something that is not this project, named so a test can assert the name.
FOREIGN_APP_TITLE = "Hexworld Deck Studio"

#: What that something serves: a valid page with no `VITE_IDENTITY_MARKER` in it.
FOREIGN_APP_BODY = f"<html><title>{FOREIGN_APP_TITLE}</title></html>"


def one_line(captured: str) -> str:
    """`captured` with every run of whitespace collapsed, undoing `rich`'s width wrapping."""
    return " ".join(captured.split())
