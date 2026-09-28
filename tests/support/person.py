"""Confirming a test client's window the way the dashboard's own page does (#1589 item 6).

The story view's decisions are refused unless they come from a window the server confirmed
(`uclone_x.ui.person`). A test that exercises a decision rather than that refusal confirms
its client first, through the same pairing route the page uses, so it does not reach past
the check.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi.testclient import TestClient

from uclone_x.ui.person import PersonGate


def pairing_code(client: TestClient) -> str:
    """A fresh pairing code from the app's gate, as the address of a window it opened."""
    gate = cast(PersonGate, cast(Any, client.app).state.person_gate)
    return gate.window_url("http://127.0.0.1:80").split("#pair=", 1)[1]


def confirm_window(client: TestClient) -> None:
    """Pair `client` as the page does; afterwards it carries the cookie on its own."""
    paired = client.post("/api/person/pair", json={"code": pairing_code(client)})
    assert paired.status_code == 200, paired.text
