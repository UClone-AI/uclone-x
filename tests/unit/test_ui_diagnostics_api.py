"""The dashboard's diagnostics endpoints: read freely, change only when asked.

The dashboard is the surface a beginner uses, so it is the surface where an
accidental default would do the most damage. These tests pin the two decisions
that matter: consent has no default, and no endpoint transmits anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uclone_x.core.failure_journal import (
    consent_path,
    journal_path,
    record_failure,
    set_consent,
)
from uclone_x.ui.app import create_ui_app


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_ui_app(static_dir=tmp_path))


def _record(client: TestClient, message: str = "boom") -> None:
    try:
        raise ValueError(message)
    except ValueError as exc:
        record_failure(exc)


def test_consent_starts_unasked(client: TestClient) -> None:
    """`unasked` is reported as itself, not collapsed into `denied`.

    The dashboard needs the difference: one state prompts, the other does not.
    """
    payload = client.get("/api/diagnostics/consent").json()

    assert payload["state"] == "unasked"
    assert payload["journal"].endswith("failures.jsonl")


def test_consent_requires_an_explicit_boolean(client: TestClient) -> None:
    """Mutation: default a missing `collect` to `True`. A malformed request
    would then turn collection on for someone who never answered."""
    assert client.post("/api/diagnostics/consent", json={}).status_code == 400
    assert client.post("/api/diagnostics/consent", json={"collect": "yes"}).status_code == 400
    assert client.get("/api/diagnostics/consent").json()["state"] == "unasked"


def test_consent_can_be_granted_and_withdrawn(client: TestClient) -> None:
    assert client.post("/api/diagnostics/consent", json={"collect": True}).json() == {
        "state": "granted"
    }
    assert client.post("/api/diagnostics/consent", json={"collect": False}).json() == {
        "state": "denied"
    }


def test_report_returns_the_text_and_a_link_but_sends_nothing(client: TestClient) -> None:
    set_consent(True)
    _record(client)

    payload = client.get("/api/diagnostics/report").json()

    assert payload["count"] == 1
    assert payload["distinct"] == 1
    assert "ValueError" in payload["body"]
    # A link the browser opens, not a request this server makes.
    assert payload["issue_url"].startswith("https://github.com/UClone-AI/uclone-x/issues/new?")
    assert payload["search_url"].startswith("https://github.com/search?")


def test_report_is_empty_and_harmless_without_consent(client: TestClient) -> None:
    _record(client)

    payload = client.get("/api/diagnostics/report").json()

    assert payload["state"] == "unasked"
    assert payload["count"] == 0
    assert "No failures recorded." in payload["body"]


def test_delete_clears_the_journal(client: TestClient) -> None:
    set_consent(True)
    _record(client)

    assert client.delete("/api/diagnostics/report").json() == {"cleared": True}
    assert client.get("/api/diagnostics/report").json()["count"] == 0


def test_cross_origin_requests_cannot_change_anything(client: TestClient) -> None:
    """This app runs with `allow_origins=["*"]` and `allow_credentials=True`.

    That predates these routes, and they must not inherit it quietly: without
    this check, any page open in the same browser while `ucx ui` is running
    could turn failure recording on and read the report back.

    Killed by: src/uclone_x/ui/app.py :: parsed.netloc == host or parsed.hostname in _LOOPBACK_HOSTS
    Becomes: True
    """
    evil = {"Origin": "https://evil.example"}

    assert (
        client.post("/api/diagnostics/consent", json={"collect": True}, headers=evil).status_code
        == 403
    )
    assert client.delete("/api/diagnostics/report", headers=evil).status_code == 403
    assert client.get("/api/diagnostics/consent").json()["state"] == "unasked"


def test_same_origin_and_originless_requests_still_work(client: TestClient) -> None:
    """A browser on this server, and a client that sends no `Origin` at all.

    The second case is the CLI and `curl`; refusing it would break every
    non-browser caller in the name of a check that does not apply to them.
    """
    same = {"Origin": "http://127.0.0.1"}

    assert (
        client.post("/api/diagnostics/consent", json={"collect": True}, headers=same).status_code
        == 200
    )
    assert client.post("/api/diagnostics/consent", json={"collect": False}).status_code == 200


def test_an_unreadable_journal_is_reported_as_such_not_as_empty(client: TestClient) -> None:
    """The API has to carry the distinction, or the dashboard invents it back."""
    set_consent(True)
    _record(client)
    journal_path().chmod(0o000)

    try:
        payload = client.get("/api/diagnostics/report").json()
    finally:
        journal_path().chmod(0o644)

    assert payload["available"] is False
    assert payload["error"] and "could not be read" in payload["error"]
    assert "could not be read" in payload["body"]
    assert "No failures recorded." not in payload["body"]


def test_reads_are_refused_cross_origin_too(client: TestClient) -> None:
    """The first version left reads open, reasoning they expose what the user
    can already see. That is the reasoning a same-origin policy exists to
    reject: the report body is the failure record, and a page that can read it
    cross-origin has taken it.
    """
    evil = {"Origin": "https://evil.example"}

    assert client.get("/api/diagnostics/report", headers=evil).status_code == 403
    assert client.get("/api/diagnostics/consent", headers=evil).status_code == 403


def test_the_dashboards_own_dev_server_is_not_locked_out(client: TestClient) -> None:
    """`vite.config.ts` proxies `/api` with `changeOrigin`, so the dashboard in
    development arrives with a loopback `Origin` against a rewritten `Host`.

    A strict origin/host comparison would 403 the very UI this serves. Loopback
    origins are admitted at any port and nothing remote can produce one.
    """
    vite = {"Origin": "http://localhost:5173", "Host": "localhost:5180"}

    assert client.get("/api/diagnostics/report", headers=vite).status_code == 200
    assert (
        client.post("/api/diagnostics/consent", json={"collect": True}, headers=vite).status_code
        == 200
    )


def test_a_failed_delete_is_an_error_not_a_successful_no_op(client: TestClient) -> None:
    """`200 {"cleared": false}` is invisible to a client checking the status.

    Which the dashboard now is, so the previous shape made its new response
    check unable to see the one failure it was added for.
    """
    set_consent(True)
    _record(client)
    journal_path().parent.chmod(0o500)

    try:
        response = client.delete("/api/diagnostics/report")
    finally:
        journal_path().parent.chmod(0o755)

    assert response.status_code == 500
    assert "could not be deleted" in response.json()["detail"]


def test_an_unreadable_preference_is_carried_to_the_client(client: TestClient) -> None:
    """The browser cannot tell "never asked" from "unreadable" on its own."""
    set_consent(True)
    consent_path().write_text("{not json", encoding="utf-8")

    payload = client.get("/api/diagnostics/consent").json()

    assert payload["state"] == "unasked"
    assert payload["error"] and "readable JSON" in payload["error"]
