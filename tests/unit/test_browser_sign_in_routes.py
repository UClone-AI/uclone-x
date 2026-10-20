"""Settings ▸ Browser ▸ Saved sign-ins and Signed-in sites over HTTP (design §3.5).

What a head is sent never carries a password or a setup key; a refusal is a code the head
words, never a sentence that repeats what was typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from uclone_x.browser.extension import ExtensionHub
from uclone_x.browser.service import BrowserService
from uclone_x.browser.sign_ins import sign_in_for
from uclone_x.ui.browser import register_browser_routes

SECRET = "sentinel-pw-5Hd1-never-sent-back"  # noqa: S105
SETUP_KEY = "JBSWY3DPEHPK3PXP"


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


@pytest.fixture
def service() -> BrowserService:
    async def _no_browser() -> Any:
        raise AssertionError("listing saved sign-ins must not start Chrome")

    return BrowserService(_no_browser)


@pytest.fixture
def client(settings: Path, service: BrowserService) -> TestClient:
    app = FastAPI()

    def refuse(request: Any) -> None:
        origin = request.headers.get("origin")
        if origin and "127.0.0.1" not in origin:
            raise HTTPException(status_code=403)

    register_browser_routes(
        app,
        hub=ExtensionHub(lambda: None),
        settings_file=settings,
        refuse_cross_origin=refuse,
        service=service,
    )
    return TestClient(app)


def _save(client: TestClient, **over: Any) -> Any:
    body = {"site": "shop.example.com", "username": "kenny", "password": SECRET, **over}
    return client.post("/api/browser/sign-ins", json=body)


def test_a_saved_sign_in_is_listed_without_its_secrets(client: TestClient, settings: Path) -> None:
    saved = _save(client, totp_key=SETUP_KEY)
    listed = client.get("/api/browser/sign-ins")

    for response in (saved, listed):
        assert response.status_code == 200
        assert response.json() == {
            "sign_ins": [{"site": "shop.example.com", "username": "kenny", "has_2fa": True}]
        }
        assert SECRET not in response.text and SETUP_KEY not in response.text
    on_disk = json.loads(settings.read_text(encoding="utf-8"))
    assert on_disk["browser_sign_ins"][0]["password"] == SECRET


def test_the_browser_signs_in_with_what_settings_saved(
    client: TestClient, service: BrowserService
) -> None:
    """Killed by: src/uclone_x/ui/browser.py :: service.use_sign_ins(_saved_sign_ins)
    Becomes: pass
    """
    _save(client)

    found = sign_in_for("https://shop.example.com/login", service._sign_ins())  # pyright: ignore[reportPrivateUsage]
    assert found is not None and found.password == SECRET


@pytest.mark.parametrize(
    ("over", "code"),
    [
        ({"site": " "}, "site_missing"),
        ({"username": ""}, "username_missing"),
        ({"password": ""}, "password_missing"),
        ({"totp_key": "not a key " + SECRET}, "setup_key_invalid"),
    ],
)
def test_a_sign_in_that_cannot_be_saved_is_refused_by_code(
    client: TestClient, over: dict[str, str], code: str
) -> None:
    response = _save(client, **over)

    assert response.status_code == 400
    assert response.json()["reason_code"] == code
    assert SECRET not in response.text
    assert client.get("/api/browser/sign-ins").json() == {"sign_ins": []}


def test_delete_removes_one_sign_in(client: TestClient) -> None:
    _save(client)
    _save(client, username="other")

    response = client.request(
        "DELETE", "/api/browser/sign-ins", json={"site": "shop.example.com", "username": "kenny"}
    )

    assert response.json() == {
        "sign_ins": [{"site": "shop.example.com", "username": "other", "has_2fa": False}]
    }


def test_a_web_page_cannot_read_or_change_saved_sign_ins(client: TestClient) -> None:
    evil = {"origin": "https://evil.example"}
    assert client.get("/api/browser/sign-ins", headers=evil).status_code == 403
    body = {"site": "a.example", "username": "u", "password": "p"}
    assert client.post("/api/browser/sign-ins", json=body, headers=evil).status_code == 403
    assert (
        client.request(
            "DELETE",
            "/api/browser/sign-ins",
            json={"site": "a.example", "username": "u"},
            headers=evil,
        ).status_code
        == 403
    )
    assert client.get("/api/browser/sessions", headers=evil).status_code == 403
    assert client.post("/api/browser/sessions/sign-out", json={}, headers=evil).status_code == 403


def test_with_no_browser_open_the_signed_in_sites_say_so_and_start_nothing(
    client: TestClient,
) -> None:
    assert client.get("/api/browser/sessions").json() == {"open": False, "sites": []}
    assert client.post("/api/browser/sessions/sign-out", json={}).json() == {
        "open": False,
        "sites": [],
    }


def test_the_cores_own_browser_reads_settings_without_being_started_early(
    settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The app registers the routes with no service: the Core's default browser must still
    sign in with what Settings saved, and registering must not create it.

    Killed by: src/uclone_x/ui/browser.py :: use_default_sign_ins(_saved_sign_ins)
    Becomes: pass
    """
    import uclone_x.browser.tool as tool

    monkeypatch.setattr(tool, "_default_service", None)
    monkeypatch.setattr(tool, "_default_sign_ins", tuple)
    app = FastAPI()
    register_browser_routes(
        app,
        hub=ExtensionHub(lambda: None),
        settings_file=settings,
        refuse_cross_origin=lambda request: None,
    )
    assert tool._default_service is None  # pyright: ignore[reportPrivateUsage]
    TestClient(app).post(
        "/api/browser/sign-ins",
        json={"site": "shop.example.com", "username": "kenny", "password": SECRET},
    )

    service = tool.default_browser_service()
    found = sign_in_for("https://shop.example.com/", service._sign_ins())  # pyright: ignore[reportPrivateUsage]
    assert found is not None and found.username == "kenny"
