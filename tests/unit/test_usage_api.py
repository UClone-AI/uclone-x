"""`GET /api/usage` and `PUT /api/usage/limits`: Settings → Usage (llm-token-gateway.md §4.5).

Driven through the dashboard app with its own storage directory, so the settings file and
the usage store are the ones the dashboard reads, and nothing touches the user's home.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uclone_x.errors import UsageLimitReachedError
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, TokenCountSource
from uclone_x.llm.usage.gate import shared_store
from uclone_x.llm.usage.limits import USAGE_LIMIT_ENV_VARS
from uclone_x.llm.usage.store import USAGE_FILE_NAME, UsageEntry
from uclone_x.ui.app import create_ui_app

NO_LIMITS = {"per_10_minutes": None, "per_5_hours": None, "per_week": None}


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = (tmp_path / "sessions").resolve()
    root.mkdir()
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(root))
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    for variable in USAGE_LIMIT_ENV_VARS.values():
        monkeypatch.delenv(variable, raising=False)
    return root


@pytest.fixture
def client(storage: Path, tmp_path: Path) -> TestClient:
    return TestClient(create_ui_app(static_dir=tmp_path, storage_dir=storage))


def _book(storage: Path, ago: timedelta, tokens: int, provider: str, model: str | None) -> None:
    shared_store(storage / USAGE_FILE_NAME).add(
        UsageEntry(
            at=datetime.now(UTC) - ago,
            tokens=tokens,
            provider=provider,
            model=model,
            count_source=TokenCountSource.PROVIDER,
        )
    )


def _window(body: dict[str, Any], name: str) -> dict[str, Any]:
    return next(w for w in body["windows"] if w["window"] == name)


def test_a_fresh_install_reports_no_limits_and_no_usage(client: TestClient) -> None:
    response = client.get("/api/usage")

    assert response.status_code == 200
    body = response.json()
    assert datetime.fromisoformat(body["checked_at"]).tzinfo is not None
    assert body["windows"] == [
        {"window": name, "used": 0, "limit": None, "available_again_at": None}
        for name in ("per_10_minutes", "per_5_hours", "per_week")
    ]
    assert body["limits"] == NO_LIMITS
    assert body["env_overrides"] == []
    assert body["providers"] == []


def test_saved_limits_round_trip_and_keep_every_other_setting(
    client: TestClient, storage: Path
) -> None:
    """A PUT saves only `usage_limits`; a later dashboard Settings save keeps them."""
    settings = storage / "settings.json"
    settings.write_text(json.dumps({"comfyui_base_url": "http://127.0.0.1:8188"}))
    wanted = {"per_10_minutes": None, "per_5_hours": 2_000_000, "per_week": 10_000_000}

    put = client.put("/api/usage/limits", json=wanted)

    assert put.status_code == 200
    assert put.json()["limits"] == wanted
    assert _window(put.json(), "per_5_hours")["limit"] == 2_000_000
    assert client.get("/api/usage").json()["limits"] == wanted

    assert client.post("/api/settings", json={"ui_language": "en"}).status_code == 200
    saved = json.loads(settings.read_text())
    assert saved["usage_limits"] == wanted
    assert saved["comfyui_base_url"] == "http://127.0.0.1:8188"

    cleared = client.put("/api/usage/limits", json=NO_LIMITS)
    assert cleared.json()["limits"] == NO_LIMITS


@pytest.mark.parametrize(
    "body",
    [
        {"per_10_minutes": 0, "per_5_hours": None, "per_week": None},
        {"per_10_minutes": None, "per_5_hours": -5, "per_week": None},
        {"per_10_minutes": None, "per_5_hours": "lots", "per_week": None},
        {"per_10_minutes": None, "per_5_hours": 1.5, "per_week": None},
        {"per_10_minutes": None, "per_5_hours": True, "per_week": None},
        {"per_minute": 100},
        {},
        {"per_10_minutes": 5000},
        [1, 2, 3],
    ],
)
def test_an_invalid_limit_is_refused_in_a_plain_sentence_and_nothing_is_saved(
    client: TestClient, storage: Path, body: object
) -> None:
    response = client.put("/api/usage/limits", json=body)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail.endswith(".")
    for internal in ("Input should", "validation", "per_5_hours", "per_minute", "{", "["):
        assert internal not in detail, detail
    assert not (storage / "settings.json").exists()


def test_a_variable_outranks_the_saved_limit_and_is_named(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.put(
        "/api/usage/limits",
        json={"per_10_minutes": None, "per_5_hours": 1_000, "per_week": None},
    )
    monkeypatch.setenv(USAGE_LIMIT_ENV_VARS[next(iter(USAGE_LIMIT_ENV_VARS))], " ")
    monkeypatch.setenv("UCLONE_USAGE_LIMIT_5_HOURS", "500")

    body = client.get("/api/usage").json()

    assert _window(body, "per_5_hours")["limit"] == 500  # the limit in effect
    assert body["limits"]["per_5_hours"] == 1_000  # what the picker edits
    assert body["env_overrides"] == ["per_5_hours"]  # a blank variable is not an override


def test_usage_counts_per_window_and_breaks_down_by_provider_for_seven_days(
    client: TestClient, storage: Path
) -> None:
    _book(storage, timedelta(minutes=1), 300, "Anthropic", "claude-a")
    _book(storage, timedelta(hours=2), 200, "Anthropic", "claude-b")
    _book(storage, timedelta(days=2), 250, "Anthropic", "claude-b")
    _book(storage, timedelta(days=1), 1_000, "OpenAI", None)  # after Anthropic's first
    _book(storage, timedelta(days=7, hours=12), 9_999, "Google", "gemini")  # past the week

    body = client.get("/api/usage").json()

    assert [_window(body, w)["used"] for w in ("per_10_minutes", "per_5_hours", "per_week")] == [
        300,
        500,
        1_750,
    ]
    assert body["providers"] == [
        {"provider": "OpenAI", "tokens": 1_000, "models": [{"model": None, "tokens": 1_000}]},
        {
            "provider": "Anthropic",
            "tokens": 750,
            "models": [
                {"model": "claude-b", "tokens": 450},
                {"model": "claude-a", "tokens": 300},
            ],
        },
    ]


def test_a_reached_window_says_when_it_lifts(client: TestClient, storage: Path) -> None:
    _book(storage, timedelta(hours=1), 600, "Anthropic", "claude-a")

    body = client.put(
        "/api/usage/limits",
        json={"per_10_minutes": None, "per_5_hours": 500, "per_week": None},
    ).json()

    reached = _window(body, "per_5_hours")
    lifts = datetime.fromisoformat(reached["available_again_at"])
    assert timedelta(hours=3, minutes=59) < lifts - datetime.now(UTC) <= timedelta(hours=4)
    assert _window(body, "per_week")["available_again_at"] is None


def test_an_unreadable_saved_limit_is_a_409_with_its_reason(
    client: TestClient, storage: Path
) -> None:
    (storage / "settings.json").write_text(json.dumps({"usage_limits": {"per_week": "lots"}}))

    response = client.get("/api/usage")

    assert response.status_code == 409
    assert "not a number of tokens" in response.json()["detail"]


def test_a_malformed_variable_refuses_the_save_before_anything_is_written(
    client: TestClient, storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the save lands and the answer is a 409, which the page reads as a failure."""
    monkeypatch.setenv("UCLONE_USAGE_LIMIT_WEEK", "lots")

    response = client.put(
        "/api/usage/limits",
        json={"per_10_minutes": None, "per_5_hours": 1_000, "per_week": None},
    )

    assert response.status_code == 409
    assert "UCLONE_USAGE_LIMIT_WEEK" in response.json()["detail"]
    assert not (storage / "settings.json").exists()


async def test_a_dashboard_with_its_own_storage_enforces_the_limit_its_panel_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedder's `storage_dir` apart from the session root: the gate follows the panel.

    Before, the panel read and saved the dashboard's own files while the gate read the
    session root's, so a limit set in the panel was shown and never applied.
    """
    session_root = (tmp_path / "session-root").resolve()
    session_root.mkdir()
    dashboard_dir = (tmp_path / "dashboard").resolve()
    dashboard_dir.mkdir()
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(session_root))
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    for variable in (*USAGE_LIMIT_ENV_VARS.values(), "LLM_PROVIDER", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(variable, "")  # recorded, so what the dashboard exports is undone
        monkeypatch.delenv(variable)
    (dashboard_dir / "settings.json").write_text(
        json.dumps(
            {
                "llm_provider": "anthropic",
                "llm_api_key": "sk-ant-test",
                "llm_api_key_provider": "anthropic",
            }
        )
    )
    provider_calls: list[LLMRequest] = []

    async def provider(_self: AnthropicConnector, request: LLMRequest) -> ModelResponse:
        provider_calls.append(request)
        raise AssertionError("the call reached the provider past the dashboard's limit")

    monkeypatch.setattr(AnthropicConnector, "generate", provider)
    app = create_ui_app(static_dir=tmp_path, storage_dir=dashboard_dir)
    client = TestClient(app)
    assert (
        client.put(
            "/api/usage/limits",
            json={"per_10_minutes": None, "per_5_hours": None, "per_week": 100},
        ).status_code
        == 200
    )
    _book(dashboard_dir, timedelta(minutes=1), 200, "Anthropic", "claude-a")
    llm = app.state.session_manager.default_llm
    assert llm is not None and llm.paid

    with pytest.raises(UsageLimitReachedError):
        await llm.generate(LLMRequest())

    assert provider_calls == []
    assert not (session_root / USAGE_FILE_NAME).exists()


def test_the_dashboard_builds_every_connector_through_its_gated_helper() -> None:
    """A direct `create_llm_connector` call in the dashboard would take the default gate.

    The test above drives one path (the connector built at startup); this holds the rest to
    the one helper that passes the dashboard's gate.
    """
    import uclone_x.ui.app as app_module

    tree = ast.parse(Path(app_module.__file__).read_text(encoding="utf-8"))
    callers = [
        function.name
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "create_llm_connector"
    ]
    assert callers == ["build_llm"]
