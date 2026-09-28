"""A person's decision, told apart from a program's on the local API (#1589 item 6).

The story view's approve and reject record `decided_in: story_view`: that a person decided.
Any program on this computer can reach the API, the model's own shell included, so both
routes ask for a secret only a window the server opened holds (`uclone_x.ui.person`). These
pin, in order of cost:

* **A decision without it changes nothing**, however the request dresses itself.
* **A decision from a confirmed window goes through**, as it did before.
* **No tool subprocess can read it**: it is in no environment a child inherits, and not in
  the page the server serves.
* **A pairing code works once**, so an address that leaked after use is worth nothing.
* **The refusal is plain words**, naming what to do and nothing of how the check works.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from tests.support.person import confirm_window, pairing_code
from uclone_x.core.failure_journal import consent_path
from uclone_x.core.secrets import is_secret_env_name
from uclone_x.llm import MockLLMConnector
from uclone_x.sandbox.models import NoIsolation
from uclone_x.story.library import StoryLibrary
from uclone_x.tools import BashRunTool, ToolContext
from uclone_x.tools.client import MCPClient
from uclone_x.tools.models import MCPConnectionConfig
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.person import (
    PAIRING_REFUSAL,
    PERSON_REFUSAL,
    WINDOW_FAILURE,
    WINDOW_INTERVAL_SECONDS,
    WINDOW_TOO_SOON,
    PersonGate,
)


@pytest.fixture(autouse=True)
def isolated_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Where the consent answer is written, away from this machine's own."""
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture
def client(tmp_path: Path, workspace: Path) -> Iterator[TestClient]:
    """An app serving a page, with no window confirmed yet: what `curl` from a shell is."""
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>UClone-X</title>", "utf-8")
    app = create_ui_app(
        static_dir=static,
        storage_dir=tmp_path / "sessions",
        workspace_dir=workspace,
        llm=MockLLMConnector(),
    )
    with TestClient(app) as started:
        yield started


def _gate(client: TestClient) -> PersonGate:
    return cast(PersonGate, cast(Any, client.app).state.person_gate)


def _secret(client: TestClient) -> str:
    """The secret, taken the one way it can be: by spending a pairing code."""
    secret = _gate(client).pair(pairing_code(client))
    assert secret is not None
    return secret


def _story_with_proposal(client: TestClient, workspace: Path) -> tuple[str, Path]:
    """A story with Vane and a pending proposal that he is wounded; its root."""
    created = client.post("/api/rooms", json={"title": "Writing room", "agent_ids": []})
    assert created.status_code == 201, created.text
    room_id = str(created.json()["room_id"])
    library = StoryLibrary(workspace)
    story_id = library.create("Night Train", room_id).story_id
    root = workspace / "stories" / story_id
    entry = root / "codex/characters/vane.yaml"
    entry.parent.mkdir(parents=True)
    entry.write_text("id: vane\nname: Vane\n", encoding="utf-8")
    digest = library.read_file(story_id, "codex/characters/vane.yaml").digest
    (root / "proposals").mkdir()
    (root / "proposals" / "p001.yaml").write_text(
        "id: p001\nkind: characters\nentry_id: vane\n"
        "change:\n  progression:\n    at: ch01.s01\n    set: {wounded: true}\n"
        f"proposed_at: '2026-09-25T10:00:00+00:00'\nroom_id: {room_id}\n"
        f"entry_digest: {digest}\n",
        encoding="utf-8",
    )
    return story_id, root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


#: What precedes the one-time code in the address of a window the server opens.
_PAIR_MARK = "#pair="

#: What a program on this computer can put on a request: every header a browser would send.
_BROWSER_DRESS = {
    "Origin": "http://127.0.0.1",
    "Referer": "http://127.0.0.1/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
}


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_a_decision_from_an_unconfirmed_caller_is_refused_and_records_nothing(
    client: TestClient, workspace: Path, decision: str
) -> None:
    """Killed by: src/uclone_x/ui/person.py :: if not self.confirms(request):
    Becomes: if False:
    """
    story_id, root = _story_with_proposal(client, workspace)
    seen = client.get(f"/api/artifacts/library/stories/{story_id}").json()["pending"][0]["digest"]
    before = _snapshot(root)

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/{decision}",
        json={"seen_digest": seen},
        headers=_BROWSER_DRESS,
    )

    assert refused.status_code == 403
    assert refused.json()["detail"] == PERSON_REFUSAL
    assert _snapshot(root) == before


def test_a_decision_from_a_confirmed_window_is_recorded_as_the_story_views(
    client: TestClient, workspace: Path
) -> None:
    """Killed by: src/uclone_x/ui/person.py :: return sent is not None and hmac.compare_digest(sent.encode(), self._secret.encode())
    Becomes: return False
    """
    story_id, root = _story_with_proposal(client, workspace)
    seen = client.get(f"/api/artifacts/library/stories/{story_id}").json()["pending"][0]["digest"]
    confirm_window(client)

    approved = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": seen},
    )

    assert approved.status_code == 200, approved.text
    assert "decided_in: story_view" in (root / "proposals" / "p001.yaml").read_text("utf-8")


def test_a_wrong_secret_is_refused(client: TestClient, workspace: Path) -> None:
    """Killed by: src/uclone_x/ui/person.py :: return sent is not None and hmac.compare_digest(sent.encode(), self._secret.encode())
    Becomes: return sent is not None
    """
    story_id, _ = _story_with_proposal(client, workspace)
    client.cookies.set(PersonGate.cookie_name(80), "guessed")

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": "x"},
    )

    assert refused.status_code == 403


def test_a_confirmed_cookie_sent_from_another_site_is_refused(
    client: TestClient, workspace: Path
) -> None:
    """A page on another port of this computer is the same site to the browser, which sends the
    cookie; `Sec-Fetch-Site` is what says the request did not come from the dashboard.

    Killed by: src/uclone_x/ui/person.py :: return site is not None and site not in _OWN_SITES
    Becomes: return False
    """
    story_id, _ = _story_with_proposal(client, workspace)
    confirm_window(client)

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": "x"},
        headers={"Sec-Fetch-Site": "same-site", "Origin": "http://127.0.0.1:3000"},
    )

    assert refused.status_code == 403


def test_a_pairing_code_confirms_one_window_only(client: TestClient) -> None:
    """Killed by: src/uclone_x/ui/person.py :: self._pairing = None
    Becomes: pass
    """
    code = pairing_code(client)

    assert client.post("/api/person/pair", json={"code": code}).status_code == 200
    again = client.post("/api/person/pair", json={"code": code})
    assert again.status_code == 403
    assert again.json()["detail"] == PAIRING_REFUSAL
    assert client.post("/api/person/pair", json={"code": "guessed"}).status_code == 403


def test_the_secret_is_in_no_page_and_no_environment_a_tool_runs_in(client: TestClient) -> None:
    """What a curl or a tool subprocess can reach: the served page, and its own environment.

    Killed by: src/uclone_x/ui/person.py :: self._secret = secrets.token_urlsafe(32)
    Becomes: self._secret = __import__("os").environ.setdefault("UCLONE_PERSON", secrets.token_urlsafe(32))
    """
    secret = _secret(client)

    for path in ("/", "/index.html"):
        page = client.get(path)
        assert page.status_code == 200, path
        assert secret not in page.text
    unconfined = BashRunTool()._build_sanitized_environment(isolation_level="none")  # pyright: ignore[reportPrivateUsage]
    assert all(secret not in value for value in unconfined.values())
    every_name = tuple(name for name in os.environ if not is_secret_env_name(name))
    mcp = MCPClient(
        MCPConnectionConfig(server_name="probe", command="true", env_allowlist=every_name)
    )
    assert all(secret not in value for value in mcp._build_environment().values())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_an_unconfined_shell_command_cannot_print_the_secret(
    client: TestClient, tmp_path: Path
) -> None:
    """The shell the issue names, run as a clone runs it with no isolation: `env` lacks it.

    Killed by: src/uclone_x/ui/person.py :: self._secret = secrets.token_urlsafe(32)
    Becomes: self._secret = __import__("os").environ.setdefault("UCLONE_PERSON", secrets.token_urlsafe(32))
    """
    secret = _secret(client)
    shell_dir = tmp_path / "shell"
    shell_dir.mkdir()
    context = ToolContext(
        agent_id="clone", session_id="s", workspace_root=shell_dir, isolation=NoIsolation()
    )

    ran = await BashRunTool().execute({"command": "env"}, context)

    assert ran.success, ran.error
    assert "PATH=" in str(ran.output)
    assert secret not in str(ran.output)


def test_the_window_opens_at_the_servers_own_address_not_the_callers(client: TestClient) -> None:
    """A caller chooses its `Host`; a window opened there could be served by the caller.

    Killed by: src/uclone_x/ui/person.py :: if not gate.open_window(_server_base_url(request)):
    Becomes: if not gate.open_window(str(request.base_url)):
    """
    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    _gate(client).opener = _open

    answered = client.post("/api/person/window", headers={"Host": "localhost:9999"})

    assert answered.status_code == 200
    address, _, code = opened[0].partition(_PAIR_MARK)
    assert (len(opened), address) == (1, "http://127.0.0.1:80/")
    assert client.post("/api/person/pair", json={"code": code}).status_code == 200


def test_a_window_that_could_not_open_is_said_plainly(client: TestClient) -> None:
    """Killed by: src/uclone_x/ui/person.py :: raise HTTPException(status_code=500, detail=WINDOW_FAILURE)
    Becomes: raise HTTPException(status_code=500, detail="webbrowser.open returned False")
    """

    def _no_window(_url: str) -> bool:
        return False

    _gate(client).opener = _no_window

    failed = client.post("/api/person/window")

    assert failed.status_code == 500
    assert failed.json()["detail"] == WINDOW_FAILURE


#: Words that would describe the mechanism rather than what the person should do.
_INTERNALS = re.compile(
    r"cookie|secret|token|header|origin|pairing|http|api|403|csrf|sec-fetch|\bcode\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("copy", [PERSON_REFUSAL, PAIRING_REFUSAL, WINDOW_FAILURE, WINDOW_TOO_SOON])
def test_the_refusals_are_plain_words(copy: str) -> None:
    """Killed by: src/uclone_x/ui/person.py :: "Only you can make this decision, and this window could not show that it is you. "
    Becomes: "Refused: no person cookie on this request. "
    """
    assert _INTERNALS.search(copy) is None, copy
    assert copy.endswith(".")


def test_the_gate_does_not_show_its_secret_when_printed() -> None:
    """Killed by: src/uclone_x/ui/person.py :: return "PersonGate(secret=<hidden>)"
    Becomes: return f"PersonGate(secret={self._secret})"
    """
    gate = PersonGate(opener=lambda _url: True)
    code = gate.window_url("http://127.0.0.1:1").split(_PAIR_MARK, 1)[1]
    secret = gate.pair(code)

    assert secret is not None
    assert secret not in repr(gate)
    assert secret not in str(gate)


@pytest.mark.parametrize("collect", [True, False])
def test_consent_from_an_unconfirmed_caller_is_refused_and_records_nothing(
    client: TestClient, collect: bool
) -> None:
    """Consent to record failures is the person's answer; a program cannot give it.

    Killed by: src/uclone_x/ui/app.py :: person_gate.require(request)
    Becomes: pass
    """
    refused = client.post(
        "/api/diagnostics/consent", json={"collect": collect}, headers=_BROWSER_DRESS
    )

    assert refused.status_code == 403
    assert refused.json()["detail"] == PERSON_REFUSAL
    assert not consent_path().exists()
    assert client.get("/api/diagnostics/consent").json()["state"] == "unasked"


def test_consent_from_a_confirmed_window_is_recorded(client: TestClient) -> None:
    """Killed by: src/uclone_x/ui/app.py :: person_gate.require(request)
    Becomes: raise HTTPException(status_code=403, detail=PERSON_REFUSAL)
    """
    confirm_window(client)

    granted = client.post("/api/diagnostics/consent", json={"collect": True})

    assert granted.status_code == 200, granted.text
    assert client.get("/api/diagnostics/consent").json()["state"] == "granted"


class _Clock:
    """Seconds that move only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _counting_opener(client: TestClient) -> tuple[list[str], _Clock]:
    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    clock = _Clock()
    _gate(client).opener = _open
    _gate(client).clock = clock
    return opened, clock


def test_a_second_window_asked_for_at_once_is_refused_plainly(client: TestClient) -> None:
    """A program asking over and over opens one window, not one per request.

    Killed by: src/uclone_x/ui/person.py :: if last is not None and now - last < WINDOW_INTERVAL_SECONDS:
    Becomes: if False:
    """
    opened, clock = _counting_opener(client)

    first = client.post("/api/person/window")
    clock.now += WINDOW_INTERVAL_SECONDS - 0.5
    again = client.post("/api/person/window")

    assert first.status_code == 200
    assert again.status_code == 429
    assert again.json()["detail"] == WINDOW_TOO_SOON
    assert len(opened) == 1


def test_a_window_opens_again_once_the_interval_has_passed(client: TestClient) -> None:
    """Killed by: src/uclone_x/ui/person.py :: if last is not None and now - last < WINDOW_INTERVAL_SECONDS:
    Becomes: if last is not None:
    """
    opened, clock = _counting_opener(client)

    assert client.post("/api/person/window").status_code == 200
    clock.now += WINDOW_INTERVAL_SECONDS
    assert client.post("/api/person/window").status_code == 200
    assert len(opened) == 2


def test_a_window_that_failed_to_open_still_counts(client: TestClient) -> None:
    """Else a program could ask as fast as a failing browser answers.

    Killed by: src/uclone_x/ui/person.py :: self._last_window = now
    Becomes: pass
    """
    tries: list[str] = []

    def _no_window(url: str) -> bool:
        tries.append(url)
        return False

    _gate(client).opener = _no_window
    _gate(client).clock = _Clock()

    assert client.post("/api/person/window").status_code == 500
    assert client.post("/api/person/window").status_code == 429
    assert len(tries) == 1
