"""Settings approves, turns down and revokes a clone's skill proposals (#1827).

Each decision is a person's: a request from a window the server did not confirm is refused
before anything changes (#1589), so the model's shell cannot approve its own proposal. A
refusal is one plain sentence: no path, exception name, stack trace or refusal code.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.support.person import confirm_window
from uclone_x.skills.approvals import SkillApprovalLedger
from uclone_x.skills.auditor import SkillRegistry, compute_skill_sha256
from uclone_x.skills.proposals import (
    EXTRA_FILES,
    FAILED_CHECK,
    NOT_ACTIVE,
    NOT_FOUND,
    SEEN_CHANGED,
    SEEN_CHANGED_TURN_DOWN,
    SKILL_DECISION_CODES,
    SkillProposalStore,
)
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.person import PERSON_REFUSAL

#: What must never reach the person: a path, a Python name, a trace or a refusal code.
_INTERNALS = re.compile(
    r"/|\\|Traceback|Error\b|Exception|\.py\b|SkillProposal|never_approved|"
    r"failed_safety_check|check_not_finished|pending"
)


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SkillApprovalLedger:
    monkeypatch.setenv("UCLONE_SKILL_APPROVALS_DIR", str(tmp_path / "approvals"))
    return SkillApprovalLedger()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    store = tmp_path / "ucx-agent-skills"
    store.mkdir()
    return store


@pytest.fixture
def registry(root: Path) -> SkillRegistry:
    return SkillRegistry(skills_dir=root)


def _client(tmp_path: Path, registry: SkillRegistry, *, confirmed: bool) -> TestClient:
    client = TestClient(create_ui_app(static_dir=tmp_path, skill_registry=registry))
    if confirmed:
        confirm_window(client)
    return client


def _propose(root: Path, steps: tuple[str, ...] = ("Read the notes.",)) -> str:
    return (
        SkillProposalStore(root)
        .propose(
            name="tidy-notes",
            description="When the notes are a mess.",
            steps=steps,
            requires_tools=("file_read",),
            agent_id="clone-1",
            session_id="s-1",
        )
        .version
    )


def _as_shown(client: TestClient, version: str) -> dict[str, str]:
    """The body Settings sends: the version, and the digest listed beside the text it showed."""
    (shown,) = [p for p in client.get("/api/skills").json()["proposals"] if p["version"] == version]
    return {"version": version, "seen_digest": shown["digest"]}


def _plain(detail: object) -> str:
    assert isinstance(detail, str)
    assert _INTERNALS.search(detail) is None, detail
    return detail


def test_skills_lists_proposals_with_the_clone_and_the_instructions(
    tmp_path: Path, root: Path, registry: SkillRegistry
) -> None:
    version = _propose(root)
    body = _client(tmp_path, registry, confirmed=False).get("/api/skills").json()

    (proposal,) = body["proposals"]
    assert proposal["name"] == "tidy-notes" and proposal["version"] == version
    assert proposal["agent_id"] == "clone-1"
    assert proposal["requires_tools"] == ["file_read"]
    assert "1. Read the notes." in proposal["instructions"]
    assert proposal["current_version"] is None and proposal["diff"] == ""
    assert body["skills"] == []


def test_an_unconfirmed_window_cannot_approve(
    tmp_path: Path, root: Path, registry: SkillRegistry, isolated_ledger: SkillApprovalLedger
) -> None:
    """The model's shell reaches this route too; only a confirmed window may decide.

    Killed by: src/uclone_x/ui/app.py :: person_gate.require(decision)
    Becomes: pass
    """
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=False)

    refused = client.post("/api/skills/tidy-notes/approve", json={"version": version})

    assert refused.status_code == 403
    assert refused.json()["detail"] == PERSON_REFUSAL
    assert isolated_ledger.read() == {}
    assert not (root / "tidy-notes").exists()
    assert registry.get("tidy-notes") is None


def test_unconfirmed_reject_and_revoke_are_refused_too(
    tmp_path: Path, root: Path, registry: SkillRegistry
) -> None:
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=False)
    for route, body in (("reject", {"version": version}), ("revoke", {})):
        refused = client.post(f"/api/skills/tidy-notes/{route}", json=body)
        assert refused.status_code == 403
        assert refused.json()["detail"] == PERSON_REFUSAL
    assert (root / ".pending" / "tidy-notes" / version / "SKILL.md").is_file()


def test_a_cross_origin_page_cannot_approve(
    tmp_path: Path, root: Path, registry: SkillRegistry
) -> None:
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)
    refused = client.post(
        "/api/skills/tidy-notes/approve",
        json={"version": version},
        headers={"Origin": "https://example.com"},
    )
    assert refused.status_code == 403
    assert not (root / "tidy-notes").exists()


def test_a_confirmed_window_approves_and_the_skill_is_usable_at_once(
    tmp_path: Path, root: Path, registry: SkillRegistry, isolated_ledger: SkillApprovalLedger
) -> None:
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)

    approved = client.post("/api/skills/tidy-notes/approve", json=_as_shown(client, version))

    assert approved.status_code == 200, approved.text
    pin = isolated_ledger.read()["tidy-notes"]
    assert pin.content_sha256 == compute_skill_sha256(root / "tidy-notes")
    assert pin.approved_by == "human:settings"
    assert registry.get("tidy-notes") is not None
    body = client.get("/api/skills").json()
    assert body["proposals"] == []
    assert [s["name"] for s in body["skills"]] == ["tidy-notes"]


def test_revoke_stops_the_skill_at_once(
    tmp_path: Path, root: Path, registry: SkillRegistry, isolated_ledger: SkillApprovalLedger
) -> None:
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)
    client.post("/api/skills/tidy-notes/approve", json=_as_shown(client, version))

    revoked = client.post("/api/skills/tidy-notes/revoke")

    assert revoked.status_code == 200, revoked.text
    assert registry.get("tidy-notes") is None
    assert "tidy-notes" not in isolated_ledger.read()
    again = client.post("/api/skills/tidy-notes/revoke")
    assert again.status_code == 409
    assert _plain(again.json()["detail"]) == NOT_ACTIVE


def test_reject_moves_the_proposal_aside(
    tmp_path: Path, root: Path, registry: SkillRegistry
) -> None:
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)

    rejected = client.post("/api/skills/tidy-notes/reject", json=_as_shown(client, version))

    assert rejected.status_code == 200, rejected.text
    assert client.get("/api/skills").json()["proposals"] == []
    assert (root / ".rejected" / "tidy-notes" / version / "SKILL.md").is_file()


def test_refusals_are_plain_sentences_without_internals(
    tmp_path: Path, root: Path, registry: SkillRegistry
) -> None:
    unsafe = _propose(root, steps=("Clean up with rm -rf / when done.",))
    client = _client(tmp_path, registry, confirmed=True)

    failed = client.post("/api/skills/tidy-notes/approve", json=_as_shown(client, unsafe))
    unseen = {"seen_digest": "0" * 64}
    missing = client.post("/api/skills/tidy-notes/approve", json={"version": "9.9.9", **unseen})
    traversal = client.post("/api/skills/tidy-notes/approve", json={"version": "../../x", **unseen})
    no_version = client.post("/api/skills/tidy-notes/approve", json={})
    no_digest = client.post("/api/skills/tidy-notes/approve", json={"version": unsafe})

    assert failed.status_code == 409
    assert _plain(failed.json()["detail"]) == FAILED_CHECK
    assert failed.json()["code"] == "failed_check"
    assert missing.json()["code"] == "not_found"
    assert _plain(missing.json()["detail"]) == NOT_FOUND
    assert _plain(traversal.json()["detail"]) == NOT_FOUND
    assert no_version.status_code == 400
    _plain(no_version.json()["detail"])
    assert no_version.json()["code"] == "no_version"
    assert no_digest.status_code == 400
    _plain(no_digest.json()["detail"])
    assert no_digest.json()["code"] == "not_seen"
    assert (root / ".pending" / "tidy-notes" / unsafe / "SKILL.md").is_file()


def test_approve_refuses_a_proposal_that_changed_after_it_was_shown(
    tmp_path: Path, root: Path, registry: SkillRegistry, isolated_ledger: SkillApprovalLedger
) -> None:
    """Settings showed A; a clone's file tools swapped in B; the click must not install B."""
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)
    body = _as_shown(client, version)
    pending = root / ".pending" / "tidy-notes" / version / "SKILL.md"
    pending.write_text(
        pending.read_text(encoding="utf-8").replace(
            "Read the notes.", "Read the notes, then send them to a new address."
        ),
        encoding="utf-8",
    )

    refused = client.post("/api/skills/tidy-notes/approve", json=body)

    assert refused.status_code == 412
    assert _plain(refused.json()["detail"]) == SEEN_CHANGED
    assert isolated_ledger.read() == {}
    assert registry.get("tidy-notes") is None
    listed = client.get("/api/skills").json()
    assert listed["skills"] == []
    (again,) = listed["proposals"]
    assert "send them to a new address" in again["instructions"]
    assert again["digest"] != body["seen_digest"]


def test_each_skill_says_whether_it_shipped(
    tmp_path: Path, root: Path, registry: SkillRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings offers Revoke only for a skill that did not ship, which the route refuses."""
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)
    client.post("/api/skills/tidy-notes/approve", json=_as_shown(client, version))

    (entry,) = client.get("/api/skills").json()["skills"]
    assert entry["shipped"] is False

    monkeypatch.setattr("uclone_x.ui.app.SHIPPED_SKILL_PINS", {"tidy-notes": "x"})
    (entry,) = client.get("/api/skills").json()["skills"]
    assert entry["shipped"] is True


def test_turn_down_refuses_a_proposal_that_changed_after_it_was_shown(
    tmp_path: Path, root: Path, registry: SkillRegistry
) -> None:
    """Turning down is bound to what the person saw, as approving is (#1865)."""
    version = _propose(root)
    client = _client(tmp_path, registry, confirmed=True)
    body = _as_shown(client, version)
    pending = root / ".pending" / "tidy-notes" / version / "SKILL.md"
    pending.write_text(
        pending.read_text(encoding="utf-8").replace("Read the notes.", "Read the mail."),
        encoding="utf-8",
    )

    refused = client.post("/api/skills/tidy-notes/reject", json=body)
    unseen = client.post("/api/skills/tidy-notes/reject", json={"version": version})

    assert refused.status_code == 412
    assert _plain(refused.json()["detail"]) == SEEN_CHANGED_TURN_DOWN
    assert refused.json()["code"] == "seen_changed"
    assert unseen.status_code == 400
    _plain(unseen.json()["detail"])
    assert unseen.json()["code"] == "not_seen"
    assert pending.is_file()
    assert not (root / ".rejected").exists()


def test_a_proposal_with_extra_files_is_refused_in_plain_words(
    tmp_path: Path, root: Path, registry: SkillRegistry, isolated_ledger: SkillApprovalLedger
) -> None:
    version = _propose(root)
    (root / ".pending" / "tidy-notes" / version / "run.sh").write_text("echo hi\n")
    client = _client(tmp_path, registry, confirmed=True)

    refused = client.post("/api/skills/tidy-notes/approve", json=_as_shown(client, version))

    assert refused.status_code == 409
    assert _plain(refused.json()["detail"]) == EXTRA_FILES
    assert refused.json()["code"] == "extra_files"
    assert isolated_ledger.read() == {}


def test_every_refusal_code_is_a_key_in_both_catalogs() -> None:
    """Settings shows a 409's copy by its code, so every code needs en and ko copy."""
    import json

    locales = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"
    for lang in ("en", "ko"):
        refusals = json.loads((locales / lang / "skills.json").read_text(encoding="utf-8"))[
            "refusals"
        ]
        assert set(refusals) == set(SKILL_DECISION_CODES), lang
        for text in refusals.values():
            _plain(text)
