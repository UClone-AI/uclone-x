"""Tests for the Files screen's HTTP routes (#1554).

The routes only translate, so these pin the translation: a refusal keeps its own plain
sentence under the status that says what kind it is, and a failure that is not a refusal
reaches the screen as a fixed sentence -- no class name, path or traceback.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.artifacts.library import READERS_NOT_CLEARED_NOTE, ArtifactLibrary
from uclone_x.errors import StaleRoomWriteError
from uclone_x.llm import MockLLMConnector
from uclone_x.room.service import RoomService
from uclone_x.story.library import StoryLibrary
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.artifacts import FILES_FAILURE_DETAIL, FILES_READ_FAILURE_DETAIL


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture
def client(tmp_path: Path, workspace: Path) -> Iterator[TestClient]:
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "sessions",
        workspace_dir=workspace,
        llm=MockLLMConnector(),
    )
    with TestClient(app) as started:
        yield started


def _room(client: TestClient, title: str) -> str:
    created = client.post("/api/rooms", json={"title": title, "agent_ids": []})
    assert created.status_code == 201, created.text
    return str(created.json()["room_id"])


def _write(workspace: Path, relative: str, text: str = "hello") -> None:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_the_survey_lists_files_with_what_it_covers(client: TestClient, workspace: Path) -> None:
    _write(workspace, "artifacts/a.md")

    body = client.get("/api/artifacts/library").json()

    assert [e["path"] for e in body["entries"]] == ["artifacts/a.md"]
    assert body["scope_note"].startswith("This list shows the files in the workspace's")
    assert body["record_gaps"] == []


def test_a_file_opens_as_text(client: TestClient, workspace: Path) -> None:
    _write(workspace, "artifacts/a.md", "# Title")

    opened = client.get("/api/artifacts/library/file", params={"path": "artifacts/a.md"})

    assert opened.status_code == 200
    assert opened.json() == {
        "path": "artifacts/a.md",
        "name": "a.md",
        "kind": "document",
        "text": "# Title",
    }


def test_archive_and_restore_round_trip(client: TestClient, workspace: Path) -> None:
    _write(workspace, "artifacts/a.md")

    archived = client.post("/api/artifacts/library/archive", json={"path": "artifacts/a.md"})
    assert archived.json() == {"path": ".archive/artifacts/a.md", "note": None}
    restored = client.post(
        "/api/artifacts/library/restore", json={"path": ".archive/artifacts/a.md"}
    )
    assert restored.json() == {"path": "artifacts/a.md"}
    assert (workspace / "artifacts" / "a.md").exists()


def test_delete_without_confirmation_is_a_400_with_the_plain_sentence(
    client: TestClient, workspace: Path
) -> None:
    _write(workspace, "artifacts/a.md")

    refused = client.post("/api/artifacts/library/delete", json={"path": "artifacts/a.md"})

    assert refused.status_code == 400
    assert refused.json()["detail"] == (
        "Deleting cannot be undone, so it needs your confirmation. Nothing was deleted."
    )
    assert (workspace / "artifacts" / "a.md").exists()

    deleted = client.post(
        "/api/artifacts/library/delete", json={"path": "artifacts/a.md", "confirm": True}
    )
    assert deleted.status_code == 200
    assert not (workspace / "artifacts" / "a.md").exists()


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("delete", {"confirm": "yes"}),
        ("delete", {"confirm": 1}),
        ("delete", {"confirm": "true"}),
    ],
)
def test_only_a_json_true_confirms_a_delete_request(
    client: TestClient, workspace: Path, route: str, body: dict[str, object]
) -> None:
    """`"yes"` and `1` are not `true`: the request is refused and nothing moves (#1578).

    Killed by: src/uclone_x/ui/artifacts.py :: confirm: StrictBool = False
    Becomes: confirm: bool = False
    """
    _write(workspace, "artifacts/a.md")

    refused = client.post(
        f"/api/artifacts/library/{route}", json={"path": "artifacts/a.md", **body}
    )

    assert refused.status_code == 422, refused.text
    assert "Traceback" not in refused.text
    assert (workspace / "artifacts" / "a.md").exists()
    assert not (workspace / ".archive").exists()


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("delete", {"confirm": True, "release_writer": "yes"}),
        ("archive", {"release_writer": 1}),
        ("archive", {"release_writer": "true"}),
    ],
)
def test_only_a_json_true_goes_ahead_over_a_writer(
    client: TestClient, workspace: Path, route: str, body: dict[str, object]
) -> None:
    """`release_writer` is as strict as `confirm`, on both routes (#1578).

    Killed by: src/uclone_x/ui/artifacts.py :: release_writer: StrictBool = False
    Becomes: release_writer: bool = False
    """
    _write(workspace, "artifacts/a.md")

    refused = client.post(
        f"/api/artifacts/library/{route}", json={"path": "artifacts/a.md", **body}
    )

    assert refused.status_code == 422, refused.text
    assert (workspace / "artifacts" / "a.md").exists()
    assert not (workspace / ".archive").exists()


@pytest.mark.parametrize(("route", "extra"), [("archive", {}), ("delete", {"confirm": True})])
def test_a_story_whose_readers_were_not_all_cleared_says_so(
    client: TestClient,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    extra: dict[str, object],
) -> None:
    """The change stands and the answer carries the plain note, not only the log (#1578).

    Killed by: src/uclone_x/ui/artifacts.py :: return {"path": moved.path, "note": moved.note}
    Becomes: return {"path": moved.path, "note": None}
    Killed by: src/uclone_x/ui/artifacts.py :: return {"deleted": body.path, "note": removed.note}
    Becomes: return {"deleted": body.path, "note": None}
    """
    room_id = _room(client, "Writing room")
    story_id = StoryLibrary(workspace).create("Night Train", room_id).story_id

    def stale(self: RoomService, story: str) -> tuple[str, ...]:
        raise StaleRoomWriteError("moved on")

    monkeypatch.setattr(RoomService, "forget_story", stale)

    done = client.post(
        f"/api/artifacts/library/{route}",
        json={"path": f"stories/{story_id}", "release_writer": True, **extra},
    )

    assert done.status_code == 200, done.text
    assert done.json()["note"] == READERS_NOT_CLEARED_NOTE
    assert not (workspace / "stories" / story_id).exists()


def test_a_missing_file_is_a_404(client: TestClient) -> None:
    refused = client.post("/api/artifacts/library/archive", json={"path": "artifacts/gone.md"})

    assert refused.status_code == 404
    assert refused.json()["detail"] == (
        "There is no file called gone.md there any more. Refresh the list to see what is there."
    )


def test_a_path_outside_the_workspace_is_refused_without_naming_it(client: TestClient) -> None:
    refused = client.post(
        "/api/artifacts/library/delete", json={"path": "../../etc/hosts", "confirm": True}
    )

    assert refused.status_code == 400
    assert refused.json()["detail"] == "That path is outside the workspace, so nothing was changed."


def test_a_story_in_use_is_a_409_until_the_caller_goes_ahead(
    client: TestClient, workspace: Path
) -> None:
    room_id = _room(client, "Writing room")
    story_id = StoryLibrary(workspace).create("Night Train", room_id).story_id

    refused = client.post("/api/artifacts/library/archive", json={"path": f"stories/{story_id}"})
    assert refused.status_code == 409
    assert refused.json()["detail"] == (
        "The conversation “Writing room” is writing this story. Delete that conversation "
        "first, or go ahead anyway to stop it writing to this story."
    )

    went_ahead = client.post(
        "/api/artifacts/library/archive",
        json={"path": f"stories/{story_id}", "release_writer": True},
    )
    assert went_ahead.status_code == 200, went_ahead.text


_ANSWERING = (
    "The conversation “Writing room” is answering right now, so it cannot be stopped "
    "from writing this story yet. Wait for the answer to finish, then try again."
)


def _refused_while(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch, predicate: str
) -> None:
    """Go ahead on a story whose writer the stack reports running by `predicate` alone."""
    room_id = _room(client, "Writing room")
    story_id = StoryLibrary(workspace).create("Night Train", room_id).story_id
    stack = cast(Any, client.app).state.room_stack
    assert not stack.turn_in_flight(room_id) and not stack.turn_unlanded(room_id)

    def running(asked: str) -> bool:
        return asked == room_id

    monkeypatch.setattr(stack, predicate, running)

    for route, extra in (("archive", {}), ("delete", {"confirm": True})):
        refused = client.post(
            f"/api/artifacts/library/{route}",
            json={"path": f"stories/{story_id}", "release_writer": True, **extra},
        )
        assert refused.status_code == 409, refused.text
        assert refused.json()["detail"] == _ANSWERING
    assert (workspace / "stories" / story_id).is_dir()
    assert StoryLibrary(workspace).load(story_id).lease is not None


def test_going_ahead_is_refused_while_the_writer_is_answering(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The route hands the library the stack's real question, not a stand-in (#1578).

    Killed by: src/uclone_x/ui/artifacts.py :: stack.turn_in_flight(room_id) or stack.turn_unlanded(room_id)
    Becomes: False or stack.turn_unlanded(room_id)
    """
    _refused_while(client, workspace, monkeypatch, "turn_in_flight")


def test_going_ahead_is_refused_while_the_writer_is_retrying(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry runs its turn inside the request, where `turn_in_flight` cannot see it.

    Killed by: src/uclone_x/ui/artifacts.py :: stack.turn_in_flight(room_id) or stack.turn_unlanded(room_id)
    Becomes: stack.turn_in_flight(room_id)
    """
    _refused_while(client, workspace, monkeypatch, "turn_unlanded")


def test_deleting_the_writer_conversation_frees_the_story(
    client: TestClient, workspace: Path
) -> None:
    """The 409 names deleting that conversation as a remedy, so it must be one."""
    room_id = _room(client, "Writing room")
    story_id = StoryLibrary(workspace).create("Night Train", room_id).story_id
    assert client.delete(f"/api/rooms/{room_id}").status_code in (200, 204)

    archived = client.post("/api/artifacts/library/archive", json={"path": f"stories/{story_id}"})

    assert archived.status_code == 200, archived.text


def test_a_story_opens_into_a_conversation(client: TestClient, workspace: Path) -> None:
    writer = _room(client, "Writing room")
    story_id = StoryLibrary(workspace).create("Night Train", writer).story_id
    client.delete(f"/api/rooms/{writer}")  # gives the lease back
    room_id = _room(client, "New conversation")

    opened = client.post(
        f"/api/artifacts/library/stories/{story_id}/open", json={"room_id": room_id}
    )

    assert opened.status_code == 200, opened.text
    assert opened.json() == {
        "room_id": room_id,
        "story_id": story_id,
        "title": "Night Train",
        "writable": True,
        "note": None,
    }
    stack = cast(Any, client.app).state.room_stack
    assert stack.service.get(room_id).story_id == story_id


def test_opening_into_a_missing_conversation_is_a_404(client: TestClient, workspace: Path) -> None:
    writer = _room(client, "Writing room")
    story_id = StoryLibrary(workspace).create("Night Train", writer).story_id

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/open", json={"room_id": "room-gone"}
    )

    assert refused.status_code == 404
    assert refused.json()["detail"] == (
        "That conversation no longer exists, so the story was not opened."
    )


def test_a_failure_that_is_not_a_refusal_is_a_plain_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fails(self: ArtifactLibrary) -> None:
        raise PermissionError(13, "Permission denied", "/Users/someone/workspace/artifacts")

    monkeypatch.setattr(ArtifactLibrary, "survey", _fails)

    failed = client.get("/api/artifacts/library")

    assert failed.status_code == 500
    assert failed.json()["detail"] == FILES_READ_FAILURE_DETAIL
    assert (
        FILES_READ_FAILURE_DETAIL == "The files could not be read. The reason is in the server log."
    )


def test_a_move_that_fails_is_a_plain_500(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(workspace, "artifacts/a.md")

    def _fails(*_: object) -> None:
        raise PermissionError(13, "Permission denied", "/Users/someone/workspace/artifacts")

    monkeypatch.setattr("uclone_x.artifacts.library.os.replace", _fails)

    failed = client.post("/api/artifacts/library/archive", json={"path": "artifacts/a.md"})

    assert failed.status_code == 500
    assert failed.json()["detail"] == FILES_FAILURE_DETAIL
    assert (
        FILES_FAILURE_DETAIL == "The files could not be changed. The reason is in the server log."
    )


# -- the story view (#1560) ---------------------------------------------------------------


def _story_with_proposal(client: TestClient, workspace: Path) -> tuple[str, str]:
    """A story written by “Writing room”, with Vane and a pending proposal he is wounded."""
    room_id = _room(client, "Writing room")
    library = StoryLibrary(workspace)
    story_id = library.create("Night Train", room_id).story_id
    root = workspace / "stories" / story_id
    _write(workspace, f"stories/{story_id}/codex/characters/vane.yaml", "id: vane\nname: Vane\n")
    entry_digest = library.read_file(story_id, "codex/characters/vane.yaml").digest
    (root / "proposals").mkdir()
    (root / "proposals" / "p001.yaml").write_text(
        "id: p001\nkind: characters\nentry_id: vane\n"
        "change:\n  progression:\n    at: ch01.s01\n    set: {wounded: true}\n"
        f"proposed_at: '2026-09-25T10:00:00+00:00'\nroom_id: {room_id}\n"
        f"entry_digest: {entry_digest}\n",
        encoding="utf-8",
    )
    return story_id, room_id


def test_the_story_view_shows_the_pending_proposal(client: TestClient, workspace: Path) -> None:
    story_id, _ = _story_with_proposal(client, workspace)

    shown = client.get(f"/api/artifacts/library/stories/{story_id}")

    assert shown.status_code == 200, shown.text
    body = shown.json()
    assert body["title"] == "Night Train"
    assert body["outline_note"] == "This story has no outline yet."
    assert [p["id"] for p in body["pending"]] == ["p001"]
    assert body["pending"][0]["changes"] == [
        {"what": "wounded", "at": "ch01.s01", "before": None, "after": True, "placed": False}
    ]


def test_approving_in_the_story_view_applies_the_change(
    client: TestClient, workspace: Path
) -> None:
    story_id, _ = _story_with_proposal(client, workspace)
    seen = client.get(f"/api/artifacts/library/stories/{story_id}").json()["pending"][0]["digest"]

    approved = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": seen},
    )

    assert approved.status_code == 200, approved.text
    assert approved.json() == {"proposal_id": "p001", "decision": "applied", "notes": []}
    entry = (workspace / "stories" / story_id / "codex/characters/vane.yaml").read_text()
    assert "wounded: true" in entry


def test_a_decision_on_a_changed_proposal_is_a_400_with_the_plain_sentence(
    client: TestClient, workspace: Path
) -> None:
    story_id, _ = _story_with_proposal(client, workspace)

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/reject",
        json={"seen_digest": "0000", "reason": "no"},
    )

    assert refused.status_code == 400
    assert refused.json()["detail"] == (
        "Proposal 'p001' changed after it was shown, so nothing was changed. Look at it again "
        "before deciding."
    )


def test_a_decision_while_the_writer_answers_is_a_409(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/artifacts.py :: if isinstance(exc, (StoryInUseError, WriterBusyError)):
    Becomes: if isinstance(exc, StoryInUseError):
    """
    story_id, room_id = _story_with_proposal(client, workspace)
    seen = client.get(f"/api/artifacts/library/stories/{story_id}").json()["pending"][0]["digest"]
    stack = cast(Any, client.app).state.room_stack

    def answering(asked: str) -> bool:
        return asked == room_id

    monkeypatch.setattr(stack, "turn_unlanded", answering)

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": seen},
    )

    assert refused.status_code == 409
    assert refused.json()["detail"] == (
        "The conversation “Writing room” is answering right now, so the decision was not "
        "saved. Wait for the answer to finish, then try again."
    )


def test_an_approval_cannot_carry_anything_but_what_was_seen(
    client: TestClient, workspace: Path
) -> None:
    story_id, _ = _story_with_proposal(client, workspace)

    refused = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": "x", "entry_digest": "y"},
    )

    assert refused.status_code == 422


def test_a_story_that_is_gone_is_a_404_in_the_story_view(client: TestClient) -> None:
    refused = client.get("/api/artifacts/library/stories/night-train")

    assert refused.status_code == 404
    assert refused.json()["detail"] == (
        "There is no story called 'night-train' any more. Refresh the list to see the stories "
        "there are."
    )


def test_a_story_view_failure_that_is_not_a_refusal_is_a_plain_500(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    story_id, _ = _story_with_proposal(client, workspace)
    seen = client.get(f"/api/artifacts/library/stories/{story_id}").json()["pending"][0]["digest"]

    def _fails(*_: object, **__: object) -> None:
        raise PermissionError(13, "Permission denied", "/Users/someone/workspace/stories")

    monkeypatch.setattr("uclone_x.story.view.apply_proposal", _fails)

    failed = client.post(
        f"/api/artifacts/library/stories/{story_id}/proposals/p001/approve",
        json={"seen_digest": seen},
    )

    assert failed.status_code == 500
    assert failed.json()["detail"] == FILES_FAILURE_DETAIL
