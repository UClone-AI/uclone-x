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

from uclone_x.artifacts.library import ArtifactLibrary
from uclone_x.llm import MockLLMConnector
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
    assert archived.json() == {"path": ".archive/artifacts/a.md"}
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
