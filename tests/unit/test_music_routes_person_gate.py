"""Every music score change needs the person's confirmed window (#2421 round 2).

The music routes once fell back from `person.require` to the cross-origin check, which lets
through any request with no `Origin` header: a plain HTTP POST from any program on the
computer approved a proposal. These tests run the real app with its real gate. A request
with no person cookie and no `Origin` is refused on every route that changes the song, and
nothing on disk moves; the same requests from a window the server confirmed go through.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.support.person import confirm_window
from tests.unit.test_music_ai_scenarios import song_with_pending_violin_patch
from uclone_x.extensions import use_extensions
from uclone_x.llm import MockLLMConnector
from uclone_x.ui.app import create_ui_app
from uclone_x_music.ucx_extension import EXTENSION as MUSIC_EXTENSION
from uclone_x_story.ucx_extension import EXTENSION as STORY_EXTENSION

SONG = "/api/artifacts/library/songs/midnight-nocturne"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def client(tmp_path: Path, workspace: Path) -> Iterator[TestClient]:
    with use_extensions([STORY_EXTENSION, MUSIC_EXTENSION]):
        app = create_ui_app(
            static_dir=tmp_path,
            storage_dir=tmp_path / "sessions",
            workspace_dir=workspace,
            llm=MockLLMConnector(),
        )
        with TestClient(app) as started:
            yield started


def _song(client: TestClient, workspace: Path) -> tuple[str, str]:
    """A song held by a new conversation, with one pending violin patch; `(room, proposal)`."""
    created = client.post("/api/rooms", json={"title": "Composer room", "agent_ids": []})
    assert created.status_code == 201, created.text
    room_id = str(created.json()["room_id"])
    _, proposal_id = asyncio.run(song_with_pending_violin_patch(workspace, room_id))
    return room_id, proposal_id


def _changes(
    client: TestClient, room_id: str, proposal_id: str
) -> list[tuple[str, dict[str, Any]]]:
    """One request for every route that changes the song, built from what the view shows."""
    shown = client.get(SONG).json()
    seen = next(p for p in shown["proposals"] if p["id"] == proposal_id)["seen_digest"]
    track = next(t for t in shown["tracks"] if t["track_id"] == "theme_a_bass_synth")
    return [
        (f"{SONG}/proposals/{proposal_id}/approve", {"seen_digest": seen, "room_id": room_id}),
        (f"{SONG}/proposals/{proposal_id}/reject", {"seen_digest": seen, "room_id": room_id}),
        (f"{SONG}/proposals/approve-all", {"room_id": room_id}),
        (
            f"{SONG}/tracks/theme_a_bass_synth/edit",
            {
                "action": "transpose_bars",
                "start_bar": 1,
                "end_bar": 1,
                "semitones": 2,
                "seen_digest": track["digest"],
            },
        ),
        # After the track edit, which was shown the track as it is before bars are added.
        (f"{SONG}/structure", {"action": "append_bars", "section_id": "theme_a", "count": 1}),
        (f"{SONG}/instruments/drum_kit/mix", {"solo": True}),
    ]


def _song_files(workspace: Path) -> dict[str, bytes]:
    root = workspace / "songs" / "midnight-nocturne"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".history" not in path.parts
    }


def test_no_route_changes_a_song_for_a_request_without_the_persons_window(
    client: TestClient, workspace: Path
) -> None:
    """No cookie and no `Origin`, as curl or any local program sends: refused, nothing moves.

    Killed by: examples/music/src/uclone_x_music/routes.py :: require_person = person.require
    Becomes: require_person = refuse_cross_origin
    """
    room_id, proposal_id = _song(client, workspace)
    before = _song_files(workspace)
    for url, body in _changes(client, room_id, proposal_id):
        refused = client.post(url, json=body)
        assert refused.status_code in (401, 403), (url, refused.status_code, refused.text)
    assert _song_files(workspace) == before


def test_every_route_still_changes_the_song_from_a_confirmed_window(
    client: TestClient, workspace: Path
) -> None:
    """The same requests from a window the server confirmed are answered, one by one."""
    room_id, proposal_id = _song(client, workspace)
    confirm_window(client)
    approve, reject, approve_all, *edits = _changes(client, room_id, proposal_id)
    for url, body in [approve, *edits]:
        resp = client.post(url, json=body)
        assert resp.status_code == 200, (url, resp.status_code, resp.text)
    # Decided once, the proposal cannot be rejected; with none pending, approve-all is a no-op.
    assert client.post(reject[0], json=reject[1]).status_code == 400
    resp = client.post(approve_all[0], json=approve_all[1])
    assert resp.status_code == 200, resp.text
    assert resp.json()["proposals"] == []
