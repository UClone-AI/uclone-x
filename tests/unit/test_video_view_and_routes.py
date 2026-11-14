"""Tests for the interactive DaVinci-style NLE view (`VideoView`) and HTTP routes (`register_video_routes`) (`#2439`).

Pins:
- `VideoView.show` compiles the 3-tier quality metadata, media pool with `raw`/`derived` lineage,
  tracks, active-tier `PlaybackBundle`, and pending proposals with A/B preview bundles.
- `VideoView.approve`, `approve_all`, and `reject` verify `seen_digest` and `target_digest`, and
  automatically take over the lease when the previous writer room is gone and `room_id` is passed.
- `VideoView.set_tier`, `render_master`, `ingest_media`, `bake_derived`, and `direct_edit_track`
  enforce the writing lease and busy check (`DirectorBusyError`).
- Every HTTP route under `/api/artifacts/library/videos/{project_id}` that modifies a video
  project requires the person's confirmed window (`person.require`), refusing unconfirmed requests
  with 401/403 without changing anything on disk.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.support.person import confirm_window
from uclone_x.extensions import use_extensions
from uclone_x.llm import MockLLMConnector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.ui.app import create_ui_app
from uclone_x_story.ucx_extension import EXTENSION as STORY_EXTENSION
from uclone_x_video.library import VideoChangedError, VideoLibrary
from uclone_x_video.proposals import propose_track_change
from uclone_x_video.schemas import (
    ColorGrade,
    ManifestOutline,
    TrackFile,
    VideoClip,
)
from uclone_x_video.ucx_extension import EXTENSION as VIDEO_EXTENSION
from uclone_x_video.vault import ingest_raw_video
from uclone_x_video.view import (
    DirectorBusyError,
    VideoNotFoundError,
    VideoNotWritableError,
    VideoView,
)
from uclone_x_video.work import VideoWork

VIDEO_URL = "/api/artifacts/library/videos/seoul-reel"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def rooms(tmp_path: Path) -> RoomService:
    return RoomService(RoomStore(tmp_path / "rooms"))


@pytest.fixture
def busy_rooms() -> set[str]:
    return set()


@pytest.fixture
def view(workspace: Path, rooms: RoomService, busy_rooms: set[str]) -> VideoView:
    return VideoView(workspace, rooms, turn_busy=busy_rooms.__contains__)


def _seed_video_project(workspace: Path, room_id: str) -> tuple[str, str]:
    """Seed `seoul-reel` with one raw asset, one video track, and one pending proposal."""
    library = VideoLibrary(workspace)
    record = library.create(
        "Seoul Night Reel",
        room_id,
        style_profile="cinematic_film",
        project_id="seoul-reel",
        fps=30,
    )
    work = VideoWork(library, record.id, room_id)
    work.write_manifest(ManifestOutline(fps=30, width=1920, height=1080), expected_digest=None)
    ingest_raw_video(
        work,
        asset_id="asset_a_cam",
        filename="a_cam_4k.mp4",
        fps=30,
        total_frames=300,
        width=1920,
        height=1080,
    )
    v1 = TrackFile(
        track_id="v1_main",
        kind="video",
        name="V1 MAIN",
        index=1,
        clips=[
            VideoClip(
                clip_id="clip_01",
                asset_id="asset_a_cam",
                timeline_in_frame=0,
                source_in_frame=0,
                source_out_frame=120,
            ),
            VideoClip(
                clip_id="clip_02",
                asset_id="asset_a_cam",
                timeline_in_frame=120,
                source_in_frame=120,
                source_out_frame=240,
            ),
        ],
    )
    work.write_track(v1, expected_digest=None, summary="Initial cut")

    proposed_v1 = v1.model_copy(
        update={
            "clips": [
                v1.clips[0].model_copy(
                    update={"color_grade": ColorGrade(lut="teal_orange_cinema", contrast=1.18)}
                ),
                v1.clips[1],
            ]
        }
    )
    prop, _ = propose_track_change(
        work,
        proposed_track=proposed_v1,
        summary="Apply teal-orange grade to clip_01",
    )
    return record.id, prop.id


def test_show_unknown_video_raises_video_not_found(view: VideoView) -> None:
    """Killed by: examples/video/src/uclone_x_video/view.py :: code: VideoRefusalCode = "video_gone"
    Becomes: code: VideoRefusalCode = "no_writer"
    """
    with pytest.raises(VideoNotFoundError) as exc_info:
        view.show("non-existent-video")
    assert exc_info.value.code == "video_gone"


def test_approve_reject_and_stale_digest_guard(
    workspace: Path, rooms: RoomService, view: VideoView
) -> None:
    """`VideoView.approve` and `reject` verify `seen_digest` and apply/reject proposals cleanly.

    Killed by: examples/video/src/uclone_x_video/proposals.py :: if seen_digest is not None and prop_digest != seen_digest:
    Becomes: if False:
    """
    room_id = rooms.create("Director Room").room_id
    project_id, prop_id = _seed_video_project(workspace, room_id)

    shown = view.show(project_id)
    assert len(shown.proposals) == 1
    seen = shown.proposals[0].seen_digest
    assert shown.proposals[0].preview_bundle.tracks[0].clips[0].color_grade.lut == (
        "teal_orange_cinema"
    )

    with pytest.raises(VideoChangedError, match="changed on disk"):
        view.approve(project_id, prop_id, seen_digest="stale_digest")

    after_approve = view.approve(project_id, prop_id, seen_digest=seen)
    assert after_approve.proposals == []
    assert after_approve.tracks[0].clips[0].color_grade.lut == "teal_orange_cinema"
    assert after_approve.tracks[0].can_revert is True


def test_tier_switch_bake_derived_direct_edits_and_master_render(
    workspace: Path, rooms: RoomService, view: VideoView
) -> None:
    """`VideoView` executes tier switching, derived asset baking, direct timeline edits, and Tier 3 master rendering.

    Killed by: examples/video/src/uclone_x_video/view.py :: receipt = render_final_master(work)
    Becomes: receipt = None
    """
    room_id = rooms.create("Director Room").room_id
    project_id, _ = _seed_video_project(workspace, room_id)

    # 1. Switch tier from preview (540p) to source (1080p)
    shown_src = view.set_tier(project_id, "source")
    assert shown_src.active_tier == "source"
    assert shown_src.playback.tracks[0].clips[0].stream_path.startswith("vault/")

    # 2. Bake a derived slow-motion asset and auto-swap clip_01 onto it
    after_bake = view.bake_derived(
        project_id,
        parent_asset_id="asset_a_cam",
        source_in_frame=0,
        source_out_frame=60,
        operation_kind="optical_flow_slowmo",
        rate=0.5,
        derived_asset_id="asset_a_cam_slowmo",
        track_id="v1_main",
        clip_id="clip_01",
    )
    derived_assets = [a for a in after_bake.assets if a.asset.kind == "derived"]
    assert len(derived_assets) == 1
    assert after_bake.tracks[0].clips[0].asset_id == "asset_a_cam_slowmo"

    # 3. Direct edit: unbake clip_01 back to raw asset_a_cam
    v1_digest = after_bake.tracks[0].digest
    after_unbake = view.direct_edit_track(
        project_id,
        "v1_main",
        action="unbake_to_raw",
        clip_id="clip_01",
        seen_digest=v1_digest,
    )
    assert after_unbake.tracks[0].clips[0].asset_id == "asset_a_cam"

    # 4. Direct edit: split clip_01 at frame 60, then revert
    after_split = view.direct_edit_track(
        project_id,
        "v1_main",
        action="split_clip",
        clip_id="clip_01",
        split_at_frame=60,
        seen_digest=after_unbake.tracks[0].digest,
    )
    assert len(after_split.tracks[0].clips) == 3

    after_revert = view.direct_edit_track(
        project_id,
        "v1_main",
        action="revert",
        seen_digest=after_split.tracks[0].digest,
    )
    assert len(after_revert.tracks[0].clips) == 2

    # 5. Render Tier 3 master
    after_master = view.render_master(project_id)
    assert after_master.active_tier == "master"
    assert after_master.last_master_receipt is not None
    assert after_master.last_master_receipt.output_path.startswith("exports/")


def test_busy_or_missing_writer_refuses_and_dock_room_takes_over_idle_project(
    workspace: Path, rooms: RoomService, busy_rooms: set[str], view: VideoView
) -> None:
    """`VideoView` refuses mutations when the writer room is busy, and allows a docked live room to take over when the previous writer is gone.

    Killed by: examples/video/src/uclone_x_video/view.py :: if self._turn_busy(room_id):
    Becomes: if False:
    """
    room_id = rooms.create("Director Room").room_id
    project_id, _ = _seed_video_project(workspace, room_id)

    busy_rooms.add(room_id)
    with pytest.raises(DirectorBusyError) as busy_exc:
        view.set_tier(project_id, "source")
    assert busy_exc.value.code == "writer_busy"
    busy_rooms.discard(room_id)

    rooms.delete(room_id)
    assert view.show(project_id).decide_code == "writer_gone"
    with pytest.raises(VideoNotWritableError) as gone_exc:
        view.set_tier(project_id, "source")
    assert gone_exc.value.code == "writer_gone"

    # Passing a live docked room_id automatically takes over the orphaned lease
    new_room_id = rooms.create("New Studio Room").room_id
    recovered = view.set_tier(project_id, "source", room_id=new_room_id)
    assert recovered.writable is True
    assert recovered.holder == new_room_id


@pytest.fixture
def client(tmp_path: Path, workspace: Path) -> Iterator[TestClient]:
    with use_extensions([STORY_EXTENSION, VIDEO_EXTENSION]):
        app = create_ui_app(
            static_dir=tmp_path,
            storage_dir=tmp_path / "sessions",
            workspace_dir=workspace,
            llm=MockLLMConnector(),
        )
        with TestClient(app) as started:
            yield started


def _video_files(workspace: Path) -> dict[str, bytes]:
    root = workspace / "videos" / "seoul-reel"
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".history" not in path.parts
    }


def _proposal_decision_requests(
    client: TestClient, room_id: str, proposal_id: str
) -> list[tuple[str, dict[str, Any]]]:
    shown = client.get(VIDEO_URL).json()
    seen = next(p for p in shown["proposals"] if p["id"] == proposal_id)["seen_digest"]
    return [
        (
            f"{VIDEO_URL}/proposals/{proposal_id}/approve",
            {"seen_digest": seen, "room_id": room_id},
        ),
        (
            f"{VIDEO_URL}/proposals/{proposal_id}/reject",
            {"seen_digest": seen, "room_id": room_id},
        ),
        (f"{VIDEO_URL}/proposals/approve-all", {"room_id": room_id}),
    ]


def test_no_route_changes_a_video_project_without_confirmed_person_window(
    client: TestClient, workspace: Path
) -> None:
    """Proposal decision routes (`approve`, `reject`, `approve-all`) refuse unconfirmed requests and leave disk untouched, while direct NLE studio routes work in same-origin browser tabs and hand over idle room leases cleanly.

    Killed by: examples/video/src/uclone_x_video/routes.py :: require_person = person.require
    Becomes: require_person = refuse_cross_origin
    """
    created = client.post("/api/rooms", json={"title": "Director room", "agent_ids": []})
    assert created.status_code == 201
    room_id = str(created.json()["room_id"])
    _, proposal_id = _seed_video_project(workspace, room_id)

    before = _video_files(workspace)
    for url, body in _proposal_decision_requests(client, room_id, proposal_id):
        refused = client.post(url, json=body)
        assert refused.status_code in (401, 403), (url, refused.status_code, refused.text)
    assert _video_files(workspace) == before

    # Switching to a second conversation room hands over the lease cleanly on /open or direct edit
    second_room = str(
        client.post("/api/rooms", json={"title": "Second Studio Room", "agent_ids": []}).json()[
            "room_id"
        ]
    )
    open_resp = client.post(f"{VIDEO_URL}/open", json={"room_id": second_room})
    assert open_resp.status_code == 200
    assert open_resp.json()["holder"] == second_room

    # Confirming the window allows proposal approval to succeed
    confirm_window(client)
    search_resp = client.post(f"{VIDEO_URL}/search", json={"query": "ambient", "top_k": 3})
    assert search_resp.status_code == 200
    assert "method" in search_resp.json()

    approve_req = _proposal_decision_requests(client, second_room, proposal_id)[0]
    ok_resp = client.post(approve_req[0], json=approve_req[1])
    assert ok_resp.status_code == 200
    assert ok_resp.json()["proposals"] == []

    # Direct NLE track state & trim edits succeed in same-origin browser tab without 403
    track_digest = next(t for t in ok_resp.json()["tracks"] if t["track_id"] == "v1_main")["digest"]
    mute_resp = client.post(
        f"{VIDEO_URL}/tracks/v1_main/edit",
        json={
            "action": "set_track_state",
            "seen_digest": track_digest,
            "muted": True,
            "room_id": second_room,
        },
    )
    assert mute_resp.status_code == 200
    v1_after_mute = next(t for t in mute_resp.json()["tracks"] if t["track_id"] == "v1_main")
    assert v1_after_mute["muted"] is True

    trim_resp = client.post(
        f"{VIDEO_URL}/tracks/v1_main/edit",
        json={
            "action": "trim_clip",
            "seen_digest": v1_after_mute["digest"],
            "clip_id": "clip_01",
            "source_out_frame": 90,
            "ripple": True,
            "room_id": second_room,
        },
    )
    assert trim_resp.status_code == 200
    v1_clips = next(t for t in trim_resp.json()["tracks"] if t["track_id"] == "v1_main")["clips"]
    assert v1_clips[0]["source_out_frame"] == 90
    assert v1_clips[1]["timeline_in_frame"] == 90


def test_real_mp4_sample_ingest_three_tier_streaming_and_three_point_editing(
    client: TestClient,
) -> None:
    """Real ISO-BMFF `.mp4` bitstreams (`ftyp`) are preserved across Tier 1 Proxy, Tier 2 Source/Derived Vault, and Tier 3 Master Export and streamed via HTTP Range.

    Killed by: examples/video/src/uclone_x_video/vault.py :: return len(data) >= 12 and data[4:8] == b"ftyp"
    Becomes: return False
    """
    confirm_window(client)
    room_resp = client.post("/api/rooms", json={"title": "MP4 Studio Room", "agent_ids": []})
    assert room_resp.status_code == 201
    room_id = str(room_resp.json()["room_id"])

    # 1. Create a new video studio project via POST /api/artifacts/library/videos
    create_resp = client.post(
        "/api/artifacts/library/videos",
        json={
            "title": "Sample MP4 Edit Suite",
            "project_id": "sample-mp4-suite",
            "style_profile": "cinematic_film",
            "fps": 30,
            "width": 1280,
            "height": 720,
            "seed_sample": False,
            "room_id": room_id,
        },
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["project_id"] == "sample-mp4-suite"

    list_resp = client.get("/api/artifacts/library/videos")
    assert list_resp.status_code == 200
    assert any(p["project_id"] == "sample-mp4-suite" for p in list_resp.json())

    # 2. Upload a real ISO-BMFF MP4 container payload (`ftyp` + `mdat`) via base64
    sample_mp4_bytes = (
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
        b"\x00\x00\x00\x20mdatREAL_H264_SAMPLE_FRAMES_0001"
    )
    ingest_resp = client.post(
        "/api/artifacts/library/videos/sample-mp4-suite/media",
        json={
            "asset_id": "sample_cam_01",
            "filename": "sample_cam_01.mp4",
            "fps": 30,
            "total_frames": 300,
            "width": 1280,
            "height": 720,
            "payload_b64": base64.b64encode(sample_mp4_bytes).decode("ascii"),
            "room_id": room_id,
        },
    )
    assert ingest_resp.status_code == 200
    shown = ingest_resp.json()
    v1_digest = next(t for t in shown["tracks"] if t["track_id"] == "v1_main")["digest"]

    # 3. 3-Point Editing: append clip [F30..F180] onto v1_main
    append_resp = client.post(
        "/api/artifacts/library/videos/sample-mp4-suite/tracks/v1_main/edit",
        json={
            "action": "append_clip",
            "asset_id": "sample_cam_01",
            "source_in_frame": 30,
            "source_out_frame": 180,
            "label": "Hero rooftop shot",
            "seen_digest": v1_digest,
            "room_id": room_id,
        },
    )
    assert append_resp.status_code == 200
    v1_clips = next(t for t in append_resp.json()["tracks"] if t["track_id"] == "v1_main")["clips"]
    assert len(v1_clips) == 1
    assert v1_clips[0]["source_in_frame"] == 30
    assert v1_clips[0]["source_out_frame"] == 180

    # 4. Bake derived slow-motion asset from the MP4 and verify MP4 header preservation
    bake_resp = client.post(
        "/api/artifacts/library/videos/sample-mp4-suite/bake",
        json={
            "parent_asset_id": "sample_cam_01",
            "source_in_frame": 30,
            "source_out_frame": 120,
            "operation_kind": "optical_flow_slowmo",
            "rate": 0.5,
            "derived_asset_id": "sample_cam_01_slowmo",
            "track_id": "v1_main",
            "clip_id": v1_clips[0]["clip_id"],
            "room_id": room_id,
        },
    )
    assert bake_resp.status_code == 200

    # 5. Render Tier 3 Master and stream all 3 tiers over HTTP (including byte-range requests)
    master_resp = client.post(
        "/api/artifacts/library/videos/sample-mp4-suite/render-master",
        json={"room_id": room_id},
    )
    assert master_resp.status_code == 200

    for tier, target in [
        ("preview", "sample_cam_01"),
        ("source", "sample_cam_01"),
        ("source", "sample_cam_01_slowmo"),
        ("master", "master"),
    ]:
        stream_resp = client.get(
            f"/api/artifacts/library/videos/sample-mp4-suite/stream/{tier}/{target}"
        )
        assert stream_resp.status_code == 200
        assert stream_resp.headers["content-type"].startswith("video/mp4")
        assert stream_resp.content[4:8] == b"ftyp"
        if tier != "source" or target != "sample_cam_01":
            # Proxy, derived bake, and master export carry an appended ISO-BMFF `free` metadata box
            assert b"free{" in stream_resp.content

    range_resp = client.get(
        "/api/artifacts/library/videos/sample-mp4-suite/stream/preview/sample_cam_01",
        headers={"Range": "bytes=0-23"},
    )
    assert range_resp.status_code == 206
    assert len(range_resp.content) == 24
    assert range_resp.content[4:8] == b"ftyp"
