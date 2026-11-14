"""Unit tests for `uclone_x_video.indexer`, `tool`, `tools`, `start`, `section_turn`, and `ucx_extension` (`#2439`).

Pins:
- `index_asset_frames` and `search_project_frames` index and rank frames via `EmbedderProtocol` +
  `BruteForceVectorStore` (`index/vectors.bin`), explicitly attribute `method`, and fail fast on
  `EmbeddingError` / `EmbeddingDimensionError` without silent lexical fallback (P6).
- `VideoLibraryTool` creates, opens, lists, takes over, and closes video projects (`OPEN_VIDEO_KEY`).
- `VideoMediaVaultTool`, `VideoSearchTool`, `VideoManifestTool`, `VideoTimelineTool`,
  `VideoFxColorTool`, `VideoContextTool`, `VideoProposalTool`, and `VideoAuditTool` execute
  non-destructive manifest edits, stage proposals for existing tracks, and bake derived assets.
- `VideoStartTool` walks through all 5 guided video production stages (`rough_cut_ready` ->
  `broll_layered` -> `derived_baked` -> `graded_and_titled` -> `completed`).
- `ProtectedRoot(dirname="videos")` refuses general `FileWriteTool` writes inside `videos/`.
- `VideoTurnHook` injects `[Video timeline context]` and advances `context.story_id`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.errors import EmbeddingDimensionError, EmbeddingError, PlainRefusalError
from uclone_x.extensions import protected_roots, use_extensions
from uclone_x.tools.builtin.filesystem import FileWriteParams, FileWriteTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import create_default_registry
from uclone_x_video import OPEN_VIDEO_KEY
from uclone_x_video.indexer import (
    FrameSeedSpec,
    index_asset_frames,
    search_project_frames,
)
from uclone_x_video.library import VideoLibrary
from uclone_x_video.schemas import BakeOperation, SequenceMarker
from uclone_x_video.section_turn import VIDEO_SECTION_HEADING, VideoTurnHook
from uclone_x_video.start import VideoStartParams, VideoStartTool
from uclone_x_video.tool import VideoLibraryParams, VideoLibraryTool
from uclone_x_video.tools import (
    VideoAuditParams,
    VideoAuditTool,
    VideoContextParams,
    VideoContextTool,
    VideoFxColorParams,
    VideoFxColorTool,
    VideoManifestParams,
    VideoManifestTool,
    VideoMediaVaultParams,
    VideoMediaVaultTool,
    VideoProposalParams,
    VideoProposalTool,
    VideoSearchParams,
    VideoSearchTool,
    VideoTimelineParams,
    VideoTimelineTool,
)
from uclone_x_video.ucx_extension import EXTENSION as VIDEO_EXTENSION
from uclone_x_video.vault import ingest_raw_video
from uclone_x_video.work import VideoWork


class _MockEmbedder:
    """Deterministic test embedder implementing `EmbedderProtocol`."""

    model_name = "mock-video-embed-v1"
    dimensions = 4

    def __init__(self, *, fail_error: Exception | None = None, bad_dims: bool = False) -> None:
        self._fail_error = fail_error
        self._bad_dims = bad_dims

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if self._fail_error is not None:
            raise self._fail_error
        if self._bad_dims:
            return tuple((1.0, 0.0) for _ in texts)
        out: list[tuple[float, ...]] = []
        for t in texts:
            low = t.casefold()
            v0 = 1.0 if ("barista" in low or "coffee" in low or "latte" in low) else 0.05
            v1 = 1.0 if ("sunset" in low or "skyline" in low or "golden" in low) else 0.05
            v2 = 1.0 if ("wide" in low or "roastery" in low) else 0.05
            v3 = 0.2
            out.append((v0, v1, v2, v3))
        return tuple(out)


def _ctx(
    tmp_path: Path,
    *,
    room_id: str = "room-director",
    project_id: str | None = None,
    approved: bool = False,
    turn_aid_sections: tuple[str, ...] = (),
) -> ToolContext:
    return ToolContext(
        agent_id="director",
        session_id="sess_video",
        workspace_root=tmp_path,
        room_id=room_id,
        story_id=project_id,
        approved_by_person=approved,
        turn_aid_sections=turn_aid_sections,
    )


@pytest.mark.asyncio
async def test_frame_indexer_embedding_and_lexical_search_and_p6_fail_fast(
    tmp_path: Path,
) -> None:
    """`index_asset_frames` and `search_project_frames` rank shots via `BruteForceVectorStore`, label `method`, and propagate `EmbeddingError` / `EmbeddingDimensionError` (P6).

    Killed by: examples/video/src/uclone_x_video/indexer.py :: if embedder is not None:
    Becomes: if False:
    Killed by: examples/video/src/uclone_x_video/indexer.py :: if embedder is not None and combined:
    Becomes: if False:
    """
    lib = VideoLibrary(tmp_path)
    lib.create("Coffee Story", "room-1", project_id="coffee-story")
    work = VideoWork(lib, "coffee-story", "room-1")

    ingest_raw_video(
        work,
        asset_id="a_cam",
        filename="a_cam.mp4",
        fps=30,
        total_frames=300,
    )
    seeds = [
        FrameSeedSpec(
            frame_no=30,
            shot_in_frame=0,
            shot_out_frame=90,
            shot_type="wide",
            camera_motion="pan",
            caption="Golden sunset skyline over the city rooftops",
            visual_tags=["sunset", "skyline", "golden"],
        ),
        FrameSeedSpec(
            frame_no=150,
            shot_in_frame=90,
            shot_out_frame=210,
            shot_type="close_up",
            camera_motion="static",
            caption="Barista smiling while pouring latte art into a coffee cup",
            visual_tags=["barista", "coffee", "latte"],
        ),
    ]

    embedder = _MockEmbedder()
    entries = await index_asset_frames(work, asset_id="a_cam", seeds=seeds, embedder=embedder)
    assert len(entries) == 2
    assert (work.project_dir / "index" / "vectors.bin").is_file()

    # 1. Semantic search with embedder reports embedding model in `method` and ranks barista shot #1
    sem_res = await search_project_frames(
        work,
        query="barista pouring latte coffee",
        embedder=embedder,
        top_k=2,
    )
    assert sem_res.method == "embedding similarity (mock-video-embed-v1)"
    assert sem_res.hits[0].shot_id == "shot_02"
    assert (sem_res.hits[0].in_frame, sem_res.hits[0].out_frame) == (90, 210)
    assert sem_res.hits[0].in_timecode == "00:00:03:00"
    assert sem_res.hits[0].out_timecode == "00:00:07:00"

    # 2. Lexical search when embedder=None explicitly reports lexical overlap
    lex_res = await search_project_frames(
        work,
        query="sunset skyline",
        embedder=None,
        top_k=2,
    )
    assert lex_res.method == "lexical overlap (no embedder configured)"
    assert lex_res.hits[0].shot_id == "shot_01"

    # 3. P6 fail-fast: embedder dimension mismatch or runtime error raises instead of silent fallback
    with pytest.raises(EmbeddingDimensionError):
        await index_asset_frames(
            work,
            asset_id="a_cam",
            seeds=seeds,
            embedder=_MockEmbedder(bad_dims=True),
        )

    with pytest.raises(EmbeddingError, match="provider down"):
        await search_project_frames(
            work,
            query="barista",
            embedder=_MockEmbedder(fail_error=EmbeddingError("provider down")),
        )


@pytest.mark.asyncio
async def test_video_library_tool_lifecycle(tmp_path: Path) -> None:
    """`VideoLibraryTool` creates, opens, takes over, lists, and closes video projects.

    Killed by: examples/video/src/uclone_x_video/tool.py :: OPEN_VIDEO_KEY: record.id,
    Becomes: OPEN_VIDEO_KEY: None,
    """
    lib_tool = VideoLibraryTool()
    ctx_a = _ctx(tmp_path, room_id="room-a")

    created = await lib_tool.run(
        VideoLibraryParams(
            action="create",
            title="Neon Documentary",
            style_profile="documentary_clean",
            project_id="neon-doc",
            fps=24,
        ),
        ctx_a,
    )
    assert created[OPEN_VIDEO_KEY] == "neon-doc"
    assert created["writable"] is True
    assert created["fps"] == 24

    ctx_b = _ctx(tmp_path, room_id="room-b")
    opened_b = await lib_tool.run(
        VideoLibraryParams(action="open", project_id="neon-doc"),
        ctx_b,
    )
    assert opened_b[OPEN_VIDEO_KEY] == "neon-doc"
    assert opened_b["writable"] is False

    taken_b = await lib_tool.run(
        VideoLibraryParams(action="take_over", project_id="neon-doc"),
        ctx_b,
    )
    assert taken_b["writable"] is True
    assert taken_b["taken_from_another_conversation"] is True

    listed = await lib_tool.run(VideoLibraryParams(action="list"), ctx_b)
    assert len(listed["projects"]) == 1
    assert listed["projects"][0]["being_written_by"] == "this conversation"

    closed = await lib_tool.run(
        VideoLibraryParams(action="close"),
        _ctx(tmp_path, room_id="room-b", project_id="neon-doc"),
    )
    assert closed[OPEN_VIDEO_KEY] is None
    assert closed["closed"] == "neon-doc"


@pytest.mark.asyncio
async def test_video_tools_vault_timeline_fx_proposal_context_and_audit(
    tmp_path: Path,
) -> None:
    """`VideoMediaVaultTool`, `VideoTimelineTool`, `VideoFxColorTool`, `VideoManifestTool`, `VideoProposalTool`, `VideoContextTool`, and `VideoAuditTool` work end-to-end.

    Killed by: examples/video/src/uclone_x_video/tools.py :: if direct_when_empty and is_empty:
    Becomes: if True:
    Killed by: examples/video/src/uclone_x_video/tools.py :: updated_clip = clip.model_copy(update={"color_grade": new_grade})
    Becomes: updated_clip = clip
    """
    lib_tool = VideoLibraryTool()
    vault_tool = VideoMediaVaultTool(embedder=_MockEmbedder())
    search_tool = VideoSearchTool(embedder=_MockEmbedder())
    manifest_tool = VideoManifestTool()
    timeline_tool = VideoTimelineTool()
    fx_tool = VideoFxColorTool()
    context_tool = VideoContextTool()
    proposal_tool = VideoProposalTool()
    audit_tool = VideoAuditTool()

    await lib_tool.run(
        VideoLibraryParams(action="create", title="City Reel", project_id="city-reel"),
        _ctx(tmp_path, room_id="room-1"),
    )
    ctx = _ctx(tmp_path, room_id="room-1", project_id="city-reel")

    # 1. Register upload into immutable vault + proxy + frame index
    up_res = await vault_tool.run(
        VideoMediaVaultParams(
            action="register_upload",
            asset_id="asset_street",
            filename="street_4k.mp4",
            fps=30,
            total_frames=300,
            width=3840,
            height=2160,
        ),
        ctx,
    )
    assert up_res["changed"] is True
    assert up_res["asset"]["kind"] == "raw"

    # 2. Bake derived slow-motion asset while keeping parent raw untouched
    bake_res = await vault_tool.run(
        VideoMediaVaultParams(
            action="bake_derived_asset",
            parent_asset_id="asset_street",
            asset_id="asset_street_slowmo",
            source_in_frame=0,
            source_out_frame=90,
            operations=[BakeOperation(op="optical_flow_slowmo", rate=0.5)],
        ),
        ctx,
    )
    assert bake_res["asset"]["kind"] == "derived"
    assert bake_res["asset"]["total_frames"] == 180

    lineage_res = await vault_tool.run(
        VideoMediaVaultParams(action="inspect_lineage", asset_id="asset_street_slowmo"),
        ctx,
    )
    assert lineage_res["immutable"] is True
    assert lineage_res["lineage"]["parent_asset_ids"] == ["asset_street"]

    # 3. Search frames via VideoSearchTool
    search_res = await search_tool.run(
        VideoSearchParams(action="search_frames", query="establishing wide shot"),
        ctx,
    )
    assert search_res["method"] == "embedding similarity (mock-video-embed-v1)"
    assert len(search_res["hits"]) >= 1

    # 4. First append_clip onto empty v1_main writes directly (proposed=False)
    first_clip_res = await timeline_tool.run(
        VideoTimelineParams(
            action="append_clip",
            track_id="v1_main",
            asset_id="asset_street",
            source_in_frame=0,
            source_out_frame=120,
            clip_id="clip_01",
        ),
        ctx,
    )
    assert first_clip_res["proposed"] is False

    # 5. Subsequent edit on non-empty v1_main stages a proposal (proposed=True)
    second_clip_res = await timeline_tool.run(
        VideoTimelineParams(
            action="append_clip",
            track_id="v1_main",
            asset_id="asset_street_slowmo",
            source_in_frame=0,
            source_out_frame=180,
            clip_id="clip_02",
        ),
        ctx,
    )
    assert second_clip_res["proposed"] is True
    p1_id = second_clip_res["proposal_id"]

    # Apply p1 via VideoProposalTool
    applied_1 = await proposal_tool.run(
        VideoProposalParams(action="apply", proposal_id=p1_id),
        _ctx(tmp_path, room_id="room-1", project_id="city-reel", approved=True),
    )
    assert applied_1["status"] == "approved"

    # 6. Apply 3-way color grade on clip_01 -> stages proposal p002, then apply it
    grade_prop = await fx_tool.run(
        VideoFxColorParams(
            action="grade_clip",
            track_id="v1_main",
            clip_id="clip_01",
            lut="teal_orange_cinema",
            contrast=1.2,
            temperature=15.0,
        ),
        ctx,
    )
    assert grade_prop["proposed"] is True
    await proposal_tool.run(
        VideoProposalParams(action="apply", proposal_id=grade_prop["proposal_id"]),
        _ctx(tmp_path, room_id="room-1", project_id="city-reel", approved=True),
    )
    read_v1 = await timeline_tool.run(
        VideoTimelineParams(action="read", track_id="v1_main"),
        ctx,
    )
    assert read_v1["clips"][0]["lut"] == "teal_orange_cinema"

    # 7. Add title on empty t1_titles -> writes directly
    title_res = await fx_tool.run(
        VideoFxColorParams(
            action="add_title",
            track_id="t1_titles",
            text="CITY NIGHTS",
            timeline_in_frame=0,
            timeline_out_frame=90,
        ),
        ctx,
    )
    assert title_res["proposed"] is False

    # 8. Switch tier, add marker, and render Tier 3 master via VideoManifestTool
    await manifest_tool.run(
        VideoManifestParams(action="set_playback_tier", tier="source"),
        ctx,
    )
    await manifest_tool.run(
        VideoManifestParams(
            action="add_marker",
            marker=SequenceMarker(marker_id="m1", frame=60, label="Beat Drop"),
        ),
        ctx,
    )
    master_out = await manifest_tool.run(
        VideoManifestParams(action="render_master"),
        ctx,
    )
    assert master_out["master_render"]["tier"] == "master"

    # 9. Context tool & deduplication when digest is already in turn_aid_sections
    ctx_bundle = await context_tool.run(VideoContextParams(), ctx)
    assert ctx_bundle["active_playback_tier"] == "source"
    dedup = await context_tool.run(
        VideoContextParams(),
        _ctx(
            tmp_path,
            room_id="room-1",
            project_id="city-reel",
            turn_aid_sections=(f"(context digest {ctx_bundle['digest']})",),
        ),
    )
    assert dedup["same_as_turn_context"] is True

    # 10. Audit passes cleanly
    audit_res = await audit_tool.run(VideoAuditParams(), ctx)
    assert audit_res["passed"] is True


@pytest.mark.asyncio
async def test_video_start_guided_five_stages_and_region_edit(tmp_path: Path) -> None:
    """`VideoStartTool` advances through all 5 stages and supports `edit_region`.

    Killed by: examples/video/src/uclone_x_video/start.py :: stage="rough_cut_ready",
    Becomes: stage="completed",
    """
    start_tool = VideoStartTool()
    ctx0 = _ctx(tmp_path, room_id="room-guided")

    s1 = await start_tool.run(
        VideoStartParams(
            action="start",
            title="Morning Espresso",
            style_profile="cinematic_film",
        ),
        ctx0,
    )
    pid = s1["project_id"]
    assert s1["stage"] == "rough_cut_ready"
    ctx = _ctx(tmp_path, room_id="room-guided", project_id=pid)

    s2 = await start_tool.run(VideoStartParams(action="continue"), ctx)
    assert s2["stage"] == "broll_layered"

    s3 = await start_tool.run(VideoStartParams(action="continue"), ctx)
    assert s3["stage"] == "derived_baked"
    assert s3["derived_asset_id"] == "asset_cafe_smile_slowmo"

    s4 = await start_tool.run(
        VideoStartParams(action="continue", title_text="MORNING ESPRESSO"),
        ctx,
    )
    assert s4["stage"] == "graded_and_titled"

    s5 = await start_tool.run(VideoStartParams(action="continue"), ctx)
    assert s5["stage"] == "completed"
    assert s5["master_render"]["output_path"].startswith("exports/")

    reg = await start_tool.run(
        VideoStartParams(
            action="edit_region",
            track_id="v1_main",
            start_frame=0,
            end_frame=60,
            lut="warm_sunset",
        ),
        ctx,
    )
    assert reg["proposed"] is True


def test_video_extension_write_guard_and_turn_hook(tmp_path: Path) -> None:
    """`VIDEO_EXTENSION` protects `videos/` from `FileWriteTool` and `VideoTurnHook` injects context and updates `story_id`.

    Killed by: examples/video/src/uclone_x_video/ucx_extension.py :: protected_roots=(ProtectedRoot(dirname=VIDEOS_DIRNAME, refusal=_WRITE_REFUSAL),),
    Becomes: protected_roots=(),
    Killed by: examples/video/src/uclone_x_video/section_turn.py :: return context.model_copy(update={"story_id": project})
    Becomes: return context
    """
    with use_extensions([VIDEO_EXTENSION]):
        roots = protected_roots()
        assert any(r.dirname == "videos" for r in roots)

        registry = create_default_registry()
        for name in (
            "video_library",
            "video_start",
            "video_media",
            "video_search",
            "video_manifest",
            "video_timeline",
            "video_fx",
            "video_context",
            "video_proposal",
            "video_audit",
        ):
            assert registry.get(name) is not None

        writer = FileWriteTool()
        ctx = _ctx(tmp_path, room_id="room-1")
        with pytest.raises(PlainRefusalError, match="video project library"):
            writer.run(
                FileWriteParams(path="videos/city-reel/manifest.yaml", content="fps: 30\n"),
                ctx,
            )

    lib = VideoLibrary(tmp_path)
    lib.create("Hook Reel", "room-1", project_id="hook-reel")
    hook = VideoTurnHook()
    rec = ToolExecutionRecord(
        tool_name="video_library",
        arguments={"action": "open", "project_id": "hook-reel"},
        output={OPEN_VIDEO_KEY: "hook-reel"},
        status=ToolResultStatus.SUCCESS,
        duration_ms=1.0,
        opens_story=True,
    )
    updated_ctx = hook.after_tool_step([rec], _ctx(tmp_path, room_id="room-1"))
    assert updated_ctx.story_id == "hook-reel"

    aid = hook.turn_aid(
        message="cut the opening shot",
        story_id="hook-reel",
        room_id="room-1",
        workspace_root=tmp_path,
        tool_names=frozenset({"video_timeline", "video_search"}),
    )
    assert aid is not None
    assert VIDEO_SECTION_HEADING in aid.section
