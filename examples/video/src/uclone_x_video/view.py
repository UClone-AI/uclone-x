"""Interactive DaVinci-style NLE view model and direct user actions (`VideoView`).

`VideoView` reads one video project whole for the interactive NLE studio view (3-tier quality
specs, immutable raw & derived media pool with lineage, tracks, clips, titles, compiled
`PlaybackBundle` for the active tier, and pending proposals with A/B preview bundles), and
applies user-confirmed proposal decisions, direct timeline cuts/grades/titles, and derived
asset baking through the project's writing lease.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.errors import RoomNotFoundError, UnreadableRoomRecordError
from uclone_x_video.audit import VideoAuditFinding, audit_project
from uclone_x_video.compiler import (
    MasterRenderResult,
    PlaybackBundle,
    compile_playback_bundle,
    render_final_master,
)
from uclone_x_video.indexer import (
    FrameSearchRanking,
    index_asset_frames_sync,
    search_project_frames,
)
from uclone_x_video.library import (
    UnknownVideoError,
    VideoError,
    VideoLibrary,
    VideoProjectRecord,
)
from uclone_x_video.proposals import (
    apply_proposal,
    preview_proposed_track,
    reject_proposal,
)
from uclone_x_video.schemas import (
    BakeOperation,
    BakeOpKind,
    ClipTransition,
    ColorGrade,
    ManifestOutline,
    MediaAsset,
    ProjectQualityTiers,
    ProposalKind,
    QualityTier,
    SequenceMarker,
    SequenceSection,
    StyleProfile,
    TitleClip,
    TrackFile,
    VideoClip,
    frames_to_timecode,
)
from uclone_x_video.vault import (
    bake_derived_asset,
    ingest_raw_video,
    swap_clip_to_derived,
    unbake_clip_to_raw,
)
from uclone_x_video.work import VideoWork

__all__ = [
    "DirectTimelineAction",
    "DirectorBusyError",
    "ShownAssetView",
    "ShownProposalView",
    "ShownTrackView",
    "ShownVideoProject",
    "VideoNotFoundError",
    "VideoNotWritableError",
    "VideoNoteCode",
    "VideoRefusalCode",
    "VideoView",
]

_MODEL = ConfigDict(frozen=True, extra="forbid", strict=True)

VideoNoteCode = Literal["no_writer", "writer_gone"]
VideoRefusalCode = Literal["video_gone", "no_writer", "writer_gone", "writer_busy"]
DirectTimelineAction = Literal[
    "append_clip",
    "split_clip",
    "trim_clip",
    "delete_clip",
    "grade_clip",
    "set_title",
    "delete_title",
    "swap_to_derived",
    "unbake_to_raw",
    "set_track_state",
    "mute_track",
    "solo_track",
    "lock_track",
    "revert",
]


class VideoNotFoundError(VideoError):
    """The video project is not in the library."""

    code: VideoRefusalCode = "video_gone"


class VideoNotWritableError(VideoError):
    """No existing conversation holds the video project's writing lease."""

    def __init__(self, message: str, code: VideoRefusalCode = "no_writer") -> None:
        super().__init__(message)
        self.code: VideoRefusalCode = code


class DirectorBusyError(VideoError):
    """The conversation directing the video project is answering right now."""

    code: VideoRefusalCode = "writer_busy"


class ShownAssetView(BaseModel):
    """A media asset in the Media Pool (`raw` or `derived` with lineage and digest)."""

    model_config = _MODEL

    asset: MediaAsset
    digest: str
    start_timecode: str
    end_timecode: str


class ShownTrackView(BaseModel):
    """A timeline track's current clips, titles, digest, and undo availability."""

    model_config = _MODEL

    track_id: str
    kind: str
    name: str
    index: int
    digest: str
    muted: bool
    solo: bool
    locked: bool
    clips: list[VideoClip]
    titles: list[TitleClip]
    can_revert: bool


class ShownProposalView(BaseModel):
    """A pending proposal with original vs proposed clips/titles and A/B preview bundle."""

    model_config = _MODEL

    id: str
    kind: ProposalKind
    summary: str
    track_id: str | None = None
    seen_digest: str
    original_clips: list[VideoClip] = Field(default_factory=list[VideoClip])
    proposed_clips: list[VideoClip] = Field(default_factory=list[VideoClip])
    original_titles: list[TitleClip] = Field(default_factory=list[TitleClip])
    proposed_titles: list[TitleClip] = Field(default_factory=list[TitleClip])
    preview_bundle: PlaybackBundle


class ShownVideoProject(BaseModel):
    """The complete video project state for `VideoView`."""

    model_config = _MODEL

    project_id: str
    title: str
    style_profile: StyleProfile
    quality_tiers: ProjectQualityTiers
    active_tier: QualityTier
    writable: bool
    holder: str | None
    decide_code: str | None
    decide_note: str | None = None
    fps: int
    width: int
    height: int
    total_frames: int
    duration_seconds: float
    total_timecode: str
    sections: list[SequenceSection]
    markers: list[SequenceMarker]
    assets: list[ShownAssetView]
    tracks: list[ShownTrackView]
    playback: PlaybackBundle
    proposals: list[ShownProposalView]
    audit_findings: list[VideoAuditFinding]
    last_master_receipt: MasterRenderResult | None = None


class VideoView:
    """Reads a video project whole and executes person-initiated NLE studio actions."""

    def __init__(
        self,
        workspace_dir: Path,
        room_service: Any = None,
        *,
        turn_busy: Callable[[str], bool] = lambda _room_id: False,
    ) -> None:
        self._library = VideoLibrary(workspace_dir.resolve())
        self._rooms = room_service
        self._turn_busy = turn_busy

    def _load_record(self, project_id: str) -> VideoProjectRecord:
        try:
            return self._library.load(project_id)
        except UnknownVideoError as exc:
            raise VideoNotFoundError(
                f"There is no video project called '{project_id}' any more. Refresh the list to see the video projects in this workspace."
            ) from exc

    def _room_status(self, room_id: str) -> tuple[bool, str | None]:
        if self._rooms is None:
            return True, room_id
        try:
            room = self._rooms.get(room_id)
            return True, getattr(room, "title", None)
        except RoomNotFoundError:
            return False, None
        except UnreadableRoomRecordError:
            return True, None

    def _check_writable_state(
        self, record: VideoProjectRecord
    ) -> tuple[bool, str | None, VideoNoteCode | None, str | None]:
        if record.lease is None or not record.lease.holder:
            return (
                False,
                None,
                "no_writer",
                f"No conversation is directing '{record.title}', so changes cannot be saved. Open the video project in a conversation first.",
            )
        holder = record.lease.holder
        if holder == "studio-ui":
            return True, holder, None, None
        exists, _ = self._room_status(holder)
        if not exists:
            return (
                False,
                holder,
                "writer_gone",
                f"The conversation that was directing '{record.title}' no longer exists. Open the video project in a conversation first.",
            )
        return True, holder, None, None

    def _require_decider(
        self, project_id: str, room_id: str | None = None
    ) -> tuple[VideoWork, str]:
        """The video project's writer, for a decision the person made in `VideoView`.

        When no conversation holds the project any more (`no_writer`, `writer_gone`) or an
        idle conversation holds it while the person acts from `room_id`, `room_id` takes the
        project over first under `_LEASE_LOCK`.
        """
        record = self._load_record(project_id)
        writable, holder, code, message = self._check_writable_state(record)
        target_room = room_id.strip() if room_id is not None and room_id.strip() else None
        if target_room and self._room_status(target_room)[0]:
            self._refuse_if_busy(target_room)
            if not writable and code in ("no_writer", "writer_gone"):
                self._library.take_over_if_held_by(project_id, target_room, expected_holder=holder)
                record = self._load_record(project_id)
                writable, holder, code, message = self._check_writable_state(record)
            elif writable and holder is not None and holder != target_room:
                self._refuse_if_busy(holder)
                self._library.take_over_if_held_by(project_id, target_room, expected_holder=holder)
                record = self._load_record(project_id)
                writable, holder, code, message = self._check_writable_state(record)
        if not writable or holder is None:
            raise VideoNotWritableError(
                message or f"Video project '{project_id}' has no active writer conversation.",
                code or "no_writer",
            )
        self._refuse_if_busy(holder)
        return VideoWork(self._library, project_id, holder), holder

    def _refuse_if_busy(self, room_id: str) -> None:
        if self._turn_busy(room_id):
            _, title = self._room_status(room_id)
            label = f"'{title}'" if title else f"'{room_id}'"
            raise DirectorBusyError(
                f"The conversation {label} is answering right now, so the change was not saved. Wait for the turn to finish, then try again."
            )

    def list_projects(self) -> list[dict[str, Any]]:
        """List all video projects in the workspace library."""
        listing = self._library.list_projects()
        return [
            {
                "project_id": p.id,
                "title": p.title,
                "style_profile": p.style_profile,
                "fps": p.fps,
                "total_timecode": "00:00:10:00",
            }
            for p in listing.projects
        ]

    def create_project(
        self,
        *,
        title: str,
        style_profile: StyleProfile = "cinematic_film",
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        project_id: str | None = None,
        room_id: str | None = None,
        seed_sample: bool = True,
    ) -> ShownVideoProject:
        """Create a new video project (optionally pre-seeded with a sample cut) and bind it to `room_id`."""
        holder = room_id.strip() if room_id and room_id.strip() else "studio-ui"
        if room_id and room_id.strip() and self._room_status(room_id)[0]:
            self._refuse_if_busy(room_id)
        record = self._library.create(
            title,
            holder,
            style_profile=style_profile,
            fps=fps,
            project_id=project_id,
        )
        work = VideoWork(self._library, record.id, holder)
        if not seed_sample:
            work.write_manifest(
                ManifestOutline(fps=fps, width=width, height=height),
                expected_digest=None,
            )
            work.write_track(
                TrackFile(track_id="v1_main", kind="video", name="V1 MAIN", index=1, clips=[]),
                expected_digest=None,
            )
        if seed_sample:
            ingest_raw_video(
                work,
                asset_id="a_cam_interview",
                filename="a_cam_interview.mp4",
                fps=fps,
                total_frames=fps * 10,
                width=1920,
                height=1080,
            )
            ingest_raw_video(
                work,
                asset_id="b_roll_rooftop",
                filename="b_roll_rooftop.mp4",
                fps=fps,
                total_frames=fps * 8,
                width=1920,
                height=1080,
            )
            index_asset_frames_sync(work, asset_id="a_cam_interview")
            index_asset_frames_sync(work, asset_id="b_roll_rooftop")
            derived, _ = bake_derived_asset(
                work,
                parent_asset_id="b_roll_rooftop",
                source_in_frame=0,
                source_out_frame=fps * 3,
                operations=[BakeOperation(op="optical_flow_slowmo", rate=0.5, strength=0.85)],
                derived_asset_id="b_roll_rooftop_slowmo_01",
            )
            index_asset_frames_sync(work, asset_id=derived.asset_id)

            manifest = ManifestOutline(
                fps=fps,
                width=1920,
                height=1080,
                active_playback_tier="preview",
                sections=[
                    SequenceSection(
                        id="sec_intro",
                        title="Intro — Establishing Shot",
                        kind="intro",
                        start_frame=0,
                        end_frame=fps * 3,
                    ),
                    SequenceSection(
                        id="sec_main",
                        title="Main — Interview & Story",
                        kind="main",
                        start_frame=fps * 3,
                        end_frame=fps * 7,
                    ),
                    SequenceSection(
                        id="sec_outro",
                        title="Finale — Slow-Mo Skyline",
                        kind="outro",
                        start_frame=fps * 7,
                        end_frame=fps * 11,
                    ),
                ],
            )
            work.write_manifest(manifest, expected_digest=None)
            v1 = TrackFile(
                track_id="v1_main",
                kind="video",
                name="V1 MAIN",
                index=1,
                clips=[
                    VideoClip(
                        clip_id="clip_01_opening",
                        asset_id="b_roll_rooftop",
                        timeline_in_frame=0,
                        source_in_frame=0,
                        source_out_frame=fps * 3,
                        label="Opening Skyline",
                        transition_out=ClipTransition(kind="cross_dissolve", duration_frames=15),
                    ),
                    VideoClip(
                        clip_id="clip_02_interview",
                        asset_id="a_cam_interview",
                        timeline_in_frame=fps * 3,
                        source_in_frame=fps * 2,
                        source_out_frame=fps * 6,
                        label="Main Interview Cut",
                    ),
                    VideoClip(
                        clip_id="clip_03_slowmo",
                        asset_id=derived.asset_id,
                        timeline_in_frame=fps * 7,
                        source_in_frame=0,
                        source_out_frame=fps * 4,
                        label="0.5x Slow-Mo Finale",
                    ),
                ],
            )
            work.write_track(v1, expected_digest=None)
            v2 = TrackFile(
                track_id="v2_broll",
                kind="video",
                name="V2 B-ROLL",
                index=2,
                clips=[
                    VideoClip(
                        clip_id="clip_broll_01",
                        asset_id="b_roll_rooftop",
                        timeline_in_frame=fps * 4,
                        source_in_frame=fps * 3,
                        source_out_frame=fps * 5,
                        label="Cutaway B-Roll",
                    )
                ],
            )
            work.write_track(v2, expected_digest=None)
            t1 = TrackFile(
                track_id="t1_titles",
                kind="title",
                name="T1 TITLES",
                index=3,
                titles=[
                    TitleClip(
                        title_id="title_01",
                        text=title.upper()[:36],
                        subtitle="Director Co-Editing Cut",
                        style_preset="lower_third_news",
                        timeline_in_frame=15,
                        timeline_out_frame=fps * 3,
                        position="lower_third_left",
                        animation_in="fade_up",
                    )
                ],
            )
            work.write_track(t1, expected_digest=None)
        if room_id and room_id.strip() and self._room_status(room_id)[0]:
            try:
                self._rooms.set_story(room_id, record.id)
            except Exception:
                pass
        return self.show(record.id)

    def open_project(
        self,
        project_id: str,
        *,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Open `project_id` in `room_id` (taking over an idle or abandoned lease if needed) and set room's `story_id`."""
        record = self._load_record(project_id)
        target_room = room_id.strip() if room_id and room_id.strip() else None
        if target_room and self._room_status(target_room)[0]:
            self._refuse_if_busy(target_room)
            writable, holder, code, _ = self._check_writable_state(record)
            if not writable and code in ("no_writer", "writer_gone"):
                self._library.take_over_if_held_by(project_id, target_room, expected_holder=holder)
            elif record.lease is not None and record.lease.holder != target_room:
                if record.lease.holder != "studio-ui":
                    self._refuse_if_busy(record.lease.holder)
                self._library.take_over_if_held_by(
                    project_id, target_room, expected_holder=record.lease.holder
                )
            elif record.lease is None:
                try:
                    self._library.open(project_id, target_room)
                except Exception:
                    pass
            try:
                self._rooms.set_story(target_room, project_id)
            except Exception:
                pass
        return self.show(project_id)

    def resolve_stream_path(
        self,
        project_id: str,
        tier: QualityTier,
        target_id: str,
    ) -> Path:
        """Resolve the on-disk media file path for `(project_id, tier, target_id)`."""
        record = self._load_record(project_id)
        holder = record.lease.holder if record.lease is not None else None
        work = VideoWork(self._library, project_id, holder)
        if tier == "master":
            exports_dir = work.project_dir / "exports"
            if target_id in ("latest", "master") and exports_dir.is_dir():
                mp4s = sorted(
                    exports_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
                )
                if mp4s:
                    return mp4s[0]
            candidate = (exports_dir / Path(target_id).name).resolve()
            if candidate.is_file() and candidate.is_relative_to(exports_dir.resolve()):
                return candidate
            raise VideoNotFoundError(
                f"Master export '{target_id}' was not found for '{project_id}'."
            )

        loaded = work.read_asset(target_id)
        if loaded is None:
            raise VideoNotFoundError(f"Media asset '{target_id}' was not found in '{project_id}'.")
        asset, _ = loaded
        rel_path = asset.tiers.preview.path if tier == "preview" else asset.tiers.source.path
        full_path = (work.project_dir / rel_path).resolve()
        if not full_path.is_file() or not full_path.is_relative_to(work.project_dir.resolve()):
            raise VideoNotFoundError(f"Media stream file '{rel_path}' is missing on disk.")
        return full_path

    def show(
        self,
        project_id: str,
        *,
        tier: QualityTier | None = None,
        last_master_receipt: MasterRenderResult | None = None,
    ) -> ShownVideoProject:
        """Compile the full `ShownVideoProject` view model for `project_id`."""
        record = self._load_record(project_id)
        writable, holder, decide_code, decide_note = self._check_writable_state(record)
        work = VideoWork(self._library, project_id, holder)

        manifest_pair = work.read_manifest()
        manifest = (
            manifest_pair[0] if manifest_pair is not None else ManifestOutline(fps=record.fps)
        )
        active_tier: QualityTier = tier if tier is not None else record.active_playback_tier
        fps = manifest.fps

        assets_with_digest = work.list_assets()
        shown_assets: list[ShownAssetView] = []
        for asset, digest in assets_with_digest:
            shown_assets.append(
                ShownAssetView(
                    asset=asset,
                    digest=digest,
                    start_timecode=frames_to_timecode(0, asset.fps),
                    end_timecode=frames_to_timecode(asset.total_frames, asset.fps),
                )
            )

        tracks_with_digest = work.list_tracks()
        track_map = {t.track_id: (t, d) for t, d in tracks_with_digest}
        raw_tracks = [t for t, _ in tracks_with_digest]
        shown_tracks: list[ShownTrackView] = []
        for track, digest in tracks_with_digest:
            rev_log = work.read_track_revisions(track.track_id)
            shown_tracks.append(
                ShownTrackView(
                    track_id=track.track_id,
                    kind=track.kind,
                    name=track.name,
                    index=track.index,
                    digest=digest,
                    muted=track.muted,
                    solo=track.solo,
                    locked=track.locked,
                    clips=list(track.clips),
                    titles=list(track.titles),
                    can_revert=len(rev_log.revisions) >= 1,
                )
            )

        playback = compile_playback_bundle(work, tier=active_tier)

        proposals_raw = work.list_proposals(status="pending")
        shown_proposals: list[ShownProposalView] = []
        for prop, prop_digest in proposals_raw:
            orig_clips: list[VideoClip] = []
            prop_clips: list[VideoClip] = []
            orig_titles: list[TitleClip] = []
            prop_titles: list[TitleClip] = []
            t_id = prop.track_id

            if prop.kind == "track_replace" and prop.proposed_track is not None:
                pt = prop.proposed_track
                t_id = t_id or pt.track_id
                if t_id in track_map:
                    orig_clips = list(track_map[t_id][0].clips)
                    orig_titles = list(track_map[t_id][0].titles)
                prop_clips = list(pt.clips)
                prop_titles = list(pt.titles)
                preview_track = preview_proposed_track(prop)
                override_list = [
                    preview_track if existing.track_id == preview_track.track_id else existing
                    for existing in raw_tracks
                ]
                if all(existing.track_id != preview_track.track_id for existing in raw_tracks):
                    override_list.append(preview_track)
                preview_bundle = compile_playback_bundle(
                    work,
                    tier=active_tier,
                    override_tracks=override_list,
                )
            elif prop.kind == "manifest_replace" and prop.proposed_manifest is not None:
                preview_bundle = compile_playback_bundle(
                    work,
                    tier=active_tier,
                    override_manifest=prop.proposed_manifest,
                )
            else:
                preview_bundle = playback

            shown_proposals.append(
                ShownProposalView(
                    id=prop.id,
                    kind=prop.kind,
                    summary=prop.summary,
                    track_id=t_id,
                    seen_digest=prop_digest,
                    original_clips=orig_clips,
                    proposed_clips=prop_clips,
                    original_titles=orig_titles,
                    proposed_titles=prop_titles,
                    preview_bundle=preview_bundle,
                )
            )

        audit_rep = audit_project(work)

        return ShownVideoProject(
            project_id=record.id,
            title=record.title,
            style_profile=record.style_profile,
            quality_tiers=record.quality_tiers,
            active_tier=active_tier,
            writable=writable,
            holder=holder,
            decide_code=decide_code,
            decide_note=decide_note,
            fps=fps,
            width=manifest.width,
            height=manifest.height,
            total_frames=playback.total_frames,
            duration_seconds=playback.duration_sec,
            total_timecode=playback.total_timecode,
            sections=list(manifest.sections),
            markers=list(manifest.markers),
            assets=shown_assets,
            tracks=shown_tracks,
            playback=playback,
            proposals=shown_proposals,
            audit_findings=audit_rep.findings,
            last_master_receipt=last_master_receipt,
        )

    def approve(
        self,
        project_id: str,
        proposal_id: str,
        *,
        seen_digest: str,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Approve pending `proposal_id` and apply its changes to the target track or manifest."""
        work, _ = self._require_decider(project_id, room_id)
        apply_proposal(
            work,
            proposal_id,
            seen_digest=seen_digest,
            decided_in="video_view",
        )
        return self.show(project_id)

    def approve_all(
        self,
        project_id: str,
        *,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Approve all pending proposals in `project_id` in order."""
        work, _ = self._require_decider(project_id, room_id)
        pending = work.list_proposals(status="pending")
        for proposal, _ in pending:
            reloaded = work.read_proposal(proposal.id)
            if reloaded is None or reloaded[0].status != "pending":
                continue
            prop_obj, prop_digest = reloaded
            apply_proposal(
                work,
                prop_obj.id,
                seen_digest=prop_digest,
                decided_in="video_view",
            )
        return self.show(project_id)

    def reject(
        self,
        project_id: str,
        proposal_id: str,
        *,
        seen_digest: str,
        reason: str | None = None,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Reject pending `proposal_id`."""
        work, _ = self._require_decider(project_id, room_id)
        reject_proposal(
            work,
            proposal_id,
            seen_digest=seen_digest,
            reason=reason,
            decided_in="video_view",
        )
        return self.show(project_id)

    def set_tier(
        self,
        project_id: str,
        tier: QualityTier,
        *,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Switch the active resolution/quality tier (`preview`, `source`, `master`)."""
        work, _ = self._require_decider(project_id, room_id)
        work.update_project_settings(active_playback_tier=tier)
        manifest_pair = work.read_manifest()
        if manifest_pair is not None:
            manifest, m_digest = manifest_pair
            if manifest.active_playback_tier != tier:
                work.write_manifest(
                    manifest.model_copy(update={"active_playback_tier": tier}),
                    expected_digest=m_digest,
                )
        return self.show(project_id, tier=tier)

    def render_master(
        self,
        project_id: str,
        *,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Render the Tier 3 `master` deliverable by relinking from `preview` proxies to Tier 2 `source` raw/derived vault assets."""
        work, _ = self._require_decider(project_id, room_id)
        receipt = render_final_master(work)
        return self.show(project_id, tier="master", last_master_receipt=receipt)

    def ingest_media(
        self,
        project_id: str,
        *,
        asset_id: str,
        filename: str,
        fps: int = 30,
        total_frames: int = 300,
        width: int = 1920,
        height: int = 1080,
        raw_bytes: bytes | None = None,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Ingest an immutable raw video into `vault/`, generate its Tier 1 540p proxy, and index its frames."""
        work, _ = self._require_decider(project_id, room_id)
        asset, _ = ingest_raw_video(
            work,
            asset_id=asset_id,
            filename=filename,
            payload=raw_bytes,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )
        index_asset_frames_sync(work, asset_id=asset.asset_id)
        return self.show(project_id)

    def bake_derived(
        self,
        project_id: str,
        *,
        parent_asset_id: str,
        source_in_frame: int,
        source_out_frame: int,
        operation_kind: BakeOpKind = "optical_flow_slowmo",
        rate: float = 0.5,
        strength: float = 0.8,
        derived_asset_id: str | None = None,
        track_id: str | None = None,
        clip_id: str | None = None,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Bake a new immutable derived asset (`kind='derived'`) from a raw asset while preserving the original raw untouched."""
        work, _ = self._require_decider(project_id, room_id)
        op = BakeOperation(op=operation_kind, rate=rate, strength=strength)
        derived, _ = bake_derived_asset(
            work,
            parent_asset_id=parent_asset_id,
            source_in_frame=source_in_frame,
            source_out_frame=source_out_frame,
            operations=[op],
            derived_asset_id=derived_asset_id,
        )
        index_asset_frames_sync(work, asset_id=derived.asset_id)
        if track_id and clip_id:
            loaded_track = work.read_track_by_id(track_id)
            if loaded_track is not None:
                track_obj, track_digest = loaded_track
                swapped = swap_clip_to_derived(
                    track_obj,
                    clip_id=clip_id,
                    derived_asset=derived,
                )
                work.write_track(
                    swapped,
                    expected_digest=track_digest,
                    summary=f"Swapped {clip_id} to derived asset {derived.asset_id}",
                )
        return self.show(project_id)

    async def search_frames(
        self,
        project_id: str,
        *,
        query: str,
        asset_id: str | None = None,
        shot_type: str | None = None,
        top_k: int = 6,
    ) -> FrameSearchRanking:
        """Search indexed frames semantically/lexically across `project_id`."""
        record = self._load_record(project_id)
        holder = record.lease.holder if record.lease is not None else None
        work = VideoWork(self._library, project_id, holder)
        return await search_project_frames(
            work,
            query=query,
            asset_id=asset_id,
            shot_type=shot_type,
            top_k=top_k,
        )

    def direct_edit_track(
        self,
        project_id: str,
        track_id: str,
        *,
        action: DirectTimelineAction,
        seen_digest: str,
        clip_id: str | None = None,
        asset_id: str | None = None,
        label: str | None = None,
        split_at_frame: int | None = None,
        timeline_in_frame: int | None = None,
        timeline_out_frame: int | None = None,
        source_in_frame: int | None = None,
        source_out_frame: int | None = None,
        ripple: bool = False,
        color_grade: dict[str, Any] | None = None,
        transition_out: dict[str, Any] | None = None,
        title_clip: dict[str, Any] | None = None,
        title_id: str | None = None,
        derived_asset_id: str | None = None,
        muted: bool | None = None,
        solo: bool | None = None,
        locked: bool | None = None,
        room_id: str | None = None,
    ) -> ShownVideoProject:
        """Execute a direct NLE timeline or color grading manipulation on `track_id`."""
        work, _ = self._require_decider(project_id, room_id)
        if action == "revert":
            work.revert_track(track_id, expected_digest=seen_digest)
            return self.show(project_id)

        loaded = work.read_track_by_id(track_id)
        if loaded is None:
            raise VideoError(f"Track '{track_id}' does not exist in video project '{project_id}'.")
        track, current_digest = loaded
        if current_digest != seen_digest:
            raise VideoError(
                f"Track '{track_id}' changed on disk since it was shown. Refresh before editing."
            )

        if action == "swap_to_derived":
            if not clip_id or not derived_asset_id:
                raise VideoError("swap_to_derived requires both 'clip_id' and 'derived_asset_id'.")
            derived_loaded = work.read_asset(derived_asset_id)
            if derived_loaded is None:
                raise VideoError(f"Derived asset '{derived_asset_id}' was not found.")
            swapped = swap_clip_to_derived(
                track,
                clip_id=clip_id,
                derived_asset=derived_loaded[0],
            )
            work.write_track(
                swapped,
                expected_digest=current_digest,
                summary=f"Swapped clip {clip_id} to derived asset {derived_asset_id}",
            )
            return self.show(project_id)

        if action == "unbake_to_raw":
            if not clip_id:
                raise VideoError("unbake_to_raw requires 'clip_id'.")
            assets_by_id = {a.asset_id: a for a, _ in work.list_assets()}
            restored = unbake_clip_to_raw(
                track,
                clip_id=clip_id,
                assets_by_id=assets_by_id,
            )
            work.write_track(
                restored,
                expected_digest=current_digest,
                summary=f"Unbaked clip {clip_id} back to original raw asset",
            )
            return self.show(project_id)

        if action in ("set_track_state", "mute_track", "solo_track", "lock_track"):
            new_muted = (
                muted
                if action in ("set_track_state", "mute_track") and muted is not None
                else track.muted
            )
            new_solo = (
                solo
                if action in ("set_track_state", "solo_track") and solo is not None
                else track.solo
            )
            new_locked = (
                locked
                if action in ("set_track_state", "lock_track") and locked is not None
                else track.locked
            )
            updated_track = track.model_copy(
                update={
                    "muted": new_muted,
                    "solo": new_solo,
                    "locked": new_locked,
                }
            )
            work.write_track(
                updated_track,
                expected_digest=current_digest,
                summary=f"Direct track state ({action}) on {track_id}",
            )
            return self.show(project_id)

        if track.locked:
            raise VideoError(f"Track '{track_id}' is locked. Unlock it before editing.")

        clips: list[VideoClip] = list(track.clips)
        titles: list[TitleClip] = list(track.titles)

        if action == "append_clip":
            if not asset_id:
                raise VideoError("append_clip requires 'asset_id'.")
            loaded_asset = work.read_asset(asset_id)
            if loaded_asset is None:
                raise VideoError(f"Media asset '{asset_id}' was not found in '{project_id}'.")
            asset_obj, _ = loaded_asset
            src_in = source_in_frame if source_in_frame is not None else 0
            src_out = (
                source_out_frame
                if source_out_frame is not None
                else min(asset_obj.total_frames, src_in + 90)
            )
            if src_out <= src_in:
                src_out = min(asset_obj.total_frames, src_in + 30)
            dur = max(1, src_out - src_in)
            tail_frame = max((c.timeline_out_frame for c in clips), default=0)
            tl_in = timeline_in_frame if timeline_in_frame is not None else tail_frame
            if any(
                not (tl_in + dur <= c.timeline_in_frame or tl_in >= c.timeline_out_frame)
                for c in clips
            ):
                tl_in = tail_frame
            existing_ids = {c.clip_id for c in clips}
            candidate_id = clip_id or f"clip_{len(clips) + 1:02d}_{asset_id[:12]}"
            suffix = 2
            while candidate_id in existing_ids:
                candidate_id = f"clip_{len(clips) + suffix:02d}_{asset_id[:12]}"
                suffix += 1
            new_clip = VideoClip(
                clip_id=candidate_id,
                asset_id=asset_id,
                source_in_frame=src_in,
                source_out_frame=src_out,
                timeline_in_frame=tl_in,
                label=label or asset_obj.filename,
            )
            clips = [*clips, new_clip]
            clips.sort(key=lambda c: c.timeline_in_frame)

        elif action == "split_clip":
            if not clip_id or split_at_frame is None:
                raise VideoError("split_clip requires 'clip_id' and 'split_at_frame'.")
            target: VideoClip | None = next((c for c in clips if c.clip_id == clip_id), None)
            if target is None:
                raise VideoError(f"Clip '{clip_id}' not found on track '{track_id}'.")
            if not (target.timeline_in_frame < split_at_frame < target.timeline_out_frame):
                raise VideoError(
                    f"split_at_frame ({split_at_frame}) must be strictly inside clip '{clip_id}' "
                    f"({target.timeline_in_frame}..{target.timeline_out_frame})."
                )
            offset = split_at_frame - target.timeline_in_frame
            source_mid = target.source_in_frame + round(offset * target.speed)
            left_clip = target.model_copy(
                update={
                    "source_out_frame": source_mid,
                }
            )
            right_id = f"{target.clip_id}_b"
            existing_ids = {c.clip_id for c in clips}
            idx = 2
            while right_id in existing_ids:
                right_id = f"{target.clip_id}_b{idx}"
                idx += 1
            right_clip = target.model_copy(
                update={
                    "clip_id": right_id,
                    "timeline_in_frame": split_at_frame,
                    "source_in_frame": source_mid,
                }
            )
            clips = [c for c in clips if c.clip_id != clip_id] + [left_clip, right_clip]
            clips.sort(key=lambda c: c.timeline_in_frame)

        elif action == "trim_clip":
            if not clip_id:
                raise VideoError("trim_clip requires 'clip_id'.")
            target = next((c for c in clips if c.clip_id == clip_id), None)
            if target is None:
                raise VideoError(f"Clip '{clip_id}' not found on track '{track_id}'.")
            old_out = target.timeline_out_frame
            new_tl_in = (
                timeline_in_frame if timeline_in_frame is not None else target.timeline_in_frame
            )
            new_src_in = max(
                0,
                source_in_frame if source_in_frame is not None else target.source_in_frame,
            )
            if source_out_frame is not None:
                new_src_out = source_out_frame
            elif timeline_out_frame is not None:
                new_src_out = new_src_in + round((timeline_out_frame - new_tl_in) * target.speed)
            else:
                new_src_out = target.source_out_frame
            if new_src_out <= new_src_in:
                new_src_out = new_src_in + max(1, round(15 * target.speed))
            trimmed = target.model_copy(
                update={
                    "timeline_in_frame": new_tl_in,
                    "source_in_frame": new_src_in,
                    "source_out_frame": new_src_out,
                }
            )
            delta = trimmed.timeline_out_frame - old_out
            updated_clips: list[VideoClip] = []
            for c in clips:
                if c.clip_id == clip_id:
                    updated_clips.append(trimmed)
                elif ripple and c.timeline_in_frame >= old_out and delta != 0:
                    updated_clips.append(
                        c.model_copy(
                            update={"timeline_in_frame": max(0, c.timeline_in_frame + delta)}
                        )
                    )
                else:
                    updated_clips.append(c)
            clips = updated_clips
            clips.sort(key=lambda c: c.timeline_in_frame)

        elif action == "delete_clip":
            if not clip_id:
                raise VideoError("delete_clip requires 'clip_id'.")
            target = next((c for c in clips if c.clip_id == clip_id), None)
            if target is None:
                raise VideoError(f"Clip '{clip_id}' not found on track '{track_id}'.")
            gap = target.duration_frames
            removed_in = target.timeline_in_frame
            remaining = [c for c in clips if c.clip_id != clip_id]
            if ripple and gap > 0:
                shifted: list[VideoClip] = []
                for c in remaining:
                    if c.timeline_in_frame >= removed_in:
                        shifted.append(
                            c.model_copy(
                                update={
                                    "timeline_in_frame": max(0, c.timeline_in_frame - gap),
                                }
                            )
                        )
                    else:
                        shifted.append(c)
                clips = shifted
            else:
                clips = remaining

        elif action == "grade_clip":
            if not clip_id:
                raise VideoError("grade_clip requires 'clip_id'.")
            target = next((c for c in clips if c.clip_id == clip_id), None)
            if target is None:
                raise VideoError(f"Clip '{clip_id}' not found on track '{track_id}'.")
            updates: dict[str, Any] = {}
            if color_grade is not None:
                merged_grade = {**target.color_grade.model_dump(), **color_grade}
                updates["color_grade"] = ColorGrade.model_validate(merged_grade)
            if transition_out is not None:
                base_trans = (
                    target.transition_out.model_dump() if target.transition_out is not None else {}
                )
                updates["transition_out"] = ClipTransition.model_validate(
                    {**base_trans, **transition_out}
                )
            graded = target.model_copy(update=updates)
            clips = [graded if c.clip_id == clip_id else c for c in clips]

        elif action == "set_title":
            if title_clip is None:
                raise VideoError("set_title requires 'title_clip'.")
            parsed_title = TitleClip.model_validate(title_clip)
            other_titles = [t for t in titles if t.title_id != parsed_title.title_id]
            t_span = parsed_title.timeline_out_frame - parsed_title.timeline_in_frame
            if any(
                not (
                    parsed_title.timeline_out_frame <= t.timeline_in_frame
                    or parsed_title.timeline_in_frame >= t.timeline_out_frame
                )
                for t in other_titles
            ):
                next_free = max((t.timeline_out_frame for t in other_titles), default=0)
                parsed_title = parsed_title.model_copy(
                    update={
                        "timeline_in_frame": next_free,
                        "timeline_out_frame": next_free + t_span,
                    }
                )
            titles = [*other_titles, parsed_title]
            titles.sort(key=lambda t: t.timeline_in_frame)

        elif action == "delete_title":
            if not title_id:
                raise VideoError("delete_title requires 'title_id'.")
            titles = [t for t in titles if t.title_id != title_id]

        updated_track = TrackFile(
            track_id=track.track_id,
            kind=track.kind,
            name=track.name,
            index=track.index,
            muted=track.muted,
            solo=track.solo,
            locked=track.locked,
            clips=clips,
            titles=titles,
        )
        work.write_track(
            updated_track,
            expected_digest=current_digest,
            summary=f"Direct edit ({action}) on {track_id}",
        )
        return self.show(project_id)
