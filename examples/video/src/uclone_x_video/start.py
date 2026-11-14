"""Guided 5-stage interactive video creation state machine (`video_start`).

Allows any user -- including someone with no prior NLE experience (P0) -- to go from raw
footage to a color-graded, titled, Tier 3 master video one step at a time:
1. `rough_cut_ready` -- Ingest immutable raw media (Tier 2 vault + Tier 1 proxy), index
   frames, and assemble the initial highlight rough cut on `v1_main`.
2. `broll_layered` -- Add B-roll and cross-dissolve transition.
3. `derived_baked` -- Bake a heavy edit (stabilize + optical-flow slow motion) into a new
   immutable derived asset while preserving the raw source untouched.
4. `graded_and_titled` -- Add animated cinema title on `t1_titles` and apply 3-way color
   grading + LUT.
5. `completed` -- Relink all clips to Tier 2 source vault files and render the Tier 3 final
   master MP4 (`exports/`).

All decisions come strictly from structured tool arguments (#2204); `request` is never
parsed for keywords.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x_video import OPEN_VIDEO_KEY
from uclone_x_video.compiler import render_final_master
from uclone_x_video.indexer import FrameSeedSpec, index_asset_frames
from uclone_x_video.library import NoConversationError, VideoError, VideoLibrary
from uclone_x_video.proposals import propose_track_change
from uclone_x_video.schemas import (
    BakeOperation,
    BakeOpKind,
    ClipTransition,
    ColorGrade,
    LutPreset,
    ManifestOutline,
    StyleProfile,
    TitleClip,
    TrackFile,
    VideoClip,
    dump_file,
    parse_file,
)
from uclone_x_video.vault import (
    bake_derived_asset,
    ingest_raw_video,
    swap_clip_to_derived,
)
from uclone_x_video.work import VideoWork

__all__ = [
    "START_FILE",
    "VideoStartParams",
    "VideoStartState",
    "VideoStartTool",
    "read_start_state",
]

START_FILE = "start.yaml"

StartStage = Literal[
    "rough_cut_ready",
    "broll_layered",
    "derived_baked",
    "graded_and_titled",
    "completed",
]


class VideoStartState(BaseModel):
    """Persisted state of the guided `video_start` workflow (`start.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stage: StartStage = "rough_cut_ready"
    style_profile: StyleProfile = "cinematic_film"
    recommended_next: str | None = "broll_and_transition"
    derived_asset_id: str | None = None
    master_output_path: str | None = None


def read_start_state(work: VideoWork) -> tuple[VideoStartState, str] | None:
    """Read `start.yaml` if present."""
    file = work.library.read_file(work.project_id, START_FILE)
    if file is None:
        return None
    state = parse_file(file.text, VideoStartState, relative_path=START_FILE)
    return state, file.digest


def _write_start_state(
    work: VideoWork, state: VideoStartState, *, expected_digest: str | None
) -> str:
    written = work.library.write_file(
        work.project_id,
        START_FILE,
        dump_file(state),
        room_id=work.room_id,
        expected_digest=expected_digest,
    )
    return written.digest


def _grade_for_style(style: StyleProfile, lut_override: LutPreset | None = None) -> ColorGrade:
    if style == "cinematic_film":
        return ColorGrade(
            lut=lut_override or "cinematic_teal_orange",
            lift=(0.0, 0.0, 0.02),
            gamma=(0.01, 0.0, -0.01),
            gain=(1.06, 1.02, 0.98),
            contrast=1.14,
            saturation=1.08,
            temperature=12.0,
            vignette=0.15,
        )
    if style == "vintage_analog":
        return ColorGrade(
            lut=lut_override or "vintage_kodak",
            lift=(0.02, 0.01, -0.01),
            gamma=(0.02, 0.01, 0.0),
            gain=(1.04, 1.0, 0.92),
            contrast=1.08,
            saturation=0.92,
            temperature=22.0,
            vignette=0.25,
        )
    if style == "fast_social_cut":
        return ColorGrade(
            lut=lut_override or "cool_cyberpunk",
            contrast=1.2,
            saturation=1.18,
            temperature=-8.0,
        )
    return ColorGrade(
        lut=lut_override or "clean_neutral",
        contrast=1.05,
        saturation=1.02,
    )


class VideoStartParams(BaseModel):
    """Parameters for `video_start`."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["start", "continue", "edit_region", "status"] = Field(
        default="start",
        description="'start' a new guided video project and build the initial rough cut; "
        "'continue' to advance one editing stage at a time; 'edit_region' to stage a "
        "regional clip/frame edit or heavy derived bake; 'status' to inspect progress.",
    )
    title: str | None = Field(
        default=None, description="Project title when starting a new guided video project."
    )
    style_profile: StyleProfile | None = Field(
        default=None,
        description="Visual style profile ('cinematic_film', 'documentary_clean', "
        "'fast_social_cut', or 'vintage_analog').",
    )
    track_id: str = Field(
        default="v1_main",
        description="Target track id for 'edit_region'.",
    )
    clip_id: str | None = Field(
        default=None,
        description="Target clip id for 'edit_region'.",
    )
    start_frame: int | None = Field(
        default=None,
        ge=0,
        description="Selected start frame for 'edit_region'.",
    )
    end_frame: int | None = Field(
        default=None,
        ge=1,
        description="Selected end frame for 'edit_region'.",
    )
    bake_op: BakeOpKind | None = Field(
        default=None,
        description="Optional heavy operation ('stabilize', 'optical_flow_slowmo', 'reverse', 'speed_ramp') to bake a new derived asset in 'edit_region'.",
    )
    lut: LutPreset | None = Field(
        default=None,
        description="Optional LUT preset to apply in 'edit_region'.",
    )
    title_text: str | None = Field(
        default=None,
        description="Optional title overlay text to add during 'continue' or 'edit_region'.",
    )


class VideoStartTool(BaseTool[VideoStartParams]):
    """Step-by-step guided video editing state machine (`video_start`)."""

    name = "video_start"
    description = (
        "Start or advance a step-by-step guided video editing workflow: (1) ingest immutable "
        "raw media, build preview proxies, index frames, and cut the initial A-roll rough cut; "
        "(2) layer B-roll with transitions; (3) bake a heavy transformation into a derived "
        "immutable asset while keeping raw media untouched; (4) apply 3-way color grading and "
        "title overlays; (5) relink to Tier 2 source vault streams and render the Tier 3 master."
    )
    params_type = VideoStartParams
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    needs_room: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"status"})

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=VideoStartParams,
        )

    async def run(self, params: VideoStartParams, context: ToolContext) -> dict[str, Any]:
        workspace = context.require_workspace()
        library = VideoLibrary(workspace)
        room_id = context.room_id

        if params.action == "status":
            work = VideoWork.for_context(context)
            state_pair = read_start_state(work)
            state = state_pair[0] if state_pair is not None else VideoStartState()
            return {
                "project_id": work.project_id,
                "stage": state.stage,
                "style_profile": state.style_profile,
                "recommended_next": state.recommended_next,
                "derived_asset_id": state.derived_asset_id,
                "master_output_path": state.master_output_path,
            }

        if room_id is None or not room_id.strip():
            raise NoConversationError("Guided video editing requires an active conversation.")

        if params.action == "start":
            style: StyleProfile = params.style_profile or "cinematic_film"
            title = (params.title or "Morning Roastery Story").strip()
            if context.story_id:
                try:
                    work = VideoWork(library, context.story_id, room_id)
                    work.record()
                except VideoError:
                    record = library.create(title, room_id, style_profile=style)
                    work = VideoWork(library, record.id, room_id)
            else:
                record = library.create(title, room_id, style_profile=style)
                work = VideoWork(library, record.id, room_id)

            await self._seed_initial_rough_cut(work, style=style)
            existing_start = read_start_state(work)
            new_state = VideoStartState(
                stage="rough_cut_ready",
                style_profile=style,
                recommended_next="broll_and_transition",
            )
            _write_start_state(
                work,
                new_state,
                expected_digest=existing_start[1] if existing_start is not None else None,
            )
            return {
                OPEN_VIDEO_KEY: work.project_id,
                "project_id": work.project_id,
                "stage": new_state.stage,
                "style_profile": style,
                "recommended_next": new_state.recommended_next,
                "summary": (
                    f"Started video project {work.project_id!r} ({style}): ingested immutable "
                    "raw assets into Tier 2 vault, generated Tier 1 540p preview proxies, "
                    "indexed shot frames, and placed the opening rough cut on track 'v1_main'."
                ),
            }

        work = VideoWork.for_context(context)
        if params.action == "edit_region":
            return await self._edit_region(work, params)

        # action == "continue"
        state_pair = read_start_state(work)
        state = (
            state_pair[0]
            if state_pair is not None
            else VideoStartState(style_profile=params.style_profile or work.record().style_profile)
        )
        prev_digest = state_pair[1] if state_pair is not None else None
        style = params.style_profile or state.style_profile

        if state.stage == "rough_cut_ready":
            # Stage 2: Add B-roll clip with cross_dissolve transition
            v1_pair = work.read_track_by_id("v1_main")
            v1 = (
                v1_pair[0]
                if v1_pair is not None
                else TrackFile(track_id="v1_main", kind="video", name="V1 MAIN", index=1)
            )
            clips = list(v1.clips)
            if clips:
                clips[-1] = clips[-1].model_copy(
                    update={
                        "transition_out": ClipTransition(kind="cross_dissolve", duration_frames=15)
                    }
                )
            end_f = max((c.timeline_out_frame for c in clips), default=0)
            broll_clip = VideoClip(
                clip_id="clip_02_sunset_broll",
                asset_id="asset_sunset_broll",
                source_in_frame=30,
                source_out_frame=150,
                timeline_in_frame=end_f,
                speed=1.0,
            )
            updated_v1 = v1.model_copy(update={"clips": [*clips, broll_clip]})
            work.write_track(
                updated_v1,
                expected_digest=v1_pair[1] if v1_pair is not None else None,
                summary="Added sunset B-roll clip with cross-dissolve transition",
            )
            next_state = state.model_copy(
                update={
                    "stage": "broll_layered",
                    "style_profile": style,
                    "recommended_next": "heavy_bake_fx",
                }
            )
            _write_start_state(work, next_state, expected_digest=prev_digest)
            return {
                OPEN_VIDEO_KEY: work.project_id,
                "project_id": work.project_id,
                "stage": next_state.stage,
                "recommended_next": next_state.recommended_next,
                "summary": (
                    "Appended 'clip_02_sunset_broll' (frames 30..150) onto 'v1_main' with a "
                    "15-frame cross-dissolve transition."
                ),
            }

        if state.stage == "broll_layered":
            # Stage 3: Bake a heavy edit (stabilize + 0.5x optical flow slow motion) into a new immutable derived asset
            derived_asset, _ = bake_derived_asset(
                work,
                parent_asset_id="asset_cafe_talk",
                source_in_frame=60,
                source_out_frame=150,
                operations=[
                    BakeOperation(op="stabilize", strength=0.8),
                    BakeOperation(op="optical_flow_slowmo", rate=0.5),
                ],
                derived_asset_id="asset_cafe_smile_slowmo",
            )
            await index_asset_frames(work, asset_id=derived_asset.asset_id)
            v1_pair = work.read_track_by_id("v1_main")
            if v1_pair is not None and v1_pair[0].clips:
                first_cid = v1_pair[0].clips[0].clip_id
                swapped = swap_clip_to_derived(
                    v1_pair[0], clip_id=first_cid, derived_asset=derived_asset
                )
                work.write_track(
                    swapped,
                    expected_digest=v1_pair[1],
                    summary=f"Swapped {first_cid} to derived slow-motion asset {derived_asset.asset_id}",
                )
            next_state = state.model_copy(
                update={
                    "stage": "derived_baked",
                    "style_profile": style,
                    "recommended_next": "titles_and_grade",
                    "derived_asset_id": derived_asset.asset_id,
                }
            )
            _write_start_state(work, next_state, expected_digest=prev_digest)
            return {
                OPEN_VIDEO_KEY: work.project_id,
                "project_id": work.project_id,
                "stage": next_state.stage,
                "derived_asset_id": derived_asset.asset_id,
                "recommended_next": next_state.recommended_next,
                "summary": (
                    f"Baked new immutable derived asset {derived_asset.asset_id!r} (stabilize + "
                    "0.5x optical-flow slow motion) while keeping raw 'asset_cafe_talk' untouched, "
                    "and linked it on 'v1_main'."
                ),
            }

        if state.stage == "derived_baked":
            # Stage 4: Apply 3-way color grading + LUT and add opening title on t1_titles
            grade = _grade_for_style(style, params.lut)
            v1_pair = work.read_track_by_id("v1_main")
            if v1_pair is not None:
                graded_clips = [
                    c.model_copy(update={"color_grade": grade}) for c in v1_pair[0].clips
                ]
                work.write_track(
                    v1_pair[0].model_copy(update={"clips": graded_clips}),
                    expected_digest=v1_pair[1],
                    summary=f"Applied {grade.lut} color grading to v1_main clips",
                )
            t1_pair = work.read_track_by_id("t1_titles")
            title_text = (params.title_text or work.record().title).strip()
            t1_track = TrackFile(
                track_id="t1_titles",
                kind="title",
                name="T1 TITLES",
                index=2,
                titles=[
                    TitleClip(
                        title_id="title_01_opening",
                        timeline_in_frame=15,
                        timeline_out_frame=105,
                        text=title_text,
                        subtitle="Director's Cut",
                        style_preset="minimal_cinema",
                        position="center",
                        animation_in="fade_up",
                    )
                ],
            )
            work.write_track(
                t1_track,
                expected_digest=t1_pair[1] if t1_pair is not None else None,
                summary=f"Added opening title {title_text!r} on t1_titles",
            )
            next_state = state.model_copy(
                update={
                    "stage": "graded_and_titled",
                    "style_profile": style,
                    "recommended_next": "master_render",
                }
            )
            _write_start_state(work, next_state, expected_digest=prev_digest)
            return {
                OPEN_VIDEO_KEY: work.project_id,
                "project_id": work.project_id,
                "stage": next_state.stage,
                "lut": grade.lut,
                "recommended_next": next_state.recommended_next,
                "summary": (
                    f"Applied 3-way color grading (LUT: {grade.lut}) to 'v1_main' and added "
                    f"animated title {title_text!r} on 't1_titles'."
                ),
            }

        # Stage 5: Render Final Master (Tier 3) relinked to Tier 2 source vault streams
        master_res = render_final_master(work)
        next_state = state.model_copy(
            update={
                "stage": "completed",
                "style_profile": style,
                "recommended_next": None,
                "master_output_path": master_res.output_path,
            }
        )
        _write_start_state(work, next_state, expected_digest=prev_digest)
        return {
            OPEN_VIDEO_KEY: work.project_id,
            "project_id": work.project_id,
            "stage": "completed",
            "recommended_next": None,
            "master_render": master_res.model_dump(mode="json"),
            "summary": (
                f"Completed guided video production: relinked all clips from preview proxies "
                f"to Tier 2 source vault files and rendered Tier 3 master to {master_res.output_path}."
            ),
        }

    async def _seed_initial_rough_cut(self, work: VideoWork, *, style: StyleProfile) -> None:
        manifest = ManifestOutline(
            fps=30,
            width=1920,
            height=1080,
            active_playback_tier="preview",
            master_grade=_grade_for_style(style),
        )
        existing_m = work.read_manifest()
        work.write_manifest(
            manifest,
            expected_digest=existing_m[1] if existing_m is not None else None,
        )

        if work.read_asset("asset_cafe_talk") is None:
            ingest_raw_video(
                work,
                asset_id="asset_cafe_talk",
                filename="cafe_interview_raw_4k.mp4",
                fps=30,
                total_frames=300,
                width=3840,
                height=2160,
            )
            await index_asset_frames(
                work,
                asset_id="asset_cafe_talk",
                seeds=[
                    FrameSeedSpec(
                        frame_no=30,
                        shot_in_frame=0,
                        shot_out_frame=60,
                        shot_type="wide",
                        camera_motion="pan",
                        caption="Wide establishing shot of the morning coffee roastery counter",
                        visual_tags=["roastery", "wide", "morning", "counter"],
                        transcript_slice="Welcome to our morning roastery.",
                    ),
                    FrameSeedSpec(
                        frame_no=105,
                        shot_in_frame=60,
                        shot_out_frame=150,
                        shot_type="close_up",
                        camera_motion="static",
                        caption="Barista smiling warmly while pouring latte art into a white ceramic cup",
                        visual_tags=["barista", "smile", "coffee", "cup", "latte", "close_up"],
                        transcript_slice="The Ethiopian natural beans have a bright floral aroma.",
                    ),
                    FrameSeedSpec(
                        frame_no=220,
                        shot_in_frame=150,
                        shot_out_frame=300,
                        shot_type="medium",
                        camera_motion="tracking",
                        caption="Steam rising from the fresh coffee cup beside the sunlit window",
                        visual_tags=["steam", "window", "sunlight", "coffee"],
                        transcript_slice="",
                    ),
                ],
            )

        if work.read_asset("asset_sunset_broll") is None:
            ingest_raw_video(
                work,
                asset_id="asset_sunset_broll",
                filename="sunset_city_broll_4k.mp4",
                fps=30,
                total_frames=240,
                width=3840,
                height=2160,
            )
            await index_asset_frames(
                work,
                asset_id="asset_sunset_broll",
                seeds=[
                    FrameSeedSpec(
                        frame_no=90,
                        shot_in_frame=30,
                        shot_out_frame=150,
                        shot_type="wide",
                        camera_motion="pan",
                        caption="Golden hour sunset skyline glowing over the city rooftops",
                        visual_tags=["sunset", "golden_hour", "skyline", "city", "warm"],
                        transcript_slice="",
                    )
                ],
            )

        existing_v1 = work.read_track_by_id("v1_main")
        v1 = TrackFile(
            track_id="v1_main",
            kind="video",
            name="V1 MAIN",
            index=1,
            clips=[
                VideoClip(
                    clip_id="clip_01_coffee_smile",
                    asset_id="asset_cafe_talk",
                    source_in_frame=60,
                    source_out_frame=150,
                    timeline_in_frame=0,
                    speed=1.0,
                )
            ],
        )
        work.write_track(
            v1,
            expected_digest=existing_v1[1] if existing_v1 is not None else None,
            summary="Seeded initial rough cut on v1_main",
        )

    async def _edit_region(self, work: VideoWork, params: VideoStartParams) -> dict[str, Any]:
        tid = params.track_id.strip()
        loaded = work.read_track_by_id(tid)
        if loaded is None or not loaded[0].clips:
            raise VideoError(f"Track {tid!r} has no clips to edit.")
        track, _ = loaded

        target_clip: VideoClip | None = None
        if params.clip_id:
            target_clip = next(
                (c for c in track.clips if c.clip_id == params.clip_id.strip()), None
            )
        elif params.start_frame is not None and params.end_frame is not None:
            target_clip = next(
                (
                    c
                    for c in track.clips
                    if c.timeline_out_frame > params.start_frame
                    and c.timeline_in_frame < params.end_frame
                ),
                None,
            )
        if target_clip is None:
            target_clip = track.clips[0]

        updated_track = track
        summary_parts: list[str] = []
        derived_id: str | None = None

        if params.bake_op is not None:
            derived_asset, _ = bake_derived_asset(
                work,
                parent_asset_id=target_clip.asset_id,
                source_in_frame=target_clip.source_in_frame,
                source_out_frame=target_clip.source_out_frame,
                operations=[BakeOperation(op=params.bake_op, rate=0.5)],
            )
            await index_asset_frames(work, asset_id=derived_asset.asset_id)
            updated_track = swap_clip_to_derived(
                updated_track, clip_id=target_clip.clip_id, derived_asset=derived_asset
            )
            derived_id = derived_asset.asset_id
            summary_parts.append(f"bake {params.bake_op} -> {derived_id}")

        if params.lut is not None:
            grade = _grade_for_style(work.record().style_profile, params.lut)
            new_clips = [
                c.model_copy(update={"color_grade": grade})
                if c.clip_id == target_clip.clip_id
                else c
                for c in updated_track.clips
            ]
            updated_track = updated_track.model_copy(update={"clips": new_clips})
            summary_parts.append(f"LUT {params.lut}")

        if not summary_parts:
            grade = _grade_for_style(work.record().style_profile, "warm_sunset")
            new_clips = [
                c.model_copy(update={"color_grade": grade})
                if c.clip_id == target_clip.clip_id
                else c
                for c in updated_track.clips
            ]
            updated_track = updated_track.model_copy(update={"clips": new_clips})
            summary_parts.append("warm_sunset grade")

        summary_text = f"Regional edit on {target_clip.clip_id} ({', '.join(summary_parts)})"
        proposal, written_prop = propose_track_change(
            work, proposed_track=updated_track, summary=summary_text
        )
        out: dict[str, Any] = {
            "proposed": True,
            "changed": True,
            "proposal_id": proposal.id,
            "track_id": tid,
            "clip_id": target_clip.clip_id,
            "summary": f"Proposed {proposal.id}: {summary_text}. Waiting for approval in VideoView.",
            "path": written_prop.relative_path,
            "digest": written_prop.digest,
        }
        if derived_id is not None:
            out["derived_asset_id"] = derived_id
        return out
