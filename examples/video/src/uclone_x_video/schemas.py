"""The shapes of a video project's files, and how a file that does not fit is reported.

A video project's files are the user's: declarative YAML manifests, tracks, and media
records a person can inspect and edit by hand. Each is validated every time it is read,
and a file that does not fit is refused with **the file and the field** named, in plain
words a person reads. Nothing is defaulted in place of a wrong value.

This module is pure: it parses text it is given and touches no file.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from uclone_x.errors import PlainRefusalError

#: The folder under a workspace that holds the video project library.
VIDEOS_DIRNAME = "videos"

__all__ = [
    "ASSET_KINDS",
    "BAKE_OP_KINDS",
    "ENTRY_ID_PATTERN",
    "LUT_PRESETS",
    "QUALITY_TIERS",
    "STYLE_PROFILES",
    "TITLE_ANIMATIONS",
    "TITLE_POSITIONS",
    "TITLE_STYLE_PRESETS",
    "TRACK_KINDS",
    "TRANSITION_KINDS",
    "VIDEOS_DIRNAME",
    "AssetKind",
    "AssetLineage",
    "BakeOpKind",
    "BakeOperation",
    "ClipTransform",
    "ClipTransition",
    "ColorGrade",
    "EntryId",
    "FrameIndexEntry",
    "LutPreset",
    "ManifestOutline",
    "MediaAsset",
    "MediaAssetTiers",
    "ProjectQualityTiers",
    "ProposalKind",
    "QualityTier",
    "QualityTierSpec",
    "RawFallbackRef",
    "Revision",
    "RevisionLog",
    "SequenceMarker",
    "SequenceSection",
    "SessionEntry",
    "SessionsFile",
    "StyleProfile",
    "TierStreamFile",
    "TimelineRegionSelection",
    "TitleAnimation",
    "TitleClip",
    "TitlePosition",
    "TitleStylePreset",
    "TrackFile",
    "TrackKind",
    "TransitionKind",
    "VideoClip",
    "VideoFileError",
    "VideoProposal",
    "default_quality_tiers",
    "describe_invalid",
    "dump_file",
    "frames_to_timecode",
    "parse_file",
    "timecode_to_frames",
]

ENTRY_ID_PATTERN = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$"
EntryId = Annotated[str, Field(pattern=ENTRY_ID_PATTERN, max_length=80)]

QualityTier = Literal["preview", "source", "master"]
QUALITY_TIERS: tuple[QualityTier, ...] = ("preview", "source", "master")

AssetKind = Literal["raw", "derived"]
ASSET_KINDS: tuple[AssetKind, ...] = ("raw", "derived")

BakeOpKind = Literal[
    "stabilize",
    "optical_flow_slowmo",
    "reverse",
    "speed_ramp",
    "auto_reframe_9_16",
    "denoise_audio",
    "audio_denoise",
    "compound_bake",
]
BAKE_OP_KINDS: tuple[BakeOpKind, ...] = (
    "stabilize",
    "optical_flow_slowmo",
    "reverse",
    "speed_ramp",
    "auto_reframe_9_16",
    "denoise_audio",
    "audio_denoise",
    "compound_bake",
)

LutPreset = Literal[
    "clean_neutral",
    "rec709_standard",
    "cinematic_teal_orange",
    "teal_orange_cinema",
    "kodak_2383_warm",
    "warm_sunset",
    "vintage_kodak",
    "vintage_sepia",
    "bleach_bypass_gritty",
    "noir_bw",
    "noir_monochrome",
    "cool_cyberpunk",
    "cyberpunk_neon",
]
LUT_PRESETS: tuple[LutPreset, ...] = (
    "clean_neutral",
    "rec709_standard",
    "cinematic_teal_orange",
    "teal_orange_cinema",
    "kodak_2383_warm",
    "warm_sunset",
    "vintage_kodak",
    "vintage_sepia",
    "bleach_bypass_gritty",
    "noir_bw",
    "noir_monochrome",
    "cool_cyberpunk",
    "cyberpunk_neon",
)

StyleProfile = Literal[
    "cinematic_film",
    "documentary_clean",
    "fast_social_cut",
    "vintage_analog",
]
STYLE_PROFILES: tuple[StyleProfile, ...] = (
    "cinematic_film",
    "documentary_clean",
    "fast_social_cut",
    "vintage_analog",
)

TransitionKind = Literal[
    "none",
    "cross_dissolve",
    "dip_to_black",
    "dip_to_white",
    "whip_pan",
    "wipe_left",
]
TRANSITION_KINDS: tuple[TransitionKind, ...] = (
    "none",
    "cross_dissolve",
    "dip_to_black",
    "dip_to_white",
    "whip_pan",
    "wipe_left",
)

TitleStylePreset = Literal[
    "minimal_cinema",
    "minimal_lower_third",
    "lower_third_news",
    "cinema_serif_center",
    "bold_pop",
    "bold_impact_headline",
    "neon_glitch",
    "neon_cyber_subtitle",
    "documentary_caption",
]
TITLE_STYLE_PRESETS: tuple[TitleStylePreset, ...] = (
    "minimal_cinema",
    "minimal_lower_third",
    "lower_third_news",
    "cinema_serif_center",
    "bold_pop",
    "bold_impact_headline",
    "neon_glitch",
    "neon_cyber_subtitle",
    "documentary_caption",
)

TitlePosition = Literal[
    "center",
    "lower_third_left",
    "lower_third_right",
    "top_banner",
    "lower_left",
    "lower_center",
    "top_left",
    "top_center",
]
TITLE_POSITIONS: tuple[TitlePosition, ...] = (
    "center",
    "lower_third_left",
    "lower_third_right",
    "top_banner",
    "lower_left",
    "lower_center",
    "top_left",
    "top_center",
)

TitleAnimation = Literal[
    "none",
    "fade_in",
    "fade_up",
    "typewriter",
    "glitch_in",
    "pop",
    "slide_left",
]
TITLE_ANIMATIONS: tuple[TitleAnimation, ...] = (
    "none",
    "fade_in",
    "fade_up",
    "typewriter",
    "glitch_in",
    "pop",
    "slide_left",
)

TrackKind = Literal["video", "title", "fx", "audio"]
TRACK_KINDS: tuple[TrackKind, ...] = ("video", "title", "fx", "audio")

ProposalKind = Literal["track_replace", "manifest_replace"]

_TIMECODE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}):(\d{2})$")
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class VideoFileError(PlainRefusalError):
    """A video project YAML file failed validation."""


def frames_to_timecode(frame: int, fps: int = 30) -> str:
    """Format an integer `frame` number at `fps` as SMPTE timecode `HH:MM:SS:FF`."""
    if fps <= 0:
        raise VideoFileError(f"Frame rate must be positive, got {fps}.")
    if frame < 0:
        raise VideoFileError(f"Frame number must be non-negative, got {frame}.")
    total_seconds, ff = divmod(frame, fps)
    minutes_total, ss = divmod(total_seconds, 60)
    hh, mm = divmod(minutes_total, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def timecode_to_frames(timecode: str, fps: int = 30) -> int:
    """Parse SMPTE timecode `HH:MM:SS:FF` into an integer frame index at `fps`."""
    if fps <= 0:
        raise VideoFileError(f"Frame rate must be positive, got {fps}.")
    match = _TIMECODE_RE.fullmatch(timecode.strip())
    if match is None:
        raise VideoFileError(
            f"Timecode {timecode!r} must be formatted as HH:MM:SS:FF (for example '00:00:04:15')."
        )
    hh, mm, ss, ff = (int(part) for part in match.groups())
    if mm >= 60 or ss >= 60 or ff >= fps:
        raise VideoFileError(
            f"Timecode {timecode!r} has out-of-range fields for {fps} fps (frames must be 0..{fps - 1})."
        )
    return ((hh * 3600) + (mm * 60) + ss) * fps + ff


class QualityTierSpec(BaseModel):
    """Resolution and encoding profile for one quality tier."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    tier: QualityTier
    width: int = Field(ge=160, le=7680)
    height: int = Field(ge=90, le=4320)
    codec: str = Field(min_length=1)
    max_keyframe_interval: int = Field(default=15, ge=1, le=300)
    bitrate_kbps: int = Field(default=2000, ge=100, le=200000)
    crf: int = Field(default=18, ge=0, le=51)
    preserve_original_bitstream: bool = False
    audio_lufs_target: float = Field(default=-14.0, ge=-30.0, le=0.0)


class ProjectQualityTiers(BaseModel):
    """The 3-tier resolution and quality pipeline (`preview`, `source`, `master`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    preview: QualityTierSpec = Field(
        default_factory=lambda: QualityTierSpec(
            tier="preview",
            width=960,
            height=540,
            codec="h264_proxy_fast",
            max_keyframe_interval=15,
            bitrate_kbps=2000,
            crf=24,
        )
    )
    source: QualityTierSpec = Field(
        default_factory=lambda: QualityTierSpec(
            tier="source",
            width=1920,
            height=1080,
            codec="source_immutable_copy",
            max_keyframe_interval=30,
            bitrate_kbps=25000,
            crf=14,
            preserve_original_bitstream=True,
        )
    )
    master: QualityTierSpec = Field(
        default_factory=lambda: QualityTierSpec(
            tier="master",
            width=3840,
            height=2160,
            codec="h264_master_10bit",
            max_keyframe_interval=30,
            bitrate_kbps=45000,
            crf=14,
            audio_lufs_target=-14.0,
        )
    )


def default_quality_tiers() -> ProjectQualityTiers:
    """Return the default 3-tier (`preview` 540p, `source` raw, `master` 4K) configuration."""
    return ProjectQualityTiers()


class BakeOperation(BaseModel):
    """One heavy transformation operation applied when baking a derived immutable asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    op: BakeOpKind
    rate: float = Field(default=1.0, gt=0.05, le=8.0)
    strength: float = Field(default=0.7, ge=0.0, le=1.0)


class AssetLineage(BaseModel):
    """Provenance record for a derived immutable media asset (`kind='derived'`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    parent_asset_ids: list[EntryId] = Field(min_length=1)
    source_in_frame: int = Field(ge=0)
    source_out_frame: int = Field(ge=1)
    recipe_kind: Literal["heavy_transform", "compound_bake", "audio_stem_clean"] = "heavy_transform"
    operations: list[BakeOperation] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_range(self) -> AssetLineage:
        if self.source_out_frame <= self.source_in_frame:
            raise ValueError(
                f"source_out_frame ({self.source_out_frame}) must be greater than "
                f"source_in_frame ({self.source_in_frame})."
            )
        return self


class TierStreamFile(BaseModel):
    """File path, SHA-256 digest, and dimensions for one tier (`source` or `preview`) of an asset."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HEX_SHA256_RE)
    width: int = Field(ge=64, le=7680)
    height: int = Field(ge=64, le=4320)
    size_bytes: int = Field(ge=1)


class MediaAssetTiers(BaseModel):
    """The immutable `source` stream in `vault/` and the fast `preview` proxy in `proxies/`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: TierStreamFile
    preview: TierStreamFile


class MediaAsset(BaseModel):
    """An immutable raw or derived media asset (`media/<asset_id>.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    asset_id: EntryId
    kind: AssetKind = "raw"
    filename: str = Field(min_length=1)
    immutable: Literal[True] = True
    fps: int = Field(default=30, ge=1, le=240)
    total_frames: int = Field(ge=1, le=10_000_000)
    duration_ms: int = Field(ge=1)
    tiers: MediaAssetTiers
    lineage: AssetLineage | None = None
    has_audio: bool = True
    index_status: Literal["indexed", "pending", "failed"] = "indexed"
    indexed_frame_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_lineage(self) -> MediaAsset:
        if self.kind == "derived" and self.lineage is None:
            raise ValueError("A derived media asset (kind='derived') must declare its 'lineage'.")
        if self.kind == "raw" and self.lineage is not None:
            raise ValueError("A raw media asset (kind='raw') must not carry a derived 'lineage'.")
        return self


class FrameIndexEntry(BaseModel):
    """One indexed frame/shot entry in `index/frames.jsonl`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    frame_id: str = Field(min_length=1)
    asset_id: EntryId
    frame_no: int = Field(ge=0)
    pts_ms: int = Field(ge=0)
    timecode: str = Field(min_length=11, max_length=11)
    shot_id: EntryId
    shot_in_frame: int = Field(ge=0)
    shot_out_frame: int = Field(ge=1)
    shot_type: Literal["wide", "medium", "close_up", "extreme_close_up", "aerial", "insert"] = (
        "medium"
    )
    camera_motion: Literal["static", "pan", "tilt", "tracking", "handheld", "zoom"] = "static"
    caption: str = Field(min_length=1)
    visual_tags: list[str] = Field(default_factory=list)
    transcript_slice: str = ""

    @model_validator(mode="after")
    def _validate_shot_bounds(self) -> FrameIndexEntry:
        if self.shot_out_frame <= self.shot_in_frame:
            raise ValueError(
                f"shot_out_frame ({self.shot_out_frame}) must exceed shot_in_frame ({self.shot_in_frame})."
            )
        if not (self.shot_in_frame <= self.frame_no < self.shot_out_frame):
            raise ValueError(
                f"frame_no ({self.frame_no}) must lie within shot bounds "
                f"{self.shot_in_frame}..{self.shot_out_frame}."
            )
        return self


class ColorGrade(BaseModel):
    """DaVinci Resolve-style 3-way color corrector (Lift/Gamma/Gain) and LUT settings."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    lut: LutPreset = "clean_neutral"
    lift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamma: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    contrast: float = Field(default=1.0, ge=0.2, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)
    temperature: float = Field(default=0.0, ge=-100.0, le=100.0)
    tint: float = Field(default=0.0, ge=-100.0, le=100.0)
    vignette: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("lift", "gamma", "gain", mode="before")
    @classmethod
    def _coerce_rgb_tuple(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
        return value


class ClipTransform(BaseModel):
    """Spatial transform (zoom, pan, rotation, opacity) for a video clip."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    scale: float = Field(default=1.0, ge=0.1, le=5.0)
    position_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    position_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    rotation_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class ClipTransition(BaseModel):
    """Outgoing transition at the end of a clip."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: TransitionKind = "cross_dissolve"
    duration_frames: int = Field(default=15, ge=1, le=120)


class RawFallbackRef(BaseModel):
    """Reference to the original raw asset and frame span before a derived asset was baked."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    asset_id: EntryId
    source_in_frame: int = Field(ge=0)
    source_out_frame: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_span(self) -> RawFallbackRef:
        if self.source_out_frame <= self.source_in_frame:
            raise ValueError(
                f"raw_fallback_ref.source_out_frame ({self.source_out_frame}) must exceed "
                f"source_in_frame ({self.source_in_frame})."
            )
        return self


class VideoClip(BaseModel):
    """One non-destructive clip placement on a video or audio timeline track."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    clip_id: EntryId
    asset_id: EntryId
    source_in_frame: int = Field(ge=0)
    source_out_frame: int = Field(ge=1)
    timeline_in_frame: int = Field(ge=0)
    speed: float = Field(default=1.0, gt=0.1, le=8.0)
    volume: int = Field(default=100, ge=0, le=150)
    muted: bool = False
    transform: ClipTransform = Field(default_factory=ClipTransform)
    color_grade: ColorGrade = Field(default_factory=ColorGrade)
    transition_out: ClipTransition | None = None
    raw_fallback_ref: RawFallbackRef | None = None
    label: str = ""

    @model_validator(mode="after")
    def _validate_frames(self) -> VideoClip:
        if self.source_out_frame <= self.source_in_frame:
            raise ValueError(
                f"Clip {self.clip_id!r} has source_out_frame ({self.source_out_frame}) <= "
                f"source_in_frame ({self.source_in_frame})."
            )
        return self

    @property
    def duration_frames(self) -> int:
        """Timeline duration in frames after applying `speed`."""
        raw_span = self.source_out_frame - self.source_in_frame
        return max(1, round(raw_span / self.speed))

    @property
    def timeline_out_frame(self) -> int:
        """Exclusive ending frame on the sequence timeline."""
        return self.timeline_in_frame + self.duration_frames


class TitleClip(BaseModel):
    """One title, subtitle, or lower-third graphic overlay on a title track."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    title_id: EntryId
    timeline_in_frame: int = Field(ge=0)
    timeline_out_frame: int = Field(ge=1)
    text: str = Field(min_length=1)
    subtitle: str = ""
    style_preset: TitleStylePreset = "minimal_cinema"
    font_family: str = "Inter"
    font_size_px: int = Field(default=48, ge=12, le=200)
    color_hex: str = Field(default="#F8FAFC", pattern=_HEX_COLOR_RE)
    position: TitlePosition = "center"
    animation_in: TitleAnimation = "fade_up"
    animation_out: TitleAnimation = "fade_in"
    backdrop_blur: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_title_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            copied = dict(data)
            if "animation" in copied:
                anim_val = copied.pop("animation")
                if "animation_in" not in copied and anim_val is not None:
                    copied["animation_in"] = anim_val
            if "backdrop_opacity" in copied:
                blur_val = copied.pop("backdrop_opacity")
                if "backdrop_blur" not in copied and blur_val is not None:
                    copied["backdrop_blur"] = blur_val
            return copied
        return data

    @model_validator(mode="after")
    def _validate_title_span(self) -> TitleClip:
        if self.timeline_out_frame <= self.timeline_in_frame:
            raise ValueError(
                f"Title {self.title_id!r} has timeline_out_frame ({self.timeline_out_frame}) <= "
                f"timeline_in_frame ({self.timeline_in_frame})."
            )
        return self


class TrackFile(BaseModel):
    """One timeline track (`tracks/<track_id>.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    track_id: EntryId
    kind: TrackKind = "video"
    name: str = ""
    index: int = Field(default=1, ge=1, le=32)
    muted: bool = False
    solo: bool = False
    locked: bool = False
    clips: list[VideoClip] = Field(default_factory=list[VideoClip])
    titles: list[TitleClip] = Field(default_factory=list[TitleClip])

    @model_validator(mode="after")
    def _validate_non_overlapping(self) -> TrackFile:
        seen_clip_ids: set[str] = set()
        for idx, clip in enumerate(self.clips):
            if clip.clip_id in seen_clip_ids:
                raise ValueError(f"Track {self.track_id!r} has duplicate clip_id {clip.clip_id!r}.")
            seen_clip_ids.add(clip.clip_id)
            if idx > 0:
                prev = self.clips[idx - 1]
                if clip.timeline_in_frame < prev.timeline_in_frame:
                    raise ValueError(
                        f"Track {self.track_id!r} clips must be ordered by timeline_in_frame "
                        f"({prev.clip_id!r} starts at {prev.timeline_in_frame}, "
                        f"{clip.clip_id!r} starts at {clip.timeline_in_frame})."
                    )
                if prev.timeline_out_frame > clip.timeline_in_frame:
                    raise ValueError(
                        f"Track {self.track_id!r} has overlapping clips: {prev.clip_id!r} ends at "
                        f"frame {prev.timeline_out_frame}, which overlaps {clip.clip_id!r} starting "
                        f"at frame {clip.timeline_in_frame}."
                    )

        seen_title_ids: set[str] = set()
        for idx, title in enumerate(self.titles):
            if title.title_id in seen_title_ids:
                raise ValueError(
                    f"Track {self.track_id!r} has duplicate title_id {title.title_id!r}."
                )
            seen_title_ids.add(title.title_id)
            if idx > 0:
                prev_t = self.titles[idx - 1]
                if prev_t.timeline_out_frame > title.timeline_in_frame:
                    raise ValueError(
                        f"Track {self.track_id!r} has overlapping titles: {prev_t.title_id!r} ends at "
                        f"frame {prev_t.timeline_out_frame}, which overlaps {title.title_id!r} starting "
                        f"at frame {title.timeline_in_frame}."
                    )
        return self


class SequenceMarker(BaseModel):
    """A named frame marker on the timeline ruler."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    marker_id: EntryId
    frame: int = Field(ge=0)
    label: str = Field(min_length=1)
    color: Literal["cyan", "amber", "emerald", "rose", "violet"] = "cyan"


class SequenceSection(BaseModel):
    """A structural chapter/section span on the video timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: EntryId
    title: str = Field(min_length=1)
    kind: Literal["intro", "main", "highlight", "broll_montage", "outro"] = "main"
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_section_span(self) -> SequenceSection:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"Section {self.id!r} has end_frame ({self.end_frame}) <= start_frame ({self.start_frame})."
            )
        return self


class ManifestOutline(BaseModel):
    """The sequence skeleton (`manifest.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fps: int = Field(default=30, ge=1, le=240)
    width: int = Field(default=1920, ge=160, le=7680)
    height: int = Field(default=1080, ge=90, le=4320)
    active_playback_tier: QualityTier = "preview"
    master_grade: ColorGrade = Field(default_factory=ColorGrade)
    sections: list[SequenceSection] = Field(default_factory=list[SequenceSection])
    markers: list[SequenceMarker] = Field(default_factory=list[SequenceMarker])


class TimelineRegionSelection(BaseModel):
    """A selected track/frame region in `VideoView` bound to the conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    track_id: EntryId
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=1)
    clip_id: EntryId | None = None

    @model_validator(mode="after")
    def _validate_region(self) -> TimelineRegionSelection:
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"Selected region end_frame ({self.end_frame}) must exceed start_frame ({self.start_frame})."
            )
        return self


class VideoProposal(BaseModel):
    """A staged timeline or manifest change awaiting person approval (`proposals/pNNN.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: EntryId
    kind: ProposalKind
    target_file: str = Field(min_length=1)
    target_digest: str | None = None
    track_id: EntryId | None = None
    summary: str = Field(min_length=1)
    proposed_track: TrackFile | None = None
    proposed_manifest: ManifestOutline | None = None
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_by: str | None = None
    created_at: str = Field(min_length=1)
    decided_in: Literal["conversation", "video_view"] | None = None
    decided_at: str | None = None
    rejection_reason: str | None = None


class Revision(BaseModel):
    """One historical revision of a track (`tracks/.history/<track_id>/revisions.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    digest: str = Field(min_length=1)
    saved_at: str = Field(min_length=1)
    room_id: str | None = None
    summary: str = ""


class RevisionLog(BaseModel):
    """The revision log for a track."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    track_id: EntryId
    revisions: list[Revision] = Field(default_factory=list[Revision])


class SessionEntry(BaseModel):
    """One recorded activity entry in `sessions.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    room_id: str = Field(min_length=1)
    at: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class SessionsFile(BaseModel):
    """The session log (`sessions.yaml`) of a video project."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sessions: list[SessionEntry] = Field(default_factory=list[SessionEntry])


def describe_invalid(relative_path: str, exc: ValidationError) -> str:
    """Format a Pydantic `ValidationError` naming `relative_path` and the first bad field."""
    errors = exc.errors()
    if not errors:
        return f"'{relative_path}' is invalid."
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ())) or "<root>"
    msg = first.get("msg", "invalid value")
    return f"'{relative_path}' has an invalid field '{loc}': {msg}"


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def parse_file(text: str, model_type: type[_ModelT], *, relative_path: str) -> _ModelT:
    """Parse YAML `text` into `model_type`, raising `VideoFileError` with plain words."""
    try:
        loaded: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise VideoFileError(f"'{relative_path}' is not valid YAML.") from exc
    if not isinstance(loaded, dict):
        raise VideoFileError(f"'{relative_path}' must be a YAML mapping at the top level.")
    try:
        return model_type.model_validate(loaded)
    except ValidationError as exc:
        raise VideoFileError(describe_invalid(relative_path, exc)) from exc


def dump_file(model: BaseModel) -> str:
    """Serialize `model` to deterministic, human-readable YAML."""
    data = model.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
