"""Pull-based score context bundle and vertical/horizontal map for music composition.

Computes the active section's polyphonic register stack, free registers, missing roles,
tail bar continuity window, and context manifest so an agent can compose a single part
without overflowing its context window or colliding with existing tracks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from uclone_x_music.library import MusicError
from uclone_x_music.schemas import (
    BarNoteEvent,
    InstrumentEntry,
    RegionSelection,
    ScoreOutline,
    ScoreSection,
    StyleProfile,
    TrackBar,
    TrackFile,
    pitch_to_name,
    resolve_track_bars,
)

if TYPE_CHECKING:
    from uclone_x_music.work import SongWork

__all__ = [
    "CORE_ROLES",
    "ContextManifest",
    "RegisterBand",
    "ScoreContextBundle",
    "TrackStackSummary",
    "build_score_context",
    "classify_register_band",
    "context_digest_line",
]


def context_digest_line(digest: str) -> str:
    """How the `[Music score context]` names its bundle's digest, which `music_context`
    looks for in the request's own turn context (#2429)."""
    return f"(context digest {digest})"


RegisterBand = Literal["low", "mid", "high", "percussion"]
FreeRegisterBand = Literal["low", "mid", "high"]
NoteDensity = Literal["empty", "sparse", "medium", "dense"]

CORE_ROLES: tuple[str, ...] = (
    "bass_foundation",
    "lead_melody",
    "harmonic_comping",
    "rhythm_percussion",
)


class TrackStackSummary(BaseModel):
    """Summary of one active track's vertical register, density, and groove."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    instrument_id: str
    instrument_name: str
    category: str
    role: str
    register_band: RegisterBand
    min_sounding_pitch: str | None
    max_sounding_pitch: str | None
    note_density: NoteDensity
    groove_swing: float


class ContextManifest(BaseModel):
    """Explicit manifest of which tracks, bars, and proposals are in the context bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    included_tracks: list[str]
    omitted_tracks: list[str]
    included_bars: list[int]
    pending_proposal_ids: list[str]


class ScoreContextBundle(BaseModel):
    """The pull-based musical context for one section or selected bar region."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    song_id: str
    title: str
    style_profile: StyleProfile
    tempo: int
    time_signature: str
    key_signature: str
    active_section_id: str | None
    section_bars: int
    chords: list[str]
    selected_region: RegionSelection | None = None
    polyphonic_stack: list[TrackStackSummary]
    free_registers: list[FreeRegisterBand]
    missing_roles: list[str]
    tail_bars: dict[str, list[TrackBar]]
    manifest: ContextManifest


def classify_register_band(
    *,
    category: str,
    role: str,
    sounding_events: list[BarNoteEvent],
) -> RegisterBand:
    """Classify a track's vertical register band (`low`, `mid`, `high`, or `percussion`)."""
    if category == "drums" or role == "rhythm_percussion":
        return "percussion"
    if not sounding_events:
        if category == "bass" or role in ("bass_foundation", "satb_bass"):
            return "low"
        if role in ("lead_melody", "satb_soprano"):
            return "high"
        return "mid"
    avg_midi = sum(ev.midi for ev in sounding_events) / len(sounding_events)
    if avg_midi < 52.0:
        return "low"
    if avg_midi < 68.0:
        return "mid"
    return "high"


def _classify_density(sounding_count: int, total_bars: int) -> NoteDensity:
    if sounding_count == 0 or total_bars <= 0:
        return "empty"
    notes_per_bar = sounding_count / total_bars
    if notes_per_bar <= 2.0:
        return "sparse"
    if notes_per_bar <= 6.0:
        return "medium"
    return "dense"


def _summarize_track(
    track: TrackFile,
    instrument: InstrumentEntry | None,
) -> TrackStackSummary:
    inst_name = instrument.name if instrument is not None else track.instrument_id
    category = instrument.category if instrument is not None else "synth"
    role = instrument.role if instrument is not None else "harmonic_comping"

    resolved = resolve_track_bars(track)
    sounding: list[BarNoteEvent] = [
        ev for bar_events in resolved.values() for ev in bar_events if ev.midi >= 0
    ]
    band = classify_register_band(category=category, role=role, sounding_events=sounding)
    min_pitch = pitch_to_name(min(ev.midi for ev in sounding)) if sounding else None
    max_pitch = pitch_to_name(max(ev.midi for ev in sounding)) if sounding else None
    density = _classify_density(len(sounding), len(track.bars))

    return TrackStackSummary(
        instrument_id=track.instrument_id,
        instrument_name=inst_name,
        category=category,
        role=role,
        register_band=band,
        min_sounding_pitch=min_pitch,
        max_sounding_pitch=max_pitch,
        note_density=density,
        groove_swing=track.groove_swing,
    )


def _select_section(
    outline: ScoreOutline,
    section_id: str | None,
    selected_region: RegionSelection | None,
) -> ScoreSection | None:
    target_id = section_id or (selected_region.section_id if selected_region is not None else None)
    if target_id is not None:
        for sec in outline.sections:
            if sec.id == target_id:
                return sec
        raise MusicError(f"Section {target_id!r} was not found in score.yaml.")
    if outline.sections:
        return outline.sections[0]
    return None


def build_score_context(
    work: SongWork,
    *,
    section_id: str | None = None,
    selected_region: RegionSelection | None = None,
    max_tracks: int = 12,
    max_window_bars: int | None = None,
) -> ScoreContextBundle:
    """Compute the horizontal and vertical musical map for the active section.

    `max_window_bars` caps the continuity window a selected region gives: a longer window
    keeps its two edges on each side (the bars around the region's boundaries), so a
    whole-section region does not put every bar of every track in the bundle (#2429).
    """
    record = work.record()
    score_pair = work.read_score()
    outline = score_pair[0] if score_pair is not None else ScoreOutline()
    section = _select_section(outline, section_id, selected_region)

    instruments_by_id = {inst.id: inst for inst, _ in work.list_instruments()}
    active_section_id = section.id if section is not None else None
    section_bars = section.bars if section is not None else 0
    chords = list(section.chords) if section is not None else []

    if active_section_id is not None:
        section_tracks = [t for t, _ in work.list_tracks(section_id=active_section_id)]
    else:
        section_tracks = []

    if selected_region is not None and selected_region.instrument_ids:
        pinned = set(selected_region.instrument_ids)
        section_tracks.sort(key=lambda t: (0 if t.instrument_id in pinned else 1, t.track_id))

    limit = max(0, max_tracks)
    included = section_tracks[:limit]
    omitted = section_tracks[limit:]

    polyphonic_stack = [
        _summarize_track(track, instruments_by_id.get(track.instrument_id)) for track in included
    ]

    occupied_bands = {summary.register_band for summary in polyphonic_stack}
    free_registers: list[FreeRegisterBand] = [
        band for band in ("low", "mid", "high") if band not in occupied_bands
    ]

    present_roles = {summary.role for summary in polyphonic_stack}
    missing_roles = [role for role in CORE_ROLES if role not in present_roles]

    if section is None or section_bars <= 0:
        included_bars: list[int] = []
        tail_bars: dict[str, list[TrackBar]] = {}
    else:
        if selected_region is not None:
            win_start = max(1, selected_region.start_bar - 1)
            win_end = min(section_bars, selected_region.end_bar + 1)
        else:
            win_start = max(1, section_bars - 1)
            win_end = section_bars
        included_bars = list(range(win_start, win_end + 1))
        if max_window_bars is not None and len(included_bars) > max_window_bars:
            edge = max(1, max_window_bars // 2)
            included_bars = [*included_bars[:edge], *included_bars[-edge:]]
        bar_set = set(included_bars)
        tail_bars = {
            track.instrument_id: [bar for bar in track.bars if bar.bar in bar_set]
            for track in included
        }

    pending_ids = [prop.id for prop, _ in work.list_proposals(status="pending")]
    manifest = ContextManifest(
        included_tracks=[t.track_id for t in included],
        omitted_tracks=[t.track_id for t in omitted],
        included_bars=included_bars,
        pending_proposal_ids=pending_ids,
    )

    return ScoreContextBundle(
        song_id=record.id,
        title=record.title,
        style_profile=record.style_axioms.profile,
        tempo=outline.tempo,
        time_signature=outline.time_signature,
        key_signature=outline.key_signature,
        active_section_id=active_section_id,
        section_bars=section_bars,
        chords=chords,
        selected_region=selected_region,
        polyphonic_stack=polyphonic_stack,
        free_registers=free_registers,
        missing_roles=missing_roles,
        tail_bars=tail_bars,
        manifest=manifest,
    )
