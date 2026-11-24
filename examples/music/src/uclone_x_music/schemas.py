"""The shapes of a song's files, and how a file that does not fit is reported.

A song's files are the user's: YAML a person can edit by hand. Each is validated every
time it is read, and a file that does not fit is refused with **the file and the field**
named, in words a person reads. Nothing is defaulted in place of a wrong value.

This module is pure: it parses text it is given and touches no file.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from uclone_x.errors import PlainRefusalError

#: The folder under a workspace that holds the song library.
SONGS_DIRNAME = "songs"

__all__ = [
    "ARTICULATIONS",
    "DRUM_PITCH_ALIASES",
    "ENTRY_ID_PATTERN",
    "INSTRUMENT_CATEGORIES",
    "INSTRUMENT_ROLES",
    "KEY_SIGNATURE_PATTERN",
    "SONGS_DIRNAME",
    "STYLE_PROFILES",
    "TIME_SIGNATURE_PATTERN",
    "Articulation",
    "BarNoteEvent",
    "EntryId",
    "InstrumentCategory",
    "InstrumentEntry",
    "InstrumentRole",
    "MusicFileError",
    "MusicProposal",
    "ProposalKind",
    "RegionSelection",
    "Revision",
    "RevisionLog",
    "ScoreOutline",
    "ScoreSection",
    "SessionEntry",
    "SessionsFile",
    "StyleAxioms",
    "StyleProfile",
    "TempoChange",
    "TrackBar",
    "TrackFile",
    "beats_per_bar_for",
    "default_style_axioms",
    "describe_invalid",
    "dump_file",
    "format_note_token",
    "normalize_bar_notes",
    "normalize_track_bar",
    "parse_file",
    "parse_note_token",
    "parse_pitch",
    "pitch_to_name",
    "resolve_track_bars",
    "validate_track_against_section",
]

ENTRY_ID_PATTERN = r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$"
EntryId = Annotated[str, Field(pattern=ENTRY_ID_PATTERN, max_length=80)]

InstrumentCategory = Literal[
    "bass",
    "strings",
    "piano",
    "drums",
    "woodwinds",
    "brass",
    "synth",
    "guitar",
    "vocal",
]
INSTRUMENT_CATEGORIES: tuple[InstrumentCategory, ...] = (
    "bass",
    "strings",
    "piano",
    "drums",
    "woodwinds",
    "brass",
    "synth",
    "guitar",
    "vocal",
)

InstrumentRole = Literal[
    "bass_foundation",
    "lead_melody",
    "counterpoint",
    "harmonic_comping",
    "arpeggio",
    "rhythm_percussion",
    "pad_texture",
    "satb_soprano",
    "satb_alto",
    "satb_tenor",
    "satb_bass",
]
INSTRUMENT_ROLES: tuple[InstrumentRole, ...] = (
    "bass_foundation",
    "lead_melody",
    "counterpoint",
    "harmonic_comping",
    "arpeggio",
    "rhythm_percussion",
    "pad_texture",
    "satb_soprano",
    "satb_alto",
    "satb_tenor",
    "satb_bass",
)

Articulation = Literal[
    "normal",
    "legato",
    "staccato",
    "tenuto",
    "accent",
    "vibrato",
    "pizzicato",
]
ARTICULATIONS: tuple[Articulation, ...] = (
    "normal",
    "legato",
    "staccato",
    "tenuto",
    "accent",
    "vibrato",
    "pizzicato",
)
_ARTICULATION_ALIASES: dict[str, Articulation] = {
    "trill": "vibrato",
    "tremolo": "vibrato",
    "fermata": "tenuto",
    "sustain": "tenuto",
    "sustained": "tenuto",
    "held": "tenuto",
    "marcato": "accent",
    "sforzando": "accent",
    "sfz": "accent",
    "portamento": "legato",
    "glissando": "legato",
    "slur": "legato",
    "spiccato": "staccato",
    "muted": "staccato",
}

StyleProfile = Literal[
    "pop_ballad",
    "hiphop_groove",
    "jazz_swing",
    "classical_strict",
]
STYLE_PROFILES: tuple[StyleProfile, ...] = (
    "pop_ballad",
    "hiphop_groove",
    "jazz_swing",
    "classical_strict",
)

ProposalKind = Literal["track_patch", "track_replace", "instrument_change"]

#: A key as the model states it: a tonic letter, an optional accidental and the mode.
KEY_SIGNATURE_PATTERN = r"^[A-G][#b]? (?:major|minor)$"
#: A meter whose lower number is a note value the bar grid can count.
TIME_SIGNATURE_PATTERN = r"^\d+/(?:2|4|8|16)$"

#: General MIDI percussion note numbers for readable drum tokens.
DRUM_PITCH_ALIASES: dict[str, int] = {
    "KICK": 36,
    "BD": 36,
    "RIM": 37,
    "SNARE": 38,
    "SD": 38,
    "CLAP": 39,
    "CLOSED_HAT": 42,
    "HIHAT": 42,
    "HH": 42,
    "LOW_TOM": 45,
    "OPEN_HAT": 46,
    "OH": 46,
    "MID_TOM": 47,
    "CRASH": 49,
    "HIGH_TOM": 50,
    "RIDE": 51,
}

_NOTE_OFFSETS: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_SHARP_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")


class MusicFileError(PlainRefusalError):
    """A song file is there and does not fit its shape; the message names file and field."""


class _Shape(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def parse_pitch(token: str) -> int:
    """Convert a pitch name (`D2`, `F#4`, `Bb3`, `KICK`, `REST`) into a MIDI integer.

    Returns `-1` for a rest (`R` or `REST`).
    Raises `ValueError` when `token` is not a valid pitch or drum alias.
    """
    cleaned = token.strip()
    upper = cleaned.upper()
    if upper in ("R", "REST"):
        return -1
    if upper in DRUM_PITCH_ALIASES:
        return DRUM_PITCH_ALIASES[upper]
    match = _PITCH_RE.fullmatch(cleaned)
    if match is None:
        raise ValueError(
            f"Invalid pitch {token!r}: use scientific pitch like 'D2', 'F#4', 'Bb3', "
            "'REST', or a drum name like 'KICK', 'SNARE', 'HIHAT'."
        )
    letter, accidental, octave_str = match.groups()
    semitone = _NOTE_OFFSETS[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    octave = int(octave_str)
    midi = (octave + 1) * 12 + semitone
    if midi < 0 or midi > 127:
        raise ValueError(f"Pitch {token!r} is outside the MIDI range 0..127 (got {midi}).")
    return midi


def pitch_to_name(midi: int) -> str:
    """Convert a MIDI integer `0..127` (or `-1` for `REST`) to scientific pitch notation."""
    if midi == -1:
        return "REST"
    if midi < 0 or midi > 127:
        raise ValueError(f"MIDI note {midi} is outside 0..127.")
    octave = (midi // 12) - 1
    name = _SHARP_NAMES[midi % 12]
    return f"{name}{octave}"


class BarNoteEvent(_Shape):
    """A parsed note or rest event inside one bar."""

    beat: float = Field(ge=1.0)
    pitch: str
    midi: int = Field(ge=-1, le=127)
    duration: float = Field(gt=0.0)
    velocity: int = Field(default=85, ge=0, le=127)
    articulation: Articulation = "normal"


def parse_note_token(token: str) -> BarNoteEvent:
    """Parse a Bar-Grid Step DSL token (`<beat>:<pitch>:<duration>[:v<vel>][:<articulation>]`)."""
    parts = [part.strip() for part in token.strip().split(":")]
    if len(parts) < 3 or len(parts) > 5:
        raise ValueError(
            f"Invalid note token {token!r}: expected '<beat>:<pitch>:<duration>[:v<velocity>][:<articulation>]'."
        )
    try:
        beat = float(parts[0])
    except ValueError as exc:
        raise ValueError(f"Invalid beat {parts[0]!r} in token {token!r}.") from exc
    if beat < 1.0:
        raise ValueError(f"Beat in token {token!r} must be >= 1.0 (1-indexed within bar).")

    pitch_str = parts[1]
    midi = parse_pitch(pitch_str)

    try:
        duration = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"Invalid duration {parts[2]!r} in token {token!r}.") from exc
    if duration <= 0.0:
        raise ValueError(f"Duration in token {token!r} must be > 0.")

    velocity = 0 if midi == -1 else 85
    articulation: Articulation = "normal"

    for extra in parts[3:]:
        if extra.lower().startswith("v") and extra[1:].isdigit():
            vel = int(extra[1:])
            if vel < 0 or vel > 127:
                raise ValueError(f"Velocity in token {token!r} must be between 0 and 127.")
            velocity = 0 if midi == -1 else vel
        elif extra.lower() in ARTICULATIONS:
            articulation = cast(Articulation, extra.lower())
        elif extra.lower() in _ARTICULATION_ALIASES:
            articulation = _ARTICULATION_ALIASES[extra.lower()]
        else:
            raise ValueError(
                f"Unknown modifier {extra!r} in token {token!r}: expected 'v0'..'v127' or "
                f"one of {ARTICULATIONS}."
            )

    return BarNoteEvent(
        beat=beat,
        pitch="REST" if midi == -1 else pitch_str,
        midi=midi,
        duration=duration,
        velocity=velocity,
        articulation=articulation,
    )


def format_note_token(event: BarNoteEvent) -> str:
    """Format a `BarNoteEvent` back into its canonical Bar-Grid Step DSL string."""
    beat_str = f"{event.beat:g}" if event.beat % 1 != 0 else f"{int(event.beat)}.0"
    dur_str = f"{event.duration:g}" if event.duration % 1 != 0 else f"{int(event.duration)}.0"
    if event.midi == -1:
        return f"{beat_str}:REST:{dur_str}"
    base = f"{beat_str}:{event.pitch}:{dur_str}:v{event.velocity}"
    if event.articulation != "normal":
        return f"{base}:{event.articulation}"
    return base


def beats_per_bar_for(time_signature: str) -> float:
    """Compute quarter-note beat count per bar for `time_signature` (e.g. `4/4` -> `4.0`)."""
    num_str, den_str = time_signature.split("/")
    numerator = int(num_str)
    denominator = int(den_str)
    if numerator <= 0 or denominator not in (2, 4, 8, 16):
        raise ValueError(f"Unsupported time signature {time_signature!r}.")
    return float(numerator) * (4.0 / float(denominator))


def normalize_bar_notes(notes: Sequence[str], time_signature: str = "4/4") -> list[str]:
    """Clamp overflowing notes and pad trailing silence with a `REST` so the bar spans `time_signature`."""
    bpb = beats_per_bar_for(time_signature)
    bar_end = round(bpb + 1.0, 6)
    clamped: list[BarNoteEvent] = []
    for raw in notes:
        if not raw or not raw.strip():
            continue
        ev = parse_note_token(raw)
        if ev.beat >= bar_end - 1e-4:
            continue
        max_dur = round(bar_end - ev.beat, 4)
        dur = min(ev.duration, max_dur)
        if dur <= 0.0:
            continue
        clamped.append(
            BarNoteEvent(
                beat=ev.beat,
                pitch=ev.pitch,
                midi=ev.midi,
                duration=dur,
                velocity=ev.velocity,
                articulation=ev.articulation,
            )
        )

    if not clamped:
        return [
            format_note_token(
                BarNoteEvent(beat=1.0, pitch="REST", midi=-1, duration=bpb, velocity=0)
            )
        ]

    max_end = max(round(ev.beat + ev.duration, 6) for ev in clamped)
    if max_end < bar_end - 1e-3:
        rem = round(bar_end - max_end, 4)
        clamped.append(
            BarNoteEvent(
                beat=round(max_end, 4),
                pitch="REST",
                midi=-1,
                duration=rem,
                velocity=0,
            )
        )

    return [format_note_token(ev) for ev in clamped]


def normalize_track_bar(bar: TrackBar, time_signature: str = "4/4") -> TrackBar:
    """Return a copy of `bar` with its `notes` normalized to fill `time_signature`."""
    if bar.repeat_bar is not None:
        return bar
    return bar.model_copy(update={"notes": normalize_bar_notes(bar.notes, time_signature)})


class TempoChange(_Shape):
    """A tempo change taking effect at `bar`."""

    bar: int = Field(ge=1)
    bpm: int = Field(ge=20, le=300)
    curve: Literal["step", "accelerando", "ritardando"] = "step"


class ScoreSection(_Shape):
    """One structural section of a song (e.g. Intro, Verse 1, Chorus)."""

    id: EntryId
    title: str = Field(min_length=1)
    kind: Literal[
        "intro",
        "verse",
        "pre_chorus",
        "chorus",
        "bridge",
        "solo",
        "outro",
        "movement",
    ] = "verse"
    bars: int = Field(ge=1, le=128)
    chords: list[str] = Field(default_factory=list[str])
    tempo_map: list[TempoChange] = Field(default_factory=list[TempoChange])

    @model_validator(mode="after")
    def _check_chords_and_tempo(self) -> ScoreSection:
        if self.chords and len(self.chords) != self.bars:
            raise ValueError(
                f"Section {self.id!r} has {self.bars} bars, but chords list has "
                f"{len(self.chords)} entries (must be empty or match bars count)."
            )
        for change in self.tempo_map:
            if change.bar > self.bars:
                raise ValueError(
                    f"Tempo change at bar {change.bar} exceeds section {self.id!r} bar count ({self.bars})."
                )
        return self


class ScoreOutline(_Shape):
    """The structural outline of a song (`score.yaml`)."""

    tempo: int = Field(default=96, ge=20, le=300)
    time_signature: str = Field(default="4/4", pattern=TIME_SIGNATURE_PATTERN)
    key_signature: str = Field(default="C major", min_length=1)
    sections: list[ScoreSection] = Field(default_factory=list[ScoreSection])

    @property
    def beats_per_bar(self) -> float:
        return beats_per_bar_for(self.time_signature)

    @model_validator(mode="after")
    def _unique_section_ids(self) -> ScoreOutline:
        seen: set[str] = set()
        for sec in self.sections:
            if sec.id in seen:
                raise ValueError(f"Duplicate section id {sec.id!r} in score.yaml.")
            seen.add(sec.id)
        return self


class InstrumentEntry(_Shape):
    """An instrument slot in the song's orchestra (`orchestra/<id>.yaml`)."""

    id: EntryId
    name: str = Field(min_length=1)
    category: InstrumentCategory
    role: InstrumentRole
    midi_program: int = Field(default=0, ge=0, le=127)
    midi_channel: int = Field(default=1, ge=1, le=16)
    soundfont_patch: str = Field(default="gm_default", min_length=1)
    min_pitch: str = "C1"
    max_pitch: str = "C7"
    transposition_semitones: int = Field(default=0, ge=-24, le=24)
    volume: int = Field(default=85, ge=0, le=100)
    pan: float = Field(default=0.0, ge=-1.0, le=1.0)
    muted: bool = False
    solo: bool = False

    @model_validator(mode="after")
    def _validate_pitch_range(self) -> InstrumentEntry:
        low = parse_pitch(self.min_pitch)
        high = parse_pitch(self.max_pitch)
        if low < 0 or high < 0 or low > high:
            raise ValueError(
                f"Instrument {self.id!r} range {self.min_pitch}..{self.max_pitch} is invalid."
            )
        return self


class TrackBar(_Shape):
    """One bar of notes inside a track (`tracks/<section>_<instrument>.yaml`)."""

    bar: int = Field(ge=1)
    chord: str | None = None
    notes: list[str] = Field(default_factory=list[str])
    repeat_bar: int | None = Field(default=None, ge=1)
    transpose_semitones: int = Field(default=0, ge=-36, le=36)
    muted: bool = False

    @model_validator(mode="after")
    def _validate_bar(self) -> TrackBar:
        if self.repeat_bar is not None and self.notes:
            raise ValueError(
                f"Bar {self.bar} cannot specify both 'notes' and 'repeat_bar' ({self.repeat_bar})."
            )
        if self.repeat_bar is None and not self.notes:
            raise ValueError(
                f"Bar {self.bar} is empty: specify 'notes' (use '1.0:REST:4.0' for silence) or 'repeat_bar'."
            )
        if self.repeat_bar == self.bar:
            raise ValueError(f"Bar {self.bar} cannot repeat itself.")
        for token in self.notes:
            parse_note_token(token)
        return self

    def parsed_events(self) -> list[BarNoteEvent]:
        """Parse all note tokens in this bar, sorted by beat."""
        events = [parse_note_token(token) for token in self.notes]
        events.sort(key=lambda item: (item.beat, item.midi))
        return events


class TrackFile(_Shape):
    """A track's bar-by-bar notes for one section and one instrument."""

    track_id: EntryId
    section_id: EntryId
    instrument_id: EntryId
    time_signature: str = Field(default="4/4", pattern=TIME_SIGNATURE_PATTERN)
    groove_swing: float = Field(default=0.5, ge=0.5, le=0.75)
    bars: list[TrackBar] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_bars_timing_and_repeats(self) -> TrackFile:
        bpb = beats_per_bar_for(self.time_signature)
        by_num: dict[int, TrackBar] = {}
        for idx, bar in enumerate(self.bars, start=1):
            if bar.bar != idx:
                raise ValueError(
                    f"Track {self.track_id!r} bars must be numbered sequentially 1..N (expected bar {idx}, got {bar.bar})."
                )
            by_num[bar.bar] = bar

        for bar in self.bars:
            if bar.repeat_bar is not None:
                if bar.repeat_bar not in by_num:
                    raise ValueError(
                        f"Track {self.track_id!r} bar {bar.bar} repeats non-existent bar {bar.repeat_bar}."
                    )
                target = by_num[bar.repeat_bar]
                if target.repeat_bar is not None:
                    raise ValueError(
                        f"Track {self.track_id!r} bar {bar.bar} repeats bar {bar.repeat_bar}, which itself is a repeat_bar (point directly to the source bar)."
                    )
                continue

            events = bar.parsed_events()
            max_end = 1.0
            for ev in events:
                end_beat = round(ev.beat + ev.duration, 6)
                if end_beat > round(bpb + 1.0, 6) + 1e-5:
                    raise ValueError(
                        f"Track {self.track_id!r} bar {bar.bar} note at beat {ev.beat} with duration "
                        f"{ev.duration} overflows the bar boundary ({bpb:g} beats per bar)."
                    )
                if end_beat > max_end:
                    max_end = end_beat

            total_span = round(max_end - 1.0, 4)
            if abs(total_span - bpb) > 1e-3:
                raise ValueError(
                    f"Track {self.track_id!r} bar {bar.bar} spans {total_span:g} beats, "
                    f"but time signature {self.time_signature} requires {bpb:g} beats per bar."
                )
        return self


def resolve_track_bars(track: TrackFile) -> dict[int, list[BarNoteEvent]]:
    """Expand all bars in `track` (resolving `repeat_bar` and `transpose_semitones`)."""
    by_num = {bar.bar: bar for bar in track.bars}
    resolved: dict[int, list[BarNoteEvent]] = {}
    for bar in track.bars:
        if bar.muted:
            bpb = beats_per_bar_for(track.time_signature)
            resolved[bar.bar] = [
                BarNoteEvent(beat=1.0, pitch="REST", midi=-1, duration=bpb, velocity=0)
            ]
            continue
        source = by_num[bar.repeat_bar] if bar.repeat_bar is not None else bar
        shift = bar.transpose_semitones
        events: list[BarNoteEvent] = []
        for ev in source.parsed_events():
            if ev.midi == -1 or shift == 0:
                events.append(ev)
            else:
                new_midi = max(0, min(127, ev.midi + shift))
                events.append(
                    BarNoteEvent(
                        beat=ev.beat,
                        pitch=pitch_to_name(new_midi),
                        midi=new_midi,
                        duration=ev.duration,
                        velocity=ev.velocity,
                        articulation=ev.articulation,
                    )
                )
        resolved[bar.bar] = events
    return resolved


def validate_track_against_section(
    track: TrackFile, section: ScoreSection, outline: ScoreOutline
) -> None:
    """Verify that `track` matches `section.bars` and `outline.time_signature`."""
    if track.section_id != section.id:
        raise MusicFileError(
            f"Track {track.track_id!r} names section {track.section_id!r}, expected {section.id!r}."
        )
    if track.time_signature != outline.time_signature:
        raise MusicFileError(
            f"Track {track.track_id!r} has time_signature {track.time_signature!r}, "
            f"which does not match score.yaml ({outline.time_signature!r})."
        )
    if len(track.bars) != section.bars:
        raise MusicFileError(
            f"Track {track.track_id!r} has {len(track.bars)} bars, "
            f"but section {section.id!r} requires {section.bars} bars."
        )


class StyleAxioms(_Shape):
    """Genre-specific composition and theory audit axioms stored on `song.yaml`."""

    profile: StyleProfile = "pop_ballad"
    forbid_parallel_fifths_octaves: bool = False
    forbid_voice_crossing: bool = False
    require_kick_bass_lock: bool = False
    default_swing: float = Field(default=0.5, ge=0.5, le=0.75)


def default_style_axioms(profile: StyleProfile = "pop_ballad") -> StyleAxioms:
    """Return the canonical `StyleAxioms` for `profile`."""
    if profile == "classical_strict":
        return StyleAxioms(
            profile="classical_strict",
            forbid_parallel_fifths_octaves=True,
            forbid_voice_crossing=True,
            require_kick_bass_lock=False,
            default_swing=0.5,
        )
    if profile == "hiphop_groove":
        return StyleAxioms(
            profile="hiphop_groove",
            forbid_parallel_fifths_octaves=False,
            forbid_voice_crossing=False,
            require_kick_bass_lock=True,
            default_swing=0.58,
        )
    if profile == "jazz_swing":
        return StyleAxioms(
            profile="jazz_swing",
            forbid_parallel_fifths_octaves=False,
            forbid_voice_crossing=False,
            require_kick_bass_lock=False,
            default_swing=0.60,
        )
    return StyleAxioms(
        profile="pop_ballad",
        forbid_parallel_fifths_octaves=False,
        forbid_voice_crossing=False,
        require_kick_bass_lock=True,
        default_swing=0.5,
    )


class RegionSelection(_Shape):
    """A bar range and optional instrument filter selected by the user in MusicView."""

    section_id: EntryId
    start_bar: int = Field(ge=1)
    end_bar: int = Field(ge=1)
    instrument_ids: list[EntryId] = Field(default_factory=list[str])

    @model_validator(mode="after")
    def _check_bounds(self) -> RegionSelection:
        if self.end_bar < self.start_bar:
            raise ValueError(
                f"RegionSelection end_bar ({self.end_bar}) must be >= start_bar ({self.start_bar})."
            )
        return self


class MusicProposal(_Shape):
    """A proposed track edit, bar-range patch, or instrument update awaiting person approval."""

    id: EntryId
    kind: ProposalKind
    target_file: str = Field(min_length=1)
    target_digest: str | None = None
    section_id: EntryId | None = None
    instrument_id: EntryId | None = None
    start_bar: int | None = Field(default=None, ge=1)
    end_bar: int | None = Field(default=None, ge=1)
    summary: str = Field(min_length=1)
    proposed_bars: list[TrackBar] = Field(default_factory=list[TrackBar])
    proposed_track: TrackFile | None = None
    proposed_instrument: InstrumentEntry | None = None
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_by: str | None = None
    created_at: str | None = None
    decided_in: Literal["conversation", "music_view"] | None = None
    decided_at: str | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> MusicProposal:
        if self.kind == "track_patch":
            if (
                self.section_id is None
                or self.instrument_id is None
                or self.start_bar is None
                or self.end_bar is None
                or not self.proposed_bars
            ):
                raise ValueError(
                    "track_patch proposal requires section_id, instrument_id, start_bar, end_bar, and proposed_bars."
                )
            if self.end_bar < self.start_bar:
                raise ValueError("track_patch end_bar must be >= start_bar.")
            expected = list(range(self.start_bar, self.end_bar + 1))
            actual = [b.bar for b in self.proposed_bars]
            if actual != expected:
                raise ValueError(
                    f"track_patch proposed_bars must cover bars {expected}, got {actual}."
                )
        elif self.kind == "track_replace":
            if self.proposed_track is None:
                raise ValueError("track_replace proposal requires proposed_track.")
        elif self.kind == "instrument_change":
            if self.proposed_instrument is None:
                raise ValueError("instrument_change proposal requires proposed_instrument.")
        return self


class Revision(_Shape):
    """One saved revision of a track file under `.history/<track_id>/revisions.yaml`."""

    digest: str = Field(min_length=1)
    saved_at: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    summary: str = ""


class RevisionLog(_Shape):
    """The revision log for a track."""

    revisions: list[Revision] = Field(default_factory=list[Revision])


class SessionEntry(_Shape):
    """One conversation's record of work on a song (`sessions.yaml`)."""

    room_id: str = Field(min_length=1)
    opened_at: str = Field(min_length=1)
    last_written_at: str | None = None
    summary: str = ""


class SessionsFile(_Shape):
    """The sessions log (`sessions.yaml`)."""

    sessions: list[SessionEntry] = Field(default_factory=list[SessionEntry])


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def describe_invalid(relative_path: str, exc: ValidationError) -> str:
    """Format a Pydantic `ValidationError` naming `relative_path` and the offending field."""
    errors = exc.errors()
    if not errors:
        return f"'{relative_path}' is invalid."
    first = errors[0]
    loc = ".".join(str(part) for part in first.get("loc", ()))
    msg = first.get("msg", "invalid value")
    if loc:
        return f"'{relative_path}' has an invalid '{loc}': {msg}"
    return f"'{relative_path}' is invalid: {msg}"


def parse_file(text: str, model: type[_ModelT], *, relative_path: str) -> _ModelT:
    """Parse YAML `text` into `model`, or raise `MusicFileError` naming `relative_path`."""
    try:
        loaded: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise MusicFileError(f"'{relative_path}' is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise MusicFileError(f"'{relative_path}' must be a YAML mapping at the top level.")
    try:
        return model.model_validate(loaded)
    except ValidationError as exc:
        raise MusicFileError(describe_invalid(relative_path, exc)) from exc


def dump_file(instance: BaseModel) -> str:
    """Serialize `instance` to clean human-editable YAML."""
    data = instance.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
