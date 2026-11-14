"""`music_start`: guided step-by-step interactive co-composition and regional bar editing.

Walks a song through four incremental layering stages so the person can listen after each
instrument is added and steer the arrangement in conversation:

1. `setup_foundation` -> writes `score.yaml` and the `bass_synth` groove track, then
   recommends `violin_solo` (`layer_melody`).
2. `layer_melody` -> adds `violin_solo` (`G3..E7`, `lead_melody`) in high register
   (`D5..A5`), then recommends `grand_piano` (`layer_harmony`).
3. `layer_harmony` -> adds `grand_piano` (`A0..C8`, `arpeggio`) in mid register
   (`C3..G4`), then recommends `drum_kit` (`layer_drums`).
4. `layer_drums` -> adds `drum_kit` (`midi_channel=10`, `rhythm_percussion`), switching to
   an 808 hip-hop groove (`groove_swing=0.58`) locked to the bass on beats 1.0 and 2.5 when
   hip-hop is requested, and marks the stage `completed`.

Also supports targeted regional fast-edits (subdividing a bar range into double-time
16th-note patterns) when the model calls `action="edit_region"` with the bars and instrument.

Every decision here is read from the tool's structured arguments. The model reads the
person's words and chooses the action, style and region; this module never guesses them from
keywords in the text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import REPLY_NOTE_KEY, ToolContext
from uclone_x_music import OPEN_SONG_KEY
from uclone_x_music.audit import audit_section
from uclone_x_music.digests import audit_digest
from uclone_x_music.groove import (
    chord_fifth_pc,
    chord_root_pc,
    chord_third_pc,
    generate_bass_groove,
    generate_hiphop_drums,
    subdivide_bars_faster,
)
from uclone_x_music.library import (
    MusicError,
    NoConversationError,
    SongFile,
    SongLibrary,
)
from uclone_x_music.schemas import (
    InstrumentEntry,
    RegionSelection,
    ScoreOutline,
    ScoreSection,
    StyleProfile,
    TrackBar,
    TrackFile,
    beats_per_bar_for,
    default_style_axioms,
    dump_file,
    parse_file,
    pitch_to_name,
    resolve_track_bars,
)
from uclone_x_music.work import SongWork

__all__ = [
    "BUNDLED_STYLES_PATH",
    "CompositionStage",
    "MusicStartParams",
    "MusicStartState",
    "MusicStartTool",
    "START_FILE",
    "load_bundled_styles",
    "read_start_state",
    "write_start_state",
]

START_FILE = "start.yaml"
BUNDLED_STYLES_PATH = Path(__file__).resolve().parent / "bundled" / "styles.yaml"

CompositionStage = Literal[
    "setup_foundation",
    "layer_melody",
    "layer_harmony",
    "layer_drums",
    "completed",
]


class MusicStartState(BaseModel):
    """Persistent state of a song's guided step-by-step composition (`start.yaml`)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    stage: CompositionStage = "setup_foundation"
    active_section_id: str = "sec_verse_01"
    style_profile: StyleProfile = "pop_ballad"
    completed_layers: list[str] = Field(default_factory=list[str])
    recommended_next: str = ""


class MusicStartParams(BaseModel):
    """Parameters for `music_start`."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["start", "continue", "status", "edit_region"] = Field(
        default="start",
        description="'start' a new guided song composition or advance the open one; "
        "'continue' to add the next instrument layer (call it only when the person agreed to "
        "go on; if they declined, do not call the tool); 'edit_region' to subdivide a bar "
        "range into a faster pattern; 'status' to inspect the current guided composition stage.",
    )
    request: str | None = Field(
        default=None,
        description="The person's request in their own words, kept as the song's logline. "
        "It is recorded, never interpreted: state the style, region and instrument in the "
        "structured fields.",
    )
    title: str | None = Field(
        default=None,
        description="Optional title when creating a new song.",
    )
    style_profile: StyleProfile | None = Field(
        default=None,
        description="Style profile the person asked for ('pop_ballad', 'hiphop_groove', "
        "'jazz_swing', 'classical_strict'). Omit to keep the song's current style.",
    )
    section_id: str | None = Field(
        default=None,
        description="Target section id (defaults to active_section_id, e.g. 'sec_verse_01').",
    )
    instrument_id: str | None = Field(
        default=None,
        description="Target instrument id for regional editing (e.g. 'violin_solo').",
    )
    selection: RegionSelection | None = Field(
        default=None,
        description="Explicit regional bar selection for targeted bar editing.",
    )
    start_bar: int | None = Field(
        default=None,
        ge=1,
        description="Start bar (1-indexed) for regional editing when 'selection' is not given.",
    )
    end_bar: int | None = Field(
        default=None,
        ge=1,
        description="End bar (1-indexed) for regional editing when 'selection' is not given.",
    )
    factor: Literal[2, 4] = Field(
        default=2,
        description="Rhythmic subdivision factor (2 or 4) for fast regional edits.",
    )


def load_bundled_styles() -> dict[str, dict[str, Any]]:
    """Load the bundled style presets from `bundled/styles.yaml`."""
    if not BUNDLED_STYLES_PATH.is_file():
        return {}
    raw = yaml.safe_load(BUNDLED_STYLES_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    styles = cast(dict[str, object], raw).get("styles")
    if not isinstance(styles, dict):
        return {}
    return cast(dict[str, dict[str, Any]], styles)


def read_start_state(work: SongWork) -> tuple[MusicStartState, str] | None:
    """Read `start.yaml` from `work`, returning `(MusicStartState, digest)` or `None`."""
    file = work.library.read_file(work.song_id, START_FILE)
    if file is None:
        return None
    state = parse_file(file.text, MusicStartState, relative_path=START_FILE)
    return state, file.digest


def write_start_state(
    work: SongWork,
    state: MusicStartState,
    *,
    expected_digest: str | None,
) -> SongFile:
    """Write `start.yaml` to `work` through the lease and digest guard."""
    return work.library.write_file(
        work.song_id,
        START_FILE,
        dump_file(state),
        room_id=work.room_id,
        expected_digest=expected_digest,
    )


def _violin_chord_tones(chord: str | None) -> tuple[str, str, str, str]:
    """Return `(root_p, third_p, fifth_p, color_p)` pitch names inside `D5..A5` (`MIDI 74..81`) for `chord`.

    Every returned pitch is strictly inside `D5..A5` and guaranteed not to form a minor-2nd
    clash against the chord's root or third.
    """
    from uclone_x_music.audit import parse_chord_pitch_classes

    sym = chord or "Dm7"
    r_pc = chord_root_pc(sym)
    t_pc = chord_third_pc(sym, r_pc)
    f_pc = chord_fifth_pc(sym, r_pc)
    chord_pcs = parse_chord_pitch_classes(sym)
    guide_pcs = {r_pc, t_pc}

    def _safe_in_d5_a5(preferred_pcs: list[int]) -> int:
        for pc in preferred_pcs:
            for m in range(74, 82):
                if m % 12 == pc % 12 and not any(abs((m % 12) - g) in (1, 11) for g in guide_pcs):
                    return m
        for m in (74, 76, 77, 79, 81, 75, 78, 80):
            if m % 12 in chord_pcs and not any(abs((m % 12) - g) in (1, 11) for g in guide_pcs):
                return m
        for m in (74, 76, 77, 79, 81):
            if not any(abs((m % 12) - g) in (1, 11) for g in guide_pcs):
                return m
        return 74

    root_m = _safe_in_d5_a5([r_pc, f_pc, t_pc])
    third_m = _safe_in_d5_a5([t_pc, (r_pc + 9) % 12, f_pc, r_pc])
    fifth_m = _safe_in_d5_a5([f_pc, t_pc, r_pc])
    color_candidates = [
        (r_pc + 2) % 12,
        (r_pc + 9) % 12,
        t_pc,
        f_pc,
        r_pc,
    ]
    color_m = _safe_in_d5_a5(color_candidates)
    # Ensure distinct melodic contour options when possible
    used = {root_m, third_m, fifth_m}
    if color_m in used:
        for m in (76, 77, 79, 81, 74):
            if m not in used and not any(abs((m % 12) - g) in (1, 11) for g in guide_pcs):
                color_m = m
                break
    return (
        pitch_to_name(root_m),
        pitch_to_name(third_m),
        pitch_to_name(fifth_m),
        pitch_to_name(color_m),
    )


def _piano_chord_tones_c3_g4(chord: str | None) -> tuple[str, str, str, str]:
    """Return `(low_root, mid_third, mid_fifth, high_color)` strictly inside `F3..G4` (`MIDI 53..67`).

    Staying in `F3..G4` (`MIDI 53..67`, a sub-window of `C3..G4`) keeps the piano accompaniment
    strictly above `bass_synth` (`<= E3`, MIDI 52) and strictly below `violin_solo` (`D5..A5`,
    MIDI 74..81), preventing voice crossing and harmonic clashes.
    """
    from uclone_x_music.audit import parse_chord_pitch_classes

    sym = chord or "Dm7"
    r_pc = chord_root_pc(sym)
    t_pc = chord_third_pc(sym, r_pc)
    f_pc = chord_fifth_pc(sym, r_pc)
    chord_pcs = parse_chord_pitch_classes(sym)
    guide_pcs = {r_pc, t_pc}

    valid_midis = [
        m
        for m in range(55, 68)
        if (m % 12 in chord_pcs or m % 12 in {2, 4, 5, 7, 9, 0})
        and not any(abs((m % 12) - g) in (1, 11) for g in guide_pcs)
    ]
    chord_midis = [m for m in range(55, 68) if m % 12 in chord_pcs]
    pool = chord_midis if len(chord_midis) >= 3 else valid_midis
    if len(pool) < 4:
        pool = [57, 60, 62, 65]

    # Pick 4 ascending chord/consonant tones in G3..G4 (MIDI 55..67)
    low_m = pool[0]
    mid_t_m = next((m for m in pool if m > low_m and m % 12 == t_pc), pool[min(1, len(pool) - 1)])
    mid_f_m = next(
        (m for m in pool if m > mid_t_m and m % 12 in (f_pc, r_pc)),
        pool[min(2, len(pool) - 1)],
    )
    high_m = pool[-1]
    return (
        pitch_to_name(low_m),
        pitch_to_name(mid_t_m),
        pitch_to_name(mid_f_m),
        pitch_to_name(high_m),
    )


def _build_violin_melody_track(
    section: ScoreSection,
    *,
    instrument_id: str = "violin_solo",
    time_signature: str = "4/4",
    swing: float = 0.5,
    style: StyleProfile = "pop_ballad",
    bass_track: TrackFile | None = None,
) -> TrackFile:
    """Build a chord-aware, style-specific, bass-responsive (`D5..A5`) violin melody over `section`."""
    bpb = beats_per_bar_for(time_signature)
    bass_resolved = resolve_track_bars(bass_track) if bass_track is not None else {}
    bars: list[TrackBar] = []

    for bar_num in range(1, section.bars + 1):
        chord = section.chords[bar_num - 1] if 1 <= bar_num <= len(section.chords) else None
        root_p, third_p, fifth_p, color_p = _violin_chord_tones(chord)
        step = (bar_num - 1) % 4

        if abs(bpb - 4.0) < 1e-4:
            b_evs = bass_resolved.get(bar_num, [])
            has_opening_rest = bool(b_evs and b_evs[0].midi == -1 and b_evs[0].duration >= 0.5)
            has_mid_rest = any(
                ev.midi == -1 and 2.0 <= ev.beat <= 3.0 and ev.duration >= 0.5 for ev in b_evs
            )
            is_custom_dense_bass = sum(1 for ev in b_evs if ev.midi >= 0) >= 5

            if has_opening_rest:
                # Call & Response: Bass rests on beat 1 -> Violin leads boldly on beats 1..2, then breathes/sustains
                notes = [
                    f"1.0:{fifth_p}:0.5:v92:accent",
                    f"1.5:{third_p}:0.5:v86:legato",
                    f"2.0:{root_p}:0.5:v88:legato",
                    f"2.5:{color_p}:1.5:v90:vibrato",
                    f"4.0:{third_p}:1.0:v82:tenuto",
                ]
            elif has_mid_rest or is_custom_dense_bass:
                # Call & Response: Bass is busy early and rests mid-bar -> Violin soars on beat 1, answers actively on beat 2.5
                notes = [
                    f"1.0:{third_p}:1.5:v88:legato",
                    f"2.5:{fifth_p}:0.5:v90:accent",
                    f"3.0:{color_p}:0.5:v86:legato",
                    f"3.5:{third_p}:0.5:v84:legato",
                    f"4.0:{root_p}:1.0:v86:vibrato",
                ]
            elif style == "classical_strict":
                classical_cells: tuple[list[str], ...] = (
                    [
                        f"1.0:{third_p}:1.0:v88:legato",
                        f"2.0:{color_p}:0.5:v82:legato",
                        f"2.5:{third_p}:0.5:v84:legato",
                        f"3.0:{fifth_p}:1.0:v90:vibrato",
                        f"4.0:{root_p}:1.0:v82:tenuto",
                    ],
                    [
                        f"1.0:{third_p}:1.5:v90:legato",
                        f"2.5:{color_p}:0.5:v84:legato",
                        f"3.0:{root_p}:1.0:v88:vibrato",
                        f"4.0:{third_p}:0.5:v82:legato",
                        f"4.5:{fifth_p}:0.5:v80:legato",
                    ],
                    [
                        f"1.0:{third_p}:0.5:v86:legato",
                        f"1.5:{color_p}:0.5:v84:legato",
                        f"2.0:{fifth_p}:1.0:v92:vibrato",
                        f"3.0:{third_p}:1.0:v88:legato",
                        f"4.0:{root_p}:1.0:v84:tenuto",
                    ],
                    [
                        f"1.0:{third_p}:2.0:v92:vibrato",
                        f"3.0:{root_p}:2.0:v86:tenuto",
                    ],
                )
                notes = list(classical_cells[step])
            elif style == "jazz_swing":
                jazz_cells: tuple[list[str], ...] = (
                    [
                        f"1.0:{third_p}:0.5:v84:staccato",
                        f"1.5:{fifth_p}:1.5:v90:legato",
                        f"3.0:{color_p}:0.5:v86:legato",
                        f"3.5:{third_p}:1.5:v88:vibrato",
                    ],
                    [
                        f"1.0:{fifth_p}:1.0:v88:legato",
                        f"2.0:{third_p}:0.5:v82:staccato",
                        f"2.5:{root_p}:1.5:v90:vibrato",
                        f"4.0:{color_p}:1.0:v84:tenuto",
                    ],
                    [
                        f"1.0:{root_p}:0.5:v86:legato",
                        f"1.5:{third_p}:0.5:v88:legato",
                        f"2.0:{fifth_p}:1.0:v92:accent",
                        f"3.0:{color_p}:1.0:v88:vibrato",
                        f"4.0:{third_p}:1.0:v82:legato",
                    ],
                    [
                        f"1.0:{fifth_p}:1.5:v90:vibrato",
                        f"2.5:{third_p}:0.5:v84:legato",
                        f"3.0:{root_p}:2.0:v86:tenuto",
                    ],
                )
                notes = list(jazz_cells[step])
            elif style == "hiphop_groove":
                hiphop_cells: tuple[list[str], ...] = (
                    [
                        f"1.0:{fifth_p}:1.0:v90:accent",
                        f"2.0:{third_p}:0.5:v84:staccato",
                        f"2.5:{root_p}:1.5:v88:vibrato",
                        f"4.0:{third_p}:1.0:v82:legato",
                    ],
                    [
                        f"1.0:{third_p}:0.5:v88:staccato",
                        f"1.5:{fifth_p}:1.0:v92:accent",
                        f"2.5:{third_p}:0.5:v84:staccato",
                        f"3.0:{root_p}:1.0:v88:vibrato",
                        f"4.0:{color_p}:1.0:v80:tenuto",
                    ],
                    [
                        f"1.0:{root_p}:1.5:v90:legato",
                        f"2.5:{third_p}:0.5:v86:legato",
                        f"3.0:{fifth_p}:1.0:v92:vibrato",
                        f"4.0:{third_p}:1.0:v84:legato",
                    ],
                    [
                        f"1.0:{fifth_p}:0.5:v90:accent",
                        f"1.5:{third_p}:0.5:v86:legato",
                        f"2.0:{root_p}:1.0:v88:legato",
                        f"3.0:{third_p}:2.0:v86:tenuto",
                    ],
                )
                notes = list(hiphop_cells[step])
            else:
                # pop_ballad: varied 4-bar lyrical phrasing (opening -> syncopated response -> arch -> cadence)
                pop_cells: tuple[list[str], ...] = (
                    [
                        f"1.0:{fifth_p}:1.5:v88:legato",
                        f"2.5:{color_p}:0.5:v82:legato",
                        f"3.0:{third_p}:1.0:v86:vibrato",
                        f"4.0:{root_p}:1.0:v80:legato",
                    ],
                    [
                        f"1.0:{third_p}:1.0:v88:legato",
                        f"2.0:{fifth_p}:0.5:v84:legato",
                        f"2.5:{root_p}:1.5:v90:vibrato",
                        f"4.0:{third_p}:1.0:v82:tenuto",
                    ],
                    [
                        f"1.0:{root_p}:1.0:v86:legato",
                        f"2.0:{third_p}:0.5:v88:legato",
                        f"2.5:{color_p}:0.5:v86:legato",
                        f"3.0:{fifth_p}:1.0:v92:vibrato",
                        f"4.0:{third_p}:1.0:v84:legato",
                    ],
                    [
                        f"1.0:{third_p}:1.5:v90:legato",
                        f"2.5:{fifth_p}:0.5:v86:legato",
                        f"3.0:{root_p}:2.0:v88:tenuto",
                    ],
                )
                notes = list(pop_cells[step])
        else:
            notes = [f"1.0:{fifth_p}:{bpb:g}:v88:legato"]
        bars.append(TrackBar(bar=bar_num, chord=chord, notes=notes))

    return TrackFile(
        track_id=f"{section.id}_{instrument_id}",
        section_id=section.id,
        instrument_id=instrument_id,
        time_signature=time_signature,
        groove_swing=swing,
        bars=bars,
    )


def _build_piano_arpeggio_track(
    section: ScoreSection,
    *,
    instrument_id: str = "grand_piano",
    time_signature: str = "4/4",
    swing: float = 0.5,
) -> TrackFile:
    """Build a chord-aware mid-register (`C3..G4`) piano arpeggio/comping track over `section`."""
    bpb = beats_per_bar_for(time_signature)
    bars: list[TrackBar] = []
    for bar_num in range(1, section.bars + 1):
        chord = section.chords[bar_num - 1] if 1 <= bar_num <= len(section.chords) else None
        low_r, mid_t, mid_f, high_c = _piano_chord_tones_c3_g4(chord)
        step = (bar_num - 1) % 4
        if abs(bpb - 4.0) < 1e-4:
            if step in (0, 2):
                notes = [
                    f"1.0:{low_r}:0.5:v76",
                    f"1.5:{mid_t}:0.5:v72",
                    f"2.0:{mid_f}:0.5:v74",
                    f"2.5:{high_c}:0.5:v78",
                    f"3.0:{mid_f}:1.0:v75",
                    f"4.0:{mid_t}:1.0:v74",
                ]
            elif step == 1:
                notes = [
                    f"1.0:{low_r}:0.5:v76",
                    f"1.5:{mid_f}:0.5:v72",
                    f"2.0:{high_c}:1.0:v78:tenuto",
                    f"3.0:{mid_f}:0.5:v74",
                    f"3.5:{mid_t}:0.5:v72",
                    f"4.0:{low_r}:1.0:v74",
                ]
            else:
                notes = [
                    f"1.0:{low_r}:0.5:v76",
                    f"1.5:{mid_t}:0.5:v72",
                    f"2.0:{mid_f}:0.5:v75",
                    f"2.5:{high_c}:0.5:v78",
                    f"3.0:{mid_t}:2.0:v76:tenuto",
                ]
        else:
            notes = [f"1.0:{mid_t}:{bpb:g}:v75"]
        bars.append(TrackBar(bar=bar_num, chord=chord, notes=notes))

    return TrackFile(
        track_id=f"{section.id}_{instrument_id}",
        section_id=section.id,
        instrument_id=instrument_id,
        time_signature=time_signature,
        groove_swing=swing,
        bars=bars,
    )


def _build_pop_drums_track(
    section: ScoreSection,
    *,
    instrument_id: str = "drum_kit",
    time_signature: str = "4/4",
) -> TrackFile:
    """Build a clean standard pop drum kit track locked to beat 1.0 and 3.0."""
    bpb = beats_per_bar_for(time_signature)
    bars: list[TrackBar] = []
    for bar_num in range(1, section.bars + 1):
        chord = section.chords[bar_num - 1] if 1 <= bar_num <= len(section.chords) else None
        if abs(bpb - 4.0) < 1e-4:
            notes = [
                "1.0:KICK:0.5:v96:accent",
                "3.0:KICK:0.5:v92",
                "2.0:SNARE:0.5:v90:accent",
                "4.0:SNARE:0.5:v90:accent",
                "1.0:HIHAT:0.5:v76",
                "1.5:HIHAT:0.5:v64",
                "2.0:HIHAT:0.5:v76",
                "2.5:HIHAT:0.5:v64",
                "3.0:HIHAT:0.5:v76",
                "3.5:HIHAT:0.5:v64",
                "4.0:HIHAT:0.5:v76",
                "4.5:HIHAT:0.5:v64",
            ]
        else:
            notes = [f"1.0:KICK:{bpb:g}:v95"]
        bars.append(TrackBar(bar=bar_num, chord=chord, notes=notes))

    return TrackFile(
        track_id=f"{section.id}_{instrument_id}",
        section_id=section.id,
        instrument_id=instrument_id,
        time_signature=time_signature,
        groove_swing=0.5,
        bars=bars,
    )


def _stage_reply_note(
    stage: CompositionStage,
    *,
    style_profile: StyleProfile,
    chords: list[str] | None = None,
) -> dict[str, str]:
    chord_str = "-".join(chords[:4]) if chords else "Dm7-Gm7-C7-Fmaj7"
    if stage == "layer_melody":
        return {
            "ko": (
                f"기본 베이스 리듬(`bass_synth`, 4마디 {chord_str})을 만들었습니다. "
                "들어보시고 마음에 드시면 '네' 또는 '바이올린 추가해줘'라고 말씀해 주세요."
            ),
            "en": (
                f"Created the base rhythm track (`bass_synth`, 4 bars {chord_str}). "
                "Listen to it and say 'yes' or ask to add the violin melody next."
            ),
        }
    if stage == "layer_harmony":
        return {
            "ko": (
                "고음역 바이올린 선율(`violin_solo`)을 올렸습니다. "
                "다음으로 중음역 피아노 화성 반주(`grand_piano`)를 추가할까요?"
            ),
            "en": (
                "Added the high-register solo violin melody (`violin_solo`). "
                "Shall I add the mid-register grand piano arpeggio (`grand_piano`) next?"
            ),
        }
    if stage == "layer_drums":
        return {
            "ko": (
                "피아노 아르페지오 화성(`grand_piano`)을 추가했습니다. "
                "마지막으로 드럼 비트(`drum_kit`, 예: '네' 또는 '힙합풍 박자 넣어줘')를 추가할까요?"
            ),
            "en": (
                "Added the grand piano arpeggio harmony (`grand_piano`). "
                "Shall I add the drum beat (`drum_kit`, e.g. pop or hip-hop groove) to complete the section?"
            ),
        }
    if stage == "completed":
        groove_ko = "힙합 스윙 비트" if style_profile == "hiphop_groove" else "드럼 비트"
        groove_en = "hip-hop swing beat" if style_profile == "hiphop_groove" else "drum beat"
        return {
            "ko": (
                f"{groove_ko}(`drum_kit`)까지 모든 파트가 완성되었습니다! "
                "특정 마디를 선택해 수정하거나(예: '3~4마디 바이올린 빠르게') 악기 믹스를 조절할 수 있습니다."
            ),
            "en": (
                f"Added the {groove_en} (`drum_kit`) and completed all layers! "
                "You can now select specific bars to edit (e.g. 'bars 3-4 violin faster') or adjust the mix."
            ),
        }
    return {
        "ko": "곡 구성을 준비 중입니다.",
        "en": "Preparing song foundation.",
    }


class MusicStartTool(BaseTool[MusicStartParams]):
    """Guided step-by-step interactive music composition and regional bar editing tool."""

    name = "music_start"
    description = (
        "Start and advance an interactive multi-track song composition one instrument layer "
        "at a time: (1) bass rhythm foundation ('bass_synth'), (2) solo violin melody "
        "('violin_solo'), (3) grand piano arpeggio harmony ('grand_piano'), and (4) drum kit "
        "('drum_kit', supporting pop or 808 hip-hop swing beats locked to the bass). "
        "Also supports targeted regional fast-edits: call action='edit_region' with "
        "'selection' (or start_bar/end_bar and instrument_id) to subdivide those bars into a "
        "16th-note double-time pattern."
    )
    params_type = MusicStartParams
    writes_files: ClassVar[bool] = True
    opens_story: ClassVar[bool] = True
    needs_room: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"status"})

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=MusicStartParams,
        )

    async def run(self, params: MusicStartParams, context: ToolContext) -> dict[str, Any]:
        workspace = context.require_workspace()
        room_id = context.room_id
        library = SongLibrary(workspace)
        logline = (params.request or "").strip()

        if params.action == "status":
            work = SongWork.for_context(context)
            state_pair = read_start_state(work)
            state = state_pair[0] if state_pair is not None else MusicStartState()
            return self._format_output(work, state, opened_new=False)

        if not room_id or not room_id.strip():
            raise NoConversationError("A guided music composition requires an active conversation.")

        opened_new = False
        released_prev: str | None = None
        song_id = context.story_id

        if song_id is None:
            detected_style = params.style_profile or "pop_ballad"
            title = (params.title or "").strip() or "Untitled Song"
            record = library.create(
                title,
                room_id,
                genre=detected_style,
                logline=logline,
                style_profile=detected_style,
            )
            song_id = record.id
            opened_new = True
        else:
            try:
                library.load(song_id)
            except MusicError:
                detected_style = params.style_profile or "pop_ballad"
                title = (params.title or "").strip() or "Untitled Song"
                record = library.create(
                    title,
                    room_id,
                    genre=detected_style,
                    logline=logline,
                    style_profile=detected_style,
                )
                released_prev = context.story_id
                song_id = record.id
                opened_new = True

        work = SongWork(library, song_id, room_id)
        state_pair = read_start_state(work)
        effective_state = state_pair[0] if state_pair is not None else None
        # A regional fast-edit is an explicit action, or an explicit bar range in the arguments.
        asks_region_edit = (
            params.action == "edit_region"
            or params.selection is not None
            or params.start_bar is not None
        )
        if effective_state is None and asks_region_edit:
            existing_score = work.read_score()
            if existing_score is not None and existing_score[0].sections:
                effective_state = MusicStartState(
                    stage="completed",
                    active_section_id=params.section_id or existing_score[0].sections[0].id,
                    style_profile=work.record().style_axioms.profile,
                    completed_layers=[inst.id for inst, _ in work.list_instruments()],
                    recommended_next="",
                )

        selection = self._resolve_selection(params, effective_state)
        if effective_state is not None and asks_region_edit:
            return self._run_regional_fast_edit(
                work,
                effective_state,
                params=params,
                selection=selection,
                opened_new=opened_new,
            )

        if state_pair is None:
            initial_style = params.style_profile or work.record().style_axioms.profile
            state = MusicStartState(
                stage="setup_foundation",
                active_section_id=params.section_id or "sec_verse_01",
                style_profile=initial_style,
                completed_layers=[],
                recommended_next="",
            )
            return self._advance_stage(
                work,
                state,
                expected_digest=None,
                style_override=params.style_profile,
                opened_new=opened_new,
                released_prev=released_prev,
            )

        state, state_digest = state_pair
        if state.stage == "completed" and params.action == "start" and params.title:
            # Explicit start of a brand-new song when previous song already completed
            detected_style = params.style_profile or "pop_ballad"
            if context.story_id is not None:
                library.release(context.story_id, room_id)
                released_prev = context.story_id
            record = library.create(
                params.title.strip(),
                room_id,
                genre=detected_style,
                logline=logline,
                style_profile=detected_style,
            )
            new_work = SongWork(library, record.id, room_id)
            fresh_state = MusicStartState(
                stage="setup_foundation",
                active_section_id=params.section_id or "sec_verse_01",
                style_profile=detected_style,
            )
            return self._advance_stage(
                new_work,
                fresh_state,
                expected_digest=None,
                style_override=params.style_profile,
                opened_new=True,
                released_prev=released_prev,
            )

        return self._advance_stage(
            work,
            state,
            expected_digest=state_digest,
            style_override=params.style_profile,
            opened_new=opened_new,
            released_prev=released_prev,
        )

    @staticmethod
    def _resolve_selection(
        params: MusicStartParams,
        state: MusicStartState | None,
    ) -> RegionSelection | None:
        default_sec = params.section_id or (
            state.active_section_id if state is not None else "sec_verse_01"
        )
        if params.selection is not None:
            return params.selection
        if params.start_bar is not None:
            end_b = params.end_bar if params.end_bar is not None else params.start_bar
            insts = [params.instrument_id] if params.instrument_id else []
            return RegionSelection(
                section_id=default_sec,
                start_bar=params.start_bar,
                end_bar=end_b,
                instrument_ids=insts,
            )
        return None

    def _advance_stage(
        self,
        work: SongWork,
        state: MusicStartState,
        *,
        expected_digest: str | None,
        style_override: StyleProfile | None,
        opened_new: bool,
        released_prev: str | None,
    ) -> dict[str, Any]:
        section_id = state.active_section_id
        detected_style = style_override

        if state.stage == "setup_foundation":
            style: StyleProfile = detected_style or state.style_profile
            presets = load_bundled_styles()
            preset = presets.get(style, {})
            tempo = int(preset.get("tempo", 96))
            time_sig = str(preset.get("time_signature", "4/4"))
            key_sig = str(preset.get("key_signature", "D minor"))
            raw_chords = preset.get("chords")
            chord_items = cast(list[object], raw_chords) if isinstance(raw_chords, list) else []
            chords = (
                [str(c) for c in chord_items]
                if len(chord_items) == 4
                else ["Dm7", "Gm7", "C7", "Fmaj7"]
            )

            score_pair = work.read_score()
            section = ScoreSection(
                id=section_id,
                title="Verse 1",
                kind="verse",
                bars=4,
                chords=chords,
            )
            outline = ScoreOutline(
                tempo=tempo,
                time_signature=time_sig,
                key_signature=key_sig,
                sections=[section],
            )
            work.write_score(
                outline,
                expected_digest=score_pair[1] if score_pair is not None else None,
            )

            bass_inst = InstrumentEntry(
                id="bass_synth",
                name="Synth Bass",
                category="bass",
                role="bass_foundation",
                midi_program=38,
                midi_channel=1,
                soundfont_patch="synth_bass_1",
                min_pitch="C1",
                max_pitch="C4",
                volume=90,
                pan=0.0,
            )
            existing_bass_inst = work.read_instrument("bass_synth")
            work.write_instrument(
                bass_inst,
                expected_digest=existing_bass_inst[1] if existing_bass_inst is not None else None,
            )

            bass_track = generate_bass_groove(
                section,
                instrument_id="bass_synth",
                style=style,
                time_signature=outline.time_signature,
            )
            existing_bass_track = work.read_track(section_id, "bass_synth")
            written_track = work.write_track(
                bass_track,
                expected_digest=existing_bass_track[1] if existing_bass_track is not None else None,
                summary="Created foundation bass rhythm track",
            )

            next_state = MusicStartState(
                stage="layer_melody",
                active_section_id=section_id,
                style_profile=style,
                completed_layers=["bass_synth"],
                recommended_next="violin_solo",
            )
            write_start_state(work, next_state, expected_digest=expected_digest)
            return self._format_output(
                work,
                next_state,
                opened_new=opened_new,
                released_prev=released_prev,
                added_instrument="bass_synth",
                written_track_path=written_track.relative_path,
            )

        if state.stage == "layer_melody":
            outline, section = self._require_section(work, section_id)
            violin_inst = InstrumentEntry(
                id="violin_solo",
                name="Solo Violin",
                category="strings",
                role="lead_melody",
                midi_program=40,
                midi_channel=2,
                soundfont_patch="strings_violin",
                min_pitch="G3",
                max_pitch="E7",
                volume=86,
                pan=-0.25,
            )
            existing_inst = work.read_instrument("violin_solo")
            work.write_instrument(
                violin_inst,
                expected_digest=existing_inst[1] if existing_inst is not None else None,
            )

            bass_pair = work.read_track(section_id, "bass_synth")
            bass_track = bass_pair[0] if bass_pair is not None else None
            violin_track = _build_violin_melody_track(
                section,
                instrument_id="violin_solo",
                time_signature=outline.time_signature,
                swing=default_style_axioms(state.style_profile).default_swing,
                style=state.style_profile,
                bass_track=bass_track,
            )
            existing_track = work.read_track(section_id, "violin_solo")
            written_track = work.write_track(
                violin_track,
                expected_digest=existing_track[1] if existing_track is not None else None,
                summary="Added high-register solo violin melody track",
            )

            completed = list(dict.fromkeys([*state.completed_layers, "violin_solo"]))
            next_state = state.model_copy(
                update={
                    "stage": "layer_harmony",
                    "completed_layers": completed,
                    "recommended_next": "grand_piano",
                }
            )
            write_start_state(work, next_state, expected_digest=expected_digest)
            return self._format_output(
                work,
                next_state,
                opened_new=opened_new,
                released_prev=released_prev,
                added_instrument="violin_solo",
                written_track_path=written_track.relative_path,
            )

        if state.stage == "layer_harmony":
            outline, section = self._require_section(work, section_id)
            piano_inst = InstrumentEntry(
                id="grand_piano",
                name="Grand Piano",
                category="piano",
                role="arpeggio",
                midi_program=0,
                midi_channel=3,
                soundfont_patch="acoustic_grand_piano",
                min_pitch="A0",
                max_pitch="C8",
                volume=82,
                pan=0.25,
            )
            existing_inst = work.read_instrument("grand_piano")
            work.write_instrument(
                piano_inst,
                expected_digest=existing_inst[1] if existing_inst is not None else None,
            )

            piano_track = _build_piano_arpeggio_track(
                section,
                instrument_id="grand_piano",
                time_signature=outline.time_signature,
                swing=default_style_axioms(state.style_profile).default_swing,
            )
            existing_track = work.read_track(section_id, "grand_piano")
            written_track = work.write_track(
                piano_track,
                expected_digest=existing_track[1] if existing_track is not None else None,
                summary="Added mid-register grand piano arpeggio track",
            )

            completed = list(dict.fromkeys([*state.completed_layers, "grand_piano"]))
            next_state = state.model_copy(
                update={
                    "stage": "layer_drums",
                    "completed_layers": completed,
                    "recommended_next": "drum_kit",
                }
            )
            write_start_state(work, next_state, expected_digest=expected_digest)
            return self._format_output(
                work,
                next_state,
                opened_new=opened_new,
                released_prev=released_prev,
                added_instrument="grand_piano",
                written_track_path=written_track.relative_path,
            )

        if state.stage == "layer_drums":
            outline, section = self._require_section(work, section_id)
            effective_style: StyleProfile = (
                detected_style if detected_style is not None else state.style_profile
            )
            use_hiphop = effective_style == "hiphop_groove"
            if use_hiphop and work.record().style_axioms.profile != "hiphop_groove":
                work.update_style_axioms(default_style_axioms("hiphop_groove"))

            drum_inst = InstrumentEntry(
                id="drum_kit",
                name="808 Hip-Hop Drum Kit" if use_hiphop else "Studio Drum Kit",
                category="drums",
                role="rhythm_percussion",
                midi_program=0,
                midi_channel=10,
                soundfont_patch="808_hiphop_kit" if use_hiphop else "standard_drum_kit",
                min_pitch="C1",
                max_pitch="C6",
                volume=90,
                pan=0.0,
            )
            existing_inst = work.read_instrument("drum_kit")
            work.write_instrument(
                drum_inst,
                expected_digest=existing_inst[1] if existing_inst is not None else None,
            )

            bass_pair = work.read_track(section_id, "bass_synth")
            bass_track = bass_pair[0] if bass_pair is not None else None
            if use_hiphop:
                drum_track = generate_hiphop_drums(
                    section,
                    bass_track=bass_track,
                    instrument_id="drum_kit",
                    substyle="boombap",
                    time_signature=outline.time_signature,
                )
            else:
                drum_track = _build_pop_drums_track(
                    section,
                    instrument_id="drum_kit",
                    time_signature=outline.time_signature,
                )

            existing_track = work.read_track(section_id, "drum_kit")
            written_track = work.write_track(
                drum_track,
                expected_digest=existing_track[1] if existing_track is not None else None,
                summary=(
                    "Added 808 hip-hop drum groove locked to bass"
                    if use_hiphop
                    else "Added pop drum kit groove"
                ),
            )

            completed = list(dict.fromkeys([*state.completed_layers, "drum_kit"]))
            next_state = state.model_copy(
                update={
                    "stage": "completed",
                    "style_profile": effective_style,
                    "completed_layers": completed,
                    "recommended_next": "",
                }
            )
            write_start_state(work, next_state, expected_digest=expected_digest)
            return self._format_output(
                work,
                next_state,
                opened_new=opened_new,
                released_prev=released_prev,
                added_instrument="drum_kit",
                written_track_path=written_track.relative_path,
            )

        return self._format_output(
            work,
            state,
            opened_new=opened_new,
            released_prev=released_prev,
        )

    def _run_regional_fast_edit(
        self,
        work: SongWork,
        state: MusicStartState,
        *,
        params: MusicStartParams,
        selection: RegionSelection | None,
        opened_new: bool,
    ) -> dict[str, Any]:
        section_id = (
            selection.section_id
            if selection is not None
            else (params.section_id or state.active_section_id)
        )
        self._require_section(work, section_id)

        target_inst = params.instrument_id
        if target_inst is None and selection is not None and selection.instrument_ids:
            target_inst = selection.instrument_ids[0]
        if target_inst is None:
            if "violin_solo" in state.completed_layers:
                target_inst = "violin_solo"
            elif state.completed_layers:
                target_inst = state.completed_layers[-1]
            else:
                target_inst = "bass_synth"

        track_pair = work.read_track(section_id, target_inst)
        if track_pair is None:
            alias_map = {
                "violin": "violin_solo",
                "piano": "grand_piano",
                "bass": "bass_synth",
                "drums": "drum_kit",
                "drum": "drum_kit",
            }
            alias = alias_map.get(target_inst.casefold())
            if alias is not None:
                alias_pair = work.read_track(section_id, alias)
                if alias_pair is not None:
                    target_inst = alias
                    track_pair = alias_pair
        if track_pair is None:
            raise MusicError(
                f"Track for section {section_id!r} and instrument {target_inst!r} does not exist yet."
            )
        track, track_digest = track_pair

        start_bar = selection.start_bar if selection is not None else (params.start_bar or 1)
        end_bar = (
            selection.end_bar if selection is not None else (params.end_bar or len(track.bars))
        )
        effective_selection = RegionSelection(
            section_id=section_id,
            start_bar=start_bar,
            end_bar=end_bar,
            instrument_ids=[target_inst],
        )

        # Subdivide selected bars into faster 16th-note double-time figurations
        patched_bars = subdivide_bars_faster(
            track,
            start_bar=start_bar,
            end_bar=end_bar,
            factor=params.factor,
        )
        # Ensure 16th-note (0.25 beat) double-time detail when all subdivided notes are still >= 0.5
        if all(
            ev.duration >= 0.5
            for bar_obj in patched_bars
            for ev in bar_obj.parsed_events()
            if ev.midi >= 0
        ):
            temp_track = TrackFile(
                track_id=track.track_id,
                section_id=track.section_id,
                instrument_id=track.instrument_id,
                time_signature=track.time_signature,
                groove_swing=track.groove_swing,
                bars=[
                    *track.bars[: start_bar - 1],
                    *patched_bars,
                    *track.bars[end_bar:],
                ],
            )
            patched_bars = subdivide_bars_faster(
                temp_track,
                start_bar=start_bar,
                end_bar=end_bar,
                factor=2,
            )

        spliced_bars = [
            *track.bars[: start_bar - 1],
            *patched_bars,
            *track.bars[end_bar:],
        ]
        updated_track = TrackFile(
            track_id=track.track_id,
            section_id=track.section_id,
            instrument_id=track.instrument_id,
            time_signature=track.time_signature,
            groove_swing=track.groove_swing,
            bars=spliced_bars,
        )
        written = work.write_track(
            updated_track,
            expected_digest=track_digest,
            summary=f"Subdivided bars {start_bar}..{end_bar} on {target_inst} into faster 16th-note pattern",
        )

        output = self._format_output(
            work,
            state,
            opened_new=opened_new,
            written_track_path=written.relative_path,
        )
        output["edited_region"] = effective_selection.model_dump(mode="json")
        output["patched_bars"] = [b.bar for b in patched_bars]
        output["instrument_id"] = target_inst
        output["track_id"] = updated_track.track_id
        output[REPLY_NOTE_KEY] = {
            "ko": (
                f"`{target_inst}` 트랙의 {start_bar}~{end_bar}마디를 16분음표 더블타임 패턴으로 "
                "빠르게 분할했습니다 (나머지 마디와 악기는 그대로 유지됨)."
            ),
            "en": (
                f"Subdivided bars {start_bar}..{end_bar} on `{target_inst}` into a 16th-note "
                "double-time pattern while keeping all other bars and tracks untouched."
            ),
        }
        return output

    @staticmethod
    def _require_section(work: SongWork, section_id: str) -> tuple[ScoreOutline, ScoreSection]:
        score_pair = work.read_score()
        if score_pair is None:
            raise MusicError("score.yaml does not exist yet; run 'setup_foundation' first.")
        outline = score_pair[0]
        for sec in outline.sections:
            if sec.id == section_id:
                return outline, sec
        if outline.sections:
            return outline, outline.sections[0]
        raise MusicError(f"Section {section_id!r} was not found in score.yaml.")

    @staticmethod
    def _format_output(
        work: SongWork,
        state: MusicStartState,
        *,
        opened_new: bool,
        released_prev: str | None = None,
        added_instrument: str | None = None,
        written_track_path: str | None = None,
    ) -> dict[str, Any]:
        record = work.record()
        score_pair = work.read_score()
        audit_summary: dict[str, Any] | None = None
        sec_chords: list[str] | None = None
        if score_pair is not None and score_pair[0].sections:
            outline = score_pair[0]
            sec = next(
                (s for s in outline.sections if s.id == state.active_section_id),
                outline.sections[0],
            )
            sec_chords = list(sec.chords)
            instruments = {inst.id: inst for inst, _ in work.list_instruments()}
            tracks = [t for t, _ in work.list_tracks(section_id=sec.id)]
            report = audit_section(sec, tracks, instruments, record.style_axioms)
            audit_summary = audit_digest(report.findings, passed=report.passed)

        result: dict[str, Any] = {
            OPEN_SONG_KEY: work.song_id,
            "song_id": work.song_id,
            "title": record.title,
            "stage": state.stage,
            "active_section_id": state.active_section_id,
            "style_profile": state.style_profile,
            "completed_layers": list(state.completed_layers),
            "recommended_next": state.recommended_next,
            "opened_new_song": opened_new,
            REPLY_NOTE_KEY: _stage_reply_note(
                state.stage, style_profile=state.style_profile, chords=sec_chords
            ),
        }
        if added_instrument in ("violin_solo", "grand_piano"):
            result["composition_hint"] = (
                f"Scaffold written to {added_instrument}. To compose a bespoke melody/part "
                "locked to the user's custom rhythm or direction, read bars 1..4 of "
                "bass_synth and call music_track(action='patch_bars', propose=False)."
            )
        if released_prev is not None:
            result["released"] = released_prev
        if added_instrument is not None:
            result["added_instrument"] = added_instrument
        if written_track_path is not None:
            result["path"] = f"songs/{work.song_id}/{written_track_path}"
        if audit_summary is not None:
            result["audit"] = audit_summary
        return result
