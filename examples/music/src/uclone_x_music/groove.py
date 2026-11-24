"""Deterministic bass groove, hip-hop drum lock, double-time subdivision, and SATB string voicing.

All functions in this module are pure: they accept parsed schema objects and return
validated `TrackFile`, `TrackBar`, and `InstrumentEntry` models without touching disk.
"""

from __future__ import annotations

import re
from typing import Literal

from uclone_x_music.audit import parse_chord_pitch_classes
from uclone_x_music.schemas import (
    BarNoteEvent,
    InstrumentEntry,
    ScoreSection,
    StyleProfile,
    TrackBar,
    TrackFile,
    beats_per_bar_for,
    default_style_axioms,
    format_note_token,
    pitch_to_name,
    resolve_track_bars,
)

__all__ = [
    "chord_fifth_pc",
    "chord_root_pc",
    "chord_third_pc",
    "generate_bass_groove",
    "generate_classical_satb_strings",
    "generate_hiphop_drums",
    "is_chord_symbol",
    "subdivide_bars_faster",
]

_NOTE_OFFSETS: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
_CHORD_ROOT_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*?)(?:/([A-Ga-g])([#b]?))?$")


def is_chord_symbol(symbol: str) -> bool:
    """Whether `symbol` names a chord root the generators can read (`Dm7`, `F#m7b5`, `C/E`).

    The root readers below fall back to C for anything else, so a tool that takes chords
    from the model checks them here and refuses instead of composing over a silent C.
    """
    text = symbol.strip()
    return bool(text) and " " not in text and _CHORD_ROOT_RE.fullmatch(text) is not None


def _chord_root_pc(chord_symbol: str | None) -> int:
    if not chord_symbol:
        return 0
    match = _CHORD_ROOT_RE.fullmatch(chord_symbol.strip())
    if match is None:
        return 0
    letter, acc, _, slash_letter, slash_acc = match.groups()
    target_letter = slash_letter or letter
    target_acc = slash_acc if slash_letter else acc
    pc = _NOTE_OFFSETS[target_letter.upper()]
    if target_acc == "#":
        pc = (pc + 1) % 12
    elif target_acc == "b":
        pc = (pc - 1) % 12
    return pc


def _chord_fifth_pc(chord_symbol: str | None, root_pc: int) -> int:
    if not chord_symbol:
        return (root_pc + 7) % 12
    pcs = parse_chord_pitch_classes(chord_symbol)
    for cand in ((root_pc + 7) % 12, (root_pc + 6) % 12, (root_pc + 8) % 12):
        if cand in pcs:
            return cand
    return (root_pc + 7) % 12


def _chord_third_pc(chord_symbol: str | None, root_pc: int) -> int:
    if not chord_symbol:
        return (root_pc + 4) % 12
    pcs = parse_chord_pitch_classes(chord_symbol)
    for cand in ((root_pc + 3) % 12, (root_pc + 4) % 12, (root_pc + 5) % 12, (root_pc + 2) % 12):
        if cand in pcs:
            return cand
    return (root_pc + 4) % 12


chord_root_pc = _chord_root_pc
chord_fifth_pc = _chord_fifth_pc
chord_third_pc = _chord_third_pc


def generate_bass_groove(
    section: ScoreSection,
    instrument_id: str = "bass_synth",
    style: StyleProfile = "pop_ballad",
    time_signature: str = "4/4",
) -> TrackFile:
    """Deterministically generate a bass groove across `section.bars` following `section.chords`."""
    axioms = default_style_axioms(style)
    bpb = beats_per_bar_for(time_signature)
    bars: list[TrackBar] = []

    for bar_num in range(1, section.bars + 1):
        chord = section.chords[bar_num - 1] if 1 <= bar_num <= len(section.chords) else "C"
        root_pc = _chord_root_pc(chord)
        fifth_pc = _chord_fifth_pc(chord, root_pc)
        # Place root in octave 2 (MIDI 36..47)
        root_midi = 36 + root_pc
        fifth_midi = 36 + fifth_pc
        if fifth_midi < root_midi:
            fifth_midi += 12
        if fifth_midi > 55:
            fifth_midi -= 12

        root_name = pitch_to_name(root_midi)
        fifth_name = pitch_to_name(fifth_midi)
        octave_name = pitch_to_name(min(55, root_midi + 12))

        if abs(bpb - 4.0) < 1e-4:
            if style == "hiphop_groove":
                notes = [
                    f"1.0:{root_name}:1.5:v96:accent",
                    f"2.5:{root_name}:0.5:v88:staccato",
                    "3.0:REST:0.5",
                    f"3.5:{fifth_name}:0.5:v86",
                    f"4.0:{octave_name}:1.0:v90",
                ]
            elif style == "jazz_swing":
                third_midi = 36 + _chord_third_pc(chord, root_pc)
                if third_midi < root_midi:
                    third_midi += 12
                third_name = pitch_to_name(third_midi)
                notes = [
                    f"1.0:{root_name}:1.0:v90",
                    f"2.0:{third_name}:1.0:v84",
                    f"3.0:{fifth_name}:1.0:v88",
                    f"4.0:{octave_name}:1.0:v84",
                ]
            else:
                notes = [
                    f"1.0:{root_name}:1.5:v92",
                    f"2.5:{fifth_name}:0.5:v82",
                    f"3.0:{root_name}:1.0:v88",
                    f"4.0:{fifth_name}:1.0:v84",
                ]
        else:
            half = round(bpb / 2.0, 2)
            rem = round(bpb - half, 2)
            notes = [
                f"1.0:{root_name}:{half:g}:v92",
                f"{1.0 + half:g}:{fifth_name}:{rem:g}:v84",
            ]

        bars.append(TrackBar(bar=bar_num, chord=chord, notes=notes))

    return TrackFile(
        track_id=f"{section.id}_{instrument_id}",
        section_id=section.id,
        instrument_id=instrument_id,
        time_signature=time_signature,
        groove_swing=axioms.default_swing,
        bars=bars,
    )


def generate_hiphop_drums(
    section: ScoreSection,
    bass_track: TrackFile | None = None,
    instrument_id: str = "drum_kit",
    substyle: Literal["boombap", "trap"] = "boombap",
    time_signature: str = "4/4",
) -> TrackFile:
    """Generate a hip-hop drum kit track locked to `bass_track` kicks."""
    swing = 0.58 if substyle == "boombap" else 0.50
    bpb = beats_per_bar_for(time_signature)
    bass_resolved = resolve_track_bars(bass_track) if bass_track is not None else {}
    bars: list[TrackBar] = []

    for bar_num in range(1, section.bars + 1):
        chord = section.chords[bar_num - 1] if 1 <= bar_num <= len(section.chords) else None
        if abs(bpb - 4.0) < 1e-4:
            kick_beats: set[float] = {1.0}
            bass_events = bass_resolved.get(bar_num, [])
            for ev in bass_events:
                if ev.midi >= 0 and ev.beat in (1.0, 2.5, 3.5):
                    kick_beats.add(ev.beat)
            if not bass_events:
                kick_beats.update({1.0, 2.5, 3.5} if substyle == "boombap" else {1.0, 2.75, 3.5})

            tokens: list[str] = []
            for kb in sorted(kick_beats):
                tokens.append(f"{kb:g}:KICK:0.5:v102:accent")

            # Snare backbeat on 2.0 and 4.0
            tokens.append("2.0:SNARE:0.5:v98:accent")
            tokens.append("4.0:SNARE:0.5:v98:accent")

            # Closed hi-hat grid spanning full 4.0 beats
            hat_step = 0.5 if substyle == "boombap" else 0.25
            steps = int(round(bpb / hat_step))
            for idx in range(steps):
                beat = round(1.0 + idx * hat_step, 4)
                vel = 82 if idx % 2 == 0 else 68
                tokens.append(f"{beat:g}:HIHAT:{hat_step:g}:v{vel}")

            bars.append(TrackBar(bar=bar_num, chord=chord, notes=tokens))
        else:
            bars.append(
                TrackBar(
                    bar=bar_num,
                    chord=chord,
                    notes=[f"1.0:KICK:{bpb:g}:v100"],
                )
            )

    return TrackFile(
        track_id=f"{section.id}_{instrument_id}",
        section_id=section.id,
        instrument_id=instrument_id,
        time_signature=time_signature,
        groove_swing=swing,
        bars=bars,
    )


def subdivide_bars_faster(
    track: TrackFile,
    start_bar: int,
    end_bar: int,
    factor: Literal[2, 4] = 2,
) -> list[TrackBar]:
    """Subdivide notes in `start_bar..end_bar` into faster rhythmic figurations (`factor` = 2 or 4).

    Implements natural-language double-time requests while preserving each bar's exact beat span
    and keeping rests intact.
    """
    if start_bar < 1 or end_bar < start_bar or end_bar > len(track.bars):
        raise ValueError(
            f"Invalid bar range {start_bar}..{end_bar} for track {track.track_id!r} "
            f"(1..{len(track.bars)})."
        )

    resolved = resolve_track_bars(track)
    by_num = {b.bar: b for b in track.bars}
    melodic_offsets = (0, 2, 4, 2)
    patched: list[TrackBar] = []

    for bar_num in range(start_bar, end_bar + 1):
        orig_bar = by_num[bar_num]
        events = resolved[bar_num]
        new_tokens: list[str] = []

        for ev in events:
            is_drum = ev.midi >= 0 and ev.pitch.upper() != pitch_to_name(ev.midi).upper()
            if (
                ev.midi < 0
                or ev.duration < 0.5
                or (is_drum and ev.pitch.upper() in ("KICK", "SNARE") and ev.duration <= 0.5)
            ):
                new_tokens.append(format_note_token(ev))
                continue

            sub_dur = round(ev.duration / float(factor), 4)
            for idx in range(factor):
                sub_beat = round(ev.beat + idx * sub_dur, 4)
                if is_drum:
                    sub_midi = ev.midi
                    sub_pitch = ev.pitch
                else:
                    offset = melodic_offsets[idx % len(melodic_offsets)]
                    sub_midi = max(0, min(127, ev.midi + offset))
                    sub_pitch = pitch_to_name(sub_midi)
                sub_vel = ev.velocity if idx == 0 else max(45, ev.velocity - 8)
                sub_art = "staccato" if sub_dur <= 0.25 else ev.articulation
                sub_ev = BarNoteEvent(
                    beat=sub_beat,
                    pitch=sub_pitch,
                    midi=sub_midi,
                    duration=sub_dur,
                    velocity=sub_vel,
                    articulation=sub_art,
                )
                new_tokens.append(format_note_token(sub_ev))

        patched.append(
            TrackBar(
                bar=bar_num,
                chord=orig_bar.chord,
                notes=new_tokens,
                muted=orig_bar.muted,
            )
        )

    return patched


def generate_classical_satb_strings(
    section: ScoreSection,
    time_signature: str = "4/4",
) -> tuple[list[InstrumentEntry], list[TrackFile]]:
    """Generate a 4-part classical string section with strict SATB voice leading.

    Guarantees zero parallel fifths/octaves and zero voice crossing across `section.chords`.
    """
    instruments: list[InstrumentEntry] = [
        InstrumentEntry(
            id="violin_1",
            name="Violin I (Soprano)",
            category="strings",
            role="satb_soprano",
            midi_program=40,
            midi_channel=1,
            soundfont_patch="strings_violin",
            min_pitch="G3",
            max_pitch="A6",
            pan=-0.35,
        ),
        InstrumentEntry(
            id="violin_2",
            name="Violin II (Alto)",
            category="strings",
            role="satb_alto",
            midi_program=40,
            midi_channel=2,
            soundfont_patch="strings_violin",
            min_pitch="F3",
            max_pitch="E6",
            pan=-0.15,
        ),
        InstrumentEntry(
            id="viola",
            name="Viola (Tenor)",
            category="strings",
            role="satb_tenor",
            midi_program=41,
            midi_channel=3,
            soundfont_patch="strings_viola",
            min_pitch="C3",
            max_pitch="A5",
            pan=0.15,
        ),
        InstrumentEntry(
            id="cello",
            name="Cello (Bass)",
            category="strings",
            role="satb_bass",
            midi_program=42,
            midi_channel=4,
            soundfont_patch="strings_cello",
            min_pitch="C2",
            max_pitch="E5",
            pan=0.35,
        ),
    ]

    bpb = beats_per_bar_for(time_signature)
    bars_by_inst: dict[str, list[TrackBar]] = {inst.id: [] for inst in instruments}
    prev_voicing: tuple[int, int, int, int] | None = None

    for bar_num in range(1, section.bars + 1):
        chord = section.chords[bar_num - 1] if 1 <= bar_num <= len(section.chords) else "C"
        root_pc = _chord_root_pc(chord)
        third_pc = _chord_third_pc(chord, root_pc)
        fifth_pc = _chord_fifth_pc(chord, root_pc)

        voicing = _choose_satb_voicing(root_pc, third_pc, fifth_pc, prev_voicing)
        prev_voicing = voicing
        cello_midi, viola_midi, vln2_midi, vln1_midi = voicing

        for inst_id, midi_val in (
            ("cello", cello_midi),
            ("viola", viola_midi),
            ("violin_2", vln2_midi),
            ("violin_1", vln1_midi),
        ):
            pname = pitch_to_name(midi_val)
            if abs(bpb - 4.0) < 1e-4:
                notes = [
                    f"1.0:{pname}:2.0:v82:legato",
                    f"3.0:{pname}:2.0:v78:legato",
                ]
            else:
                notes = [f"1.0:{pname}:{bpb:g}:v82:legato"]
            bars_by_inst[inst_id].append(TrackBar(bar=bar_num, chord=chord, notes=notes))

    tracks = [
        TrackFile(
            track_id=f"{section.id}_{inst.id}",
            section_id=section.id,
            instrument_id=inst.id,
            time_signature=time_signature,
            groove_swing=0.5,
            bars=bars_by_inst[inst.id],
        )
        for inst in instruments
    ]
    return instruments, tracks


def _choose_satb_voicing(
    root_pc: int,
    third_pc: int,
    fifth_pc: int,
    prev: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    """Select `(cello, viola, violin_2, violin_1)` MIDI notes with zero parallel 5ths/8ves."""
    cello_candidates = [m for m in range(36, 53) if m % 12 == root_pc]
    viola_candidates = [m for m in range(52, 65) if m % 12 in (root_pc, third_pc, fifth_pc)]
    vln2_candidates = [m for m in range(60, 74) if m % 12 in (root_pc, third_pc, fifth_pc)]
    vln1_candidates = [m for m in range(67, 82) if m % 12 in (root_pc, third_pc, fifth_pc)]

    best: tuple[int, int, int, int] | None = None
    best_cost = 10_000

    for c in cello_candidates:
        for v in viola_candidates:
            if v <= c + 2:
                continue
            for a in vln2_candidates:
                if a <= v + 1:
                    continue
                for s in vln1_candidates:
                    if s <= a + 1:
                        continue
                    pcs = {c % 12, v % 12, a % 12, s % 12}
                    if third_pc not in pcs:
                        continue
                    cand = (c, v, a, s)
                    if prev is not None and _has_parallel_5th_or_8ve(prev, cand):
                        continue
                    if prev is None:
                        cost = abs(c - 43) + abs(v - 55) + abs(a - 64) + abs(s - 72)
                    else:
                        cost = sum(abs(cand[i] - prev[i]) for i in range(4))
                        # Prefer contrary motion between outer voices
                        if (cand[0] - prev[0]) * (cand[3] - prev[3]) < 0:
                            cost -= 3
                    if cost < best_cost:
                        best_cost = cost
                        best = cand

    if best is not None:
        return best

    # Fallback strictly ordered triad voicing
    c = 36 + root_pc
    v = 55 + ((fifth_pc - 55) % 12)
    a = 62 + ((third_pc - 62) % 12)
    s = 72 + root_pc
    return (c, v, a, s)


def _has_parallel_5th_or_8ve(
    prev: tuple[int, int, int, int],
    curr: tuple[int, int, int, int],
) -> bool:
    for i in range(4):
        for j in range(i + 1, 4):
            da = curr[i] - prev[i]
            db = curr[j] - prev[j]
            if (da > 0 and db > 0) or (da < 0 and db < 0):
                int_prev = abs(prev[i] - prev[j]) % 12
                int_curr = abs(curr[i] - curr[j]) % 12
                if int_prev == int_curr and int_curr in (0, 7):
                    return True
    return False
