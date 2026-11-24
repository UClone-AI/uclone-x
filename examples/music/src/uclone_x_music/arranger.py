"""Context-aware multi-track bar synthesis and structural section arrangement operations.

Provides:
- `ArrangementIntent`: the structured intent the model states as a tool argument
  (`gentle_intro`, `climax_ornate`, `hiphop_groove`, `outro_cadence`, `continuation`).
  It is never guessed from the person's words.
- `synthesize_bars_for_intent`: generates chord-coherent, role-aware `TrackBar` sequences
  for `bass`, `piano`, `strings`, `drums`, or other instruments without blindly subdividing
  into 16th-note staccato notes when a soft/sustained intro or outro was requested.
- `insert_section_bars`, `append_section_bars`, `delete_section_bars`, `duplicate_section_bars`:
  atomic structural bar transformations across `score.yaml` and all tracks in a section.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from uclone_x_music.audit import parse_chord_pitch_classes
from uclone_x_music.groove import (
    chord_fifth_pc,
    chord_root_pc,
    chord_third_pc,
    subdivide_bars_faster,
)
from uclone_x_music.library import MusicError
from uclone_x_music.schemas import (
    InstrumentCategory,
    InstrumentRole,
    ScoreOutline,
    ScoreSection,
    TempoChange,
    TrackBar,
    TrackFile,
    beats_per_bar_for,
    normalize_bar_notes,
    pitch_to_name,
)
from uclone_x_music.work import SongWork, track_id_for

__all__ = [
    "ArrangementIntent",
    "StructureAction",
    "append_section_bars",
    "delete_section_bars",
    "duplicate_section_bars",
    "insert_section_bars",
    "synthesize_bars_for_intent",
]

ArrangementIntent = Literal[
    "gentle_intro",
    "climax_ornate",
    "hiphop_groove",
    "outro_cadence",
    "continuation",
]

StructureAction = Literal[
    "insert_bars",
    "append_bars",
    "delete_bars",
    "duplicate_bars",
]


def _chord_tones_for_octave(chord: str | None, base_midi: int) -> tuple[str, str, str, str]:
    """Return `(root, third, fifth, octave_root)` pitch strings near `base_midi`."""
    sym = chord or "Dm7"
    pcs = parse_chord_pitch_classes(sym)
    root_pc = chord_root_pc(sym) if pcs else 2
    third_pc = chord_third_pc(sym, root_pc)
    fifth_pc = chord_fifth_pc(sym, root_pc)

    root_midi = base_midi + root_pc
    if base_midi <= 40 and root_midi > 47:
        root_midi -= 12
    third_midi = base_midi + third_pc
    while third_midi < root_midi:
        third_midi += 12
    fifth_midi = base_midi + fifth_pc
    while fifth_midi < third_midi:
        fifth_midi += 12
    if base_midi <= 40 and fifth_midi > 55:
        fifth_midi -= 12
    oct_midi = min(108, root_midi + 12)
    return (
        pitch_to_name(max(24, min(108, root_midi))),
        pitch_to_name(max(24, min(108, third_midi))),
        pitch_to_name(max(24, min(108, fifth_midi))),
        pitch_to_name(max(24, min(108, oct_midi))),
    )


def _beats_token(beats: float) -> str:
    return f"{beats:g}" if beats % 1 != 0 else f"{int(beats)}.0"


def _synthesize_single_bar(
    bar_num: int,
    chord: str,
    *,
    category: InstrumentCategory,
    role: InstrumentRole,
    intent: ArrangementIntent,
    time_signature: str,
    is_final_bar: bool = False,
) -> TrackBar:
    bpb = beats_per_bar_for(time_signature)
    if abs(bpb - 4.0) > 1e-3:
        # Non-4/4 fallback: full-bar root or rest
        root, third, _, _ = _chord_tones_for_octave(chord, 36 if category == "bass" else 60)
        pitch = "HIHAT" if category == "drums" else (root if category == "bass" else third)
        vel = 64 if intent in ("gentle_intro", "outro_cadence") else 84
        notes = normalize_bar_notes([f"1.0:{pitch}:{bpb:g}:v{vel}:tenuto"], time_signature)
        return TrackBar(bar=bar_num, chord=chord, notes=notes)

    if category == "drums" or role == "rhythm_percussion":
        if intent == "gentle_intro":
            raw_notes = [
                "1.0:HIHAT:2.0:v54",
                "3.0:HIHAT:2.0:v50",
            ]
        elif intent == "outro_cadence":
            raw_notes = (
                ["1.0:KICK:2.0:v72", "3.0:REST:2.0"]
                if is_final_bar
                else ["1.0:KICK:1.0:v78", "2.0:SNARE:1.0:v72", "3.0:HIHAT:1.0:v62", "4.0:REST:1.0"]
            )
        elif intent in ("hiphop_groove", "climax_ornate"):
            raw_notes = [
                "1.0:KICK:0.5:v102:accent",
                "2.5:KICK:0.5:v100:accent",
                "2.0:SNARE:0.5:v98:accent",
                "4.0:SNARE:0.5:v98:accent",
                "1.0:HIHAT:0.5:v82",
                "1.5:HIHAT:0.5:v68",
                "2.0:HIHAT:0.5:v82",
                "2.5:HIHAT:0.5:v68",
                "3.0:HIHAT:0.5:v82",
                "3.5:HIHAT:0.5:v68",
                "4.0:HIHAT:0.5:v82",
                "4.5:HIHAT:0.5:v68",
            ]
        else:
            raw_notes = [
                "1.0:KICK:1.0:v94:accent",
                "2.0:SNARE:1.0:v90",
                "3.0:KICK:1.0:v92",
                "4.0:SNARE:1.0:v92:accent",
            ]
        return TrackBar(
            bar=bar_num,
            chord=chord,
            notes=normalize_bar_notes(raw_notes, time_signature),
        )

    if category == "bass" or role in ("bass_foundation", "satb_bass"):
        root, third, fifth, oct_root = _chord_tones_for_octave(chord, 36)
        if intent == "gentle_intro":
            raw_notes = [
                f"1.0:{root}:2.0:v62:legato",
                f"3.0:{fifth}:2.0:v58:tenuto",
            ]
        elif intent == "outro_cadence":
            raw_notes = (
                [f"1.0:{root}:4.0:v64:tenuto"]
                if is_final_bar
                else [f"1.0:{root}:2.0:v72:legato", f"3.0:{fifth}:2.0:v66:tenuto"]
            )
        elif intent == "hiphop_groove":
            raw_notes = [
                f"1.0:{root}:1.5:v96:accent",
                f"2.5:{fifth}:0.5:v84:staccato",
                f"3.0:{oct_root}:1.0:v90",
                f"4.0:{fifth}:1.0:v86",
            ]
        elif intent == "climax_ornate":
            raw_notes = [
                f"1.0:{root}:1.0:v96:accent",
                f"2.0:{third}:0.5:v86",
                f"2.5:{fifth}:0.5:v88:staccato",
                f"3.0:{oct_root}:1.0:v94:tenuto",
                f"4.0:{fifth}:1.0:v90",
            ]
        else:
            raw_notes = [
                f"1.0:{root}:1.5:v92",
                f"2.5:{fifth}:0.5:v80",
                f"3.0:{oct_root}:1.0:v88",
                f"4.0:{fifth}:1.0:v84",
            ]
        return TrackBar(
            bar=bar_num,
            chord=chord,
            notes=normalize_bar_notes(raw_notes, time_signature),
        )

    if category in ("piano", "guitar", "synth") or role in (
        "harmonic_comping",
        "arpeggio",
        "pad_texture",
    ):
        root, third, fifth, oct_root = _chord_tones_for_octave(chord, 60)
        if intent == "gentle_intro":
            raw_notes = [
                f"1.0:{third}:2.0:v64:legato",
                f"3.0:{fifth}:2.0:v60:tenuto",
            ]
        elif intent == "outro_cadence":
            raw_notes = (
                [f"1.0:{third}:4.0:v60:tenuto"]
                if is_final_bar
                else [f"1.0:{fifth}:2.0:v68:legato", f"3.0:{third}:2.0:v62:tenuto"]
            )
        elif intent == "climax_ornate":
            raw_notes = [
                f"1.0:{root}:0.5:v84:legato",
                f"1.5:{third}:0.5:v80:legato",
                f"2.0:{fifth}:0.5:v86:accent",
                f"2.5:{oct_root}:0.5:v88:tenuto",
                f"3.0:{fifth}:1.0:v84:legato",
                f"4.0:{third}:1.0:v80:tenuto",
            ]
        else:
            raw_notes = [
                f"1.0:{third}:1.0:v76",
                f"2.0:{fifth}:0.5:v72",
                f"2.5:{oct_root}:1.5:v80:legato",
                f"4.0:{fifth}:1.0:v74",
            ]
        return TrackBar(
            bar=bar_num,
            chord=chord,
            notes=normalize_bar_notes(raw_notes, time_signature),
        )

    # Lead melody / strings / brass / woodwinds
    root, third, fifth, oct_root = _chord_tones_for_octave(chord, 60)
    if intent == "gentle_intro":
        raw_notes = [
            f"1.0:{fifth}:2.5:v66:legato",
            f"3.5:{third}:1.5:v62:tenuto",
        ]
    elif intent == "outro_cadence":
        raw_notes = (
            [f"1.0:{root}:4.0:v68:vibrato"]
            if is_final_bar
            else [
                f"1.0:{fifth}:1.5:v78:legato",
                f"2.5:{third}:0.5:v72",
                f"3.0:{root}:2.0:v74:tenuto",
            ]
        )
    elif intent == "climax_ornate":
        raw_notes = [
            f"1.0:{third}:0.75:v90:legato",
            f"1.75:{fifth}:0.75:v86:legato",
            f"2.5:{oct_root}:0.5:v94:accent",
            f"3.0:{oct_root}:1.0:v96:vibrato",
            f"4.0:{fifth}:0.5:v88:staccato",
            f"4.5:{third}:0.5:v84:legato",
        ]
    else:
        phrase_offset = 2 if (role == "counterpoint" or category == "woodwinds") else 0
        phrase_step = (bar_num - 1 + phrase_offset) % 4
        if phrase_step == 0:
            raw_notes = [
                f"1.0:{root}:1.0:v86:legato",
                f"2.0:{third}:0.5:v82:legato",
                f"2.5:{fifth}:1.5:v90:vibrato",
                f"4.0:{third}:1.0:v84:tenuto",
            ]
        elif phrase_step == 1:
            raw_notes = [
                f"1.0:{fifth}:1.5:v88:legato",
                f"2.5:{oct_root}:0.5:v90:legato",
                f"3.0:{fifth}:0.5:v84:legato",
                f"3.5:{third}:0.5:v82:legato",
                f"4.0:{root}:1.0:v85:tenuto",
            ]
        elif phrase_step == 2:
            raw_notes = [
                f"1.0:{third}:0.5:v84:legato",
                f"1.5:{fifth}:0.5:v88:legato",
                f"2.0:{oct_root}:1.5:v94:vibrato",
                f"3.5:{fifth}:0.5:v86:legato",
                f"4.0:{third}:1.0:v84:legato",
            ]
        else:
            raw_notes = [
                f"1.0:{fifth}:1.0:v88:legato",
                f"2.0:{third}:1.0:v84:legato",
                f"3.0:{root}:2.0:v90:vibrato",
            ]
    return TrackBar(
        bar=bar_num,
        chord=chord,
        notes=normalize_bar_notes(raw_notes, time_signature),
    )


def synthesize_bars_for_intent(
    track: TrackFile,
    start_bar: int,
    end_bar: int,
    *,
    chords: Sequence[str] | None = None,
    category: InstrumentCategory = "strings",
    role: InstrumentRole = "lead_melody",
    intent: ArrangementIntent = "continuation",
) -> list[TrackBar]:
    """Generate `TrackBar` objects for `start_bar..end_bar` matching `intent` and `chords`."""
    if (
        intent == "climax_ornate"
        and start_bar >= 1
        and end_bar <= len(track.bars)
        and category != "drums"
    ):
        patched = subdivide_bars_faster(track, start_bar=start_bar, end_bar=end_bar, factor=2)
        if all(
            ev.duration >= 0.5
            for bar_obj in patched
            for ev in bar_obj.parsed_events()
            if ev.midi >= 0
        ):
            temp = TrackFile(
                track_id=track.track_id,
                section_id=track.section_id,
                instrument_id=track.instrument_id,
                time_signature=track.time_signature,
                groove_swing=track.groove_swing,
                bars=[*track.bars[: start_bar - 1], *patched, *track.bars[end_bar:]],
            )
            patched = subdivide_bars_faster(temp, start_bar=start_bar, end_bar=end_bar, factor=2)
        return patched

    fallback_chords = (
        list(chords)
        if chords
        else [b.chord for b in track.bars if b.chord] or ["Dm7", "Gm7", "A7", "Dm7"]
    )
    result: list[TrackBar] = []
    for bar_num in range(start_bar, end_bar + 1):
        chord = fallback_chords[(bar_num - 1) % len(fallback_chords)]
        result.append(
            _synthesize_single_bar(
                bar_num,
                chord,
                category=category,
                role=role,
                intent=intent,
                time_signature=track.time_signature,
                is_final_bar=(bar_num == end_bar),
            )
        )
    return result


def _resolve_section(outline: ScoreOutline, section_id: str | None) -> ScoreSection:
    if not outline.sections:
        raise MusicError("score.yaml has no sections yet.")
    if not section_id or not section_id.strip():
        return outline.sections[0]
    sec = next((s for s in outline.sections if s.id == section_id.strip()), None)
    if sec is None:
        raise MusicError(f"Section {section_id!r} was not found in score.yaml.")
    return sec


def insert_section_bars(
    work: SongWork,
    section_id: str | None = None,
    *,
    at_bar: int = 1,
    count: int = 2,
    chords: Sequence[str] | None = None,
    intent: ArrangementIntent = "gentle_intro",
    sketch: bool = True,
) -> ScoreSection:
    """Insert `count` bars at `at_bar` across `score.yaml` and all tracks in `section_id`.

    Existing bars `>= at_bar` are shifted right by `+count` without overwriting them. The new
    bars take `chords` when given, otherwise the section's own progression in order (never a
    fixed progression from another key). Each track's new bars are a full-bar rest unless
    `sketch` asks for a generic draft in the stated `intent` and the section has chords to
    draft over.
    """
    if count < 1 or count > 32:
        raise MusicError("insert_section_bars 'count' must be between 1 and 32.")
    score_pair = work.read_score()
    if score_pair is None:
        raise MusicError("score.yaml does not exist yet.")
    outline, score_digest = score_pair
    sec = _resolve_section(outline, section_id)
    insert_pos = max(1, min(sec.bars + 1, at_bar))

    given = [c for c in (chords or []) if c.strip()]
    source = given or list(sec.chords)
    inserted_chords: list[str | None] = [
        source[i % len(source)] if source else None for i in range(count)
    ]
    if sec.chords or given:
        filler = list(sec.chords) or given
        existing_chords = [
            sec.chords[i] if i < len(sec.chords) else filler[i % len(filler)]
            for i in range(sec.bars)
        ]
        new_chords = [
            *existing_chords[: insert_pos - 1],
            *[c or filler[0] for c in inserted_chords],
            *existing_chords[insert_pos - 1 :],
        ]
    else:
        new_chords = []
    # A sketch follows chords; a section without any gets rests, as it did before chords
    # stopped being invented for it.
    drafts = sketch and bool(new_chords)
    shifted_tempo_map = [
        TempoChange(
            bar=(tc.bar + count if tc.bar >= insert_pos else tc.bar),
            bpm=tc.bpm,
            curve=tc.curve,
        )
        for tc in sec.tempo_map
    ]
    new_sec = ScoreSection(
        id=sec.id,
        title=sec.title,
        kind=sec.kind,
        bars=sec.bars + count,
        chords=new_chords,
        tempo_map=shifted_tempo_map,
    )
    updated_outline = outline.model_copy(
        update={"sections": [new_sec if s.id == sec.id else s for s in outline.sections]}
    )
    work.write_score(updated_outline, expected_digest=score_digest)

    inst_map = {inst.id: inst for inst, _ in work.list_instruments()}
    for track, track_digest in work.list_tracks():
        if track.section_id != sec.id:
            continue
        inst = inst_map.get(track.instrument_id)
        category: InstrumentCategory = inst.category if inst is not None else "strings"
        role: InstrumentRole = inst.role if inst is not None else "lead_melody"

        before_bars = [
            b.model_copy(update={"bar": idx})
            for idx, b in enumerate(track.bars[: insert_pos - 1], start=1)
        ]
        rest = f"1.0:REST:{_beats_token(beats_per_bar_for(outline.time_signature))}"
        synth_bars = [
            (
                _synthesize_single_bar(
                    insert_pos + i,
                    new_chords[insert_pos - 1 + i],
                    category=category,
                    role=role,
                    intent=intent,
                    time_signature=outline.time_signature,
                    is_final_bar=(i == count - 1),
                )
                if drafts
                else TrackBar(
                    bar=insert_pos + i,
                    chord=new_chords[insert_pos - 1 + i] if new_chords else None,
                    notes=[rest],
                )
            )
            for i in range(count)
        ]
        after_bars = [
            b.model_copy(
                update={
                    "bar": insert_pos + count + offset,
                    "repeat_bar": (
                        b.repeat_bar + count
                        if b.repeat_bar is not None and b.repeat_bar >= insert_pos
                        else b.repeat_bar
                    ),
                }
            )
            for offset, b in enumerate(track.bars[insert_pos - 1 :])
        ]
        all_bars = [*before_bars, *synth_bars, *after_bars]
        updated_track = TrackFile(
            track_id=track_id_for(sec.id, track.instrument_id),
            section_id=sec.id,
            instrument_id=track.instrument_id,
            time_signature=outline.time_signature,
            groove_swing=track.groove_swing,
            bars=all_bars,
        )
        work.write_track(
            updated_track,
            expected_digest=track_digest,
            summary=f"Inserted {count} bars at bar {insert_pos} in {track.track_id}",
        )

    return new_sec


def append_section_bars(
    work: SongWork,
    section_id: str | None = None,
    *,
    count: int = 4,
    chords: Sequence[str] | None = None,
    intent: ArrangementIntent = "continuation",
    sketch: bool = True,
) -> ScoreSection:
    """Append `count` bars at the end of `section_id` across `score.yaml` and all tracks."""
    score_pair = work.read_score()
    if score_pair is None:
        raise MusicError("score.yaml does not exist yet.")
    outline, _ = score_pair
    sec = _resolve_section(outline, section_id)
    return insert_section_bars(
        work,
        sec.id,
        at_bar=sec.bars + 1,
        count=count,
        chords=chords,
        intent=intent,
        sketch=sketch,
    )


def delete_section_bars(
    work: SongWork,
    section_id: str | None = None,
    *,
    start_bar: int,
    end_bar: int,
) -> ScoreSection:
    """Delete `start_bar..end_bar` from `section_id` across `score.yaml` and all tracks, renumbering 1..N."""
    score_pair = work.read_score()
    if score_pair is None:
        raise MusicError("score.yaml does not exist yet.")
    outline, score_digest = score_pair
    sec = _resolve_section(outline, section_id)
    if start_bar < 1 or end_bar < start_bar or end_bar > sec.bars:
        raise MusicError(
            f"Delete range {start_bar}..{end_bar} is outside section {sec.id!r} (1..{sec.bars})."
        )
    remove_count = end_bar - start_bar + 1
    if sec.bars - remove_count < 1:
        raise MusicError(
            f"Cannot delete all bars from section {sec.id!r}; at least 1 bar must remain."
        )

    # Chords are all-or-none per section; a chordless section stays chordless.
    existing_chords = list(sec.chords)
    new_chords = [
        *existing_chords[: start_bar - 1],
        *existing_chords[end_bar:],
    ]
    new_bars_count = sec.bars - remove_count
    new_tempo_map = [
        TempoChange(
            bar=(tc.bar - remove_count if tc.bar > end_bar else tc.bar),
            bpm=tc.bpm,
            curve=tc.curve,
        )
        for tc in sec.tempo_map
        if not (start_bar <= tc.bar <= end_bar)
        and (tc.bar - (remove_count if tc.bar > end_bar else 0)) <= new_bars_count
    ]
    new_sec = ScoreSection(
        id=sec.id,
        title=sec.title,
        kind=sec.kind,
        bars=new_bars_count,
        chords=new_chords,
        tempo_map=new_tempo_map,
    )
    updated_outline = outline.model_copy(
        update={"sections": [new_sec if s.id == sec.id else s for s in outline.sections]}
    )
    work.write_score(updated_outline, expected_digest=score_digest)

    for track, track_digest in work.list_tracks():
        if track.section_id != sec.id:
            continue
        kept_raw = [b for b in track.bars if not (start_bar <= b.bar <= end_bar)]
        renumbered: list[TrackBar] = []
        for idx, b in enumerate(kept_raw, start=1):
            renumbered.append(
                b.model_copy(
                    update={
                        "bar": idx,
                        "repeat_bar": None if b.repeat_bar is not None else None,
                    }
                )
            )
        updated_track = TrackFile(
            track_id=track.track_id,
            section_id=sec.id,
            instrument_id=track.instrument_id,
            time_signature=outline.time_signature,
            groove_swing=track.groove_swing,
            bars=renumbered,
        )
        work.write_track(
            updated_track,
            expected_digest=track_digest,
            summary=f"Deleted bars {start_bar}..{end_bar} from {track.track_id}",
        )

    return new_sec


def duplicate_section_bars(
    work: SongWork,
    section_id: str | None = None,
    *,
    start_bar: int,
    end_bar: int,
    insert_at: int | None = None,
) -> ScoreSection:
    """Duplicate `start_bar..end_bar` across `score.yaml` and all tracks in `section_id`."""
    score_pair = work.read_score()
    if score_pair is None:
        raise MusicError("score.yaml does not exist yet.")
    outline, score_digest = score_pair
    sec = _resolve_section(outline, section_id)
    if start_bar < 1 or end_bar < start_bar or end_bar > sec.bars:
        raise MusicError(
            f"Duplicate range {start_bar}..{end_bar} is outside section {sec.id!r} (1..{sec.bars})."
        )
    dup_count = end_bar - start_bar + 1
    target_pos = end_bar + 1 if insert_at is None else max(1, min(sec.bars + 1, insert_at))

    # Chords are all-or-none per section; a chordless section stays chordless.
    existing_chords = list(sec.chords)
    copied_chords = existing_chords[start_bar - 1 : end_bar]
    new_chords = [
        *existing_chords[: target_pos - 1],
        *copied_chords,
        *existing_chords[target_pos - 1 :],
    ]
    new_sec = ScoreSection(
        id=sec.id,
        title=sec.title,
        kind=sec.kind,
        bars=sec.bars + dup_count,
        chords=new_chords,
        tempo_map=[
            TempoChange(
                bar=(tc.bar + dup_count if tc.bar >= target_pos else tc.bar),
                bpm=tc.bpm,
                curve=tc.curve,
            )
            for tc in sec.tempo_map
        ],
    )
    updated_outline = outline.model_copy(
        update={"sections": [new_sec if s.id == sec.id else s for s in outline.sections]}
    )
    work.write_score(updated_outline, expected_digest=score_digest)

    for track, track_digest in work.list_tracks():
        if track.section_id != sec.id:
            continue
        copied_bars = [
            b.model_copy(update={"repeat_bar": None}) for b in track.bars[start_bar - 1 : end_bar]
        ]
        combined = [
            *track.bars[: target_pos - 1],
            *copied_bars,
            *track.bars[target_pos - 1 :],
        ]
        renumbered = [
            b.model_copy(update={"bar": idx, "repeat_bar": None})
            for idx, b in enumerate(combined, start=1)
        ]
        updated_track = TrackFile(
            track_id=track.track_id,
            section_id=sec.id,
            instrument_id=track.instrument_id,
            time_signature=outline.time_signature,
            groove_swing=track.groove_swing,
            bars=renumbered,
        )
        work.write_track(
            updated_track,
            expected_digest=track_digest,
            summary=f"Duplicated bars {start_bar}..{end_bar} in {track.track_id}",
        )

    return new_sec
