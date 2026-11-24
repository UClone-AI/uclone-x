"""Music composition agent tools: score, orchestra, track, context, and proposal tools.

Each tool operates on the song open in `ToolContext.story_id` and accepts no filesystem
paths from the model. Every file write goes through `SongWork` and `SongLibrary.write_file`,
enforcing both the single-writer conversation lease and SHA-256 digest guards.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x_music.arranger import (
    ArrangementIntent,
    append_section_bars,
    delete_section_bars,
    duplicate_section_bars,
    insert_section_bars,
    synthesize_bars_for_intent,
)
from uclone_x_music.audit import audit_bar_patch_continuity, audit_section
from uclone_x_music.context import build_score_context, context_digest_line
from uclone_x_music.digests import (
    LIST_ITEMS_SHOWN,
    NOTHING_CHANGED,
    audit_digest,
    bar_view,
    bundle_view,
    changed_bar_numbers,
    digest_of_value,
    instrument_line,
    proposal_line,
    section_line,
    write_result,
)
from uclone_x_music.groove import is_chord_symbol, subdivide_bars_faster
from uclone_x_music.library import MusicError
from uclone_x_music.proposals import (
    apply_proposal,
    preview_patched_track,
    propose_instrument_change,
    propose_track_patch,
    reject_proposal,
)
from uclone_x_music.schemas import (
    KEY_SIGNATURE_PATTERN,
    TIME_SIGNATURE_PATTERN,
    BarNoteEvent,
    InstrumentCategory,
    InstrumentEntry,
    InstrumentRole,
    MusicFileError,
    MusicProposal,
    RegionSelection,
    ScoreOutline,
    ScoreSection,
    StyleProfile,
    TempoChange,
    TrackBar,
    TrackFile,
    beats_per_bar_for,
    default_style_axioms,
    describe_invalid,
    format_note_token,
    normalize_bar_notes,
    normalize_track_bar,
    parse_note_token,
    pitch_to_name,
    validate_track_against_section,
)
from uclone_x_music.start import read_start_state, write_start_state
from uclone_x_music.work import (
    SCORE_FILE,
    SongWork,
    instrument_file,
    proposal_file,
    track_file,
    track_id_for,
)

__all__ = [
    "APPLY_PROPOSAL_NOT_APPROVED_NOTE",
    "MusicContextParams",
    "MusicContextTool",
    "MusicOrchestraParams",
    "MusicOrchestraTool",
    "MusicProposalParams",
    "MusicProposalTool",
    "MusicScoreParams",
    "MusicScoreTool",
    "MusicTheoryAuditParams",
    "MusicTheoryAuditTool",
    "MusicTrackParams",
    "MusicTrackTool",
    "ensure_section_and_track_seeded",
    "synthesize_bars_for_track",
]

_ARGUMENTS = ConfigDict(extra="forbid")

APPLY_PROPOSAL_NOT_APPROVED_NOTE = (
    "The proposal was not applied, and nothing in the song changed. Only the person can "
    "approve a proposal, and they do that in MusicView. Tell them it is waiting there for "
    "them to approve or reject, and do not make the change another way."
)

#: The most bars one `music_track` read returns (#2429); a longer range is cut and says so.
_TRACK_READ_BAR_CAP = 8
#: The most characters of bar data one read returns, for bars dense with notes.
_TRACK_READ_CHAR_CAP = 1_400
#: The continuity window `music_context` gives a selected region, in bars.
_CONTEXT_WINDOW_BARS = 2

SectionKind = Literal[
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "bridge",
    "solo",
    "outro",
    "movement",
]


def _format_beats(beats: float) -> str:
    return f"{beats:g}" if beats % 1 != 0 else f"{int(beats)}.0"


_INSTRUMENT_ALIASES: dict[str, str] = {
    "violin": "violin_solo",
    "piano": "grand_piano",
    "bass": "bass_synth",
    "drums": "drum_kit",
    "drum": "drum_kit",
}


def _resolve_instrument_id(work: SongWork, section_id: str | None, instrument_id: str) -> str:
    """Resolve short instrument names (e.g. 'violin' -> 'violin_solo') when present in the song."""
    cleaned = instrument_id.strip()
    if work.read_instrument(cleaned) is not None:
        return cleaned
    if section_id and work.read_track(section_id, cleaned) is not None:
        return cleaned
    alias = _INSTRUMENT_ALIASES.get(cleaned.casefold())
    if alias is not None:
        if work.read_instrument(alias) is not None or (
            section_id and work.read_track(section_id, alias) is not None
        ):
            return alias
    return cleaned


def _resolve_target_instruments(
    work: SongWork,
    section_id: str,
    raw_instrument_id: str,
) -> list[str]:
    """Expand 'all'/'*' or comma-separated instrument IDs into concrete instrument IDs."""
    cleaned = raw_instrument_id.strip()
    if cleaned.casefold() in {"all", "*", "ensemble", "full"}:
        inst_ids = [inst.id for inst, _ in work.list_instruments()]
        for track, _ in work.list_tracks():
            if track.section_id == section_id and track.instrument_id not in inst_ids:
                inst_ids.append(track.instrument_id)
        if not inst_ids:
            inst_ids = ["violin_solo", "grand_piano", "bass_synth", "drum_kit"]
        return inst_ids
    if "," in cleaned:
        resolved: list[str] = []
        for part in cleaned.split(","):
            if part.strip():
                rid = _resolve_instrument_id(work, section_id, part.strip())
                if rid not in resolved:
                    resolved.append(rid)
        return resolved or ["violin_solo"]
    return [_resolve_instrument_id(work, section_id, cleaned)]


def _normalize_patch_bars(
    bars: Sequence[TrackBar],
    start_bar: int,
    end_bar: int,
    time_signature: str,
) -> list[TrackBar]:
    """Normalize each bar's note beat sum to `time_signature` and align bar numbers to `start_bar..end_bar`."""
    expected_len = max(1, end_bar - start_bar + 1)
    source = list(bars)
    if not source:
        return []
    normalized: list[TrackBar] = []
    for idx in range(expected_len):
        tmpl = source[idx] if idx < len(source) else source[idx % len(source)]
        norm_bar = normalize_track_bar(tmpl, time_signature)
        target_bar_num = start_bar + idx
        if norm_bar.bar != target_bar_num:
            norm_bar = norm_bar.model_copy(
                update={
                    "bar": target_bar_num,
                    "repeat_bar": None if idx >= len(source) else norm_bar.repeat_bar,
                }
            )
        normalized.append(norm_bar)
    return normalized


def _synthesize_bars_for_track(
    work: SongWork,
    track: TrackFile,
    start_bar: int,
    end_bar: int,
    intent: ArrangementIntent = "climax_ornate",
) -> list[TrackBar]:
    """Synthesize musical bars for `start_bar..end_bar` for the stated arrangement `intent`."""
    if intent == "climax_ornate":
        return _auto_ornate_bars(track, start_bar, end_bar)
    score_pair = work.read_score()
    chords: list[str] | None = None
    if score_pair is not None:
        sec = next((s for s in score_pair[0].sections if s.id == track.section_id), None)
        if sec is not None and sec.chords:
            chords = list(sec.chords)
    inst_pair = work.read_instrument(track.instrument_id)
    category: InstrumentCategory = inst_pair[0].category if inst_pair is not None else "strings"
    role: InstrumentRole = inst_pair[0].role if inst_pair is not None else "lead_melody"
    return synthesize_bars_for_intent(
        track,
        start_bar,
        end_bar,
        chords=chords,
        category=category,
        role=role,
        intent=intent,
    )


def _auto_ornate_bars(track: TrackFile, start_bar: int, end_bar: int) -> list[TrackBar]:
    """Generate an ornate subdivided melody/rhythm variation for `start_bar..end_bar`."""
    patched = subdivide_bars_faster(track, start_bar=start_bar, end_bar=end_bar, factor=2)
    if all(
        ev.duration >= 0.5 for bar_obj in patched for ev in bar_obj.parsed_events() if ev.midi >= 0
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


def _extend_section_and_tracks_to_bar(
    work: SongWork,
    section_id: str,
    target_bars: int,
    proposed_bars: list[TrackBar] | None = None,
) -> None:
    """Extend `section_id` in `score.yaml` and all its existing tracks up to `target_bars`."""
    score_pair = work.read_score()
    if score_pair is None:
        return
    outline, score_digest = score_pair
    sec = next((s for s in outline.sections if s.id == section_id), None)
    if sec is None:
        return

    proposed_chord_map = {
        b.bar: b.chord for b in (proposed_bars or []) if b.chord and b.chord.strip()
    }
    base_chords = list(sec.chords) or [proposed_chord_map[k] for k in sorted(proposed_chord_map)]
    new_chords = list(sec.chords)
    while base_chords and len(new_chords) < target_bars:
        bar_num = len(new_chords) + 1
        chord = proposed_chord_map.get(bar_num) or base_chords[(bar_num - 1) % len(base_chords)]
        new_chords.append(chord)

    if sec.bars < target_bars or len(sec.chords) < target_bars:
        updated_sec = ScoreSection(
            id=sec.id,
            title=sec.title,
            kind=sec.kind,
            bars=max(sec.bars, target_bars),
            chords=new_chords[: max(sec.bars, target_bars)] if new_chords else [],
            tempo_map=list(sec.tempo_map),
        )
        new_sections = [updated_sec if s.id == section_id else s for s in outline.sections]
        updated_outline = outline.model_copy(update={"sections": new_sections})
        work.write_score(updated_outline, expected_digest=score_digest)

    for track, track_digest in work.list_tracks():
        if track.section_id != section_id or len(track.bars) >= target_bars:
            continue
        extended_bars = [normalize_track_bar(b, outline.time_signature) for b in track.bars]
        source_bars = list(extended_bars)
        while len(extended_bars) < target_bars:
            bar_num = len(extended_bars) + 1
            template = source_bars[(bar_num - 1) % len(source_bars)]
            chord = new_chords[bar_num - 1] if bar_num - 1 < len(new_chords) else template.chord
            extended_bars.append(
                TrackBar(
                    bar=bar_num,
                    chord=chord,
                    notes=normalize_bar_notes(template.notes, outline.time_signature),
                    repeat_bar=None,
                    transpose_semitones=template.transpose_semitones,
                    muted=template.muted,
                )
            )
        extended_track = track.model_copy(
            update={"time_signature": outline.time_signature, "bars": extended_bars}
        )
        work.write_track(
            extended_track,
            expected_digest=track_digest,
            summary=f"Extended {track.track_id} to {target_bars} bars",
        )


def _require_new_bar_chords(
    outline: ScoreOutline,
    section_id: str,
    chords: list[str] | None,
    count: int,
    action: str,
) -> None:
    """Refuse new bars in a chorded section unless the model wrote their chords."""
    sec = next((s for s in outline.sections if s.id == section_id), None)
    if sec is None or not sec.chords:
        return
    if chords is None or len(chords) != count:
        got = "none" if chords is None else str(len(chords))
        raise MusicError(
            f"Action {action!r} adds {count} bars to {section_id!r}, whose bars carry chords, "
            f"so it needs 'chords' with exactly {count} symbols, one per new bar (got {got}). "
            "Write the progression the new bars should follow. Nothing was changed."
        )
    bad = [c for c in chords if not is_chord_symbol(c)]
    if bad:
        raise MusicError(f"Not chord symbols: {bad}. Nothing was changed.")


def _ensure_section_and_track_seeded(
    work: SongWork,
    section_id: str,
    instrument_id: str,
    min_bars: int,
    *,
    rests_only: bool = False,
) -> tuple[TrackFile, str] | None:
    """Ensure `section_id` and `tracks/<section_id>_<instrument_id>.yaml` exist with at least `min_bars` bars.

    A new track repeats the instrument's bars from another section, or holds full-bar rests
    when it has none or `rests_only` is set.
    """
    score_pair = work.read_score()
    outline = score_pair[0] if score_pair is not None else ScoreOutline()
    score_digest = score_pair[1] if score_pair is not None else None
    sec = next((s for s in outline.sections if s.id == section_id), None)
    target_bars = max(min_bars, sec.bars if sec is not None else 4)
    if sec is None:
        # A section the score does not have yet gets no chords: the progression is the
        # model's to write (`music_score` upsert_section), never a fixed one from another key.
        chords: list[str] = []
        kind: SectionKind = "intro" if "intro" in section_id.lower() else "verse"
        sec = ScoreSection(
            id=section_id,
            title=section_id.replace("_", " ").title(),
            kind=kind,
            bars=target_bars,
            chords=chords,
            tempo_map=[],
        )
        new_sections = [sec, *outline.sections] if kind == "intro" else [*outline.sections, sec]
        outline = outline.model_copy(update={"sections": new_sections})
        work.write_score(outline, expected_digest=score_digest)

    loaded = work.read_track(section_id, instrument_id)
    if loaded is not None:
        existing_track, existing_digest = loaded
        if existing_track.time_signature != outline.time_signature:
            re_bars = [normalize_track_bar(b, outline.time_signature) for b in existing_track.bars]
            re_track = existing_track.model_copy(
                update={"time_signature": outline.time_signature, "bars": re_bars}
            )
            written_re = work.write_track(
                re_track,
                expected_digest=existing_digest,
                summary=f"Normalized {re_track.track_id} to {outline.time_signature}",
            )
            return re_track, written_re.digest
        return loaded

    donor_track = (
        None
        if rests_only
        else next(
            (t for t, _ in work.list_tracks() if t.instrument_id == instrument_id and t.bars),
            None,
        )
    )
    bpb = beats_per_bar_for(outline.time_signature)
    rest_token = f"1.0:REST:{_format_beats(bpb)}"
    seeded_bars: list[TrackBar] = []
    for bar_num in range(1, sec.bars + 1):
        chord = sec.chords[bar_num - 1] if bar_num - 1 < len(sec.chords) else None
        if donor_track is not None:
            tmpl = donor_track.bars[(bar_num - 1) % len(donor_track.bars)]
            seeded_bars.append(
                TrackBar(
                    bar=bar_num,
                    chord=chord,
                    notes=normalize_bar_notes(tmpl.notes, outline.time_signature),
                    repeat_bar=None,
                    transpose_semitones=tmpl.transpose_semitones,
                    muted=tmpl.muted,
                )
            )
        else:
            seeded_bars.append(TrackBar(bar=bar_num, chord=chord, notes=[rest_token]))

    record = work.record()
    new_track = TrackFile(
        track_id=track_id_for(section_id, instrument_id),
        section_id=section_id,
        instrument_id=instrument_id,
        time_signature=outline.time_signature,
        groove_swing=record.style_axioms.default_swing,
        bars=seeded_bars,
    )
    written = work.write_track(
        new_track,
        expected_digest=None,
        summary=f"Seeded track {new_track.track_id}",
    )
    return new_track, written.digest


synthesize_bars_for_track = _synthesize_bars_for_track
ensure_section_and_track_seeded = _ensure_section_and_track_seeded


def _score_read(
    work: SongWork, outline: ScoreOutline, digest: str | None, params: MusicScoreParams
) -> dict[str, Any]:
    """The outline as one line per section, or one section's chords sliced to a bar range."""
    record = work.record()
    result: dict[str, Any] = {
        "song_id": record.id,
        "title": record.title,
        "style_axioms": record.style_axioms.model_dump(mode="json"),
        "tempo": outline.tempo,
        "time_signature": outline.time_signature,
        "key_signature": outline.key_signature,
        "digest": digest,
    }
    wanted = params.section_id.strip() if params.section_id else None
    if wanted is None:
        shown = outline.sections[:LIST_ITEMS_SHOWN]
        result["sections"] = [
            {"id": s.id, "title": s.title, "kind": s.kind, "bars": s.bars} for s in shown
        ]
        if len(outline.sections) > len(shown):
            result["sections_not_shown"] = len(outline.sections) - len(shown)
        result["note"] = "Read with section_id for a section's chords and tempo map."
        return result
    sec = next((s for s in outline.sections if s.id == wanted), None)
    if sec is None:
        raise MusicError(f"Section {wanted!r} was not found in score.yaml.")
    first = params.start_bar or 1
    last = min(sec.bars, params.end_bar or sec.bars, first + 31)
    result["section"] = {
        "id": sec.id,
        "title": sec.title,
        "kind": sec.kind,
        "bars": sec.bars,
        "chords_from_bar": first,
        "chords": list(sec.chords[first - 1 : last]),
        "tempo_map": [t.model_dump(mode="json") for t in sec.tempo_map],
    }
    if last < sec.bars:
        result["note"] = f"Chords for bars {last + 1}..{sec.bars} not shown; ask with start_bar."
    return result


# -- music_score -------------------------------------------------------------------------


class MusicScoreParams(BaseModel):
    """Parameters for `music_score`."""

    model_config = _ARGUMENTS

    action: Literal[
        "read",
        "set_metadata",
        "upsert_section",
        "set_tempo_map",
        "insert_bars",
        "append_bars",
        "delete_bars",
        "duplicate_bars",
    ] = Field(
        description="'read' the score outline and style axioms (every section's id, kind "
        "and length; with 'section_id', that section's chords and tempo map, sliced by "
        "start_bar..end_bar); 'set_metadata' to update "
        "tempo, time_signature, key_signature, or style_profile; 'upsert_section' to add "
        "or update a structural section; 'set_tempo_map' to set tempo changes on a section; "
        "'insert_bars' to insert/prepend bars at 'at_bar' across all tracks; 'append_bars' "
        "to append bars at the end of a section across all tracks; 'delete_bars' to delete "
        "start_bar..end_bar; 'duplicate_bars' to duplicate start_bar..end_bar."
    )
    tempo: int | None = Field(
        default=None, ge=20, le=300, description="Global tempo in BPM (20..300)."
    )
    time_signature: str | None = Field(
        default=None,
        pattern=TIME_SIGNATURE_PATTERN,
        description="Global time signature (e.g. '4/4', '3/4', '6/8').",
    )
    key_signature: str | None = Field(
        default=None,
        pattern=KEY_SIGNATURE_PATTERN,
        description="Global key, '<tonic> major' or '<tonic> minor' (e.g. 'D minor', 'C major').",
    )
    style_profile: StyleProfile | None = Field(
        default=None,
        description="Genre style profile ('pop_ballad', 'hiphop_groove', 'jazz_swing', "
        "'classical_strict').",
    )
    section_id: str | None = Field(
        default=None, description="Target section id (e.g. 'verse_1', 'chorus_1')."
    )
    title: str | None = Field(default=None, description="Display title for 'upsert_section'.")
    kind: SectionKind | None = Field(default=None, description="Section kind for 'upsert_section'.")
    bars: int | None = Field(
        default=None, ge=1, le=128, description="Bar count for 'upsert_section'."
    )
    at_bar: int | None = Field(
        default=None,
        ge=1,
        description="1-indexed insertion point for 'insert_bars' (1 prepends at the start of the section).",
    )
    count: int | None = Field(
        default=None,
        ge=1,
        le=64,
        description="Number of bars to insert or append for 'insert_bars' or 'append_bars'.",
    )
    start_bar: int | None = Field(
        default=None, ge=1, description="Start bar for 'delete_bars' or 'duplicate_bars'."
    )
    end_bar: int | None = Field(
        default=None, ge=1, description="End bar for 'delete_bars' or 'duplicate_bars'."
    )
    intent: ArrangementIntent | None = Field(
        default=None,
        description="Arrangement intent for the synthesized bars of 'insert_bars' or "
        "'append_bars': 'gentle_intro', 'continuation', 'outro_cadence', 'hiphop_groove' or "
        "'climax_ornate'. Defaults to 'gentle_intro' when inserting at bar 1, otherwise "
        "'continuation'.",
    )
    chords: list[str] | None = Field(
        default=None,
        description="Per-bar chord symbols for 'upsert_section', 'insert_bars', or 'append_bars'.",
    )
    tempo_map: list[TempoChange] | None = Field(
        default=None,
        description="Tempo changes within the section for 'upsert_section' or 'set_tempo_map'.",
    )
    sketch: bool = Field(
        default=False,
        description="For 'insert_bars', 'append_bars' and a new section in 'upsert_section': let the "
        "tool fill every track's new bars with a generic draft in 'intent'. Without it "
        "they are full-bar rests that you then compose with music_track patch_bars.",
    )


class MusicScoreTool(BaseTool[MusicScoreParams]):
    """Read or update the open song's structural outline (`score.yaml`) and style axioms."""

    name = "music_score"
    description = (
        "Read or update the open song's structural score outline (tempo, time_signature, "
        "key_signature, sections, chords, tempo_map) and genre style profile."
    )
    params_type = MusicScoreParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"read"})
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(name=self.name, description=self.description, params_type=MusicScoreParams)

    async def run(self, params: MusicScoreParams, context: ToolContext) -> dict[str, Any]:
        work = SongWork.for_context(context)
        loaded = work.read_score()
        outline = loaded[0] if loaded is not None else ScoreOutline()
        digest = loaded[1] if loaded is not None else None

        if params.action == "read":
            return _score_read(work, outline, digest, params)

        if params.action == "set_metadata":
            record = work.record()
            style_changed = False
            if params.style_profile is not None:
                new_axioms = default_style_axioms(params.style_profile)
                style_changed = new_axioms != record.style_axioms
                if style_changed:
                    record = work.update_style_axioms(new_axioms)
                    start_pair = read_start_state(work)
                    if (
                        start_pair is not None
                        and start_pair[0].style_profile != params.style_profile
                    ):
                        write_start_state(
                            work,
                            start_pair[0].model_copy(
                                update={"style_profile": params.style_profile}
                            ),
                            expected_digest=start_pair[1],
                        )
            updates: dict[str, Any] = {}
            if params.tempo is not None:
                updates["tempo"] = params.tempo
            if params.time_signature is not None:
                updates["time_signature"] = params.time_signature
            if params.key_signature is not None:
                updates["key_signature"] = params.key_signature
            try:
                updated_outline = outline.model_copy(update=updates)
                ScoreOutline.model_validate(updated_outline.model_dump(mode="json"))
            except ValidationError as exc:
                raise MusicFileError(describe_invalid(SCORE_FILE, exc)) from exc
            written = work.write_score(updated_outline, expected_digest=digest)
            if updated_outline.time_signature != outline.time_signature:
                for trk, trk_digest in work.list_tracks():
                    if trk.time_signature != updated_outline.time_signature:
                        norm_bars = [
                            normalize_track_bar(b, updated_outline.time_signature) for b in trk.bars
                        ]
                        norm_trk = trk.model_copy(
                            update={
                                "time_signature": updated_outline.time_signature,
                                "bars": norm_bars,
                            }
                        )
                        work.write_track(
                            norm_trk,
                            expected_digest=trk_digest,
                            summary=f"Normalized {trk.track_id} to {updated_outline.time_signature}",
                        )
            meta = (
                f"tempo {updated_outline.tempo}, {updated_outline.time_signature}, "
                f"{updated_outline.key_signature}, style {record.style_axioms.profile}"
            )
            result = write_result(written.relative_path, digest, written.digest, f"Score: {meta}")
            if style_changed and not result["changed"]:
                result["changed"] = True
                result["summary"] = f"Style axioms set; score: {meta}"
            return result

        if params.action in {"insert_bars", "append_bars", "delete_bars", "duplicate_bars"}:
            default_sec_id = outline.sections[0].id if outline.sections else "theme_a"
            sec_id = (
                params.section_id.strip()
                if params.section_id and params.section_id.strip()
                else default_sec_id
            )
            if params.action == "insert_bars":
                at_bar = params.at_bar if params.at_bar is not None else 1
                count = (
                    params.count
                    if params.count is not None
                    else (params.bars if params.bars is not None else 2)
                )
                _require_new_bar_chords(outline, sec_id, params.chords, count, params.action)
                updated_sec = insert_section_bars(
                    work,
                    sec_id,
                    at_bar=at_bar,
                    count=count,
                    chords=params.chords,
                    intent=params.intent or ("gentle_intro" if at_bar == 1 else "continuation"),
                    sketch=params.sketch,
                )
            elif params.action == "append_bars":
                count = (
                    params.count
                    if params.count is not None
                    else (params.bars if params.bars is not None else 4)
                )
                _require_new_bar_chords(outline, sec_id, params.chords, count, params.action)
                updated_sec = append_section_bars(
                    work,
                    sec_id,
                    count=count,
                    chords=params.chords,
                    intent=params.intent or "continuation",
                    sketch=params.sketch,
                )
            elif params.action == "delete_bars":
                s_bar = params.start_bar if params.start_bar is not None else 1
                e_bar = params.end_bar if params.end_bar is not None else s_bar
                updated_sec = delete_section_bars(work, sec_id, start_bar=s_bar, end_bar=e_bar)
            else:
                s_bar = params.start_bar if params.start_bar is not None else 1
                e_bar = params.end_bar if params.end_bar is not None else s_bar
                updated_sec = duplicate_section_bars(
                    work,
                    sec_id,
                    start_bar=s_bar,
                    end_bar=e_bar,
                    insert_at=params.at_bar,
                )
            after = work.read_score()
            return write_result(
                SCORE_FILE,
                digest,
                after[1] if after is not None else "",
                f"Section {sec_id} ({params.action}): {section_line(updated_sec)}",
            )

        if params.section_id is None or not params.section_id.strip():
            raise MusicError(f"Action {params.action!r} requires 'section_id'.")
        sec_id = params.section_id.strip()

        if params.action == "upsert_section":
            existing_sec = next((s for s in outline.sections if s.id == sec_id), None)
            bars = (
                params.bars
                if params.bars is not None
                else (existing_sec.bars if existing_sec else None)
            )
            if bars is None:
                raise MusicError("Adding a new section with 'upsert_section' requires 'bars'.")
            title = (
                params.title.strip()
                if params.title and params.title.strip()
                else (existing_sec.title if existing_sec else sec_id.replace("_", " ").title())
            )
            kind: SectionKind = (
                params.kind
                if params.kind is not None
                else (existing_sec.kind if existing_sec else "verse")
            )
            chords = (
                params.chords
                if params.chords is not None
                else (list(existing_sec.chords) if existing_sec else [])
            )
            tempo_map = (
                params.tempo_map
                if params.tempo_map is not None
                else (list(existing_sec.tempo_map) if existing_sec else [])
            )
            try:
                new_sec = ScoreSection(
                    id=sec_id,
                    title=title,
                    kind=kind,
                    bars=bars,
                    chords=chords,
                    tempo_map=tempo_map,
                )
            except ValidationError as exc:
                raise MusicFileError(describe_invalid(SCORE_FILE, exc)) from exc

            new_sections: list[ScoreSection] = []
            replaced = False
            for s in outline.sections:
                if s.id == sec_id:
                    new_sections.append(new_sec)
                    replaced = True
                else:
                    new_sections.append(s)
            if not replaced:
                if new_sec.kind == "intro":
                    new_sections = [new_sec, *new_sections]
                else:
                    new_sections.append(new_sec)

            has_existing_tracks = bool(work.list_tracks())
            updated_outline = outline.model_copy(update={"sections": new_sections})
            written = work.write_score(updated_outline, expected_digest=digest)
            if replaced:
                _extend_section_and_tracks_to_bar(work, sec_id, new_sec.bars)
            elif has_existing_tracks and not params.sketch:
                # Every stem gets the new section as full-bar rests, so bar counts stay in
                # step; the notes are the model's to write with music_track.
                for inst_entry, _ in work.list_instruments():
                    _ensure_section_and_track_seeded(
                        work, sec_id, inst_entry.id, new_sec.bars, rests_only=True
                    )
            elif has_existing_tracks:
                intent: ArrangementIntent = (
                    "gentle_intro" if new_sec.kind == "intro" else "continuation"
                )
                for inst_entry, _ in work.list_instruments():
                    seeded = _ensure_section_and_track_seeded(
                        work, sec_id, inst_entry.id, new_sec.bars
                    )
                    if seeded is not None:
                        seeded_trk, seeded_dig = seeded
                        synth_bars = _synthesize_bars_for_track(
                            work, seeded_trk, 1, new_sec.bars, intent
                        )
                        synth_trk = seeded_trk.model_copy(update={"bars": synth_bars})
                        work.write_track(
                            synth_trk,
                            expected_digest=seeded_dig,
                            summary=f"Synthesized {new_sec.kind} section {sec_id} for {inst_entry.id}",
                        )
            return write_result(
                written.relative_path,
                digest,
                written.digest,
                f"Section {sec_id}: {section_line(new_sec)}",
                created=not replaced,
            )

        # set_tempo_map
        if params.tempo_map is None:
            raise MusicError("Action 'set_tempo_map' requires 'tempo_map'.")
        existing_sec = next((s for s in outline.sections if s.id == sec_id), None)
        if existing_sec is None:
            raise MusicError(f"Section {sec_id!r} was not found in score.yaml.")
        try:
            updated_sec = ScoreSection(
                id=existing_sec.id,
                title=existing_sec.title,
                kind=existing_sec.kind,
                bars=existing_sec.bars,
                chords=list(existing_sec.chords),
                tempo_map=list(params.tempo_map),
            )
        except ValidationError as exc:
            raise MusicFileError(describe_invalid(SCORE_FILE, exc)) from exc

        new_sections = [updated_sec if s.id == sec_id else s for s in outline.sections]
        updated_outline = outline.model_copy(update={"sections": new_sections})
        written = work.write_score(updated_outline, expected_digest=digest)
        return write_result(
            written.relative_path,
            digest,
            written.digest,
            f"Section {sec_id} tempo map: {section_line(updated_sec)}",
        )


# -- music_orchestra ---------------------------------------------------------------------


class MusicOrchestraParams(BaseModel):
    """Parameters for `music_orchestra`."""

    model_config = _ARGUMENTS

    action: Literal["list", "add_instrument", "adjust_mix", "propose_instrument"] = Field(
        description="'list' instruments in orchestra/; 'add_instrument' to register a new "
        "instrument; 'adjust_mix' to update volume, pan, muted, or solo; "
        "'propose_instrument' to stage an instrument change proposal."
    )
    instrument: InstrumentEntry | None = Field(
        default=None,
        description="Full InstrumentEntry object for 'add_instrument' or 'propose_instrument'.",
    )
    instrument_id: str | None = Field(
        default=None, description="Instrument id (e.g. 'bass', 'violin_solo')."
    )
    name: str | None = Field(default=None, description="Display name of the instrument.")
    category: InstrumentCategory | None = Field(default=None, description="Instrument category.")
    role: InstrumentRole | None = Field(default=None, description="Musical role.")
    midi_program: int | None = Field(default=None, ge=0, le=127)
    midi_channel: int | None = Field(default=None, ge=1, le=16)
    soundfont_patch: str | None = Field(default=None)
    min_pitch: str | None = Field(default=None)
    max_pitch: str | None = Field(default=None)
    transposition_semitones: int | None = Field(default=None, ge=-24, le=24)
    volume: int | None = Field(default=None, ge=0, le=100)
    pan: float | None = Field(default=None, ge=-1.0, le=1.0)
    muted: bool | None = Field(default=None)
    solo: bool | None = Field(default=None)
    summary: str | None = Field(default=None, description="Summary for 'propose_instrument'.")


def _default_gm_patch_for_instrument(
    inst_id: str,
    name: str,
    category: InstrumentCategory,
    role: InstrumentRole,
) -> tuple[int, int, str]:
    """Return `(midi_program, midi_channel, soundfont_patch)` defaults for an instrument."""
    hint = f"{inst_id} {name} {role}".casefold()
    if category == "drums":
        if "808" in hint or "hiphop" in hint or "trap" in hint or "boom" in hint:
            return (0, 10, "808_hiphop_kit")
        if "jazz" in hint or "brush" in hint:
            return (0, 10, "jazz_brush_kit")
        return (0, 10, "standard_drum_kit")
    if category == "strings":
        if "cello" in hint or role == "satb_bass":
            return (42, 1, "cello")
        if "viola" in hint or role == "satb_tenor":
            return (41, 1, "viola")
        if "contrabass" in hint or "double" in hint:
            return (43, 1, "contrabass")
        if "ensemble" in hint or "section" in hint:
            return (48, 1, "string_ensemble_1")
        return (40, 1, "violin")
    if category == "bass":
        if "upright" in hint or "acoustic" in hint:
            return (32, 1, "acoustic_bass")
        if "808" in hint or "sub" in hint or "synth" in hint:
            return (38, 1, "synth_bass_1")
        return (33, 1, "electric_bass_finger")
    if category == "piano":
        if "rhodes" in hint or "electric" in hint or "ep" in hint:
            return (4, 1, "electric_piano_1")
        if "harpsichord" in hint:
            return (6, 1, "harpsichord")
        return (0, 1, "acoustic_grand_piano")
    if category == "brass":
        if "trombone" in hint:
            return (57, 1, "trombone")
        if "horn" in hint:
            return (60, 1, "french_horn")
        if "sax" in hint:
            return (65, 1, "alto_sax")
        return (56, 1, "trumpet")
    if category == "woodwinds":
        if "flute" in hint:
            return (73, 1, "flute")
        if "oboe" in hint:
            return (68, 1, "oboe")
        if "sax" in hint:
            return (66, 1, "tenor_sax")
        return (71, 1, "clarinet")
    if category == "guitar":
        if "jazz" in hint or "hollow" in hint:
            return (26, 1, "electric_guitar_jazz")
        if "nylon" in hint or "classic" in hint:
            return (24, 1, "acoustic_guitar_nylon")
        return (25, 1, "acoustic_guitar_steel")
    if category == "vocal":
        return (52, 1, "choir_aahs")
    return (89, 1, "pad_2_warm")


def _build_instrument_entry(params: MusicOrchestraParams) -> InstrumentEntry:
    if params.instrument is not None:
        return params.instrument
    if not params.instrument_id or not params.name or not params.category or not params.role:
        raise MusicError(
            "Provide either 'instrument' or ('instrument_id', 'name', 'category', 'role')."
        )
    default_prog, default_chan, default_patch = _default_gm_patch_for_instrument(
        params.instrument_id.strip(),
        params.name.strip(),
        params.category,
        params.role,
    )
    data: dict[str, Any] = {
        "id": params.instrument_id.strip(),
        "name": params.name.strip(),
        "category": params.category,
        "role": params.role,
        "midi_program": default_prog,
        "midi_channel": default_chan,
        "soundfont_patch": default_patch,
    }
    for key in (
        "midi_program",
        "midi_channel",
        "soundfont_patch",
        "min_pitch",
        "max_pitch",
        "transposition_semitones",
        "volume",
        "pan",
        "muted",
        "solo",
    ):
        val = getattr(params, key)
        if val is not None:
            data[key] = val
    try:
        return InstrumentEntry.model_validate(data)
    except ValidationError as exc:
        raise MusicFileError(
            describe_invalid(instrument_file(params.instrument_id.strip()), exc)
        ) from exc


class MusicOrchestraTool(BaseTool[MusicOrchestraParams]):
    """Manage the open song's orchestra instruments and mixer parameters."""

    name = "music_orchestra"
    description = (
        "List, add, adjust mix (volume, pan, mute, solo), or propose changes to instruments "
        "in the open song's orchestra."
    )
    params_type = MusicOrchestraParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"list"})
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=MusicOrchestraParams
        )

    async def run(self, params: MusicOrchestraParams, context: ToolContext) -> dict[str, Any]:
        work = SongWork.for_context(context)
        if params.action == "list":
            items = work.list_instruments()
            listed: dict[str, Any] = {
                "instruments": [instrument_line(entry) for entry, _ in items[:LIST_ITEMS_SHOWN]]
            }
            if len(items) > LIST_ITEMS_SHOWN:
                listed["not_shown"] = [entry.id for entry, _ in items[LIST_ITEMS_SHOWN:]]
            return listed

        if params.action == "add_instrument":
            entry = _build_instrument_entry(params)
            existing = work.read_instrument(entry.id)
            expected_digest = existing[1] if existing is not None else None
            written = work.write_instrument(entry, expected_digest=expected_digest)
            return write_result(
                written.relative_path,
                expected_digest,
                written.digest,
                f"Instrument {entry.id}: {entry.name}, {entry.category}, {entry.role}",
            )

        if params.action == "adjust_mix":
            if not params.instrument_id or not params.instrument_id.strip():
                raise MusicError("Action 'adjust_mix' requires 'instrument_id'.")
            updates: dict[str, Any] = {}
            for key in ("volume", "pan", "muted", "solo"):
                val = getattr(params, key)
                if val is not None:
                    updates[key] = val
            if not updates:
                raise MusicError(
                    "Specify at least one of 'volume', 'pan', 'muted', or 'solo' for 'adjust_mix'."
                )
            raw_inst = params.instrument_id.strip()
            if raw_inst.casefold() in {"all", "*", "ensemble", "full"}:
                changed_ids: list[str] = []
                for item_entry, item_digest in work.list_instruments():
                    upd = item_entry.model_copy(update=updates)
                    w = work.write_instrument(upd, expected_digest=item_digest)
                    if w.digest != item_digest:
                        changed_ids.append(upd.id)
                mix = ", ".join(f"{k} {v}" for k, v in updates.items())
                return {
                    "changed": bool(changed_ids),
                    "changed_instruments": changed_ids,
                    "summary": (
                        f"Set {mix} on {len(changed_ids)} instruments."
                        if changed_ids
                        else f"{NOTHING_CHANGED}: every instrument already had {mix}."
                    ),
                }
            inst_id = _resolve_instrument_id(work, None, raw_inst)
            existing = work.read_instrument(inst_id)
            if existing is None:
                raise MusicError(f"Instrument {inst_id!r} was not found in orchestra/.")
            entry, digest = existing
            if updates.get("solo") is True:
                for other_entry, other_digest in work.list_instruments():
                    if other_entry.id != inst_id and other_entry.solo:
                        work.write_instrument(
                            other_entry.model_copy(update={"solo": False}),
                            expected_digest=other_digest,
                        )
            updated = entry.model_copy(update=updates)
            written = work.write_instrument(updated, expected_digest=digest)
            mix = ", ".join(f"{k} {v}" for k, v in updates.items())
            return write_result(written.relative_path, digest, written.digest, f"{inst_id}: {mix}")

        # propose_instrument
        entry = _build_instrument_entry(params)
        summary = (
            params.summary.strip()
            if params.summary and params.summary.strip()
            else f"Update instrument {entry.id}"
        )
        proposal, written = propose_instrument_change(work, instrument=entry, summary=summary)
        return {
            "proposed": True,
            "proposal_id": proposal.id,
            **proposal_line(proposal, written.digest),
            "path": written.relative_path,
        }


# -- music_track -------------------------------------------------------------------------


class MusicTrackParams(BaseModel):
    """Parameters for `music_track`."""

    model_config = _ARGUMENTS

    action: Literal[
        "read",
        "write_track",
        "patch_bars",
        "clear_bars",
        "transpose_bars",
        "revert",
    ] = Field(
        description="'read' a track's bars (start_bar..end_bar, at most 8 bars a read); "
        "'write_track' to create a new track (or overwrite "
        "when force_direct=True); 'patch_bars' to patch start_bar..end_bar (proposes by "
        "default when propose=True, or writes directly when propose=False); 'clear_bars' to "
        "replace bars with full-bar rests; 'transpose_bars' to shift sounding notes by "
        "'semitones'; 'revert' to restore the previous revision from .history/."
    )
    section_id: str = Field(description="Section id (e.g. 'verse_1').")
    instrument_id: str = Field(description="Instrument id (e.g. 'bass').")
    bars: list[TrackBar] | None = Field(
        default=None,
        description="For 'patch_bars': only the bars that change, each with its bar number, "
        "within start_bar..end_bar; never the whole track. For 'write_track': every bar of a "
        "new track.",
    )
    start_bar: int | None = Field(
        default=None, ge=1, description="First bar of the target range (1-indexed)."
    )
    end_bar: int | None = Field(
        default=None, ge=1, description="Last bar of the target range (inclusive)."
    )
    semitones: int | None = Field(
        default=None,
        ge=-36,
        le=36,
        description="Semitone shift for 'transpose_bars'.",
    )
    groove_swing: float | None = Field(
        default=None, ge=0.5, le=0.75, description="Groove swing ratio (0.5..0.75)."
    )
    propose: bool | None = Field(
        default=None,
        description="For 'patch_bars', 'clear_bars', or 'transpose_bars': whether to stage "
        "a MusicProposal instead of writing directly (defaults to True for 'patch_bars', "
        "False for 'clear_bars' and 'transpose_bars').",
    )
    force_direct: bool = Field(
        default=False,
        description="For 'write_track': allow overwriting an existing track directly.",
    )
    summary: str | None = Field(default=None, description="Revision or proposal summary.")
    intent: ArrangementIntent | None = Field(
        default=None,
        description="Arrangement intent for bars generated by 'patch_bars' when 'bars' is "
        "omitted and 'sketch' is set: 'gentle_intro', 'outro_cadence', 'hiphop_groove', "
        "'continuation', or 'climax_ornate' (the default).",
    )
    sketch: bool = Field(
        default=False,
        description="For 'patch_bars' without 'bars': let the tool write a generic draft of "
        "start_bar..end_bar in 'intent'. Without it, write the bars yourself.",
    )


def _transpose_bar_tokens(bar: TrackBar, semitones: int) -> TrackBar:
    if bar.repeat_bar is not None:
        new_shift = bar.transpose_semitones + semitones
        if new_shift < -36 or new_shift > 36:
            raise MusicError(f"Bar {bar.bar} transpose_semitones ({new_shift}) exceeds -36..36.")
        return bar.model_copy(update={"transpose_semitones": new_shift})

    shifted_notes: list[str] = []
    for token in bar.notes:
        ev = parse_note_token(token)
        if ev.midi == -1:
            shifted_notes.append(format_note_token(ev))
        else:
            new_midi = ev.midi + semitones
            if new_midi < 0 or new_midi > 127:
                raise MusicError(
                    f"Transposing {ev.pitch} in bar {bar.bar} by {semitones} semitones "
                    f"exceeds MIDI range 0..127 (got {new_midi})."
                )
            shifted_ev = BarNoteEvent(
                beat=ev.beat,
                pitch=pitch_to_name(new_midi),
                midi=new_midi,
                duration=ev.duration,
                velocity=ev.velocity,
                articulation=ev.articulation,
            )
            shifted_notes.append(format_note_token(shifted_ev))
    return bar.model_copy(update={"notes": shifted_notes})


class MusicTrackTool(BaseTool[MusicTrackParams]):
    """Read, write, patch, clear, transpose, or revert an instrument track for a section."""

    name = "music_track"
    description = (
        "Read, write, patch, clear, transpose, or revert a track "
        "(tracks/<section_id>_<instrument_id>.yaml). Bar patches preserve all bars outside "
        "start_bar..end_bar untouched, and 'clear_bars' inserts full-bar rests so bar beat "
        "counts stay valid."
    )
    params_type = MusicTrackParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"read"})
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(name=self.name, description=self.description, params_type=MusicTrackParams)

    async def run(self, params: MusicTrackParams, context: ToolContext) -> dict[str, Any]:
        if (
            params.action == "patch_bars"
            and not params.bars
            and not (params.sketch and params.start_bar is not None)
        ):
            # Refused before any seeding or extension: a refusal changes nothing.
            raise MusicError(
                "Action 'patch_bars' requires 'bars' you composed for start_bar..end_bar, "
                "or sketch=true with start_bar for a generic draft. Nothing was changed."
            )
        work = SongWork.for_context(context)
        sec_id = params.section_id.strip()
        if params.action in {"patch_bars", "clear_bars", "transpose_bars"}:
            target_insts = _resolve_target_instruments(work, sec_id, params.instrument_id)
            if len(target_insts) > 1:
                results: list[dict[str, Any]] = []
                for target_inst in target_insts:
                    sub_res = await self.run(
                        params.model_copy(update={"instrument_id": target_inst}),
                        context,
                    )
                    results.append({"instrument_id": target_inst, **sub_res})
                return _many_tracks_result(results)

        inst_id = (
            _resolve_instrument_id(work, sec_id, params.instrument_id)
            if params.action != "write_track"
            else params.instrument_id.strip()
        )
        tid = track_id_for(sec_id, inst_id)

        if params.action == "read":
            loaded = work.read_track(sec_id, inst_id)
            if loaded is None:
                raise MusicError(f"Track {tid!r} does not exist yet.")
            track, digest = loaded
            return _track_read(track, digest, track_file(sec_id, inst_id), params)

        if params.action == "revert":
            loaded = work.read_track(sec_id, inst_id)
            if loaded is None:
                raise MusicError(f"Track {tid!r} does not exist yet.")
            _, current_digest = loaded
            restored, written = work.revert_track(tid, expected_digest=current_digest)
            return write_result(
                written.relative_path,
                current_digest,
                written.digest,
                f"Reverted {tid} to its previous revision ({len(restored.bars)} bars).",
                reverted=tid,
            )

        if params.action == "write_track":
            if not params.bars:
                raise MusicError("Action 'write_track' requires 'bars'.")
            score_pair = work.read_score()
            outline = score_pair[0] if score_pair is not None else ScoreOutline()
            section = next((s for s in outline.sections if s.id == sec_id), None)
            if section is None:
                raise MusicError(
                    f"Section {sec_id!r} does not exist in score.yaml; create it with 'music_score' first."
                )
            existing = work.read_track(sec_id, inst_id)
            if existing is not None and not params.force_direct:
                raise MusicError(
                    f"Track {tid!r} already exists. Use 'patch_bars' to propose or apply a "
                    "bar-range update, or set force_direct=True to overwrite."
                )
            record = work.record()
            swing = (
                params.groove_swing
                if params.groove_swing is not None
                else record.style_axioms.default_swing
            )
            normalized_bars = [normalize_track_bar(b, outline.time_signature) for b in params.bars]
            try:
                new_track = TrackFile(
                    track_id=tid,
                    section_id=sec_id,
                    instrument_id=inst_id,
                    time_signature=outline.time_signature,
                    groove_swing=swing,
                    bars=normalized_bars,
                )
            except ValidationError as exc:
                raise MusicFileError(describe_invalid(track_file(sec_id, inst_id), exc)) from exc
            validate_track_against_section(new_track, section, outline)
            expected_digest = existing[1] if existing is not None else None
            written = work.write_track(
                new_track,
                expected_digest=expected_digest,
                summary=params.summary or f"Wrote track {tid}",
            )
            changed_bars = (
                changed_bar_numbers(existing[0].bars, new_track.bars)
                if existing is not None
                else [b.bar for b in new_track.bars]
            )
            verb = "Overwrote" if existing is not None else "Created"
            return write_result(
                written.relative_path,
                expected_digest,
                written.digest,
                f"{verb} {tid}: {len(new_track.bars)} bars.",
                changed_bars=changed_bars,
            )

        # Actions that operate on an existing track's bar range: patch_bars, clear_bars, transpose_bars
        loaded = work.read_track(sec_id, inst_id)
        before_digest = loaded[1] if loaded is not None else None
        if loaded is None and params.action == "patch_bars":
            req_end_seed = (
                params.end_bar
                if params.end_bar is not None
                else (params.bars[-1].bar if params.bars else (params.start_bar or 4))
            )
            loaded = _ensure_section_and_track_seeded(work, sec_id, inst_id, req_end_seed)
        if loaded is None:
            raise MusicError(
                f"Track {tid!r} does not exist yet; create it with 'write_track' first."
            )
        track, current_digest = loaded
        total_bars = len(track.bars)

        if params.action == "patch_bars":
            req_start = (
                params.start_bar
                if params.start_bar is not None
                else (params.bars[0].bar if params.bars else 1)
            )
            req_end = (
                params.end_bar
                if params.end_bar is not None
                else (params.bars[-1].bar if params.bars else req_start)
            )
            if req_end > total_bars and req_start <= total_bars + 1:
                _extend_section_and_tracks_to_bar(
                    work, sec_id, req_end, list(params.bars) if params.bars else None
                )
                reloaded = work.read_track(sec_id, inst_id)
                if reloaded is not None:
                    track, current_digest = reloaded
                    total_bars = len(track.bars)
            if params.bars:
                resolved_bars = _normalize_patch_bars(
                    params.bars, req_start, req_end, track.time_signature
                )
            else:
                # Only a sketch with start_bar gets here; anything else was refused up front.
                resolved_bars = _synthesize_bars_for_track(
                    work, track, req_start, req_end, params.intent or "climax_ornate"
                )
            start_bar = params.start_bar if params.start_bar is not None else resolved_bars[0].bar
            end_bar = params.end_bar if params.end_bar is not None else resolved_bars[-1].bar
            use_propose = True if params.propose is None else params.propose
            return _stage_or_write(
                work,
                track,
                before_digest,
                current_digest,
                (start_bar, end_bar),
                resolved_bars,
                params.summary,
                "Patch",
                propose=use_propose,
            )

        if params.start_bar is None or params.end_bar is None:
            raise MusicError(f"Action {params.action!r} requires 'start_bar' and 'end_bar'.")
        start_bar = params.start_bar
        end_bar = params.end_bar
        if start_bar < 1 or end_bar < start_bar or end_bar > total_bars:
            raise MusicError(
                f"Bar range {start_bar}..{end_bar} is outside track {tid!r} (1..{total_bars})."
            )

        if params.action == "clear_bars":
            bpb = beats_per_bar_for(track.time_signature)
            rest_token = f"1.0:REST:{_format_beats(bpb)}"
            cleared_bars = [
                TrackBar(
                    bar=b.bar,
                    chord=b.chord,
                    notes=[rest_token],
                    repeat_bar=None,
                    transpose_semitones=0,
                    muted=b.muted,
                )
                for b in track.bars[start_bar - 1 : end_bar]
            ]
            return _stage_or_write(
                work,
                track,
                before_digest,
                current_digest,
                (start_bar, end_bar),
                cleared_bars,
                params.summary,
                "Clear",
                propose=False if params.propose is None else params.propose,
            )

        # transpose_bars
        if params.semitones is None:
            raise MusicError("Action 'transpose_bars' requires 'semitones'.")
        transposed_bars = [
            _transpose_bar_tokens(b, params.semitones) for b in track.bars[start_bar - 1 : end_bar]
        ]
        return _stage_or_write(
            work,
            track,
            before_digest,
            current_digest,
            (start_bar, end_bar),
            transposed_bars,
            params.summary,
            f"Transpose by {params.semitones:+d} semitones,",
            propose=False if params.propose is None else params.propose,
        )


def _track_read(
    track: TrackFile, digest: str, path: str, params: MusicTrackParams
) -> dict[str, Any]:
    """The bars of `track` the read asked for, at most `_TRACK_READ_BAR_CAP` of them.

    A read with no range starts at bar 1. A range longer than the cap, or bars dense
    enough to pass `_TRACK_READ_CHAR_CAP`, is cut, and the result names the bars it left
    out so the model can ask for them.
    """
    total = len(track.bars)
    first = max(1, params.start_bar or 1)
    last = min(total, params.end_bar or total)
    shown: list[dict[str, Any]] = []
    used = 0
    for bar in [b for b in track.bars if first <= b.bar <= last]:
        view = bar_view(bar)
        cost = len(json.dumps(view, separators=(",", ":")))
        if shown and (len(shown) >= _TRACK_READ_BAR_CAP or used + cost > _TRACK_READ_CHAR_CAP):
            break
        shown.append(view)
        used += cost
    result: dict[str, Any] = {
        "track_id": track.track_id,
        "path": path,
        "digest": digest,
        "time_signature": track.time_signature,
        "groove_swing": track.groove_swing,
        "bars_total": total,
        "bars": shown,
    }
    shown_last = shown[-1]["bar"] if shown else first - 1
    if shown_last < last:
        result["not_shown"] = (
            f"Bars {shown_last + 1}..{last} were not returned; read them with "
            f"start_bar={shown_last + 1}."
        )
    return result


def _stage_or_write(
    work: SongWork,
    track: TrackFile,
    before_digest: str | None,
    current_digest: str,
    bar_range: tuple[int, int],
    new_bars: list[TrackBar],
    summary: str | None,
    verb: str,
    *,
    propose: bool,
) -> dict[str, Any]:
    """Propose or write `new_bars` over `bar_range`, keeping only the bars that change.

    Bars identical to the track's own are dropped and the range narrowed to the ones that
    differ, so a patch that resent unchanged bars stages or writes only its changes. When
    no bar differs nothing is proposed or written, and the result says so.
    """
    path = track_file(track.section_id, track.instrument_id)
    # A patch bar that names no chord keeps the bar's own: the patch carries what changes.
    chords = {b.bar: b.chord for b in track.bars}
    new_bars = [
        b.model_copy(update={"chord": chords[b.bar]})
        if b.chord is None and chords.get(b.bar) is not None
        else b
        for b in new_bars
    ]
    changed_bars = changed_bar_numbers(track.bars, new_bars)
    if not changed_bars:
        result = write_result(
            path,
            before_digest,
            current_digest,
            f"{track.track_id} was created or extended, but bars "
            f"{bar_range[0]}..{bar_range[1]} already held these notes, so nothing was "
            "proposed or written to them.",
            proposed=False,
            changed_bars=[],
        )
        if not result["changed"]:
            result["summary"] = (
                f"{NOTHING_CHANGED}: bars {bar_range[0]}..{bar_range[1]} of "
                f"{track.track_id} already hold these notes, so nothing was proposed or written."
            )
        return result
    start_bar, end_bar = min(changed_bars), max(changed_bars)
    new_bars = [b for b in new_bars if start_bar <= b.bar <= end_bar]
    text = (
        summary.strip()
        if summary and summary.strip()
        else f"{verb} {track.track_id} bars {start_bar}..{end_bar}"
    )
    if propose:
        proposal, written_prop = propose_track_patch(
            work,
            section_id=track.section_id,
            instrument_id=track.instrument_id,
            start_bar=start_bar,
            end_bar=end_bar,
            proposed_bars=new_bars,
            summary=text,
        )
        return {
            "proposed": True,
            "changed": True,
            "proposal_id": proposal.id,
            "changed_bars": changed_bars,
            "summary": f"Proposed {proposal.id}: {text}. Waiting for approval in MusicView.",
            "path": written_prop.relative_path,
            "digest": written_prop.digest,
        }
    temp_proposal = MusicProposal(
        id="p000",
        kind="track_patch",
        target_file=path,
        target_digest=current_digest,
        section_id=track.section_id,
        instrument_id=track.instrument_id,
        start_bar=start_bar,
        end_bar=end_bar,
        summary=text,
        proposed_bars=list(new_bars),
    )
    patched = preview_patched_track(track, temp_proposal)
    written = work.write_track(patched, expected_digest=current_digest, summary=text)
    return write_result(
        written.relative_path,
        before_digest,
        written.digest,
        f"{text}: bars {', '.join(str(b) for b in changed_bars)} changed.",
        proposed=False,
        changed_bars=changed_bars,
    )


def _many_tracks_result(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One instrument-range call's results, one short line per instrument."""
    lines = [
        {k: r[k] for k in ("instrument_id", "changed", "proposal_id", "changed_bars") if k in r}
        for r in results
    ]
    proposal_ids = [r["proposal_id"] for r in results if "proposal_id" in r]
    changed = [r["instrument_id"] for r in results if r.get("changed")]
    out: dict[str, Any] = {
        "changed": bool(changed),
        "proposed": bool(proposal_ids),
        "tracks": lines,
        "summary": (
            f"{len(changed)} of {len(results)} tracks changed"
            + (f"; proposals {', '.join(proposal_ids)} wait in MusicView." if proposal_ids else ".")
            if changed
            else f"{NOTHING_CHANGED} on any of the {len(results)} tracks."
        ),
    }
    if proposal_ids:
        out["proposal_ids"] = proposal_ids
    return out


# -- music_context -----------------------------------------------------------------------


class MusicContextParams(BaseModel):
    """Parameters for `music_context`."""

    model_config = _ARGUMENTS

    section_id: str | None = Field(
        default=None,
        description="Section id to inspect (defaults to the first section in score.yaml).",
    )
    selected_region: RegionSelection | None = Field(
        default=None,
        description="Optional bar range and instrument filter selected in MusicView.",
    )
    max_tracks: int = Field(
        default=12,
        ge=1,
        le=64,
        description="Maximum number of tracks to include in the polyphonic stack.",
    )


class MusicContextTool(BaseTool[MusicContextParams]):
    """Pull the vertical register map, free registers, missing roles, and tail bars for a section."""

    name = "music_context"
    description = (
        "Pull the active section's musical context bundle: chord progression, polyphonic "
        "register stack, free registers, missing roles, tail bar continuity window, and "
        "context manifest."
    )
    params_type = MusicContextParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=MusicContextParams
        )

    async def run(self, params: MusicContextParams, context: ToolContext) -> dict[str, Any]:
        work = SongWork.for_context(context)
        bundle = build_score_context(
            work,
            section_id=params.section_id,
            selected_region=params.selected_region,
            max_tracks=params.max_tracks,
            max_window_bars=_CONTEXT_WINDOW_BARS,
        )
        payload = bundle_view(bundle)
        digest = digest_of_value(payload)
        mark = context_digest_line(digest)
        if any(mark in section for section in context.turn_aid_sections):
            return {
                "same_as_turn_context": True,
                "digest": digest,
                "summary": "This is the [Music score context] already in this request "
                f"(context digest {digest[:12]}); it is not repeated here.",
            }
        return {**payload, "digest": digest}


# -- music_proposal ----------------------------------------------------------------------


class MusicProposalParams(BaseModel):
    """Parameters for `music_proposal`."""

    model_config = _ARGUMENTS

    action: Literal["list", "propose", "apply", "reject"] = Field(
        description="'list' proposals; 'propose' a new bar-range melody/rhythm proposal on a track; "
        "'apply' a pending proposal (requires person approval in conversation); "
        "'reject' a pending proposal."
    )
    proposal_id: str | None = Field(
        default=None, description="Proposal id (e.g. 'p001') for 'apply' or 'reject'."
    )
    section_id: str | None = Field(
        default=None, description="Target section id (e.g. 'theme_a') for 'propose'."
    )
    instrument_id: str | None = Field(
        default=None, description="Target instrument id (e.g. 'violin_solo') for 'propose'."
    )
    start_bar: int | None = Field(
        default=None, ge=1, description="First bar of the proposed edit (1-indexed) for 'propose'."
    )
    end_bar: int | None = Field(
        default=None, ge=1, description="Last bar of the proposed edit (inclusive) for 'propose'."
    )
    bars: list[TrackBar] | None = Field(
        default=None,
        description="Replacement TrackBar list for 'propose': the bars you composed for "
        "start_bar..end_bar. Omit it only with sketch=true.",
    )
    sketch: bool = Field(
        default=False,
        description="For 'propose' without 'bars': let the tool generate a generic variation "
        "of start_bar..end_bar in 'intent'.",
    )
    summary: str | None = Field(default=None, description="Human-readable summary for 'propose'.")
    intent: ArrangementIntent | None = Field(
        default=None,
        description="Arrangement intent for the bars generated when 'bars' is omitted: "
        "'gentle_intro' (soft, sustained), 'outro_cadence', 'hiphop_groove', 'continuation', "
        "or 'climax_ornate' (the default, a faster ornate variation).",
    )
    status: Literal["pending", "approved", "rejected"] | None = Field(
        default=None, description="Optional status filter for 'list'."
    )
    reason: str | None = Field(default=None, description="Optional rejection reason for 'reject'.")
    seen_digest: str | None = Field(
        default=None,
        description="Optional SHA-256 digest of the proposal file when it was shown.",
    )


class MusicProposalTool(BaseTool[MusicProposalParams]):
    """List, propose, apply, or reject staged music proposals (`proposals/pNNN.yaml`)."""

    name = "music_proposal"
    description = (
        "List, propose, apply, or reject staged music proposals. Use action='propose' to stage "
        "a regional bar melody/rhythm edit proposal for MusicView review, or 'list', 'apply', 'reject'."
    )
    params_type = MusicProposalParams
    writes_files: ClassVar[bool] = True
    read_actions: ClassVar[frozenset[str]] = frozenset({"list"})
    needs_room: ClassVar[bool] = True
    approval_actions: ClassVar[frozenset[str]] = frozenset({"apply"})
    approval_timeout_note: ClassVar[str | None] = APPLY_PROPOSAL_NOT_APPROVED_NOTE
    approval_unavailable_note: ClassVar[str | None] = APPLY_PROPOSAL_NOT_APPROVED_NOTE
    approval_unavailable_ends_turn: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name, description=self.description, params_type=MusicProposalParams
        )

    async def run(self, params: MusicProposalParams, context: ToolContext) -> dict[str, Any]:
        work = SongWork.for_context(context)
        if params.action == "list":
            items = work.list_proposals(status=params.status or "pending")
            listed: dict[str, Any] = {
                "status": params.status or "pending",
                "proposals": [proposal_line(p, d) for p, d in items[-LIST_ITEMS_SHOWN:]],
            }
            if len(items) > LIST_ITEMS_SHOWN:
                listed["older_not_shown"] = len(items) - LIST_ITEMS_SHOWN
            return listed

        if params.action == "propose":
            if not params.bars and not params.sketch:
                # Refused before any seeding or extension: a refusal changes nothing.
                raise MusicError(
                    "Action 'propose' needs 'bars' you composed for start_bar..end_bar, or "
                    "sketch=true for a generic variation. Nothing was proposed or changed."
                )
            score_pair = work.read_score()
            default_sec = (
                score_pair[0].sections[0].id
                if score_pair is not None and score_pair[0].sections
                else "theme_a"
            )
            sec_id = (params.section_id or default_sec).strip()
            raw_inst = (params.instrument_id or "violin_solo").strip()
            target_insts = _resolve_target_instruments(work, sec_id, raw_inst)
            if len(target_insts) > 1:
                results: list[dict[str, Any]] = []
                for target_inst in target_insts:
                    sub_res = await self.run(
                        params.model_copy(update={"instrument_id": target_inst}),
                        context,
                    )
                    results.append(sub_res)
                return {
                    "proposed": True,
                    "changed": True,
                    "proposal_ids": [r["proposal_id"] for r in results],
                    "summary": f"Proposed {len(results)} changes, one per instrument; they "
                    "wait for approval in MusicView.",
                }

            inst_id = target_insts[0]
            s_bar = params.start_bar if params.start_bar is not None else 1
            synth_intent: ArrangementIntent = params.intent or "climax_ornate"
            loaded = work.read_track(sec_id, inst_id)
            if loaded is None:
                loaded = _ensure_section_and_track_seeded(
                    work, sec_id, inst_id, params.end_bar or max(s_bar, 2)
                )
            if loaded is None:
                raise MusicError(f"Track {track_id_for(sec_id, inst_id)!r} does not exist yet.")
            track, _ = loaded
            e_bar = (
                params.end_bar
                if params.end_bar is not None
                else min(len(track.bars), max(s_bar, 2))
            )
            if e_bar > len(track.bars) and s_bar <= len(track.bars) + 1:
                _extend_section_and_tracks_to_bar(
                    work, sec_id, e_bar, list(params.bars) if params.bars else None
                )
                reloaded = work.read_track(sec_id, inst_id)
                if reloaded is not None:
                    track, _ = reloaded
            if params.bars:
                try:
                    norm_bars = _normalize_patch_bars(
                        params.bars, s_bar, e_bar, track.time_signature
                    )
                    temp_prop = MusicProposal(
                        id="p000",
                        kind="track_patch",
                        target_file=track_file(sec_id, inst_id),
                        section_id=sec_id,
                        instrument_id=inst_id,
                        start_bar=s_bar,
                        end_bar=e_bar,
                        summary="validate",
                        proposed_bars=norm_bars,
                    )
                    preview_patched_track(track, temp_prop)
                except (MusicError, MusicFileError, ValidationError, ValueError) as exc:
                    # The model's bars are refused as they are, never swapped for generated ones.
                    raise MusicError(
                        f"The proposed bars for {inst_id} bars {s_bar}..{e_bar} do not fit "
                        f"the track: {exc} Nothing was proposed."
                    ) from exc
                proposed_bars = norm_bars
            else:
                # Only a sketch gets here; a call without bars or sketch was refused up front.
                proposed_bars = _synthesize_bars_for_track(work, track, s_bar, e_bar, synth_intent)
            summary = (
                params.summary.strip()
                if params.summary and params.summary.strip()
                else f"Ornate melody variation on {inst_id} bars {s_bar}..{e_bar}"
            )
            proposal, written_prop = propose_track_patch(
                work,
                section_id=sec_id,
                instrument_id=inst_id,
                start_bar=s_bar,
                end_bar=e_bar,
                proposed_bars=proposed_bars,
                summary=summary,
            )
            return {
                "proposed": True,
                "changed": True,
                "proposal_id": proposal.id,
                "proposal_ids": [proposal.id],
                **_proposal_summary(proposal, written_prop.digest),
                "path": written_prop.relative_path,
            }

        if not params.proposal_id or not params.proposal_id.strip():
            raise MusicError(f"Action {params.action!r} requires 'proposal_id'.")
        pid = params.proposal_id.strip()

        if params.action == "apply":
            if not context.approved_by_person:
                raise MusicError(APPLY_PROPOSAL_NOT_APPROVED_NOTE)
            approved, written = apply_proposal(
                work,
                pid,
                decided_in="conversation",
                seen_digest=params.seen_digest,
            )
            return {
                "applied": approved.id,
                "changed": True,
                **_proposal_summary(approved, written.digest),
                "path": proposal_file(approved.id),
            }

        # reject
        rejected, written = reject_proposal(
            work,
            pid,
            decided_in="conversation",
            reason=params.reason,
            seen_digest=params.seen_digest,
        )
        return {
            "rejected": rejected.id,
            "changed": True,
            **_proposal_summary(rejected, written.digest),
            "path": proposal_file(rejected.id),
        }


def _proposal_summary(proposal: MusicProposal, digest: str) -> dict[str, Any]:
    """A proposal as its one-line summary: id, target, bar range; never its bars."""
    return proposal_line(proposal, digest)


# -- MusicTheoryAuditTool ---------------------------------------------------------------


class MusicTheoryAuditParams(BaseModel):
    """Arguments for `music_audit`."""

    model_config = _ARGUMENTS

    section_id: str | None = Field(
        default=None,
        description="Section id to audit. Defaults to the first section in score.yaml.",
    )
    instrument_id: str | None = Field(
        default=None,
        description="Optional instrument id when checking boundary continuity of a bar range.",
    )
    start_bar: int | None = Field(
        default=None,
        ge=1,
        description="Optional start bar for boundary continuity check.",
    )
    end_bar: int | None = Field(
        default=None,
        ge=1,
        description="Optional end bar for boundary continuity check.",
    )


class MusicTheoryAuditTool(BaseTool[MusicTheoryAuditParams]):
    """Audit harmonic clashes, voice leading, instrument ranges, and kick-bass lock."""

    name = "music_audit"
    description = (
        "Run deterministic music theory, voice-leading, instrument range, and groove "
        "audits on the open song's section or bar range."
    )
    params_type = MusicTheoryAuditParams
    writes_files: ClassVar[bool] = False
    needs_room: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__(
            name=self.name,
            description=self.description,
            params_type=MusicTheoryAuditParams,
        )

    async def run(self, params: MusicTheoryAuditParams, context: ToolContext) -> dict[str, Any]:
        work = SongWork.for_context(context)
        record = work.record()
        score_loaded = work.read_score()
        if score_loaded is None:
            raise MusicError("The song has no 'score.yaml' yet.")
        outline, _ = score_loaded
        if not outline.sections:
            raise MusicError("'score.yaml' has no sections to audit.")

        target_sec_id = params.section_id or outline.sections[0].id
        section = next((s for s in outline.sections if s.id == target_sec_id), None)
        if section is None:
            raise MusicError(f"Section {target_sec_id!r} was not found in 'score.yaml'.")

        instruments = {inst.id: inst for inst, _ in work.list_instruments()}
        tracks = [t for t, _ in work.list_tracks(section_id=section.id)]
        report = audit_section(section, tracks, instruments, record.style_axioms)
        findings = list(report.findings)

        if (
            params.instrument_id is not None
            and params.start_bar is not None
            and params.end_bar is not None
        ):
            track_loaded = work.read_track(section.id, params.instrument_id)
            if track_loaded is not None:
                track, _ = track_loaded
                findings.extend(audit_bar_patch_continuity(track, params.start_bar, params.end_bar))

        passed = all(f.severity != "error" for f in findings)
        return {
            "section_id": section.id,
            "style_profile": record.style_axioms.profile,
            **audit_digest(findings, passed=passed),
        }
