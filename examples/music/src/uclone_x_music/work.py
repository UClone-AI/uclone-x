"""One song's files, read as their shapes and written through its lease.

`SongLibrary` knows where a song is and who may write it; this module knows what is in it:
the score outline, the orchestra instruments, the tracks with their `.history/` revisions,
the proposals, and the session log. Every read validates (`uclone_x_music.schemas`) and
every write goes through `SongLibrary.write_file`, so the lease and the digest check apply
to each of them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from uclone_x_music.library import (
    MusicError,
    SongFile,
    SongLibrary,
    SongRecord,
)
from uclone_x_music.schemas import (
    ENTRY_ID_PATTERN,
    InstrumentEntry,
    MusicProposal,
    Revision,
    RevisionLog,
    ScoreOutline,
    SessionEntry,
    SessionsFile,
    StyleAxioms,
    TrackFile,
    dump_file,
    parse_file,
)

if TYPE_CHECKING:
    from uclone_x.tools.models import ToolContext

__all__ = [
    "HISTORY_DIR",
    "ORCHESTRA_DIR",
    "PROPOSALS_DIR",
    "SCORE_FILE",
    "SESSIONS_FILE",
    "TRACKS_DIR",
    "SongWork",
    "instrument_file",
    "proposal_file",
    "track_file",
    "track_file_by_id",
    "track_id_for",
]

SCORE_FILE = "score.yaml"
SESSIONS_FILE = "sessions.yaml"
ORCHESTRA_DIR = "orchestra"
TRACKS_DIR = "tracks"
HISTORY_DIR = f"{TRACKS_DIR}/.history"
PROPOSALS_DIR = "proposals"

_ID = re.compile(ENTRY_ID_PATTERN)
_PROPOSAL_NAME = re.compile(r"^p(\d+)\.yaml$")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check_id(value: str, kind: str) -> str:
    if not value or _ID.fullmatch(value) is None:
        raise MusicError(
            f"{value!r} is not a valid {kind} id (use lowercase letters, digits, '.', '_', or '-')."
        )
    return value


def track_id_for(section_id: str, instrument_id: str) -> str:
    """Return the canonical `track_id` for `section_id` and `instrument_id`."""
    _check_id(section_id, "section")
    _check_id(instrument_id, "instrument")
    return f"{section_id}_{instrument_id}"


def instrument_file(instrument_id: str) -> str:
    """Return the song-relative path for `instrument_id`."""
    _check_id(instrument_id, "instrument")
    return f"{ORCHESTRA_DIR}/{instrument_id}.yaml"


def track_file(section_id: str, instrument_id: str) -> str:
    """Return the song-relative path for `(section_id, instrument_id)`."""
    tid = track_id_for(section_id, instrument_id)
    return f"{TRACKS_DIR}/{tid}.yaml"


def track_file_by_id(track_id: str) -> str:
    """Return the song-relative path for `track_id`."""
    _check_id(track_id, "track")
    return f"{TRACKS_DIR}/{track_id}.yaml"


def proposal_file(proposal_id: str) -> str:
    """Return the song-relative path for `proposal_id`."""
    _check_id(proposal_id, "proposal")
    return f"{PROPOSALS_DIR}/{proposal_id}.yaml"


class SongWork:
    """The typed files of one song in one workspace's library."""

    def __init__(self, library: SongLibrary, song_id: str, room_id: str | None = None) -> None:
        self._library = library
        self._song_id = song_id
        self._room_id = room_id

    @classmethod
    def for_context(cls, context: ToolContext) -> SongWork:
        """Build a `SongWork` for the song `context` currently has open."""
        library, song_id, room_id = SongLibrary.for_context(context)
        return cls(library, song_id, room_id)

    @property
    def library(self) -> SongLibrary:
        return self._library

    @property
    def song_id(self) -> str:
        return self._song_id

    @property
    def room_id(self) -> str | None:
        return self._room_id

    def record(self) -> SongRecord:
        """Load the song's `SongRecord` (`song.yaml`)."""
        return self._library.load(self._song_id)

    def update_style_axioms(self, axioms: StyleAxioms) -> SongRecord:
        """Update `style_axioms` on `song.yaml` (requires lease)."""
        record = self._library.require_writable(self._song_id, self._room_id)
        updated = record.model_copy(update={"style_axioms": axioms, "genre": axioms.profile})
        self._library.save_record(updated)
        return updated

    # -- score.yaml -------------------------------------------------------------------------

    def read_score(self) -> tuple[ScoreOutline, str] | None:
        """Read `score.yaml`, returning `(ScoreOutline, digest)` or `None` if absent."""
        file = self._library.read_file(self._song_id, SCORE_FILE)
        if file is None:
            return None
        outline = parse_file(file.text, ScoreOutline, relative_path=SCORE_FILE)
        return outline, file.digest

    def write_score(self, outline: ScoreOutline, *, expected_digest: str | None) -> SongFile:
        """Write `score.yaml` through the lease and digest guard."""
        text = dump_file(outline)
        written = self._library.write_file(
            self._song_id,
            SCORE_FILE,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )
        self.record_session("Updated score.yaml")
        return written

    # -- orchestra/*.yaml -------------------------------------------------------------------

    def read_instrument(self, instrument_id: str) -> tuple[InstrumentEntry, str] | None:
        """Read `orchestra/<instrument_id>.yaml`, returning `(InstrumentEntry, digest)` or `None`."""
        rel = instrument_file(instrument_id)
        file = self._library.read_file(self._song_id, rel)
        if file is None:
            return None
        entry = parse_file(file.text, InstrumentEntry, relative_path=rel)
        if entry.id != instrument_id:
            raise MusicError(
                f"'{rel}' has id {entry.id!r}, which does not match its filename {instrument_id!r}."
            )
        return entry, file.digest

    def list_instruments(self) -> list[tuple[InstrumentEntry, str]]:
        """List all valid instrument entries in `orchestra/` sorted by id."""
        song_dir = self._library.song_dir(self._song_id)
        orch_dir = song_dir / ORCHESTRA_DIR
        if not orch_dir.is_dir():
            return []
        items: list[tuple[InstrumentEntry, str]] = []
        for path in sorted(orch_dir.iterdir(), key=lambda p: p.name):
            if path.name.startswith(".") or not path.is_file() or path.suffix != ".yaml":
                continue
            inst_id = path.stem
            if _ID.fullmatch(inst_id) is None:
                continue
            loaded = self.read_instrument(inst_id)
            if loaded is not None:
                items.append(loaded)
        return items

    def write_instrument(self, entry: InstrumentEntry, *, expected_digest: str | None) -> SongFile:
        """Write `orchestra/<entry.id>.yaml` through the lease and digest guard."""
        rel = instrument_file(entry.id)
        text = dump_file(entry)
        written = self._library.write_file(
            self._song_id,
            rel,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )
        self.record_session(f"Updated instrument {entry.id}")
        return written

    # -- tracks/*.yaml and .history/ --------------------------------------------------------

    def read_track(self, section_id: str, instrument_id: str) -> tuple[TrackFile, str] | None:
        """Read `tracks/<section_id>_<instrument_id>.yaml`."""
        return self.read_track_by_id(track_id_for(section_id, instrument_id))

    def read_track_by_id(self, track_id: str) -> tuple[TrackFile, str] | None:
        """Read `tracks/<track_id>.yaml`."""
        rel = track_file_by_id(track_id)
        file = self._library.read_file(self._song_id, rel)
        if file is None:
            return None
        track = parse_file(file.text, TrackFile, relative_path=rel)
        if track.track_id != track_id:
            raise MusicError(
                f"'{rel}' has track_id {track.track_id!r}, which does not match {track_id!r}."
            )
        return track, file.digest

    def list_tracks(self, section_id: str | None = None) -> list[tuple[TrackFile, str]]:
        """List all tracks in `tracks/`, optionally filtered by `section_id`."""
        song_dir = self._library.song_dir(self._song_id)
        t_dir = song_dir / TRACKS_DIR
        if not t_dir.is_dir():
            return []
        result: list[tuple[TrackFile, str]] = []
        for path in sorted(t_dir.iterdir(), key=lambda p: p.name):
            if path.name.startswith(".") or not path.is_file() or path.suffix != ".yaml":
                continue
            tid = path.stem
            if _ID.fullmatch(tid) is None:
                continue
            loaded = self.read_track_by_id(tid)
            if loaded is not None:
                track, digest = loaded
                if section_id is None or track.section_id == section_id:
                    result.append((track, digest))
        return result

    def read_revision_log(self, track_id: str) -> tuple[RevisionLog, str | None]:
        """Read `tracks/.history/<track_id>/revisions.yaml`."""
        _check_id(track_id, "track")
        rel = f"{HISTORY_DIR}/{track_id}/revisions.yaml"
        file = self._library.read_file(self._song_id, rel)
        if file is None:
            return RevisionLog(revisions=[]), None
        return parse_file(file.text, RevisionLog, relative_path=rel), file.digest

    def _record_track_revision(
        self, track_id: str, content: str, digest: str, summary: str
    ) -> None:
        snap_rel = f"{HISTORY_DIR}/{track_id}/{digest}.yaml"
        if self._library.read_file(self._song_id, snap_rel) is None:
            self._library.write_file(
                self._song_id,
                snap_rel,
                content,
                room_id=self._room_id,
                expected_digest=None,
            )
        log, log_digest = self.read_revision_log(track_id)
        entry = Revision(
            digest=digest,
            saved_at=_now_iso(),
            room_id=self._room_id or "unknown",
            summary=summary,
        )
        updated_log = RevisionLog(revisions=[*log.revisions, entry])
        self._library.write_file(
            self._song_id,
            f"{HISTORY_DIR}/{track_id}/revisions.yaml",
            dump_file(updated_log),
            room_id=self._room_id,
            expected_digest=log_digest,
        )

    def write_track(
        self,
        track: TrackFile,
        *,
        expected_digest: str | None,
        summary: str = "",
    ) -> SongFile:
        """Write `tracks/<track.track_id>.yaml` and preserve its revision in `.history/`."""
        expected_tid = track_id_for(track.section_id, track.instrument_id)
        if track.track_id != expected_tid:
            raise MusicError(
                f"Track id {track.track_id!r} must be '{expected_tid}' for section "
                f"{track.section_id!r} and instrument {track.instrument_id!r}."
            )
        rel = track_file_by_id(track.track_id)
        text = dump_file(track)
        written = self._library.write_file(
            self._song_id,
            rel,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )
        if written.digest == expected_digest:
            return written
        self._record_track_revision(
            track.track_id,
            written.text,
            written.digest,
            summary or f"Updated {track.track_id}",
        )
        self.record_session(summary or f"Wrote track {track.track_id}")
        return written

    def revert_track(self, track_id: str, *, expected_digest: str) -> tuple[TrackFile, SongFile]:
        """Revert `track_id` to its previous revision in `.history/<track_id>/revisions.yaml`."""
        log, _ = self.read_revision_log(track_id)
        if len(log.revisions) < 2:
            raise MusicError(f"Track {track_id!r} has no previous revision to revert to.")
        prev_rev = log.revisions[-2]
        snap_rel = f"{HISTORY_DIR}/{track_id}/{prev_rev.digest}.yaml"
        snap_file = self._library.read_file(self._song_id, snap_rel)
        if snap_file is None:
            raise MusicError(
                f"Previous revision snapshot {prev_rev.digest[:12]} for {track_id!r} is missing."
            )
        restored = parse_file(snap_file.text, TrackFile, relative_path=snap_rel)
        written = self.write_track(
            restored,
            expected_digest=expected_digest,
            summary=f"Reverted to {prev_rev.digest[:12]}",
        )
        return restored, written

    # -- proposals/*.yaml -------------------------------------------------------------------

    def next_proposal_id(self) -> str:
        """Return the next unused proposal id (`p001`, `p002`, ...)."""
        song_dir = self._library.song_dir(self._song_id)
        p_dir = song_dir / PROPOSALS_DIR
        highest = 0
        if p_dir.is_dir():
            for path in p_dir.iterdir():
                match = _PROPOSAL_NAME.fullmatch(path.name)
                if match is not None:
                    highest = max(highest, int(match.group(1)))
        return f"p{highest + 1:03d}"

    def read_proposal(self, proposal_id: str) -> tuple[MusicProposal, str] | None:
        """Read `proposals/<proposal_id>.yaml`."""
        rel = proposal_file(proposal_id)
        file = self._library.read_file(self._song_id, rel)
        if file is None:
            return None
        prop = parse_file(file.text, MusicProposal, relative_path=rel)
        if prop.id != proposal_id:
            raise MusicError(
                f"'{rel}' has id {prop.id!r}, which does not match its filename {proposal_id!r}."
            )
        return prop, file.digest

    def list_proposals(self, status: str | None = None) -> list[tuple[MusicProposal, str]]:
        """List proposals in `proposals/`, optionally filtered by `status`."""
        song_dir = self._library.song_dir(self._song_id)
        p_dir = song_dir / PROPOSALS_DIR
        if not p_dir.is_dir():
            return []
        result: list[tuple[MusicProposal, str]] = []
        for path in sorted(p_dir.iterdir(), key=lambda p: p.name):
            if path.name.startswith(".") or not path.is_file() or path.suffix != ".yaml":
                continue
            pid = path.stem
            if _ID.fullmatch(pid) is None:
                continue
            loaded = self.read_proposal(pid)
            if loaded is not None:
                prop, digest = loaded
                if status is None or prop.status == status:
                    result.append((prop, digest))
        return result

    def write_proposal(self, proposal: MusicProposal, *, expected_digest: str | None) -> SongFile:
        """Write `proposals/<proposal.id>.yaml`."""
        rel = proposal_file(proposal.id)
        text = dump_file(proposal)
        return self._library.write_file(
            self._song_id,
            rel,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )

    # -- sessions.yaml ----------------------------------------------------------------------

    def record_session(self, summary: str = "") -> None:
        """Record the active room's write activity in `sessions.yaml`."""
        if not self._room_id:
            return
        file = self._library.read_file(self._song_id, SESSIONS_FILE)
        if file is None:
            sessions = SessionsFile(sessions=[])
            expected: str | None = None
        else:
            sessions = parse_file(file.text, SessionsFile, relative_path=SESSIONS_FILE)
            expected = file.digest

        now = _now_iso()
        updated_list: list[SessionEntry] = []
        found = False
        for item in sessions.sessions:
            if item.room_id == self._room_id:
                updated_list.append(
                    SessionEntry(
                        room_id=self._room_id,
                        opened_at=item.opened_at,
                        last_written_at=now,
                        summary=summary or item.summary,
                    )
                )
                found = True
            else:
                updated_list.append(item)
        if not found:
            updated_list.append(
                SessionEntry(
                    room_id=self._room_id,
                    opened_at=now,
                    last_written_at=now,
                    summary=summary,
                )
            )
        self._library.write_file(
            self._song_id,
            SESSIONS_FILE,
            dump_file(SessionsFile(sessions=updated_list)),
            room_id=self._room_id,
            expected_digest=expected,
        )
