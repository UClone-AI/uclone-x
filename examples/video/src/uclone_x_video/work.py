"""One video project's files, read as their shapes and written through its lease.

`VideoLibrary` knows where a video project is and who may write it; this module knows what
is in it: the sequence manifest, the immutable media asset records, the timeline tracks with
their `.history/` revisions, the frame index, the proposals, and the session log.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from uclone_x_video.library import (
    VideoChangedError,
    VideoError,
    VideoFile,
    VideoLibrary,
    VideoProjectRecord,
)
from uclone_x_video.schemas import (
    ENTRY_ID_PATTERN,
    FrameIndexEntry,
    ManifestOutline,
    MediaAsset,
    QualityTier,
    Revision,
    RevisionLog,
    SessionEntry,
    SessionsFile,
    StyleProfile,
    TrackFile,
    VideoFileError,
    VideoProposal,
    dump_file,
    parse_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from uclone_x.tools.models import ToolContext

__all__ = [
    "EXPORTS_DIR",
    "FRAMES_INDEX_FILE",
    "HISTORY_DIR",
    "INDEX_DIR",
    "MANIFEST_FILE",
    "MEDIA_DIR",
    "PROPOSALS_DIR",
    "PROXIES_DIR",
    "SESSIONS_FILE",
    "TRACKS_DIR",
    "VAULT_DIR",
    "VECTORS_INDEX_FILE",
    "VideoWork",
    "asset_file",
    "proposal_file",
    "track_file_by_id",
]

MANIFEST_FILE = "manifest.yaml"
SESSIONS_FILE = "sessions.yaml"
MEDIA_DIR = "media"
VAULT_DIR = "vault"
PROXIES_DIR = "proxies"
EXPORTS_DIR = "exports"
INDEX_DIR = "index"
FRAMES_INDEX_FILE = f"{INDEX_DIR}/frames.jsonl"
VECTORS_INDEX_FILE = f"{INDEX_DIR}/vectors.bin"
TRACKS_DIR = "tracks"
HISTORY_DIR = f"{TRACKS_DIR}/.history"
PROPOSALS_DIR = "proposals"

_ID = re.compile(ENTRY_ID_PATTERN)
_PROPOSAL_NAME = re.compile(r"^p(\d+)\.yaml$")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _check_id(value: str, kind: str) -> str:
    if not value or _ID.fullmatch(value) is None:
        raise VideoError(
            f"{value!r} is not a valid {kind} id (use lowercase letters, digits, '.', '_', or '-')."
        )
    return value


def asset_file(asset_id: str) -> str:
    """Return the project-relative path for `asset_id`."""
    _check_id(asset_id, "asset")
    return f"{MEDIA_DIR}/{asset_id}.yaml"


def track_file_by_id(track_id: str) -> str:
    """Return the project-relative path for `track_id`."""
    _check_id(track_id, "track")
    return f"{TRACKS_DIR}/{track_id}.yaml"


def proposal_file(proposal_id: str) -> str:
    """Return the project-relative path for `proposal_id`."""
    _check_id(proposal_id, "proposal")
    return f"{PROPOSALS_DIR}/{proposal_id}.yaml"


class VideoWork:
    """The typed files of one video project in one workspace's library."""

    def __init__(self, library: VideoLibrary, project_id: str, room_id: str | None = None) -> None:
        self._library = library
        self._project_id = project_id
        self._room_id = room_id

    @classmethod
    def for_context(cls, context: ToolContext) -> VideoWork:
        """Build a `VideoWork` for the video project `context` currently has open."""
        library, project_id, room_id = VideoLibrary.for_context(context)
        return cls(library, project_id, room_id)

    @property
    def library(self) -> VideoLibrary:
        return self._library

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def room_id(self) -> str | None:
        return self._room_id

    @property
    def project_dir(self) -> Path:
        return self._library.project_dir(self._project_id)

    def record(self) -> VideoProjectRecord:
        """Load the project's `VideoProjectRecord` (`project.yaml`)."""
        return self._library.load(self._project_id)

    def update_project_settings(
        self,
        *,
        style_profile: StyleProfile | None = None,
        active_playback_tier: QualityTier | None = None,
        fps: int | None = None,
    ) -> VideoProjectRecord:
        """Update `project.yaml` settings (requires lease)."""
        record = self._library.require_writable(self._project_id, self._room_id)
        updates: dict[str, object] = {}
        if style_profile is not None:
            updates["style_profile"] = style_profile
        if active_playback_tier is not None:
            updates["active_playback_tier"] = active_playback_tier
        if fps is not None:
            updates["fps"] = fps
        updated = record.model_copy(update=updates)
        self._library.save_record(updated)
        return updated

    # -- manifest.yaml ----------------------------------------------------------------------

    def read_manifest(self) -> tuple[ManifestOutline, str] | None:
        """Read `manifest.yaml`, returning `(ManifestOutline, digest)` or `None` if absent."""
        file = self._library.read_file(self._project_id, MANIFEST_FILE)
        if file is None:
            return None
        outline = parse_file(file.text, ManifestOutline, relative_path=MANIFEST_FILE)
        return outline, file.digest

    def write_manifest(self, outline: ManifestOutline, *, expected_digest: str | None) -> VideoFile:
        """Write `manifest.yaml` through the lease and digest guard."""
        text = dump_file(outline)
        written = self._library.write_file(
            self._project_id,
            MANIFEST_FILE,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )
        self.record_session("Updated manifest.yaml")
        return written

    # -- media/*.yaml -----------------------------------------------------------------------

    def read_asset(self, asset_id: str) -> tuple[MediaAsset, str] | None:
        """Read `media/<asset_id>.yaml`, returning `(MediaAsset, digest)` or `None`."""
        rel = asset_file(asset_id)
        file = self._library.read_file(self._project_id, rel)
        if file is None:
            return None
        asset = parse_file(file.text, MediaAsset, relative_path=rel)
        if asset.asset_id != asset_id:
            raise VideoError(
                f"'{rel}' has asset_id {asset.asset_id!r}, which does not match its filename {asset_id!r}."
            )
        return asset, file.digest

    def list_assets(self) -> list[tuple[MediaAsset, str]]:
        """List all valid media assets in `media/` sorted by `asset_id`."""
        m_dir = self.project_dir / MEDIA_DIR
        if not m_dir.is_dir():
            return []
        items: list[tuple[MediaAsset, str]] = []
        for path in sorted(m_dir.iterdir(), key=lambda p: p.name):
            if path.name.startswith(".") or not path.is_file() or path.suffix != ".yaml":
                continue
            aid = path.stem
            if _ID.fullmatch(aid) is None:
                continue
            loaded = self.read_asset(aid)
            if loaded is not None:
                items.append(loaded)
        return items

    def write_asset(self, asset: MediaAsset, *, expected_digest: str | None) -> VideoFile:
        """Write `media/<asset.asset_id>.yaml` through the lease and digest guard."""
        rel = asset_file(asset.asset_id)
        text = dump_file(asset)
        written = self._library.write_file(
            self._project_id,
            rel,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )
        self.record_session(f"Registered media asset {asset.asset_id} ({asset.kind})")
        return written

    # -- index/frames.jsonl -----------------------------------------------------------------

    def read_frame_index(self) -> list[FrameIndexEntry]:
        """Read all indexed frames from `index/frames.jsonl`."""
        file = self._library.read_file(self._project_id, FRAMES_INDEX_FILE)
        if file is None or not file.text.strip():
            return []
        entries: list[FrameIndexEntry] = []
        for line_no, raw_line in enumerate(file.text.splitlines(), start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                entries.append(FrameIndexEntry.model_validate(data))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise VideoFileError(
                    f"'{FRAMES_INDEX_FILE}' line {line_no} is not a valid frame index entry."
                ) from exc
        return entries

    def write_frame_index(self, entries: Sequence[FrameIndexEntry]) -> VideoFile:
        """Write `index/frames.jsonl` atomically through the lease guard."""
        current = self._library.read_file(self._project_id, FRAMES_INDEX_FILE)
        expected = current.digest if current is not None else None
        lines = [
            json.dumps(entry.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            for entry in entries
        ]
        content = ("\n".join(lines) + "\n") if lines else ""
        return self._library.write_file(
            self._project_id,
            FRAMES_INDEX_FILE,
            content,
            room_id=self._room_id,
            expected_digest=expected,
        )

    # -- tracks/*.yaml and .history/ --------------------------------------------------------

    def read_track_by_id(self, track_id: str) -> tuple[TrackFile, str] | None:
        """Read `tracks/<track_id>.yaml`."""
        rel = track_file_by_id(track_id)
        file = self._library.read_file(self._project_id, rel)
        if file is None:
            return None
        track = parse_file(file.text, TrackFile, relative_path=rel)
        if track.track_id != track_id:
            raise VideoError(
                f"'{rel}' has track_id {track.track_id!r}, which does not match {track_id!r}."
            )
        return track, file.digest

    def list_tracks(self) -> list[tuple[TrackFile, str]]:
        """List all tracks in `tracks/` ordered by `(track.index, track.track_id)`."""
        t_dir = self.project_dir / TRACKS_DIR
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
                result.append(loaded)
        result.sort(key=lambda pair: (pair[0].index, pair[0].track_id))
        return result

    def write_track(
        self,
        track: TrackFile,
        *,
        expected_digest: str | None,
        summary: str = "Updated track",
    ) -> VideoFile:
        """Write `tracks/<track_id>.yaml`, archiving the previous version under `.history/`."""
        rel = track_file_by_id(track.track_id)
        previous = self._library.read_file(self._project_id, rel)
        if previous is not None:
            self._archive_track_revision(track.track_id, previous, summary=summary)
        text = dump_file(track)
        written = self._library.write_file(
            self._project_id,
            rel,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )
        self.record_session(f"{summary} ({track.track_id})")
        return written

    def _archive_track_revision(self, track_id: str, previous: VideoFile, *, summary: str) -> None:
        hist_dir = f"{HISTORY_DIR}/{track_id}"
        snap_rel = f"{hist_dir}/{previous.digest}.yaml"
        existing_snap = self._library.read_file(self._project_id, snap_rel)
        if existing_snap is None:
            self._library.write_file(
                self._project_id,
                snap_rel,
                previous.text,
                room_id=self._room_id,
                expected_digest=None,
            )
        log_rel = f"{hist_dir}/revisions.yaml"
        existing_log = self._library.read_file(self._project_id, log_rel)
        if existing_log is None:
            rev_log = RevisionLog(track_id=track_id, revisions=[])
            expected_log_digest: str | None = None
        else:
            rev_log = parse_file(existing_log.text, RevisionLog, relative_path=log_rel)
            expected_log_digest = existing_log.digest
        entry = Revision(
            digest=previous.digest,
            saved_at=_now_iso(),
            room_id=self._room_id,
            summary=summary,
        )
        updated_log = rev_log.model_copy(update={"revisions": [*rev_log.revisions, entry]})
        self._library.write_file(
            self._project_id,
            log_rel,
            dump_file(updated_log),
            room_id=self._room_id,
            expected_digest=expected_log_digest,
        )

    def read_track_revisions(self, track_id: str) -> RevisionLog:
        """Read the revision history for `track_id`."""
        _check_id(track_id, "track")
        log_rel = f"{HISTORY_DIR}/{track_id}/revisions.yaml"
        file = self._library.read_file(self._project_id, log_rel)
        if file is None:
            return RevisionLog(track_id=track_id, revisions=[])
        return parse_file(file.text, RevisionLog, relative_path=log_rel)

    def revert_track(
        self,
        track_id: str,
        *,
        target_digest: str | None = None,
        expected_digest: str,
    ) -> tuple[TrackFile, VideoFile]:
        """Restore `track_id` from `.history/<track_id>/<digest>.yaml`."""
        rev_log = self.read_track_revisions(track_id)
        if not rev_log.revisions:
            raise VideoError(f"Track {track_id!r} has no saved revision history to revert to.")
        chosen_digest = target_digest or rev_log.revisions[-1].digest
        snap_rel = f"{HISTORY_DIR}/{track_id}/{chosen_digest}.yaml"
        snap_file = self._library.read_file(self._project_id, snap_rel)
        if snap_file is None:
            raise VideoError(
                f"Historical snapshot {chosen_digest[:12]!r} for track {track_id!r} was not found."
            )
        restored = parse_file(snap_file.text, TrackFile, relative_path=snap_rel)
        written = self.write_track(
            restored,
            expected_digest=expected_digest,
            summary=f"Reverted {track_id} to {chosen_digest[:12]}",
        )
        return restored, written

    # -- proposals/*.yaml -------------------------------------------------------------------

    def next_proposal_id(self) -> str:
        """Return the next sequential proposal id (`p001`, `p002`, ...)."""
        p_dir = self.project_dir / PROPOSALS_DIR
        if not p_dir.is_dir():
            return "p001"
        highest = 0
        for child in p_dir.iterdir():
            match = _PROPOSAL_NAME.match(child.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
        return f"p{highest + 1:03d}"

    def read_proposal(self, proposal_id: str) -> tuple[VideoProposal, str] | None:
        """Read `proposals/<proposal_id>.yaml`."""
        rel = proposal_file(proposal_id)
        file = self._library.read_file(self._project_id, rel)
        if file is None:
            return None
        prop = parse_file(file.text, VideoProposal, relative_path=rel)
        if prop.id != proposal_id:
            raise VideoError(f"'{rel}' has id {prop.id!r}, which does not match {proposal_id!r}.")
        return prop, file.digest

    def list_proposals(
        self, *, status: Literal["pending", "approved", "rejected"] | None = None
    ) -> list[tuple[VideoProposal, str]]:
        """List proposals in `proposals/` sorted by id, optionally filtered by `status`."""
        p_dir = self.project_dir / PROPOSALS_DIR
        if not p_dir.is_dir():
            return []
        result: list[tuple[VideoProposal, str]] = []
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

    def write_proposal(self, proposal: VideoProposal, *, expected_digest: str | None) -> VideoFile:
        """Write `proposals/<proposal.id>.yaml` through the lease and digest guard."""
        rel = proposal_file(proposal.id)
        text = dump_file(proposal)
        return self._library.write_file(
            self._project_id,
            rel,
            text,
            room_id=self._room_id,
            expected_digest=expected_digest,
        )

    # -- sessions.yaml ----------------------------------------------------------------------

    def record_session(self, summary: str) -> None:
        """Append an entry to `sessions.yaml` when `room_id` is set."""
        if not self._room_id:
            return
        try:
            existing = self._library.read_file(self._project_id, SESSIONS_FILE)
            if existing is None:
                sessions = SessionsFile(sessions=[])
                expected: str | None = None
            else:
                sessions = parse_file(existing.text, SessionsFile, relative_path=SESSIONS_FILE)
                expected = existing.digest
            entry = SessionEntry(room_id=self._room_id, at=_now_iso(), summary=summary)
            updated = sessions.model_copy(update={"sessions": [*sessions.sessions, entry]})
            self._library.write_file(
                self._project_id,
                SESSIONS_FILE,
                dump_file(updated),
                room_id=self._room_id,
                expected_digest=expected,
            )
        except VideoChangedError:
            pass
