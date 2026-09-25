"""One story's files, read as their shapes and written through its lease (#1556).

`StoryLibrary` knows where a story is and who may write it; this knows what is in it: the
outline, the codex, the manuscript with its history, and the record of the conversations
that wrote it. Every read validates (`uclone_x.story.schemas`) and every write goes through
`StoryLibrary.write_file`, so the lease and the digest check apply to each of them.

Paths named in messages are relative to the story folder, which is all a person needs to
find the file.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from uclone_x.story.context import CodexIndex, CodexItem, UnreadableFile
from uclone_x.story.library import (
    STORIES_DIRNAME,
    NoConversationError,
    NoOpenStoryError,
    StoryError,
    StoryFile,
    StoryLibrary,
    StoryRecord,
)
from uclone_x.story.schemas import (
    CODEX_KINDS,
    ENTRY_ID_PATTERN,
    CodexEntry,
    CodexKind,
    Outline,
    Revision,
    RevisionLog,
    SessionEntry,
    SessionsFile,
    StoryFileError,
    dump_file,
    entry_model,
    parse_file,
)

if TYPE_CHECKING:
    from uclone_x.tools.models import ToolContext

__all__ = ["OUTLINE_FILE", "SESSIONS_FILE", "StoryWork", "manuscript_file"]

OUTLINE_FILE = "outline.yaml"
SESSIONS_FILE = "sessions.yaml"
CODEX_DIR = "codex"
MANUSCRIPT_DIR = "manuscript"
HISTORY_DIR = f"{MANUSCRIPT_DIR}/.history"

_ID = re.compile(ENTRY_ID_PATTERN)


def manuscript_file(scene_id: str) -> str:
    """The story-relative file holding the text of `scene_id`."""
    return f"{MANUSCRIPT_DIR}/{scene_id}.md"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StoryWork:
    """The files of one story, in one workspace's library."""

    def __init__(self, library: StoryLibrary, story_id: str) -> None:
        self._library = library
        self._story_id = story_id

    @classmethod
    def open_in(cls, context: ToolContext) -> StoryWork:
        """The story the call's conversation has open, for reading; refuse when there is none.

        Reading needs no lease: a story open read-only reads like any other.

        Raises:
            NoConversationError: the call is not part of a conversation.
            NoOpenStoryError: the conversation has no story open.
        """
        if context.room_id is None:
            raise NoConversationError(
                "Stories are read inside a conversation, and this call is not part of one."
            )
        if context.story_id is None:
            raise NoOpenStoryError(
                "No story is open in this conversation. Open or create a story first."
            )
        return cls(StoryLibrary(context.require_workspace()), context.story_id)

    @property
    def story_id(self) -> str:
        return self._story_id

    def workspace_path(self, relative: str) -> str:
        """`relative` as a workspace-relative path, the form a room's file record keeps."""
        return f"{STORIES_DIRNAME}/{self._story_id}/{relative}"

    # -- reading ---------------------------------------------------------------------

    def record(self) -> StoryRecord:
        return self._library.load(self._story_id)

    def read(self, relative: str) -> StoryFile | None:
        return self._library.read_file_if_present(self._story_id, relative)

    def outline(self) -> tuple[Outline, str] | None:
        """The outline and its digest, or `None` when the story has none yet.

        Raises:
            StoryFileError: `outline.yaml` does not fit; the message names the field.
        """
        found = self.read(OUTLINE_FILE)
        if found is None:
            return None
        return parse_file(Outline, found.text, OUTLINE_FILE), found.digest

    def require_outline(self) -> tuple[Outline, str]:
        found = self.outline()
        if found is None:
            raise StoryError("The story has no outline yet. Start one with story_outline 'init'.")
        return found

    def codex(self) -> CodexIndex:
        """Every codex entry of every kind; a file that does not load is reported."""
        items: list[CodexItem] = []
        unreadable: list[UnreadableFile] = []
        for kind in CODEX_KINDS:
            for relative in self._library.files_in(self._story_id, f"{CODEX_DIR}/{kind}"):
                name = PurePosixPath(relative)
                if name.suffix != ".yaml":
                    continue
                try:
                    items.append(CodexItem(kind, self._entry_at(kind, relative)))
                except (StoryError, StoryFileError) as exc:
                    unreadable.append(UnreadableFile(relative, str(exc)))
        return CodexIndex(tuple(items), tuple(unreadable))

    def _entry_at(self, kind: CodexKind, relative: str) -> CodexEntry:
        stem = PurePosixPath(relative).stem
        if not _ID.fullmatch(stem):
            raise StoryFileError(
                f"{relative} is not named as an id: use lowercase letters and digits, "
                "joined by '.', '_' or '-'."
            )
        found = self.read(relative)
        if found is None:
            raise StoryError(f"The story has no file '{relative}'.")
        entry = parse_file(entry_model(kind), found.text, relative)
        if entry.id != stem:
            raise StoryFileError(
                f"{relative} says its id is '{entry.id}', but the file is named '{stem}'. "
                "Make the two the same."
            )
        return entry

    def manuscript(self, scene_id: str) -> StoryFile | None:
        return self.read(manuscript_file(scene_id))

    def written_scenes(self) -> list[str]:
        """The ids of the scenes that have text, sorted."""
        return sorted(
            PurePosixPath(relative).stem
            for relative in self._library.files_in(self._story_id, MANUSCRIPT_DIR)
            if relative.endswith(".md")
        )

    def sessions(self) -> tuple[SessionsFile, str | None]:
        """The record of the conversations that wrote the story, and its digest.

        Raises:
            StoryFileError: `sessions.yaml` does not fit; the message names the field.
        """
        found = self.read(SESSIONS_FILE)
        if found is None:
            return SessionsFile(), None
        return parse_file(SessionsFile, found.text, SESSIONS_FILE), found.digest

    # -- writing ---------------------------------------------------------------------

    def write(self, relative: str, text: str, *, room_id: str, expected_digest: str | None) -> str:
        """Write a story file as `room_id`; the lease and digest checks apply."""
        return self._library.write_file(
            self._story_id,
            relative,
            text,
            conversation_id=room_id,
            expected_digest=expected_digest,
        )

    def save_outline(self, outline: Outline, *, room_id: str, expected_digest: str | None) -> str:
        return self.write(
            OUTLINE_FILE, dump_file(outline), room_id=room_id, expected_digest=expected_digest
        )

    def write_scene(
        self,
        scene_id: str,
        text: str,
        *,
        room_id: str,
        expected_digest: str | None,
        agent_id: str | None,
        turn_index: int | None,
    ) -> tuple[str, list[str]]:
        """Write a scene's text, keeping the text it replaces; return the digest and notes.

        The replaced text is copied to `manuscript/.history/<scene_id>/<digest>.md` *before*
        the scene is written, so a refused write leaves no history of a change that did not
        happen, and a written one never loses the text before it. The revision is then
        logged. A log that could not be written does not undo the scene: the reason comes
        back in the notes, so the result says it (P6).

        Raises:
            StoryReadOnlyError: `room_id` does not hold the lease.
            StoryChangedError: the scene is not at `expected_digest`.
        """
        target = manuscript_file(scene_id)
        current = self.read(target)
        replaced: str | None = None
        if current is not None and current.digest == expected_digest:
            replaced = f"{HISTORY_DIR}/{scene_id}/{current.digest}.md"
            if self.read(replaced) is None:
                self.write(replaced, current.text, room_id=room_id, expected_digest=None)
        # A digest that does not match is refused here, before anything else is written.
        digest = self.write(target, text, room_id=room_id, expected_digest=expected_digest)
        notes: list[str] = []
        try:
            self._log_revision(
                scene_id,
                Revision(
                    digest=digest,
                    written_at=_now_iso(),
                    room_id=room_id,
                    agent_id=agent_id,
                    turn_index=turn_index,
                    replaced=replaced,
                ),
                room_id=room_id,
            )
        except (StoryError, StoryFileError) as exc:
            notes.append(f"The scene was written, but its revision was not logged: {exc}")
        return digest, notes

    def _log_revision(self, scene_id: str, revision: Revision, *, room_id: str) -> None:
        relative = f"{HISTORY_DIR}/{scene_id}/revisions.yaml"
        found = self.read(relative)
        if found is None:
            log = RevisionLog(scene_id=scene_id)
        else:
            log = parse_file(RevisionLog, found.text, relative)
        updated = log.model_copy(update={"revisions": [*log.revisions, revision]})
        self.write(
            relative,
            dump_file(updated),
            room_id=room_id,
            expected_digest=found.digest if found else None,
        )

    def note_session(
        self,
        room_id: str,
        *,
        scene_written: str | None = None,
        summary: str | None = None,
        closing: bool = False,
    ) -> None:
        """Record in `sessions.yaml` that `room_id` is working on the story.

        The conversation's open entry -- its latest one not yet closed -- is updated, or a
        new one is started. `scene_written` is added to its scenes, `summary` replaces its
        summary, and `closing` closes it.

        Raises:
            StoryFileError: `sessions.yaml` does not fit.
            StoryReadOnlyError, StoryChangedError: see `StoryLibrary.write_file`.
        """
        sessions, digest = self.sessions()
        entries = list(sessions.sessions)
        index = next(
            (
                i
                for i in range(len(entries) - 1, -1, -1)
                if entries[i].room_id == room_id and entries[i].closed_at is None
            ),
            None,
        )
        if index is None:
            if closing:
                return  # nothing open to close
            entries.append(SessionEntry(room_id=room_id, opened_at=_now_iso()))
            index = len(entries) - 1
        entry = entries[index]
        scenes = list(entry.scenes_written)
        if scene_written is not None and scene_written not in scenes:
            scenes.append(scene_written)
        entries[index] = entry.model_copy(
            update={
                "scenes_written": scenes,
                "summary": summary if summary is not None else entry.summary,
                "closed_at": _now_iso() if closing else None,
            }
        )
        self.write(
            SESSIONS_FILE,
            dump_file(SessionsFile(sessions=entries)),
            room_id=room_id,
            expected_digest=digest,
        )
