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
    StoryChangedError,
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
    Proposal,
    Revision,
    RevisionLog,
    SessionEntry,
    SessionsFile,
    StoryAxiom,
    StoryFileError,
    dump_file,
    entry_model,
    parse_file,
    story_axioms,
)

if TYPE_CHECKING:
    from uclone_x.tools.models import ToolContext

__all__ = [
    "OUTLINE_FILE",
    "PROPOSALS_DIR",
    "SESSIONS_FILE",
    "StoryWork",
    "entry_file",
    "manuscript_file",
    "proposal_file",
]

OUTLINE_FILE = "outline.yaml"
SESSIONS_FILE = "sessions.yaml"
CODEX_DIR = "codex"
MANUSCRIPT_DIR = "manuscript"
HISTORY_DIR = f"{MANUSCRIPT_DIR}/.history"
PROPOSALS_DIR = "proposals"
STORY_FILE = "story.yaml"
#: How many times a new proposal's number is retried when another write took it first.
_NUMBER_TRIES = 5
_PROPOSAL_NAME = re.compile(r"^p(\d+)\.yaml$")

_ID = re.compile(ENTRY_ID_PATTERN)


def _rename_target(relative: str) -> str | None:
    """The entry file name `relative`, a file not ending in '.yaml', plainly means, if any.

    `mara.yml`, `mara.yaml~` and `mara.yaml.bak` all mean `mara.yaml`; a name that is not
    an entry id (`Old Map.txt`) means none.
    """
    base = PurePosixPath(PurePosixPath(relative).stem)
    if base.suffix in (".yaml", ".yml"):
        base = PurePosixPath(base.stem)
    return f"{base}.yaml" if _ID.fullmatch(str(base)) else None


def _rename_advice(relative: str, beside: list[str]) -> str:
    """What to rename `relative`, a codex file not ending in '.yaml', to if it is an entry.

    A name is suggested only when `_rename_target` gives one, no other file in the folder
    has that name in any mix of capitals, and `_rename_target` gives no other file there
    the same name, so two files are never both told to become it. Names are compared ignoring case because a Mac or Windows disk treats `Mara.yaml`
    and `mara.yaml` as one file, so the rename would replace it (#1595).
    """
    path = PurePosixPath(relative)
    target = _rename_target(relative)
    if target is not None:
        wanted = target.casefold()
        others = [PurePosixPath(b).name for b in beside if b != relative]
        taken = any(name.casefold() == wanted for name in others)
        shared = any(_rename_target(name) == target for name in others)
        if not taken and not shared:
            return f"If it is an entry, rename it to '{target}'."
    return (
        f"If it is an entry, give it a name ending in '.yaml' that no other file in "
        f"'{path.parent}/' has, even with different capitals."
    )


def manuscript_file(scene_id: str) -> str:
    """The story-relative file holding the text of `scene_id`."""
    return f"{MANUSCRIPT_DIR}/{scene_id}.md"


def entry_file(kind: CodexKind, entry_id: str) -> str:
    """The story-relative file of the codex entry `entry_id` of `kind`."""
    return f"{CODEX_DIR}/{kind}/{entry_id}.yaml"


def proposal_file(proposal_id: str) -> str:
    """The story-relative file of the proposal `proposal_id`."""
    return f"{PROPOSALS_DIR}/{proposal_id}.yaml"


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
        """Every codex entry of every kind; a file that does not load is reported.

        So is a file or folder the codex does not read: one outside the four kind folders,
        a folder inside one, or a file not named `.yaml`. A person who saved `mara.yml` is
        told why Mara is missing, rather than finding nothing (#1576, P6).
        """
        items: list[CodexItem] = []
        unreadable: list[UnreadableFile] = []
        kinds = ", ".join(f"{CODEX_DIR}/{kind}/" for kind in CODEX_KINDS)
        for relative in self._library.files_in(self._story_id, CODEX_DIR):
            unreadable.append(
                UnreadableFile(
                    relative,
                    f"'{relative}' was not read: codex entries are kept in one of the "
                    f"folders {kinds}. Move it into the right one.",
                )
            )
        for relative in self._library.folders_in(self._story_id, CODEX_DIR):
            if PurePosixPath(relative).name in CODEX_KINDS:
                continue
            unreadable.append(
                UnreadableFile(
                    relative,
                    f"The folder '{relative}/' was not read: the codex reads only the "
                    f"folders {kinds}. Move its entries into the right one.",
                )
            )
        for kind in CODEX_KINDS:
            folder = f"{CODEX_DIR}/{kind}"
            for relative in self._library.folders_in(self._story_id, folder):
                unreadable.append(
                    UnreadableFile(
                        relative,
                        f"The folder '{relative}/' was not read: codex entries are files "
                        f"directly in '{folder}/'. Move its entries up into '{folder}/'.",
                    )
                )
            files = self._library.files_in(self._story_id, folder)
            for relative in files:
                name = PurePosixPath(relative)
                if name.suffix != ".yaml":
                    unreadable.append(
                        UnreadableFile(
                            relative,
                            f"'{relative}' was not read: codex entries are read only from "
                            f"files ending in '.yaml'. {_rename_advice(relative, files)}",
                        )
                    )
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

    def entry(self, kind: CodexKind, entry_id: str) -> tuple[CodexEntry, str] | None:
        """The codex entry and its file's digest, or `None` when there is no such file.

        Raises:
            StoryFileError: the file does not fit; the message names the field.
        """
        relative = entry_file(kind, entry_id)
        found = self.read(relative)
        if found is None:
            return None
        return self._entry_at(kind, relative), found.digest

    def axioms(self) -> tuple[list[StoryAxiom], bool]:
        """The rules the story's facts are checked under, and whether they are the defaults.

        Raises:
            StoryFileError: `story.yaml`'s `axioms` does not fit; the message names the field.
        """
        extra = self.record().model_extra or {}
        return story_axioms(extra.get("axioms"), present="axioms" in extra, where=STORY_FILE)

    def proposals(self) -> tuple[list[tuple[Proposal, str]], list[UnreadableFile]]:
        """Every proposal with its digest, oldest first; a file that does not load is reported."""
        loaded: list[tuple[Proposal, str]] = []
        unreadable: list[UnreadableFile] = []
        for relative in self._library.files_in(self._story_id, PROPOSALS_DIR):
            if not relative.endswith(".yaml"):
                continue
            try:
                found = self.read(relative)
                if found is None:
                    continue
                loaded.append((parse_file(Proposal, found.text, relative), found.digest))
            except (StoryError, StoryFileError) as exc:
                problem = UnreadableFile(relative, str(exc))
                unreadable.append(problem)
        return loaded, unreadable

    def proposal(self, proposal_id: str) -> tuple[Proposal, str]:
        """The proposal `proposal_id` and its digest.

        Raises:
            StoryError: there is no such proposal.
            StoryFileError: its file does not fit; the message names the field.
        """
        relative = proposal_file(proposal_id)
        found = self.read(relative) if _ID.fullmatch(proposal_id) else None
        if found is None:
            raise StoryError(f"The story has no proposal '{proposal_id}'.")
        return parse_file(Proposal, found.text, relative), found.digest

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

    def add_proposal(self, draft: Proposal, *, room_id: str) -> Proposal:
        """Save `draft` under the next free number (`p001`, `p002`, ...) and return it.

        The file is written as one that must not exist yet: a number that is already taken
        is refused rather than overwritten, and the next one is tried.

        Raises:
            StoryReadOnlyError: `room_id` does not hold the lease.
        """
        taken = [
            int(match.group(1))
            for relative in self._library.files_in(self._story_id, PROPOSALS_DIR)
            if (match := _PROPOSAL_NAME.match(PurePosixPath(relative).name))
        ]
        number = max(taken, default=0) + 1
        for _ in range(_NUMBER_TRIES):
            proposal = draft.model_copy(update={"id": f"p{number:03d}"})
            try:
                self.write(
                    proposal_file(proposal.id),
                    dump_file(proposal, compact=True),
                    room_id=room_id,
                    expected_digest=None,
                )
            except StoryChangedError:
                number += 1
                continue
            return proposal
        raise StoryError(
            "Other proposals were being saved at the same moment, so this one was not saved. "
            "Try again."
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
        taken_from: str | None = None,
    ) -> None:
        """Record in `sessions.yaml` that `room_id` is working on the story.

        The conversation's open entry -- its latest one not yet closed -- is updated, or a
        new one is started. `scene_written` is added to its scenes, `summary` replaces its
        summary, and `closing` closes it. `taken_from` names the conversation `room_id`
        took the story over from: its open entry is closed in the same write, since it
        can no longer write the story and cannot close its own entry (#1576).

        Raises:
            StoryFileError: `sessions.yaml` does not fit.
            StoryReadOnlyError, StoryChangedError: see `StoryLibrary.write_file`.
        """
        sessions, digest = self.sessions()
        entries = list(sessions.sessions)
        if taken_from is not None and taken_from != room_id:
            entries = [
                e.model_copy(update={"closed_at": _now_iso()})
                if e.room_id == taken_from and e.closed_at is None
                else e
                for e in entries
            ]
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
