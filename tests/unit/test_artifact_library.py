"""Tests for the Files screen's Core service, `ArtifactLibrary` (#1554).

What they pin, in order of what it would cost to get wrong:

* **A deleted conversation's files are still there to find.** Listed, openable, and shown as
  not linked to a conversation that still exists -- never as written by nobody.
* **A story being written is not taken silently.** Archiving or deleting a story whose
  writer still exists is refused, naming the conversation, until the caller says to go
  ahead; a writer that no longer exists holds nothing.
* **Nothing outside the artifact folders moves.** A path outside the workspace, outside
  the folders, the folder itself, or one file of a story is refused, and the file stays.
* **Archive is reversible and delete asks first.** Both refusals say what happened in
  plain words, which the tests pin exactly.
* **The list says what it covers.** The scope sentence is pinned, and an unreadable
  conversation is counted rather than dropped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from uclone_x.artifacts.library import (
    SCOPE_NOTE,
    ArtifactEntry,
    ArtifactError,
    ArtifactLibrary,
    ArtifactNotFoundError,
    DeleteNotConfirmedError,
    StoryInUseError,
)
from uclone_x.room.models import RoomWrittenFile
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.story.library import StoryLibrary


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def store(tmp_path: Path) -> RoomStore:
    return RoomStore(tmp_path / "rooms")


@pytest.fixture
def rooms(store: RoomStore) -> RoomService:
    return RoomService(store)


@pytest.fixture
def answering() -> set[str]:
    """The conversations the fixture library is told are answering right now."""
    return set()


@pytest.fixture
def library(workspace: Path, rooms: RoomService, answering: set[str]) -> ArtifactLibrary:
    return ArtifactLibrary(workspace, rooms, turn_in_flight=answering.__contains__)


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.write_text("", encoding="utf-8")
    try:
        return (directory / "CASE-PROBE").exists()
    finally:
        probe.unlink()


def _write(workspace: Path, relative: str, text: str = "hello") -> Path:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _room_that_wrote(rooms: RoomService, store: RoomStore, title: str, *paths: str) -> str:
    state = rooms.create(title)
    written = tuple(
        RoomWrittenFile(path=p, participant_id="writer", tool_name="write_file", turn_id="t1")
        for p in paths
    )
    store.save(state.model_copy(update={"written_files": written}))
    return state.room_id


def _entry(library: ArtifactLibrary, path: str) -> ArtifactEntry:
    matches = [e for e in library.survey().entries if e.path == path]
    assert len(matches) == 1, [e.path for e in library.survey().entries]
    return matches[0]


# -- listing ----------------------------------------------------------------------------


def test_a_deleted_conversations_file_is_listed_and_opens(
    workspace: Path, rooms: RoomService, store: RoomStore, library: ArtifactLibrary
) -> None:
    _write(workspace, "artifacts/notes/plan.md", "# Plan")
    room_id = _room_that_wrote(rooms, store, "Planning", "artifacts/notes/plan.md")
    assert _entry(library, "artifacts/notes/plan.md").conversations[0].title == "Planning"

    rooms.delete(room_id)

    entry = _entry(library, "artifacts/notes/plan.md")
    assert entry.conversations == ()
    assert (entry.kind, entry.archived, entry.managed) == ("document", False, True)
    assert library.open_file("artifacts/notes/plan.md").text == "# Plan"


def test_the_list_states_what_it_covers(library: ArtifactLibrary) -> None:
    survey = library.survey()
    assert survey.scope_note == SCOPE_NOTE
    assert SCOPE_NOTE == (
        "This list shows the files in the workspace's artifacts and stories folders, and "
        "files a current conversation saved by name somewhere else. A file saved somewhere "
        "else by a conversation that was since deleted, or written by a shell command, may "
        "not appear here. A file is linked to a conversation only while that conversation "
        "exists."
    )
    assert survey.record_gaps == ()


def test_an_unreadable_conversation_is_counted_not_dropped(
    tmp_path: Path, library: ArtifactLibrary
) -> None:
    (tmp_path / "rooms").mkdir(exist_ok=True)
    (tmp_path / "rooms" / "broken.json").write_text("{not json", encoding="utf-8")

    assert library.survey().record_gaps == (
        "1 conversation(s) could not be read, so the files they saved are not linked to them here.",
    )


def test_a_file_a_live_conversation_saved_elsewhere_is_listed_but_not_managed(
    workspace: Path, rooms: RoomService, store: RoomStore, library: ArtifactLibrary
) -> None:
    _write(workspace, "scene_01.txt", "Chapter one")
    _room_that_wrote(rooms, store, "Draft", "scene_01.txt")

    entry = _entry(library, "scene_01.txt")
    assert entry.managed is False
    assert library.open_file("scene_01.txt").text == "Chapter one"
    with pytest.raises(ArtifactError) as refused:
        library.archive("scene_01.txt")
    assert str(refused.value) == (
        "Only files in the artifacts and stories folders can be archived or deleted here."
    )


def test_a_file_nobody_recorded_outside_the_folders_is_neither_listed_nor_opened(
    workspace: Path, library: ArtifactLibrary
) -> None:
    _write(workspace, "secrets.txt")

    assert [e.path for e in library.survey().entries] == []
    with pytest.raises(ArtifactError) as refused:
        library.open_file("secrets.txt")
    assert str(refused.value) == "That file is not one the clones saved, so it is not opened here."


def test_links_and_dot_names_are_not_followed_and_the_gap_says_so(
    workspace: Path, tmp_path: Path, library: ArtifactLibrary
) -> None:
    outside = _write(tmp_path, "elsewhere.md")
    _write(workspace, "artifacts/.hidden.md")
    (workspace / "artifacts" / "link.md").symlink_to(outside)

    survey = library.survey()
    assert [e.path for e in survey.entries] == []
    assert survey.record_gaps == (
        "1 linked file(s) or folder(s) in the artifact folders were not followed, so what "
        "they point to is not listed.",
    )


def test_a_file_that_is_not_text_is_refused_plainly(
    workspace: Path, library: ArtifactLibrary
) -> None:
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "blob.bin").write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ArtifactError) as refused:
        library.open_file("artifacts/blob.bin")
    assert str(refused.value) == "blob.bin is not text, so it cannot be shown here."


def test_an_image_opens_as_its_kind_without_text(workspace: Path, library: ArtifactLibrary) -> None:
    (workspace / "artifacts" / "images").mkdir(parents=True)
    (workspace / "artifacts" / "images" / "a.png").write_bytes(b"\x89PNG")

    opened = library.open_file("artifacts/images/a.png")
    assert (opened.kind, opened.text) == ("image", None)


# -- archive, restore, delete ----------------------------------------------------------


def test_archive_is_reversible(workspace: Path, library: ArtifactLibrary) -> None:
    _write(workspace, "artifacts/a.md", "kept")

    moved = library.archive("artifacts/a.md")

    assert moved == ".archive/artifacts/a.md"
    assert not (workspace / "artifacts" / "a.md").exists()
    entry = _entry(library, ".archive/artifacts/a.md")
    assert (entry.archived, entry.original_path) == (True, "artifacts/a.md")
    assert library.open_file(".archive/artifacts/a.md").text == "kept"

    assert library.restore(".archive/artifacts/a.md") == "artifacts/a.md"
    assert (workspace / "artifacts" / "a.md").read_text(encoding="utf-8") == "kept"
    assert not (workspace / ".archive" / "artifacts" / "a.md").exists()


def test_neither_archive_nor_restore_overwrites(workspace: Path, library: ArtifactLibrary) -> None:
    _write(workspace, "artifacts/a.md", "second")
    _write(workspace, ".archive/artifacts/a.md", "first")

    with pytest.raises(ArtifactError) as archive_refused:
        library.archive("artifacts/a.md")
    assert str(archive_refused.value) == (
        "An archived copy of a.md is already there. Restore or delete that copy first."
    )
    with pytest.raises(ArtifactError) as restore_refused:
        library.restore(".archive/artifacts/a.md")
    assert str(restore_refused.value) == (
        "Something is already where a.md was. Move or delete it, then restore."
    )
    assert (workspace / "artifacts" / "a.md").read_text(encoding="utf-8") == "second"
    assert (workspace / ".archive" / "artifacts" / "a.md").read_text(encoding="utf-8") == "first"


def test_delete_needs_confirmation(workspace: Path, library: ArtifactLibrary) -> None:
    """Deleting cannot be undone, so an unconfirmed request removes nothing.

    Killed by: src/uclone_x/artifacts/library.py :: if not confirm:
    Becomes: if False:
    """
    path = _write(workspace, "artifacts/a.md")

    with pytest.raises(DeleteNotConfirmedError) as refused:
        library.delete("artifacts/a.md", confirm=False)
    assert str(refused.value) == (
        "Deleting cannot be undone, so it needs your confirmation. Nothing was deleted."
    )
    assert path.exists()

    library.delete("artifacts/a.md", confirm=True)
    assert not path.exists()


def test_an_archived_file_can_be_deleted(workspace: Path, library: ArtifactLibrary) -> None:
    _write(workspace, "artifacts/a.md")
    library.archive("artifacts/a.md")

    library.delete(".archive/artifacts/a.md", confirm=True)

    assert library.survey().entries == ()


@pytest.mark.parametrize(
    ("path", "sentence"),
    [
        ("../outside.md", "That path is outside the workspace, so nothing was changed."),
        (
            "notes.md",
            "Only files in the artifacts and stories folders can be archived or deleted here.",
        ),
        (
            "artifacts",
            "Only files in the artifacts and stories folders can be archived or deleted here.",
        ),
        ("", "No file was named, so nothing was changed."),
    ],
)
def test_nothing_outside_the_artifact_folders_moves(
    workspace: Path, tmp_path: Path, library: ArtifactLibrary, path: str, sentence: str
) -> None:
    _write(tmp_path, "outside.md")
    _write(workspace, "notes.md")
    _write(workspace, "artifacts/a.md")

    with pytest.raises(ArtifactError) as refused:
        library.delete(path, confirm=True)
    assert str(refused.value) == sentence
    assert (tmp_path / "outside.md").exists()
    assert (workspace / "notes.md").exists()
    assert (workspace / "artifacts" / "a.md").exists()


def test_a_link_is_not_changed(workspace: Path, library: ArtifactLibrary) -> None:
    target = _write(workspace, "artifacts/real.md")
    (workspace / "artifacts" / "alias.md").symlink_to(target)

    with pytest.raises(ArtifactError) as refused:
        library.delete("artifacts/alias.md", confirm=True)
    assert (
        str(refused.value) == "That path is a link to somewhere else, so it was not changed here."
    )
    assert target.exists()


def test_a_missing_file_says_to_refresh(library: ArtifactLibrary) -> None:
    with pytest.raises(ArtifactNotFoundError) as refused:
        library.archive("artifacts/gone.md")
    assert str(refused.value) == (
        "There is no file called gone.md there any more. Refresh the list to see what is there."
    )


# -- stories ----------------------------------------------------------------------------


def _story(workspace: Path, rooms: RoomService, title: str) -> tuple[str, str]:
    """A story whose lease is held by a new conversation titled `title`; (story, room)."""
    room_id = rooms.create(title).room_id
    record = StoryLibrary(workspace).create("Night Train", room_id)
    rooms.set_story(room_id, record.story_id)
    _write(workspace, f"stories/{record.story_id}/chapters/01.md", "It was late.")
    return record.story_id, room_id


def _lease_holder(folder: Path) -> str | None:
    raw = yaml.safe_load((folder / "story.yaml").read_text(encoding="utf-8"))
    lease = raw.get("lease")
    return None if lease is None else str(lease["holder"])


def test_a_story_is_one_entry_with_its_files_and_writer(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, room_id = _story(workspace, rooms, "Writing room")

    entry = _entry(library, f"stories/{story_id}")
    assert entry.kind == "story"
    assert entry.name == "Night Train"
    assert entry.story is not None
    assert [f.name for f in entry.story.files] == ["chapters/01.md", "story.yaml"]
    assert entry.story.writer is not None
    assert (entry.story.writer.room_id, entry.story.writer.title, entry.story.writer.exists) == (
        room_id,
        "Writing room",
        True,
    )
    assert [c.room_id for c in entry.conversations] == [room_id]


def test_one_file_of_a_story_is_not_moved_alone(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, _ = _story(workspace, rooms, "Writing room")

    with pytest.raises(ArtifactError) as refused:
        library.delete(f"stories/{story_id}/chapters/01.md", confirm=True)
    assert str(refused.value) == (
        f"01.md is part of the story '{story_id}'. Archive or delete the whole story instead."
    )


def test_a_story_being_written_is_not_archived_silently(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    """A story a live conversation holds is refused, not moved out from under it.

    Killed by: src/uclone_x/artifacts/library.py :: if exists and not release_writer:
    Becomes: if False:
    """
    story_id, room_id = _story(workspace, rooms, "Writing room")

    with pytest.raises(StoryInUseError) as refused:
        library.archive(f"stories/{story_id}")
    assert str(refused.value) == (
        "The conversation “Writing room” is writing this story. Delete that conversation "
        "first, or go ahead anyway to stop it writing to this story."
    )
    assert refused.value.room_id == room_id
    assert (workspace / "stories" / story_id).is_dir()

    with pytest.raises(StoryInUseError):
        library.delete(f"stories/{story_id}", confirm=True)
    assert (workspace / "stories" / story_id).is_dir()


def test_going_ahead_releases_the_writer_and_clears_its_story(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, room_id = _story(workspace, rooms, "Writing room")

    moved = library.archive(f"stories/{story_id}", release_writer=True)

    assert moved == f".archive/stories/{story_id}"
    assert _lease_holder(workspace / moved) is None
    assert rooms.get(room_id).story_id is None
    assert _entry(library, moved).story is not None

    assert library.restore(moved) == f"stories/{story_id}"
    assert (workspace / "stories" / story_id / "chapters" / "01.md").exists()


def test_going_ahead_is_refused_while_the_writer_is_answering(
    workspace: Path,
    rooms: RoomService,
    store: RoomStore,
    library: ArtifactLibrary,
    answering: set[str],
) -> None:
    """Releasing mid-turn would change the room under the turn and lose its reply.

    Killed by: src/uclone_x/artifacts/library.py :: if exists and self._turn_in_flight(holder):
    Becomes: if False:
    """
    story_id, room_id = _story(workspace, rooms, "Writing room")
    held = rooms.get(room_id)  # what the running turn loaded, and will save against
    answering.add(room_id)

    for attempt in (
        lambda: library.archive(f"stories/{story_id}", release_writer=True),
        lambda: library.delete(f"stories/{story_id}", confirm=True, release_writer=True),
    ):
        with pytest.raises(StoryInUseError) as refused:
            attempt()
        assert str(refused.value) == (
            "The conversation “Writing room” is answering right now, so it cannot be stopped "
            "from writing this story yet. Wait for the answer to finish, then try again."
        )
    assert (workspace / "stories" / story_id).is_dir()
    assert _lease_holder(workspace / "stories" / story_id) == room_id

    store.save(held)  # the turn lands: the room was not changed under it
    assert rooms.get(room_id).story_id == story_id

    answering.clear()
    assert library.archive(f"stories/{story_id}", release_writer=True) == (
        f".archive/stories/{story_id}"
    )


def test_a_story_named_in_another_case_is_still_the_story(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    """On a disk that ignores case, the lease is checked on the folder, not the typing."""
    story_id, room_id = _story(workspace, rooms, "Writing room")
    if not _case_insensitive(workspace):
        pytest.skip("this disk tells the cases apart, so the other spelling names nothing")

    shouted = f"stories/{story_id.upper()}"
    with pytest.raises(StoryInUseError):
        library.archive(shouted)
    with pytest.raises(StoryInUseError):
        library.delete(shouted, confirm=True)
    with pytest.raises(ArtifactError) as member:
        library.delete(f"STORIES/{story_id.upper()}/chapters/01.md", confirm=True)
    assert str(member.value).startswith("01.md is part of the story")
    assert _lease_holder(workspace / "stories" / story_id) == room_id

    assert library.archive(shouted, release_writer=True) == f".archive/stories/{story_id}"
    assert _entry(library, f".archive/stories/{story_id}").story is not None


def test_a_move_through_a_planted_link_is_refused(
    tmp_path: Path, workspace: Path, library: ArtifactLibrary
) -> None:
    """Archive and restore land only inside the workspace, where they can be undone.

    Killed by: src/uclone_x/artifacts/library.py :: resolved = PathValidator().resolve_safe_path(target, self._workspace)
    Becomes: resolved = target
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(workspace, "artifacts/c.md", "mine")
    (workspace / ".archive").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactError) as refused:
        library.archive("artifacts/c.md")
    assert str(refused.value) == (
        "The place c.md would move to is reached through a link, so it was not moved."
    )
    assert (workspace / "artifacts" / "c.md").read_text(encoding="utf-8") == "mine"
    assert list(outside.iterdir()) == []

    (workspace / ".archive").unlink()
    _write(workspace, ".archive/artifacts/sub/d.md", "archived")
    (workspace / "artifacts" / "sub").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactError) as restore_refused:
        library.restore(".archive/artifacts/sub/d.md")
    assert str(restore_refused.value) == (
        "The place d.md would move to is reached through a link, so it was not moved."
    )
    assert (workspace / ".archive" / "artifacts" / "sub" / "d.md").is_file()
    assert list(outside.iterdir()) == []


def test_a_link_that_stays_inside_the_workspace_is_refused_too(
    workspace: Path, library: ArtifactLibrary
) -> None:
    """Inside is not enough: a linked `.archive` would file the copy where restore can't find it.

    Killed by: src/uclone_x/artifacts/library.py :: if resolved != target:
    Becomes: if False:
    """
    _write(workspace, "artifacts/c.md", "mine")
    (workspace / "elsewhere").mkdir()
    (workspace / ".archive").symlink_to(workspace / "elsewhere", target_is_directory=True)

    with pytest.raises(ArtifactError) as refused:
        library.archive("artifacts/c.md")
    assert str(refused.value) == (
        "The place c.md would move to is reached through a link, so it was not moved."
    )
    assert (workspace / "artifacts" / "c.md").is_file()
    assert list((workspace / "elsewhere").iterdir()) == []


def test_a_writer_that_no_longer_exists_holds_nothing(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, room_id = _story(workspace, rooms, "Writing room")
    rooms.delete(room_id)

    entry = _entry(library, f"stories/{story_id}")
    assert entry.story is not None and entry.story.writer is not None
    assert (entry.story.writer.exists, entry.story.writer.title) == (False, None)

    library.delete(f"stories/{story_id}", confirm=True)
    assert not (workspace / "stories" / story_id).exists()


def test_an_unreadable_story_is_listed_with_its_reason(
    workspace: Path, library: ArtifactLibrary
) -> None:
    _write(workspace, "stories/broken/story.yaml", "- not a mapping")

    entry = _entry(library, "stories/broken")
    assert entry.story is not None
    assert entry.story.title is None
    assert (
        entry.story.unreadable_reason == "stories/broken/story.yaml is not a set of named fields."
    )


def test_a_story_opens_writable_in_a_new_conversation(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, old_room = _story(workspace, rooms, "Writing room")
    library.archive(f"stories/{story_id}", release_writer=True)
    library.restore(f".archive/stories/{story_id}")
    new_room = rooms.create("New conversation").room_id

    opened = library.open_story_in_conversation(story_id, new_room)

    assert (opened.writable, opened.note, opened.title) == (True, None, "Night Train")
    assert rooms.get(new_room).story_id == story_id
    assert _lease_holder(workspace / "stories" / story_id) == new_room
    assert rooms.get(old_room).story_id is None


def test_a_story_another_live_conversation_is_writing_opens_to_read(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, old_room = _story(workspace, rooms, "Writing room")
    new_room = rooms.create("New conversation").room_id

    opened = library.open_story_in_conversation(story_id, new_room)

    assert opened.writable is False
    assert opened.note == (
        "The conversation “Writing room” is writing this story, so this one can read it but "
        "not change it."
    )
    assert rooms.get(new_room).story_id == story_id
    assert _lease_holder(workspace / "stories" / story_id) == old_room


def test_a_stale_lease_is_taken_over_when_the_story_opens(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    story_id, old_room = _story(workspace, rooms, "Writing room")
    rooms.delete(old_room)
    new_room = rooms.create("New conversation").room_id

    opened = library.open_story_in_conversation(story_id, new_room)

    assert (opened.writable, opened.note) == (True, None)
    assert _lease_holder(workspace / "stories" / story_id) == new_room


def test_opening_a_missing_story_or_into_a_missing_conversation_is_refused_plainly(
    workspace: Path, rooms: RoomService, library: ArtifactLibrary
) -> None:
    room_id = rooms.create("New conversation").room_id
    with pytest.raises(ArtifactNotFoundError) as no_story:
        library.open_story_in_conversation("no-such-story", room_id)
    assert str(no_story.value) == (
        "There is no story called 'no-such-story' any more. Refresh the list to see the "
        "stories there are."
    )

    story_id, _ = _story(workspace, rooms, "Writing room")
    with pytest.raises(ArtifactNotFoundError) as no_room:
        library.open_story_in_conversation(story_id, "room-gone")
    assert str(no_room.value) == "That conversation no longer exists, so the story was not opened."


def test_the_survey_is_newest_first(workspace: Path, library: ArtifactLibrary) -> None:
    older = _write(workspace, "artifacts/older.md")
    newer = _write(workspace, "artifacts/newer.md")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert [e.path for e in library.survey().entries] == [
        "artifacts/newer.md",
        "artifacts/older.md",
    ]
