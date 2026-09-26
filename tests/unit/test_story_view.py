"""Tests for the story view's Core, `StoryView` (#1560).

What they pin, in order of what it would cost to get wrong:

* **A person's approval applies exactly what they were shown.** Approving writes the
  change and closes the proposal; a proposal whose file changed after it was shown, or
  whose entry changed after it was made, is refused and nothing is written.
* **The lease is honoured.** A decision is written as the conversation writing the story.
  With no writer, a writer that was deleted, or a writer answering right now, it is
  refused with the remedy, and the view says so before anyone presses a button.
* **The view shows what approving would do.** Each value before and after, the file's line
  diff, the quote it rests on and whether the scene still has it, and the conversation
  that proposed it.
* **What cannot be read is said.** No outline, an outline that does not read, and a codex
  or proposal file that does not load each come with their reason.
* **No story or file tool reaches the approval path.** Only the UI routes import this
  module. That says nothing about a shell: a persona with `bash_run` can call the local
  API (#1589).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.story.library import StoryError, StoryLibrary, StoryReadOnlyError
from uclone_x.story.proposals import apply_proposal
from uclone_x.story.schemas import Chapter, Outline, Proposal, Scene, dump_file
from uclone_x.story.view import (
    StoryNotFoundError,
    StoryNotWritableError,
    StoryView,
    WriterBusyError,
)
from uclone_x.story.work import StoryWork

SRC = Path(__file__).resolve().parents[2] / "src" / "uclone_x"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def rooms(tmp_path: Path) -> RoomService:
    return RoomService(RoomStore(tmp_path / "rooms"))


@pytest.fixture
def answering() -> set[str]:
    return set()


@pytest.fixture
def view(workspace: Path, rooms: RoomService, answering: set[str]) -> StoryView:
    return StoryView(workspace, rooms, turn_busy=answering.__contains__)


def _story_dir(workspace: Path, story_id: str) -> Path:
    return workspace / "stories" / story_id


def _put(workspace: Path, story_id: str, relative: str, text: str) -> Path:
    path = _story_dir(workspace, story_id) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _story(workspace: Path, rooms: RoomService) -> tuple[str, str]:
    """The Salt Road, written by “Writing room”: Vane, alive, and a proposal he is wounded.

    Reading order ch01.s01 (day 3), ch01.s02 (no time: continues s01), ch01.s03 (day 1).
    """
    room_id = rooms.create("Writing room").room_id
    story_id = StoryLibrary(workspace).create("The Salt Road", room_id).story_id
    outline = Outline(
        chapters=[
            Chapter(
                id="ch01",
                title="The Crossing",
                act="I",
                scenes=[
                    Scene(id="ch01.s01", title="The gate", story_time="day 3"),
                    Scene(id="ch01.s02", title="The bridge"),
                    Scene(id="ch01.s03", title="Years before", story_time="day 1"),
                ],
            )
        ]
    )
    _put(workspace, story_id, "outline.yaml", dump_file(outline))
    _put(workspace, story_id, "manuscript/ch01.s01.md", "An arrow grazed Lord Vane at the gate.")
    _put(
        workspace,
        story_id,
        "codex/characters/vane.yaml",
        yaml.safe_dump({"id": "vane", "name": "Lord Vane", "state": {"status": "alive"}}),
    )
    _put(
        workspace,
        story_id,
        "codex/places/gate.yaml",
        yaml.safe_dump({"id": "gate", "name": "The Salt Gate"}),
    )
    _propose(workspace, story_id, room_id)
    return story_id, room_id


def _propose(workspace: Path, story_id: str, room_id: str, number: int = 1) -> str:
    library = StoryLibrary(workspace)
    entry_digest = library.read_file(story_id, "codex/characters/vane.yaml").digest
    proposal = Proposal.model_validate(
        {
            "id": f"p{number:03d}",
            "kind": "characters",
            "entry_id": "vane",
            "change": {
                "progression": {
                    "at": "ch01.s01",
                    "set": {"wounded": True},
                    "note": "The arrow grazed him.",
                }
            },
            "evidence": [{"scene_id": "ch01.s01", "quote": "An arrow grazed Lord Vane"}],
            "proposed_at": "2026-09-25T10:00:00+00:00",
            "room_id": room_id,
            "agent_id": "writer",
            "entry_digest": entry_digest,
        }
    )
    _put(workspace, story_id, f"proposals/p{number:03d}.yaml", dump_file(proposal, compact=True))
    return proposal.id


def _vane(workspace: Path, story_id: str) -> str:
    return (_story_dir(workspace, story_id) / "codex/characters/vane.yaml").read_text()


# -- showing ----------------------------------------------------------------------------


def test_the_outline_is_shown_in_reading_order_and_in_story_time(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/view.py :: order = [p.scene_id for p in sorted(placements.values(), key=lambda p: p.position)]
    Becomes: order = list(placements)
    """
    story_id, _ = _story(workspace, rooms)

    shown = view.show(story_id)

    assert shown.title == "The Salt Road"
    assert shown.outline is not None
    chapter = shown.outline.chapters[0]
    assert (chapter.title, chapter.act) == ("The Crossing", "I")
    assert [(s.id, s.written) for s in chapter.scenes] == [
        ("ch01.s01", True),
        ("ch01.s02", False),
        ("ch01.s03", False),
    ]
    assert shown.outline.story_order == ["ch01.s03", "ch01.s01", "ch01.s02"]
    assert shown.outline.assumptions == [
        "ch01.s02 is placed at the same time as 'ch01.s01': it has no story_time of its own."
    ]
    assert shown.outline_note is None


def test_the_codex_is_grouped_by_kind(workspace: Path, rooms: RoomService, view: StoryView) -> None:
    story_id, _ = _story(workspace, rooms)

    groups = {g.kind: [e.name for e in g.entries] for g in view.show(story_id).codex}

    assert groups == {
        "characters": ["Lord Vane"],
        "places": ["The Salt Gate"],
        "items": [],
        "threads": [],
    }


def test_a_pending_proposal_shows_what_approving_would_do(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    story_id, room_id = _story(workspace, rooms)

    shown = view.show(story_id)

    assert shown.decide_note is None
    assert [p.id for p in shown.pending] == ["p001"]
    proposal = shown.pending[0]
    assert proposal.entry_name == "Lord Vane"
    assert proposal.note == "The arrow grazed him."
    assert proposal.proposed_in.model_dump() == {
        "room_id": room_id,
        "title": "Writing room",
        "exists": True,
    }
    assert [(c.what, c.at, c.before, c.after, c.placed) for c in proposal.changes] == [
        ("wounded", "ch01.s01", None, True, True)
    ]
    assert proposal.diff == [
        "--- codex/characters/vane.yaml (now)",
        "+++ codex/characters/vane.yaml (if approved)",
        "@@ -3,2 +3,7 @@",
        " state:",
        "   status: alive",
        "+progressions:",
        "+- at: ch01.s01",
        "+  set:",
        "+    wounded: true",
        "+  note: The arrow grazed him.",
    ]
    evidence = proposal.evidence[0]
    assert (evidence.scene_title, evidence.quote, evidence.still_in_scene) == (
        "The gate",
        "An arrow grazed Lord Vane",
        True,
    )
    assert proposal.blocked is None
    assert (
        proposal.digest == StoryLibrary(workspace).read_file(story_id, "proposals/p001.yaml").digest
    )


def test_a_change_to_how_a_character_looks_shows_the_looks_before_and_after(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/proposals.py :: after_tags = kept + [t for t in vp.add_tags if t not in kept]
    Becomes: after_tags = list(vp.add_tags)
    """
    story_id, room_id = _story(workspace, rooms)
    _put(
        workspace,
        story_id,
        "codex/characters/mara.yaml",
        yaml.safe_dump({"id": "mara", "name": "Mara", "visual": {"tags": ["cloak", "red hair"]}}),
    )
    digest = StoryLibrary(workspace).read_file(story_id, "codex/characters/mara.yaml").digest
    proposal = Proposal.model_validate(
        {
            "id": "p002",
            "kind": "characters",
            "entry_id": "mara",
            "change": {
                "visual_progression": {
                    "at": "ch01.s02",
                    "add_tags": ["scar"],
                    "remove_tags": ["cloak"],
                },
                "visual": {"prose": "tall and grim"},
            },
            "proposed_at": "2026-09-25T10:00:00+00:00",
            "room_id": room_id,
            "entry_digest": digest,
        }
    )
    _put(workspace, story_id, "proposals/p002.yaml", dump_file(proposal, compact=True))

    shown = [p for p in view.show(story_id).pending if p.id == "p002"][0]

    assert [(c.what, c.at, c.before, c.after) for c in shown.changes] == [
        ("looks", "ch01.s02", ["cloak", "red hair"], ["red hair", "scar"]),
        ("looks: prose", None, None, "tall and grim"),
    ]
    assert shown.evidence == []


def test_a_quote_the_scene_no_longer_has_is_shown_as_gone(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/view.py :: still_in_scene=passage_in(item.quote, text.text) if text else None,
    Becomes: still_in_scene=True if text else None,
    """
    story_id, _ = _story(workspace, rooms)
    _put(workspace, story_id, "manuscript/ch01.s01.md", "Lord Vane passed the gate unharmed.")

    assert view.show(story_id).pending[0].evidence[0].still_in_scene is False


def test_a_passage_quoted_before_quotes_were_whole_words_is_still_in_the_scene(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """#1601: a proposal made before #1597 may quote two letters or part of a word.

    Those words are still in the scene, so the view does not say they are gone.

    Killed by: src/uclone_x/story/quotes.py :: return bool(needle) and needle in folded(text)
    Becomes: return quote_found(passage, text)
    """
    story_id, _ = _story(workspace, rooms)
    path = _story_dir(workspace, story_id) / "proposals/p001.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["evidence"] = [
        {"scene_id": "ch01.s01", "quote": "An"},
        {"scene_id": "ch01.s01", "quote": "arrow graz"},
        {"scene_id": "ch01.s01", "quote": "arrow missed"},
    ]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    evidence = view.show(story_id).pending[0].evidence
    assert [e.still_in_scene for e in evidence] == [True, True, False]


def test_an_entry_edited_since_the_proposal_is_shown_blocked_and_not_applied(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/proposals.py :: if digest == proposal.entry_digest
    Becomes: if True
    """
    story_id, _ = _story(workspace, rooms)
    _put(
        workspace,
        story_id,
        "codex/characters/vane.yaml",
        yaml.safe_dump({"id": "vane", "name": "Vane the Grey"}),
    )
    shown = view.show(story_id).pending[0]
    assert shown.blocked == (
        "It cannot be approved: the entry 'vane' changed after this was proposed. Reject it, "
        "and ask for the change again if it is still wanted."
    )

    with pytest.raises(StoryError) as refused:
        view.approve(story_id, "p001", seen_digest=shown.digest)

    assert str(refused.value) == (
        "The entry 'vane' changed after proposal 'p001' was made, so it was not applied. "
        "Reject it and propose the change again."
    )
    assert "Vane the Grey" in _vane(workspace, story_id)


def test_absences_state_their_cause(workspace: Path, rooms: RoomService, view: StoryView) -> None:
    story_id, _ = _story(workspace, rooms)
    (_story_dir(workspace, story_id) / "outline.yaml").unlink()
    _put(workspace, story_id, "codex/items/sword.yaml", "id: sword\n")
    _put(workspace, story_id, "proposals/p002.yaml", "not: [a proposal\n")

    shown = view.show(story_id)

    assert shown.outline is None
    assert shown.outline_note == "This story has no outline yet."
    assert [u.file for u in shown.codex_unreadable] == ["codex/items/sword.yaml"]
    assert "name" in shown.codex_unreadable[0].reason
    assert [u.file for u in shown.proposals_unreadable] == ["proposals/p002.yaml"]
    assert shown.proposals_unreadable[0].reason.startswith("proposals/p002.yaml could not be read")

    _put(workspace, story_id, "outline.yaml", "chapters: 3\n")
    assert view.show(story_id).outline_note is not None
    assert view.show(story_id).outline_note.startswith("The outline could not be read: ")  # type: ignore[union-attr]


def test_a_story_that_is_gone_is_named(view: StoryView) -> None:
    with pytest.raises(StoryNotFoundError) as refused:
        view.show("night-train")

    assert str(refused.value) == (
        "There is no story called 'night-train' any more. Refresh the list to see the stories "
        "there are."
    )


# -- deciding ---------------------------------------------------------------------------


def test_approving_applies_the_change_as_the_writer_and_records_where(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/proposals.py :: update={"status": "applied", "decided_at": _now(), "decided_in": decided_in}
    Becomes: update={"status": "applied", "decided_at": _now()}
    """
    story_id, _ = _story(workspace, rooms)
    seen = view.show(story_id).pending[0].digest

    decided = view.approve(story_id, "p001", seen_digest=seen)

    assert (decided.decision, decided.notes) == ("applied", [])
    vane = yaml.safe_load(_vane(workspace, story_id))
    assert vane["progressions"] == [
        {"at": "ch01.s01", "set": {"wounded": True}, "note": "The arrow grazed him."}
    ]
    shown = view.show(story_id)
    assert shown.pending == []
    assert [(p.id, p.status, p.decided_in) for p in shown.decided] == [
        ("p001", "applied", "story_view")
    ]
    with pytest.raises(StoryError) as again:
        view.approve(story_id, "p001", seen_digest=seen)
    assert str(again.value) == "Proposal 'p001' was already applied, so nothing was changed."


def test_a_proposal_changed_after_it_was_shown_is_not_applied(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """The person decides what they saw, not what the file says a moment later.

    Killed by: src/uclone_x/story/proposals.py :: if expected_digest is not None and digest != expected_digest:
    Becomes: if False:
    """
    story_id, _ = _story(workspace, rooms)
    seen = view.show(story_id).pending[0].digest
    path = _story_dir(workspace, story_id) / "proposals/p001.yaml"
    path.write_text(path.read_text().replace("wounded: true", "wounded: false"))
    before = _vane(workspace, story_id)

    for decide in (
        lambda: view.approve(story_id, "p001", seen_digest=seen),
        lambda: view.reject(story_id, "p001", seen_digest=seen, reason=None),
    ):
        with pytest.raises(StoryError) as refused:
            decide()
        assert str(refused.value) == (
            "Proposal 'p001' changed after it was shown, so nothing was changed. Look at it "
            "again before deciding."
        )
    assert _vane(workspace, story_id) == before
    assert "status:" not in path.read_text()  # still pending: the default is not written


def test_rejecting_keeps_the_reason_and_changes_no_entry(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/view.py :: reason=reason.strip() if reason and reason.strip() else None,
    Becomes: reason=reason,
    """
    story_id, _ = _story(workspace, rooms)
    before = _vane(workspace, story_id)
    seen = view.show(story_id).pending[0].digest

    decided = view.reject(story_id, "p001", seen_digest=seen, reason="  Too early.  ")

    assert decided.decision == "rejected"
    assert _vane(workspace, story_id) == before
    rejected = view.show(story_id).decided[0]
    assert (rejected.status, rejected.reason, rejected.decided_in) == (
        "rejected",
        "Too early.",
        "story_view",
    )


def test_with_no_writer_nothing_is_saved_and_the_view_says_why(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    story_id, room_id = _story(workspace, rooms)
    StoryLibrary(workspace).release(story_id, room_id)
    shown = view.show(story_id)
    expected = (
        "No conversation is writing “The Salt Road”, so a decision cannot be saved. Open the "
        "story in a conversation, then decide here."
    )
    assert shown.writer is None
    assert shown.decide_note == expected

    with pytest.raises(StoryNotWritableError) as refused:
        view.approve(story_id, "p001", seen_digest=shown.pending[0].digest)

    assert str(refused.value) == expected
    assert "progressions" not in _vane(workspace, story_id)


def test_a_writer_that_was_deleted_holds_nothing_to_write_with(
    workspace: Path, rooms: RoomService, view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/view.py :: if note is not None or writer is None:
    Becomes: if writer is None:
    """
    story_id, room_id = _story(workspace, rooms)
    seen = view.show(story_id).pending[0].digest
    rooms.delete(room_id)
    expected = (
        "The conversation that was writing “The Salt Road” no longer exists, so a decision "
        "cannot be saved. Open the story in a conversation, then decide here."
    )
    assert view.show(story_id).decide_note == expected

    with pytest.raises(StoryNotWritableError) as refused:
        view.reject(story_id, "p001", seen_digest=seen, reason=None)

    assert str(refused.value) == expected


def test_a_writer_answering_right_now_is_waited_for(
    workspace: Path, rooms: RoomService, answering: set[str], view: StoryView
) -> None:
    """Killed by: src/uclone_x/story/view.py :: if self._turn_busy(writer.room_id):
    Becomes: if False:
    """
    story_id, room_id = _story(workspace, rooms)
    seen = view.show(story_id).pending[0].digest
    answering.add(room_id)

    with pytest.raises(WriterBusyError) as refused:
        view.approve(story_id, "p001", seen_digest=seen)

    assert str(refused.value) == (
        "The conversation “Writing room” is answering right now, so the decision was not "
        "saved. Wait for the answer to finish, then try again."
    )
    assert "progressions" not in _vane(workspace, story_id)


def test_the_write_itself_still_checks_the_lease(workspace: Path, rooms: RoomService) -> None:
    """The Core write refuses a conversation that does not hold the lease, whoever calls it."""
    story_id, _ = _story(workspace, rooms)
    other = rooms.create("Another room").room_id

    with pytest.raises(StoryReadOnlyError):
        apply_proposal(
            StoryWork(StoryLibrary(workspace), story_id),
            "p001",
            room_id=other,
            decided_in="story_view",
        )
    assert "progressions" not in _vane(workspace, story_id)


def test_no_tool_reaches_the_approval_path() -> None:
    """No tool module imports the approval path: only the UI's routes import this module.

    This pins imports, not reachability: a persona with an unconfined shell can still
    call the local API (#1589 item 6)."""
    importers = sorted(
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if "from uclone_x.story.view import" in path.read_text(encoding="utf-8")
        or "import uclone_x.story.view" in path.read_text(encoding="utf-8")
    )
    assert importers == ["ui/artifacts.py"]
