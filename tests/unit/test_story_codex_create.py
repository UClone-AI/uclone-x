"""New codex entries, proposed by the Writer and added by a person (#1808).

A story the Writer starts has no codex, and no file tool writes one, so the scene
context, the audit and the conflict checks had nothing to read. `story_codex 'create'`
proposes new entries through the same proposals a change goes through.

What these pin, in order of what it would cost to get wrong:

* **Nothing is added until a person approves it**, and a rejected entry writes nothing.
* **An added entry is an entry**: `get`, `for_scene` and the audit read it.
* **No second entry answers to a name one already has**, in the codex or waiting.
* **A person sees what they approve**: every value, and the whole file, as added.
* **Refusals are plain words.**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.story.library import StoryLibrary
from uclone_x.story.names import id_for_name
from uclone_x.story.proposals import preview_proposal
from uclone_x.story.tool import StoryLibraryTool
from uclone_x.story.tools import (
    StoryAuditTool,
    StoryCodexTool,
    StoryContextTool,
    StoryManuscriptTool,
    StoryOutlineTool,
)
from uclone_x.story.view import StoryView
from uclone_x.story.work import StoryWork
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import NoIsolation, ToolContext, ToolResult

ROOM = "room_a"


def _ctx(workspace: Path, *, story: str | None = None, approved: bool = False) -> ToolContext:
    return ToolContext(
        agent_id="writer",
        session_id=f"sess_room__{ROOM}__writer",
        workspace_root=workspace,
        room_id=ROOM,
        story_id=story,
        isolation=NoIsolation(),
        approved_by_person=approved,
    )


async def _call(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> ToolResult:
    return await tool.execute(args, ctx)


async def _ok(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> dict[str, Any]:
    result = await _call(tool, ctx, **args)
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


async def _refused(tool: BaseTool[Any], ctx: ToolContext, **args: Any) -> str:
    result = await _call(tool, ctx, **args)
    assert not result.success, result.output
    assert result.error is not None
    return result.error


MARA: dict[str, Any] = {
    "kind": "characters",
    "name": "Mara",
    "aliases": ["the ferrywoman"],
    "profile": "Keeps the ferry at the salt crossing.",
    "state": {"status": "alive"},
    "appearance": "Grey braid, a burn on her left hand.",
    "gender": "female",
}
CROSSING: dict[str, Any] = {"kind": "places", "name": "Salt Crossing"}


async def _new_story(workspace: Path) -> tuple[str, ToolContext]:
    """A story with no codex, one chapter of two scenes, and the second scene written."""
    out = await _ok(StoryLibraryTool(), _ctx(workspace), action="create", title="The Salt Road")
    story_id = out["open_story_id"]
    ctx = _ctx(workspace, story=story_id)
    outline = StoryOutlineTool()
    await _ok(outline, ctx, action="init", chapter_titles=["The Crossing"])
    await _ok(
        outline, ctx, action="set_scene", chapter_id="ch01", title="Dusk", characters=["mara"]
    )
    await _ok(outline, ctx, action="set_scene", chapter_id="ch01", title="Night")
    await _ok(
        StoryManuscriptTool(),
        ctx,
        action="write",
        scene_id="ch01.s02",
        text="Mara lay still in the reeds, and she did not rise again.",
    )
    return story_id, ctx


def _codex_dir(workspace: Path, story_id: str) -> Path:
    return workspace / "stories" / story_id / "codex"


def _entry_files(workspace: Path, story_id: str) -> list[str]:
    root = _codex_dir(workspace, story_id)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.yaml"))


class TestCreateProposesAndAPersonAdds:
    async def test_create_proposes_and_writes_no_entry(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/tools.py :: saved = [work.add_proposal(draft, room_id=room_id) for draft in drafts]
        Becomes: saved = [draft for draft in drafts]
        """
        story_id, ctx = await _new_story(tmp_path)
        out = await _ok(StoryCodexTool(), ctx, action="create", entries=[MARA, CROSSING])
        assert out["proposed"] == ["p001", "p002"]
        assert [p["entry_id"] for p in out["proposals"]] == ["mara", "salt_crossing"]
        assert _entry_files(tmp_path, story_id) == []
        listed = await _ok(StoryCodexTool(), ctx, action="proposals")
        assert [p["id"] for p in listed["pending"]] == ["p001", "p002"]

    async def test_apply_writes_the_entry_and_get_context_and_audit_read_it(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/proposals.py :: work.write(relative, dump_file(entry, compact=True), room_id=room_id, expected_digest=None)
        Becomes: pass
        Killed by: src/uclone_x/story/proposals.py :: return _apply_new(work, proposal, proposal_digest, room_id=room_id, decided_in=decided_in)
        Becomes: raise StoryError("x")
        """
        story_id, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        fact = {
            "subject": "Mara",
            "predicate": "status",
            "object": "dead",
            "quote": "she did not rise again",
        }
        before = await _ok(StoryAuditTool(), ctx, action="check", scene_id="ch01.s02", facts=[fact])
        assert before["contradictions"] == []

        applied = await _ok(
            codex, _ctx(tmp_path, story=story_id, approved=True), action="apply", proposal_id="p001"
        )
        assert applied["entry"] == {"kind": "characters", "id": "mara"}
        assert _entry_files(tmp_path, story_id) == ["characters/mara.yaml"]
        written = yaml.safe_load(
            (_codex_dir(tmp_path, story_id) / "characters" / "mara.yaml").read_text("utf-8")
        )
        assert written == {
            "id": "mara",
            "name": "Mara",
            "aliases": ["the ferrywoman"],
            "profile": "Keeps the ferry at the salt crossing.",
            "state": {"status": "alive"},
            "visual": {"prose": "Grey braid, a burn on her left hand.", "gender": "female"},
        }

        got = await _ok(codex, ctx, action="get", entry_id="the ferrywoman")
        assert [e["id"] for e in got["entries"]] == ["mara"]
        scene = await _ok(StoryContextTool(), ctx, action="for_scene", scene_id="ch01.s01")
        [mara] = [e for e in scene["codex"] if e["id"] == "mara"]
        assert mara["appearance"] == "Grey braid, a burn on her left hand."
        after = await _ok(StoryAuditTool(), ctx, action="check", scene_id="ch01.s02", facts=[fact])
        assert len(after["contradictions"]) == 1

    async def test_reject_writes_no_entry(self, tmp_path: Path) -> None:
        story_id, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        await _ok(codex, ctx, action="reject", proposal_id="p001", reason="not this one")
        assert _entry_files(tmp_path, story_id) == []
        listed = await _ok(codex, ctx, action="proposals")
        assert listed["decided"] == [{"id": "p001", "status": "rejected", "entry_id": "mara"}]

    async def test_apply_without_a_persons_approval_writes_nothing(self, tmp_path: Path) -> None:
        story_id, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        await _refused(codex, ctx, action="apply", proposal_id="p001")
        assert _entry_files(tmp_path, story_id) == []


class TestNoSecondEntryForOneName:
    async def test_a_name_an_entry_has_is_refused_and_nothing_is_proposed(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/proposals.py :: for item in names.matches(key):
        Becomes: for item in []:
        """
        story_id, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        await _ok(
            codex, _ctx(tmp_path, story=story_id, approved=True), action="apply", proposal_id="p001"
        )
        # A place, under another id, that one of Mara's aliases already names; with it, a
        # new place that would be fine alone: neither is proposed.
        clash = {"kind": "places", "name": "The Ferrywoman", "id": "ferry"}
        error = await _refused(codex, ctx, action="create", entries=[CROSSING, clash])
        assert error.startswith(
            "'The Ferrywoman' was not proposed, and nor were the others: the codex already "
            "has the character 'Mara', also called 'The Ferrywoman'."
        )
        listed = await _ok(codex, ctx, action="proposals")
        assert listed["pending"] == []

    async def test_an_entry_waiting_for_a_decision_holds_its_name(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/proposals.py :: if wanted & {folded(k) for k in (other.id, other.name, *other.aliases)}:
        Becomes: if False:
        """
        _, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        error = await _refused(codex, ctx, action="create", entries=[{**MARA, "id": "mara2"}])
        assert "proposal 'p001' already proposes the new character 'Mara'" in error
        # Two of the same in one call: the second clashes with the first.
        error = await _refused(codex, ctx, action="create", entries=[CROSSING, CROSSING])
        assert "already proposes the new place 'Salt Crossing'" in error

    async def test_an_entry_added_since_blocks_applying_the_proposal(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/proposals.py :: if clash is not None:
        Becomes: if False:
        """
        story_id, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        path = _codex_dir(tmp_path, story_id) / "places" / "ferry.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("id: ferry\nname: Mara\n", encoding="utf-8")
        error = await _refused(
            codex, _ctx(tmp_path, story=story_id, approved=True), action="apply", proposal_id="p001"
        )
        assert error.startswith(
            "Proposal 'p001' was not applied: the codex already has the place 'Mara'."
        )
        assert _entry_files(tmp_path, story_id) == ["places/ferry.yaml"]
        work = StoryWork(StoryLibrary(tmp_path), story_id)
        proposal, _ = work.proposal("p001")
        blocked = preview_proposal(work, proposal, None).blocked
        assert blocked is not None
        assert blocked.startswith("It cannot be approved: the codex already has the place 'Mara'.")


class TestAPersonSeesWhatTheyApprove:
    async def test_the_preview_shows_every_value_and_the_file_as_added(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/story/proposals.py :: return _preview_new(work, proposal)
        Becomes: return ProposalPreview(None, "", [], [], None)
        Killed by: src/uclone_x/story/proposals.py :: lines.extend(ChangeLine(name, None, None, v) for name, v in state.items())
        Becomes: pass
        """
        story_id, ctx = await _new_story(tmp_path)
        await _ok(StoryCodexTool(), ctx, action="create", entries=[MARA])
        work = StoryWork(StoryLibrary(tmp_path), story_id)
        proposal, _ = work.proposal("p001")
        preview = preview_proposal(work, proposal, None)
        assert preview.entry_name == "Mara"
        assert preview.blocked is None
        assert [(c.what, c.at, c.before, c.after) for c in preview.changes] == [
            ("name", None, None, "Mara"),
            ("aliases", None, None, ["the ferrywoman"]),
            ("profile", None, None, "Keeps the ferry at the salt crossing."),
            ("status", None, None, "alive"),
            ("looks: prose", None, None, "Grey braid, a burn on her left hand."),
            ("looks: gender", None, None, "female"),
        ]
        added = [
            line for line in preview.diff if line.startswith("+") and not line.startswith("+++")
        ]
        assert added[:2] == ["+id: mara", "+name: Mara"]
        assert not [line for line in preview.diff[2:] if line.startswith("-")]

    async def test_a_rejected_new_entry_is_listed_by_its_name(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/story/view.py :: entry_name = current[0].name if current is not None else _new_name(proposal)
        Becomes: entry_name = current[0].name if current is not None else None
        """
        story_id, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        await _ok(codex, ctx, action="create", entries=[MARA])
        await _ok(codex, ctx, action="reject", proposal_id="p001")
        view = StoryView(tmp_path, RoomService(RoomStore(tmp_path / "rooms")), turn_busy=bool)
        [decided] = view.show(story_id).decided
        assert (decided.id, decided.entry_name) == ("p001", "Mara")


class TestIdsAndRefusals:
    def test_an_id_is_made_from_the_name_as_the_codex_writes_ids(self) -> None:
        """Killed by: src/uclone_x/story/names.py :: out.append(_FINALS[index % 28])
        Becomes: out.append("")
        """
        assert id_for_name("라온") == "raon"
        assert id_for_name("예린") == "yerin"
        assert id_for_name("은월성") == "eunwolseong"
        assert id_for_name("Lord Vane") == "lord_vane"
        assert id_for_name("Émile") == "emile"
        assert id_for_name("月光") is None

    async def test_refusals_are_plain_words(self, tmp_path: Path) -> None:
        _, ctx = await _new_story(tmp_path)
        codex = StoryCodexTool()
        errors = [
            await _refused(codex, ctx, action="create"),
            await _refused(
                codex, ctx, action="create", entries=[{"kind": "items", "name": "月光"}]
            ),
            await _refused(
                codex,
                ctx,
                action="create",
                entries=[{"kind": "places", "name": "Salt Crossing", "gender": "female"}],
            ),
            await _refused(codex, ctx, action="create", entries=[CROSSING] * 9),
        ]
        assert errors[1].startswith("No id could be made from '月光', so nothing was proposed.")
        assert errors[2].startswith("'Salt Crossing' is a place, and only a character has")
        for error in errors:
            for internal in ("`", "Traceback", "Error", "new_entry", "entry_id", "pattern", "["):
                assert internal not in error, (internal, error)
