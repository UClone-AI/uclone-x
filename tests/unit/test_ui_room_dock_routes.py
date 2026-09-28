"""Tests for the dock's room-scoped reads (#1353, #1354, #1355, #1357).

What they pin, in order of what it would cost to get wrong:

* **A seat's knowledge is the seat's.** The shared engine the chat surface reads describes
  nothing a seat learned; answering from it is showing the wrong agent's memory.
* **An absence says what kind it is.** A turn that raised has no record of its tools, and
  `[]` would claim it used none. A seat not running in this process has no engine to read,
  and an empty graph would claim it learned nothing (P6).
* **The room's files are the room's.** Only the paths its seats wrote through tools, not the
  workspace's whole listing.
* **An idle seat is still in the graph.** A topology that shows only seats that spoke reads
  as "no agents" for a room nobody has addressed yet.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast, get_args

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, ToolExecutionRecord
from uclone_x.agent.session import SessionState
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, ToolCallRequest
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.models import OntologyRelation
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomMessage,
    RoomState,
    RoomToolUse,
    RoomWrittenFile,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.room_dock import (
    HistoryGapCode,
    LogFailureKind,
    ToolCallGapCode,
    TopologyReasonCode,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "sessions",
        workspace_dir=tmp_path / "workspace",
        llm=MockLLMConnector(),
    )
    with TestClient(app) as started:
        yield started


def _create(client: TestClient, agent_ids: list[str]) -> str:
    created = client.post("/api/rooms", json={"title": "Dock", "agent_ids": agent_ids})
    assert created.status_code == 201, created.text
    return str(created.json()["room_id"])


def _stack(client: TestClient) -> Any:
    return cast(Any, client.app).state.room_stack


def _seed(client: TestClient, room_id: str, **update: Any) -> None:
    """Write the room's record directly, the way the orchestrator would have after turns."""
    stack = _stack(client)
    state: RoomState = stack.service.get(room_id)
    stack.store.save(state.model_copy(update=update))


def _utterance(
    seq: int,
    sender: str,
    *,
    turn_id: str | None,
    tools_recorded: bool,
    error: str | None = None,
    completed: bool = True,
    persist_error: str | None = None,
) -> RoomMessage:
    return RoomMessage(
        seq=seq,
        sender_id=sender,
        content=f"row {seq}",
        turn_id=turn_id,
        tools_recorded=tools_recorded,
        error=error,
        completed=completed,
        persist_error=persist_error,
    )


def _use(turn_id: str, sender: str = "scout", **extra: Any) -> RoomToolUse:
    fields: dict[str, Any] = {
        "turn_id": turn_id,
        "participant_id": sender,
        "tool_name": "file_write",
        "status": "success",
    }
    fields.update(extra)
    return RoomToolUse(**fields)


def _wait_for_rows(client: TestClient, room_id: str, rows: int) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if len(client.get(f"/api/rooms/{room_id}").json()["transcript"]) >= rows:
            return
        time.sleep(0.05)
    raise AssertionError(f"room {room_id} never reached {rows} rows")


class _Seat:
    """A seat whose one turn reports the given tools, or raises: the orchestrator's input."""

    def __init__(self, agent_id: str, session_id: str) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.tool_executions: tuple[Any, ...] = ()
        self.fail_with: Exception | None = None

    async def execute_turn(self, prompt: str, *, stream_callback: Any = None, **kwargs: Any) -> Any:
        from uclone_x.agent.models import TurnResult

        if self.fail_with is not None:
            raise self.fail_with
        return TurnResult(
            turn_index=1, content="done", provenance=None, tool_executions=self.tool_executions
        )

    def checkpoint_turn(self, session_id: str | None = None) -> Any:
        from uclone_x.agent.session import SessionState

        return SessionState(session_id=session_id or self.session_id, agent_id=self.agent_id)

    def roll_back_turn(self, checkpoint: Any, *, reason: str) -> int:
        return 0

    def persist_session(self, session_id: str | None = None) -> Any:
        from uclone_x.agent.session import SessionState

        return SessionState(session_id=session_id or self.session_id, agent_id=self.agent_id)


def _take_one_turn(client: TestClient, room_id: str, seat: _Seat) -> None:
    """Run one real orchestrator turn for `seat` against the app's own room store.

    Through `RoomOrchestrator`, not `_seed`: what the room records about a turn it could
    not see is written there, and a seeded record would test the seed.
    """
    import asyncio

    from uclone_x.room.models import (
        SelectionVerdict,
        SpeakerDecision,
        SpeakerRequest,
    )
    from uclone_x.room.orchestrator import RoomOrchestrator

    class _Once:
        name = "once"

        def __init__(self) -> None:
            self.spoke = False

        async def select(self, request: SpeakerRequest) -> SpeakerDecision:
            if self.spoke:
                return SpeakerDecision(verdict=SelectionVerdict.ABSTAIN, selector=self.name)
            self.spoke = True
            return SpeakerDecision(
                verdict=SelectionVerdict.SPEAK, speaker_id=seat.agent_id, selector=self.name
            )

    class _Resolver:
        async def resolve(self, participant: Any, *, one_seat: bool = False) -> Any:
            return seat

    orch = RoomOrchestrator(store=_stack(client).store, selectors=[_Once()], resolver=_Resolver())
    asyncio.run(orch.post(room_id, "user", "go"))


def _shell_call() -> Any:
    """A successful call to a tool that writes and names no path: a shell."""
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.tools.models import ToolResultStatus

    return ToolExecutionRecord(
        tool_name="shell",
        arguments={"command": "echo hi > notes.md"},
        output={"stdout": "", "exit_code": 0},
        status=ToolResultStatus.SUCCESS,
        tool_call_id="call_sh",
        writes_files=True,
    )


#: Any wording that states, as a fact, that no file was written (#1366). The room cannot
#: know that -- a shell, an MCP server or a helper can write without naming a path -- so no
#: dock read may ever say it, in any field.
#:
#: It also forbids the negations a sentence can carry the same claim in (#1388 N1): "has
#: not written", "no file written", "never wrote", and the claims about tool use that
#: imply it -- "without using a tool", "no tool calls were made", "0 tool calls".
_NONE_WRITTEN = re.compile(
    r"\bno files? ((has|have|had|was|were|got) )?(been )?(written|saved|created)"
    r"|\bnothing ((has|have|had) been |was |were |got )?(written|saved)"
    r"|\bnone ((were|was|has|have) (been )?)?written"
    r"|\bwrote (nothing|no files?)"
    r"|\b(did not|didn't|never|has not|have not|had not|hasn't|haven't|hadn't)"
    r" (yet )?(write|wrote|save|saved|written|created)\b"
    r"|\bwithout (using|calling|running) (a |any )?tools?\b"
    r"|\bused no tools?\b"
    # "No tool calls listed" is honest -- it is a statement about the list -- so a claim
    # followed by listed/shown/recorded is not one.
    r"|\bno tool (use|uses|calls?)\b(?! (are |were )?(listed|shown|recorded))"
    r"|(?<![\w.])0 tool calls?\b(?! (are |were )?(listed|shown|recorded))"
    r"|whether any file was written",
    re.IGNORECASE,
)

#: What an empty file list says when the room knows of no gap (#1366).
_NONE_LISTED = (
    "No files are listed yet. This list shows files the clones saved by name; a file "
    "written another way, such as by a shell command or a helper, may not appear here."
)


#: A lost turn, in the words every dock read uses. "Or still running elsewhere" because the
#: started/landed count cannot tell a lost turn from one another process is running (#1388 N5).
_UNSAVED_ONE = (
    "1 turn(s) started but stopped before their results were saved, or are still running "
    "in another copy of the app"
)


def _claims_none_written(body: dict[str, Any]) -> bool:
    """True when any field of a dock response says no file was written (#1366)."""
    return _NONE_WRITTEN.search(json.dumps(body)) is not None


# --------------------------------------------------------------------------------------
# GET /api/rooms/{room_id}/seats/{participant_id}/history  (#1353)
# --------------------------------------------------------------------------------------


class TestASeatsHistory:
    def test_each_turn_carries_its_own_tools_and_only_that_seats(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: uses = [u for u in state.tool_uses if u.participant_id == seat.id]
        Becomes: uses = list(state.tool_uses)
        """
        room_id = _create(client, ["scout", "critic"])
        _seed(
            client,
            room_id,
            transcript=(
                RoomMessage(seq=1, sender_id="user", content="go"),
                _utterance(2, "scout", turn_id="t_a", tools_recorded=True),
                _utterance(3, "critic", turn_id="t_b", tools_recorded=True),
                _utterance(4, "scout", turn_id="t_c", tools_recorded=True),
            ),
            tool_uses=(
                _use("t_a", tool_call_id="c1"),
                _use("t_b", "critic", tool_call_id="c2"),
                _use("t_a", tool_call_id="c3", tool_name="web_search"),
            ),
        )

        body = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()

        assert [t["seq"] for t in body["turns"]] == [2, 4]
        assert [u["tool_call_id"] for u in body["turns"][0]["tools"]] == ["c1", "c3"]
        assert body["turns"][1]["tools"] == [], "a recorded turn with no tools is an empty list"
        assert [u["tool_call_id"] for u in body["tool_uses"]] == ["c1", "c3"]
        assert body["tool_uses"][0]["seq"] == 2
        assert body["reason"] is None

    def test_a_turn_that_did_not_report_its_tools_says_so_rather_than_none(
        self, client: TestClient
    ) -> None:
        """`[]` for a raised turn would claim it used no tools; nobody recorded that.

        Killed by: src/uclone_x/ui/room_dock.py :: recorded = message.tools_recorded and message.turn_id is not None
        Becomes: recorded = True
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(1, "scout", turn_id="t_x", tools_recorded=False, error="boom"),
                _utterance(2, "scout", turn_id=None, tools_recorded=False),
            ),
        )

        turns = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()["turns"]

        assert [t["tools"] for t in turns] == [None, None]
        assert turns[0]["status"] == "failed"
        assert "failed or was stopped" in turns[0]["tools_not_recorded_reason"]
        assert "before the conversation kept" in turns[1]["tools_not_recorded_reason"]

    def test_a_seat_that_has_not_spoken_says_so(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: "live": stack.live_agent(state.room_id, seat.session_id) is not None,
        Becomes: "live": True,
        """
        room_id = _create(client, ["scout"])

        body = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()

        assert body["turns"] == []
        assert "has not taken a turn" in body["reason"]
        assert body["live"] is False

    def test_a_seat_whose_turns_were_cleared_is_not_said_to_have_none(
        self, client: TestClient
    ) -> None:
        """After a clear, "has not taken a turn yet" is a claim the room can no longer make.

        Killed by: src/uclone_x/room/service.py :: record = record.model_copy(update={"clears": record.clears + 1})
        Becomes: record = record
        """
        room_id = _create(client, ["scout"])
        _take_one_turn(client, room_id, _Seat("scout", "s"))
        assert client.delete(f"/api/rooms/{room_id}/history").status_code == 200

        body = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()

        assert body["turns"] == []
        assert "has not taken a turn" not in body["reason"]
        assert "cleared" in body["reason"]

    def test_an_unknown_seat_is_refused_naming_the_roster(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: raise _roster_refusal(state, participant_id)
        Becomes: return state.participants[0]
        """
        room_id = _create(client, ["scout"])

        refused = client.get(f"/api/rooms/{room_id}/seats/ghost/history")

        assert refused.status_code == 404
        assert "scout" in refused.json()["detail"]

    def test_an_unknown_room_is_a_404(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: raise _http_error(exc) from exc
        Becomes: return RoomState(room_id=room_id, title="stand-in")
        """
        refused = client.get("/api/rooms/room_missing/seats/scout/history")

        assert refused.status_code == 404
        # The room's own refusal, not the seat's: a missing room is not an empty roster.
        assert "No room 'room_missing'" in refused.json()["detail"]


# --------------------------------------------------------------------------------------
# GET /api/rooms/{room_id}/artifacts  (#1354)
# --------------------------------------------------------------------------------------


class TestARoomsFiles:
    def test_only_what_the_room_wrote_is_listed_once_per_path(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: entry: dict[str, Any] | None = entries.get(written.path)
        Becomes: entry: dict[str, Any] | None = None
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        (workspace / "plan.md").write_text("# plan", encoding="utf-8")
        (workspace / "unrelated.md").write_text("not this room's", encoding="utf-8")
        room_id = _create(client, ["scout", "critic"])
        _seed(
            client,
            room_id,
            written_files=(
                RoomWrittenFile(
                    path="plan.md",
                    participant_id="scout",
                    tool_name="file_write",
                    turn_id="t1",
                    written_at="2026-09-22T00:00:01+00:00",
                ),
                RoomWrittenFile(
                    path="plan.md",
                    participant_id="critic",
                    tool_name="file_edit",
                    turn_id="t2",
                    written_at="2026-09-22T00:00:02+00:00",
                ),
                RoomWrittenFile(
                    path="gone.png",
                    participant_id="scout",
                    tool_name="image_generate",
                    turn_id="t1",
                    written_at="2026-09-22T00:00:00+00:00",
                ),
            ),
        )

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert [a["path"] for a in body["artifacts"]] == ["plan.md", "gone.png"]
        plan, gone = body["artifacts"]
        assert plan["write_count"] == 2
        assert plan["writers"] == ["scout", "critic"]
        assert plan["participant_id"] == "critic", "the latest write names the entry"
        assert plan["exists"] is True and plan["type"] == "document"
        assert plan["id"].startswith("art_")
        assert gone["exists"] is False, "a recorded file since deleted is not listed as present"
        assert gone["type"] == "image"
        assert body["total"] == 2

    def test_a_room_with_no_known_gap_says_what_the_list_covers_not_that_none_were_written(
        self, client: TestClient
    ) -> None:
        """Every turn reported its tools and none could write unnamed -- and still no claim.

        A tool can write without the room seeing it, so the empty list says what it covers
        instead of "no file has been written" (#1366, fifth review).

        Killed by: src/uclone_x/ui/room_dock.py :: reason = "No files are listed yet. " + FILES_SCOPE_NOTE
        Becomes: reason = "No file has been written through a tool in this conversation yet."
        Killed by: src/uclone_x/ui/room_dock.py :: "scope_note": FILES_SCOPE_NOTE,
        Becomes: "scope_note": None,
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
            tool_uses=(_use("t1", tool_name="search"),),
        )

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert body["artifacts"] == []
        assert body["unattributed_writes"] == 0 and body["unrecorded_turns"] == 0
        assert body["record_gaps"] == []
        assert body["reason"] == _NONE_LISTED
        assert "may not appear here" in body["scope_note"]
        assert "record_complete" not in body, "no field may read as 'nothing was written'"
        assert not _claims_none_written(body)

    def test_a_room_with_no_files_but_unnamed_writes_does_not_claim_none_were_written(
        self, client: TestClient
    ) -> None:
        """A shell is a tool: "No file has been written" would be a claim nobody recorded.

        Killed by: src/uclone_x/ui/room_dock.py :: if file_record.unattributed_writes:
        Becomes: if False:
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            tool_uses=(_use("t1", tool_name="shell", wrote_unnamed=True),),
            file_record=RoomFileRecord(kept_since_creation=True, unattributed_writes=1),
        )

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert body["artifacts"] == []
        assert body["unattributed_writes"] == 1
        assert "without naming them" in body["unattributed_note"]
        assert not _claims_none_written(body)
        assert body["reason"].startswith(_NONE_LISTED)
        assert "1 tool call(s) could have written files without naming them" in body["reason"]

    def test_both_unknown_causes_are_named_in_the_reason(self, client: TestClient) -> None:
        """An unreported turn and an unnamed write together: the reason names each.

        Killed by: src/uclone_x/ui/room_dock.py :: + _join(causes)
        Becomes: + causes[0]
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t_x", tools_recorded=False),),
            tool_uses=(_use("t0", tool_name="shell", wrote_unnamed=True),),
            file_record=RoomFileRecord(
                kept_since_creation=True, unrecorded_turns=1, unattributed_writes=1
            ),
        )

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert body["artifacts"] == []
        assert "1 turn(s) ended before reporting their tools" in body["reason"]
        assert "1 tool call(s) could have written files without naming them" in body["reason"]
        assert not _claims_none_written(body)

    def test_a_turn_that_ended_unreported_is_not_read_as_no_files(self, client: TestClient) -> None:
        """A Stop after a `file_write` leaves the file on disk and outside the record.

        "No file has been written" would then be a claim nobody recorded (P6).

        Killed by: src/uclone_x/ui/room_dock.py :: if file_record.unrecorded_turns:
        Becomes: if False:
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(1, "scout", turn_id="t_x", tools_recorded=False, completed=False),
                _utterance(2, "scout", turn_id="t_y", tools_recorded=True),
            ),
            file_record=RoomFileRecord(kept_since_creation=True, unrecorded_turns=1),
        )

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert body["artifacts"] == []
        assert body["unrecorded_turns"] == 1
        assert "It may also be missing files because" in body["reason"]
        assert not _claims_none_written(body)
        assert "before reporting their tools" in body["unrecorded_note"]

    def test_a_turn_that_failed_partway_is_not_said_to_have_none_listed(
        self, client: TestClient
    ) -> None:
        """Some of a partial turn's files are listed; the note must not say none are (#1388 N2).

        Killed by: src/uclone_x/ui/room_dock.py :: f"stopped) before reporting their tools, so the files they wrote may not "
        Becomes: f"stopped) before reporting their tools, so any files they wrote are not "
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t_p", tools_recorded=False, error="boom"),),
            tool_uses=(_use("t_p", tool_call_id="c1"),),
            written_files=(
                RoomWrittenFile(
                    path="half.md",
                    participant_id="scout",
                    tool_name="file_write",
                    turn_id="t_p",
                    tool_call_id="c1",
                ),
            ),
            file_record=RoomFileRecord(kept_since_creation=True, unrecorded_turns=1),
        )

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert [a["path"] for a in body["artifacts"]] == ["half.md"]
        assert "may not all be listed here" in body["unrecorded_note"]
        assert "are not listed here" not in body["unrecorded_note"]

    def test_a_recorded_path_outside_the_workspace_is_never_read(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """`exists` and `size_bytes` go through the workspace boundary, not a bare `stat`.

        Killed by: src/uclone_x/ui/room_dock.py :: resolved = PathValidator().resolve_safe_path(Path(path), workspace)
        Becomes: resolved = workspace / path
        """
        (tmp_path / "workspace").mkdir(exist_ok=True)
        (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            written_files=(
                RoomWrittenFile(
                    path="../secret.txt",
                    participant_id="scout",
                    tool_name="mcp_write",
                    turn_id="t1",
                ),
            ),
        )

        (entry,) = client.get(f"/api/rooms/{room_id}/artifacts").json()["artifacts"]

        # Not False: the file is there, and it was not looked at. `null` with the cause.
        assert entry["exists"] is None
        assert entry["size_bytes"] is None
        assert "outside the workspace" in entry["exists_reason"]

    def test_a_path_the_os_cannot_name_is_unknown_not_a_crash(self, client: TestClient) -> None:
        """A recorded path the OS rejects outright is `null` with a cause, not a 500.

        Killed by: src/uclone_x/ui/room_dock.py :: except ValueError:  # a path the OS cannot name
        Becomes: except ZeroDivisionError:  # a path the OS cannot name
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            written_files=(
                RoomWrittenFile(
                    path="bad\x00name.txt",
                    participant_id="scout",
                    tool_name="mcp_write",
                    turn_id="t1",
                ),
            ),
        )

        response = client.get(f"/api/rooms/{room_id}/artifacts")

        assert response.status_code == 200
        (entry,) = response.json()["artifacts"]
        assert entry["exists"] is None
        assert entry["exists_reason"] == "This path cannot be checked on this computer."

    def test_a_clear_after_a_shell_turn_does_not_turn_unknown_into_none(
        self, client: TestClient
    ) -> None:
        """Reproduction 1 (#1366 review): the shell's files stay on disk after a clear.

        The clear removes the `tool_uses` row that showed the unnamed write, so a reason
        read off the remaining rows said "none written" without learning anything new.

        Killed by: src/uclone_x/room/orchestrator.py :: + sum(1 for u in uses if u.wrote_unnamed),
        Becomes: + 0,
        """
        room_id = _create(client, ["scout"])
        seat = _Seat("scout", "s")
        seat.tool_executions = (_shell_call(),)
        _take_one_turn(client, room_id, seat)
        assert client.delete(f"/api/rooms/{room_id}/history").status_code == 200

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert body["artifacts"] == []
        assert not _claims_none_written(body)
        assert "1 tool call(s) could have written files without naming them" in body["reason"]
        assert "its history was cleared" in body["reason"]
        assert body["record_gaps"]

    def test_a_rewind_past_a_stopped_turn_does_not_turn_unknown_into_none(
        self, client: TestClient
    ) -> None:
        """Reproduction 2: a turn that raised reported no tools; rewinding past it hid that.

        Killed by: src/uclone_x/room/orchestrator.py :: + (0 if tools_recorded else 1),
        Becomes: + 0,
        Killed by: src/uclone_x/room/service.py :: record = record.model_copy(update={"rewinds": record.rewinds + 1})
        Becomes: record = record
        """
        room_id = _create(client, ["scout"])
        seat = _Seat("scout", "s")
        seat.fail_with = RuntimeError("stopped mid-write")
        _take_one_turn(client, room_id, seat)
        transcript = client.get(f"/api/rooms/{room_id}").json()["transcript"]
        asked = next(
            m["seq"] for m in transcript if m["sender_id"] == "user" and m["content"] == "go"
        )
        rewound = client.post(f"/api/rooms/{room_id}/history/truncate", json={"seq": asked})
        assert rewound.status_code == 200, rewound.text

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert not _claims_none_written(body)
        assert "1 turn(s) ended before reporting their tools" in body["reason"]
        assert "its history was rewound" in body["reason"]
        assert body["unrecorded_turns"] == 1

    def test_a_room_stored_before_the_record_does_not_claim_none(self, client: TestClient) -> None:
        """Reproduction 3: agent turns stored before this record existed wrote unknown files.

        The room file is written without `file_record`, as an older build left it, and it
        must load as incomplete -- not as a fresh room with nothing to hide.

        Killed by: src/uclone_x/ui/room_dock.py :: if not file_record.kept_since_creation:
        Becomes: if False:
        """
        room_id = _create(client, ["scout"])
        store = _stack(client).store
        stored = json.loads(store.room_path(room_id).read_text(encoding="utf-8"))
        stored.pop("file_record", None)
        stored["transcript"].append(
            {"seq": len(stored["transcript"]) + 1, "sender_id": "scout", "content": "old reply"}
        )
        store.room_path(room_id).write_text(json.dumps(stored), encoding="utf-8")

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert not _claims_none_written(body)
        assert "began before files written by tools were recorded" in body["reason"]

    def test_an_agent_row_without_a_turn_id_is_a_gap_and_a_humans_is_not(
        self, client: TestClient
    ) -> None:
        """Only agent rows count: a human's row has no turn id either, and ran no tool.

        Killed by: src/uclone_x/ui/room_dock.py :: if m.is_utterance and m.turn_id is None and m.sender_id not in humans
        Becomes: if m.is_utterance and m.turn_id is None
        """
        room_id = _create(client, ["scout"])
        human_only = (RoomMessage(seq=1, sender_id="user", content="hello"),)
        _seed(client, room_id, transcript=human_only)
        human_body = client.get(f"/api/rooms/{room_id}/artifacts").json()
        assert human_body["record_gaps"] == []
        assert human_body["reason"] == _NONE_LISTED

        _seed(
            client,
            room_id,
            transcript=(*human_only, _utterance(2, "scout", turn_id=None, tools_recorded=False)),
        )
        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert not _claims_none_written(body)
        assert "1 turn(s) were saved without a record of their tools" in body["reason"]

    def test_a_turn_still_running_is_named_rather_than_none(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running turn's writes reach the record when it lands, not before.

        Killed by: src/uclone_x/ui/room_dock.py :: running = stack.turn_in_flight(state.room_id)
        Becomes: running = False
        """
        room_id = _create(client, ["scout"])
        stack = _stack(client)

        def running_here(rid: str) -> bool:
            return rid == room_id

        monkeypatch.setattr(stack, "turn_in_flight", running_here)

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        assert not _claims_none_written(body)
        assert "a turn is still running" in body["reason"]
        assert body["turn_running"] is True
        assert body["record_gaps"] == [], "a running turn is not a gap in the record"

    def test_a_file_the_disk_would_not_answer_for_is_unknown_not_absent(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """`exists: False` only when `stat` said so; a permission error said nothing.

        Killed by: src/uclone_x/ui/room_dock.py :: except FileNotFoundError:
        Becomes: except OSError:
        """
        import os

        locked = tmp_path / "workspace" / "locked"
        locked.mkdir(parents=True)
        (locked / "a.md").write_text("x", encoding="utf-8")
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            written_files=(
                RoomWrittenFile(
                    path="locked/a.md", participant_id="scout", tool_name="file_write", turn_id="t1"
                ),
                RoomWrittenFile(
                    path="gone.md", participant_id="scout", tool_name="file_write", turn_id="t1"
                ),
            ),
        )
        os.chmod(locked, 0)
        try:
            if os.access(locked / "a.md", os.F_OK):  # pragma: no cover - running as root
                pytest.skip("a mode-0 directory is still readable to this user")
            artifacts = client.get(f"/api/rooms/{room_id}/artifacts").json()["artifacts"]
        finally:
            os.chmod(locked, 0o700)

        by_path = {a["path"]: a for a in artifacts}
        assert by_path["locked/a.md"]["exists"] is None
        assert "Could not check this file" in by_path["locked/a.md"]["exists_reason"]
        assert by_path["gone.md"]["exists"] is False
        assert by_path["gone.md"]["exists_reason"] is None

    def test_an_unknown_room_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/rooms/room_missing/artifacts").status_code == 404


# --------------------------------------------------------------------------------------
# GET /api/rooms/{room_id}/topology  (#1355)
# --------------------------------------------------------------------------------------


class TestARoomsTopology:
    def test_an_idle_room_still_shows_every_seat(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: for participant in seated.values():
        Becomes: for participant in [p for p in seated.values() if p.id in spoken]:
        """
        room_id = _create(client, ["scout", "critic"])

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        seats = [n for n in body["nodes"] if n["kind"] == "seat"]
        assert [(n["id"], n["status"]) for n in seats] == [
            ("seat:scout", "idle"),
            ("seat:critic", "idle"),
        ]
        assert body["edges"] == []
        assert body["summary"] == {"seats": 2, "turns": 0, "tool_calls": 0, "subagents": 0}

    def test_turns_tools_and_subagents_are_linked(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: if use.subagent_id is not None:
        Becomes: if False:
        """
        room_id = _create(client, ["scout", "critic"])
        _seed(
            client,
            room_id,
            transcript=(
                RoomMessage(seq=1, sender_id="user", content="go"),
                _utterance(2, "scout", turn_id="t_a", tools_recorded=True),
                _utterance(3, "critic", turn_id=None, tools_recorded=False),
            ),
            tool_uses=(_use("t_a", tool_name="delegate", subagent_id="sub_1", tool_call_id="c1"),),
        )

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        ids = {n["id"]: n for n in body["nodes"]}
        assert ids["seat:scout"]["status"] == "answered"
        assert ids["seat:scout"]["turn_count"] == 1
        assert "turn:t_a" in ids and "turn:seq-3" in ids
        assert "turn:seq-1" not in ids, "the human's message is not a seat's turn"
        assert ids["tool:t_a:0"]["tool_name"] == "delegate"
        assert ids["subagent:sub_1"]["parent_participant_id"] == "scout"
        edges = {(e["source"], e["target"], e["kind"]) for e in body["edges"]}
        assert ("seat:scout", "turn:t_a", "took_turn") in edges
        assert ("turn:t_a", "turn:seq-3", "followed_by") in edges
        assert ("turn:t_a", "tool:t_a:0", "called") in edges
        assert ("seat:scout", "subagent:sub_1", "spawned") in edges
        assert body["summary"] == {"seats": 2, "turns": 2, "tool_calls": 1, "subagents": 1}

    def test_a_rewind_is_a_named_gap_in_the_graphs_turns(self, client: TestClient) -> None:
        """A rewind removes turn rows, so the graph must not read as complete (#1388 N3).

        Killed by: src/uclone_x/ui/room_dock.py :: history_gaps.append("its history was rewound")
        Becomes: pass
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(client, room_id, file_record=RoomFileRecord(kept_since_creation=True, rewinds=1))

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        assert body["history_gaps"] == ["its history was rewound"]
        assert body["history_complete"] is False

    def test_complete_turns_do_not_hide_unrecorded_tool_calls(self, client: TestClient) -> None:
        """`history_complete` speaks for turn rows; tool-level gaps are named apart (#1388 N3).

        Every turn row is present, but one turn ended before reporting its tools and one
        was saved by a build that kept no tool record, so the tool-call total is low.

        Killed by: src/uclone_x/ui/room_dock.py :: "tool_call_gaps": [text for _, _, text in tool_gaps],
        Becomes: "tool_call_gaps": [],
        Killed by: src/uclone_x/ui/room_dock.py :: "tool_call_gap_codes": [{"code": code, "count": count} for code, count, _ in tool_gaps],
        Becomes: "tool_call_gap_codes": [],
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(1, "scout", turn_id="t_a", tools_recorded=False, error="boom"),
                _utterance(2, "scout", turn_id=None, tools_recorded=False),
            ),
            file_record=RoomFileRecord(kept_since_creation=True, unrecorded_turns=1),
        )

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        assert body["history_complete"] is True
        assert body["history_gaps"] == []
        assert body["tool_call_gaps"] == [
            "1 turn(s) were saved without a record of their tools",
            "1 turn(s) ended before reporting their tools",
        ]
        # The same gaps as codes with their counts, in the same order (#1911).
        assert body["tool_call_gap_codes"] == [
            {"code": "saved_without_tools", "count": 1},
            {"code": "unreported", "count": 1},
        ]

    def test_a_room_with_no_known_tool_gap_names_none(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: if record.unrecorded_turns:
        Becomes: if True:
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t_a", tools_recorded=True),),
        )

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        assert body["tool_call_gaps"] == []

    def test_an_unknown_room_is_a_404(self, client: TestClient) -> None:
        assert client.get("/api/rooms/room_missing/topology").status_code == 404

    def test_each_turn_gap_is_sent_as_a_code_beside_its_clause(self, client: TestClient) -> None:
        """The head words a code in its reader's language; the clause stays English (#1911).

        Codes and clauses are in the same order, one for one, and a count travels apart.

        Killed by: src/uclone_x/ui/room_dock.py :: history_codes.append({"code": "before_record", "count": None})
        Becomes: history_codes.append({"code": "cleared", "count": None})
        Killed by: src/uclone_x/ui/room_dock.py :: history_codes.append({"code": "cleared", "count": None})
        Becomes: history_codes.append({"code": "rewound", "count": None})
        Killed by: src/uclone_x/ui/room_dock.py :: history_codes.append({"code": "rewound", "count": None})
        Becomes: history_codes.append({"code": "cleared", "count": None})
        Killed by: src/uclone_x/ui/room_dock.py :: return {"code": "uncounted", "count": None}
        Becomes: return {"code": "unsaved", "count": None}
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            file_record=RoomFileRecord(clears=1, rewinds=2, turns_started=0, turns_landed=1),
        )

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        assert body["history_gaps"] == [
            "this conversation began before its history was fully recorded",
            "its history was cleared",
            "its history was rewound",
            "some turns were taken before this conversation counted turns as they started",
        ]
        assert body["history_gap_codes"] == [
            {"code": "before_record", "count": None},
            {"code": "cleared", "count": None},
            {"code": "rewound", "count": None},
            {"code": "uncounted", "count": None},
        ]
        assert body["reason_code"] is None

    def test_turns_lost_before_saving_are_coded_with_their_count(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: return {"code": "unsaved", "count": unsaved}
        Becomes: return {"code": "unsaved", "count": None}
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            file_record=RoomFileRecord(kept_since_creation=True, turns_started=3, turns_landed=1),
        )

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        assert len(body["history_gaps"]) == 1
        assert body["history_gaps"][0].startswith("2 turn(s) started but stopped")
        assert body["history_gap_codes"] == [{"code": "unsaved", "count": 2}]

    def test_a_room_with_no_seat_is_coded_no_seat(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: reason_code: TopologyReasonCode | None = None if seated else "no_seat"
        Becomes: reason_code: TopologyReasonCode | None = None
        """
        room_id = _create(client, ["scout"])
        _seed(client, room_id, participants=())

        body = client.get(f"/api/rooms/{room_id}/topology").json()

        assert body["reason"] == "No agent is seated in this conversation."
        assert body["reason_code"] == "no_seat"


# --------------------------------------------------------------------------------------
# GET /api/rooms/{room_id}/knowledge?agent_id=  (#1357)
# --------------------------------------------------------------------------------------


class TestASeatsKnowledge:
    def test_a_seat_answers_from_its_clones_engine_not_the_managers(
        self, client: TestClient
    ) -> None:
        """The engine a running seat reasons with is its clone's, and the route reads that one.

        Killed by: src/uclone_x/ui/room_dock.py :: engine = stack.session_manager().ontology_for(seat.id)
        Becomes: engine = stack.session_manager().ontology_for("default")
        """
        room_id = _create(client, ["scout"])
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_rows(client, room_id, 2)
        room = client.get(f"/api/rooms/{room_id}").json()
        session_id = next(p["session_id"] for p in room["participants"] if p["id"] == "scout")
        live = _stack(client).live_agent(room_id, session_id)
        manager = _stack(client).session_manager()
        assert live is not None and live.ontology is manager.ontology_for("scout")
        live.ontology.register_relation(
            OntologyRelation(source_entity="Postgres", predicate="is_a", target_entity="Database")
        )
        other_clone = manager.ontology_for("default")
        assert isinstance(other_clone, OntologyEngine)
        assert not any(r.source_entity == "Postgres" for r in other_clone.list_relations())

        body = client.get(f"/api/rooms/{room_id}/knowledge", params={"agent_id": "scout"}).json()

        assert body["status"] == "ok" and body["reason"] is None
        assert body["participant_id"] == "scout"
        assert ("Postgres", "is_a", "Database") in {
            (t["subject"], t["predicate"], t["object"]) for t in body["triples"]
        }

    def test_two_clones_in_one_room_never_read_each_others_rules(self, client: TestClient) -> None:
        """Per clone, and read without building the seat: nobody has spoken here yet."""
        room_id = _create(client, ["scout", "critic"])
        manager = _stack(client).session_manager()
        manager.ontology_for("scout").register_relation(
            OntologyRelation(source_entity="tide", predicate="part_of", target_entity="sea")
        )

        scout = _knowledge(client, room_id, "scout")
        critic = _knowledge(client, room_id, "critic")

        assert [(t["subject"], t["object"]) for t in scout["triples"]] == [("tide", "sea")]
        assert critic["triples"] == []
        assert scout["status"] == critic["status"] == "ok"

    def test_what_the_rules_work_out_is_listed_with_the_facts_it_rests_on(
        self, client: TestClient
    ) -> None:
        """Computed on the read from the clone's facts and rules, never saved (§3.1).

        A second clone holding the same facts under no rule works out nothing: the rule is
        the first clone's alone.

        Killed by: src/uclone_x/ui/room_dock.py :: axioms = engine.list_axioms() if isinstance(engine, OntologyEngine) else []
        Becomes: axioms = []
        """
        room_id = _create(client, ["scout", "critic"])
        _stack(client).session_manager().ontology_for("scout").teach_axiom(
            name="part_of_is_transitive",
            subject_entity="part_of",
            predicate="transitive",
            object_value="true",
        )
        wheel = _record("scout", "wheel", "part_of", "car", "a-chat")
        car = _record("scout", "car", "part_of", "fleet", "a-chat")
        _record("critic", "wheel", "part_of", "car", "a-chat")
        _record("critic", "car", "part_of", "fleet", "a-chat")

        scout = _knowledge(client, room_id, "scout")
        critic = _knowledge(client, room_id, "critic")

        assert scout["worked_out"] == [
            {"statement": "wheel part of fleet", "because": sorted([wheel, car])}
        ]
        assert critic["worked_out"] == []
        assert "wheel part of fleet" not in [f["statement"] for f in scout["facts"]], (
            "worked out is not saved as a fact"
        )

    def test_a_missing_agent_id_is_refused_naming_the_seats(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: if not agent_id:
        Becomes: if False:
        """
        room_id = _create(client, ["scout", "critic"])

        refused = client.get(f"/api/rooms/{room_id}/knowledge")

        assert refused.status_code == 400
        assert "scout, critic" in refused.json()["detail"]

    def test_a_non_member_is_a_404_naming_the_roster(self, client: TestClient) -> None:
        room_id = _create(client, ["scout"])

        refused = client.get(f"/api/rooms/{room_id}/knowledge", params={"agent_id": "ghost"})

        assert refused.status_code == 404
        assert "scout" in refused.json()["detail"]


# --------------------------------------------------------------------------------------
# Two paths to a false "none written" with a file on disk (#1366, review 5790615363)
# --------------------------------------------------------------------------------------


class _NoteParams(BaseModel):
    text: str = Field(default="")


class _RealWriter(BaseTool[_NoteParams]):
    """A real tool, run by a real `BaseAgent`, that puts `out.md` on disk."""

    name = "scribble"
    description = "Writes out.md."
    writes_files = True

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace

    def run(self, params: _NoteParams, context: ToolContext) -> dict[str, Any]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "out.md").write_text("hello", encoding="utf-8")
        return {"path": "out.md"}


class _WriteThenFail(MockLLMConnector):
    """Step 1 asks for the writer; step 2, the answer, fails at the provider."""

    def __init__(self) -> None:
        super().__init__(
            default_response="ok",
            tool_calls=(ToolCallRequest(id="c1", name="scribble", arguments={"text": "x"}),),
        )
        self.calls = 0

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls += 1
        if self.calls >= 2:
            raise RuntimeError("provider went away")
        return await super().generate(request)


class _CasWriter(_Seat):
    """Writes `cas.md` and reports it, so only the landing save stands between the two."""

    def __init__(self, session_id: str, workspace: Path) -> None:
        super().__init__("scout", session_id)
        self.workspace = workspace
        self.tool_executions = (
            ToolExecutionRecord(
                tool_name="file_write",
                arguments={"path": "cas.md"},
                output={"path": "cas.md"},
                status=ToolResultStatus.SUCCESS,
                tool_call_id="w1",
                writes_files=True,
            ),
        )

    async def execute_turn(self, prompt: str, *, stream_callback: Any = None, **kwargs: Any) -> Any:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "cas.md").write_text("x", encoding="utf-8")
        return await super().execute_turn(prompt, stream_callback=stream_callback, **kwargs)


class TestAFileOnDiskIsNeverReportedAsNoneWritten:
    """The reviewer's two repros, which reached the plain sentence at d208ddd7."""

    def test_a_provider_error_after_a_write_lists_the_write(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Path 2: `execute_turn`'s error handler returned `tool_executions=()`.

        Killed by: src/uclone_x/agent/turn_executor.py :: tool_executions=tuple(tool_executions),  # ran before the failure
        Becomes: tool_executions=(),  # ran before the failure
        """
        room_id = _create(client, ["scout"])
        workspace = tmp_path / "workspace"
        registry = ToolRegistry()
        registry.register(cast(Any, _RealWriter(workspace)))
        llm = _WriteThenFail()
        agent = BaseAgent(
            config=AgentConfig(agent_id="scout", name="Scout"), tools=registry, llm=llm
        )
        _take_one_turn(client, room_id, cast(Any, agent))

        assert (workspace / "out.md").exists()
        assert llm.calls == 2
        body = client.get(f"/api/rooms/{room_id}/artifacts").json()
        assert not _claims_none_written(body)
        assert [a["path"] for a in body["artifacts"]] == ["out.md"]
        history = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()
        assert history["turns"][-1]["error"]
        assert [t["tool_name"] for t in history["turns"][-1]["tools"]] == ["scribble"]

    def test_a_lost_landing_save_is_named_as_a_gap(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Path 1: the row, the tools and the files were one save, and losing it left nothing.

        Killed by: src/uclone_x/room/orchestrator.py :: update={"turns_started": record.turns_started + 1}
        Becomes: update={"turns_started": record.turns_started + 0}
        Killed by: src/uclone_x/ui/room_dock.py :: lost = _unsaved_gap(unsaved)
        Becomes: lost = None
        """
        from uclone_x.errors import StaleRoomWriteError

        room_id = _create(client, ["scout"])
        workspace = tmp_path / "workspace"
        stack = _stack(client)
        store = stack.store
        real_save = store.save
        seat = _CasWriter(stack.service.get(room_id).participants[-1].session_id, workspace)

        def _lose_the_landing(state: Any) -> Any:
            if any(m.turn_id for m in state.transcript):
                raise StaleRoomWriteError("simulated lost compare-and-swap")
            return real_save(state)

        store.save = _lose_the_landing
        try:
            with pytest.raises(StaleRoomWriteError):
                _take_one_turn(client, room_id, seat)
        finally:
            store.save = real_save

        assert (workspace / "cas.md").exists()
        body = client.get(f"/api/rooms/{room_id}/artifacts").json()
        assert not _claims_none_written(body)
        assert _UNSAVED_ONE in body["reason"]
        assert body["record_gaps"] == [_UNSAVED_ONE]
        assert body["unsaved_turns"] == 1
        assert body["turn_running"] is False

        history = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()
        assert history["turns"] == []
        assert "has not taken a turn" not in history["reason"]
        assert _UNSAVED_ONE in history["reason"]
        # The count is per process: another copy of the app running a turn on this room is
        # counted the same way, so the cause names both (#1388 N5).
        assert "or are still running in another copy of the app" in history["unsaved_note"]
        assert history["unsaved_turns"] == 1

        topology = client.get(f"/api/rooms/{room_id}/topology").json()
        assert topology["history_complete"] is False
        assert topology["history_gaps"] == [_UNSAVED_ONE]

    def test_the_turn_still_running_is_not_counted_as_lost(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Started and not landed is a gap only when no turn is running -- a retry included.

        Killed by: src/uclone_x/ui/room_dock.py :: return record.turns_started - record.turns_landed - (1 if turn_unlanded else 0)
        Becomes: return record.turns_started - record.turns_landed
        """
        room_id = _create(client, ["scout"])
        stack = _stack(client)
        record = stack.service.get(room_id).file_record
        _seed(client, room_id, file_record=record.model_copy(update={"turns_started": 1}))

        def _running(room_id: str) -> bool:
            return True

        monkeypatch.setattr(stack, "turn_unlanded", _running)

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()
        assert body["unsaved_turns"] == 0
        assert body["record_gaps"] == []
        assert body["turn_running"] is True
        assert not _claims_none_written(body)
        history = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()
        assert "A turn is in progress now" in history["reason"]

    def test_a_refused_start_is_a_503_that_keeps_its_cause(self) -> None:
        """A store fault the caller can retry, not a 500 and not the caller's error.

        Killed by: src/uclone_x/ui/rooms.py :: if isinstance(exc, TurnNotStartedError):
        Becomes: if False:
        """
        from uclone_x.errors import TurnNotStartedError
        from uclone_x.ui.rooms import _http_error  # pyright: ignore[reportPrivateUsage]

        refusal = _http_error(TurnNotStartedError("Scout's turn was not started, because ..."))
        assert refusal.status_code == 503
        assert refusal.detail == "Scout's turn was not started, because ..."


# --------------------------------------------------------------------------------------
# The dock never says that no file was written (#1366, fifth review)
# --------------------------------------------------------------------------------------


class _DelegateThenWrite(MockLLMConnector):
    """The seat starts a helper; the helper runs the real writer; then both answer."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.calls = 0

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            self._tool_calls = [
                ToolCallRequest(
                    id="d1",
                    name="delegate_subagent",
                    arguments={"role": "helper", "goal": "write", "prompt": "write out.md"},
                )
            ]
        elif self.calls == 2:
            self._tool_calls = [ToolCallRequest(id="c1", name="scribble", arguments={"text": "x"})]
        else:
            self._tool_calls = []
        return await super().generate(request)


class TestTheDockNeverSaysNothingWasWritten:
    """The reviewer's two repros at 69fb8dfa, and a guard over every wording the dock has."""

    def test_a_helpers_write_is_counted_as_possible_not_hidden(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """A helper's file is on disk and outside the seat's tool record.

        The delegation result does not carry the helper's tool records, so the call that
        started it is counted as a possible unnamed write, whatever its status.

        Killed by: src/uclone_x/room/orchestrator.py :: or execution.spawns_subagents,
        Becomes: or False,
        """
        from uclone_x.tools.builtin.subagent import SubagentDelegationTool

        room_id = _create(client, ["scout"])
        workspace = tmp_path / "workspace"
        registry = ToolRegistry()
        registry.register(cast(Any, _RealWriter(workspace)))
        registry.register(cast(Any, SubagentDelegationTool()))
        llm = _DelegateThenWrite()
        agent = BaseAgent(
            config=AgentConfig(agent_id="scout", name="Scout"), tools=registry, llm=llm
        )
        _take_one_turn(client, room_id, cast(Any, agent))

        assert (workspace / "out.md").exists(), "the helper wrote the file"
        body = client.get(f"/api/rooms/{room_id}/artifacts").json()
        assert body["artifacts"] == []
        assert body["unattributed_writes"] == 1
        assert "1 tool call(s) could have written files without naming them" in body["reason"]
        assert not _claims_none_written(body)
        history = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()
        (delegate,) = history["turns"][-1]["tools"]
        assert delegate["tool_name"] == "delegate_subagent"
        assert delegate["wrote_unnamed"] is True
        assert "helper" in history["tools_note"]
        assert not _claims_none_written(history)

    def test_a_shell_that_wrote_then_failed_is_counted(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """A shell that exits non-zero may already have written; its status says nothing.

        Killed by: src/uclone_x/room/orchestrator.py :: wrote_unnamed=(written_path is None and execution.writes_files)
        Becomes: wrote_unnamed=(written_path is None and execution.writes_files and succeeded)
        """
        from uclone_x.tools.builtin.shell import BashRunTool

        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        room_id = _create(client, ["scout"])
        registry = ToolRegistry()
        registry.register(cast(Any, BashRunTool()))
        command = f"echo hi > {workspace}/notes.md; exit 3"
        llm = MockLLMConnector(
            default_response="ok",
            tool_calls=(
                ToolCallRequest(
                    id="s1", name="bash_run", arguments={"action": "run", "command": command}
                ),
            ),
        )
        agent = BaseAgent(
            config=AgentConfig(agent_id="scout", name="Scout", workspace_dir=workspace),
            tools=registry,
            llm=llm,
        )
        _take_one_turn(client, room_id, cast(Any, agent))

        assert (workspace / "notes.md").exists(), "the shell wrote before it failed"
        body = client.get(f"/api/rooms/{room_id}/artifacts").json()
        assert body["artifacts"] == []
        assert body["unattributed_writes"] == 1
        assert not _claims_none_written(body)
        history = client.get(f"/api/rooms/{room_id}/seats/scout/history").json()
        (shell,) = history["turns"][-1]["tools"]
        assert shell["status"] == "error"
        assert shell["wrote_unnamed"] is True

    def test_turns_counted_before_the_start_counter_are_a_named_gap(
        self, client: TestClient
    ) -> None:
        """More turns landed than were counted starting: a lost one from then would not show.

        Killed by: src/uclone_x/ui/room_dock.py :: if unsaved < 0:
        Becomes: if unsaved < -9:
        """
        room_id = _create(client, ["scout"])
        record = _stack(client).service.get(room_id).file_record
        _seed(client, room_id, file_record=record.model_copy(update={"turns_landed": 2}))

        body = client.get(f"/api/rooms/{room_id}/artifacts").json()

        gap = "some turns were taken before this conversation counted turns as they started"
        assert body["record_gaps"] == [gap]
        assert gap in body["reason"]
        assert body["unsaved_turns"] == 0

    def test_no_response_in_any_state_says_no_file_was_written(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every state the file list and a seat's history can be in, and no categorical claim.

        Killed by: src/uclone_x/ui/room_dock.py :: reason = "No files are listed yet. " + FILES_SCOPE_NOTE
        Becomes: reason = "No file has been written through a tool in this conversation yet."
        """
        from uclone_x.room.models import RoomFileRecord

        room_id = _create(client, ["scout"])
        stack = _stack(client)
        states: list[dict[str, Any]] = [
            {},
            {"transcript": (_utterance(1, "scout", turn_id="t1", tools_recorded=True),)},
            {"file_record": RoomFileRecord(kept_since_creation=False)},
            {"file_record": RoomFileRecord(kept_since_creation=True, unattributed_writes=2)},
            {"file_record": RoomFileRecord(kept_since_creation=True, unrecorded_turns=1)},
            {"file_record": RoomFileRecord(kept_since_creation=True, clears=1, rewinds=1)},
            {"file_record": RoomFileRecord(kept_since_creation=True, turns_started=3)},
            {"file_record": RoomFileRecord(kept_since_creation=True, turns_landed=3)},
        ]
        seen: list[dict[str, Any]] = []
        for running in (False, True):

            def _in_flight(rid: str, answer: bool = running) -> bool:
                return answer

            monkeypatch.setattr(stack, "turn_in_flight", _in_flight)
            for update in states:
                _seed(client, room_id, **update)
                seen.append(client.get(f"/api/rooms/{room_id}/artifacts").json())
                seen.append(client.get(f"/api/rooms/{room_id}/seats/scout/history").json())

        assert len(seen) == 32
        claims = [b for b in seen if _claims_none_written(b)]
        assert claims == []
        assert all("may not appear here" in b["scope_note"] for b in seen if "artifacts" in b)

    def test_the_guard_recognises_the_claims_it_forbids(self) -> None:
        """So a passing guard is not a guard that matches nothing."""
        for claim in (
            "No file has been written through a tool in this conversation yet.",
            "No files were written.",
            "Nothing was written here.",
            "Scout wrote nothing.",
            "The seat did not write any files.",
            "so whether any file was written is not known.",
            # The negations #1388 N1 found passing the guard, and the tool-use claims.
            "They have not written any files.",
            "Scout has not written anything.",
            "Scout hasn't written a file yet.",
            "The critics haven't saved a file.",
            "No file written in this conversation.",
            "Nothing written yet.",
            "Scout never wrote to the workspace.",
            "Scout wrote nothing.",
            "Scout answered without using a tool.",
            "There was no tool use in this turn.",
            "No tool calls were made.",
            "0 tool calls.",
            "The seat used no tools.",
        ):
            assert _claims_none_written({"reason": claim}), claim
        for honest in (
            _NONE_LISTED,
            "Scout has not taken a turn in this conversation yet.",
            "so the files they wrote may not all be listed here.",
            "10 tool calls listed",
            "0 tool calls listed",
            "No tool calls listed.",
            "2 turn(s) started but stopped before their results were saved, or are still "
            "running in another copy of the app",
        ):
            assert not _claims_none_written({"reason": honest}), honest

    def test_no_string_the_dock_can_return_says_no_file_was_written(self) -> None:
        """Every literal in the module, reached by a test or not (docstrings excepted)."""
        import ast

        import uclone_x.ui.room_dock as dock

        tree = ast.parse(Path(dock.__file__).read_text(encoding="utf-8"))
        docstrings = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        }
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ]
        assert any("may not appear here" in s for s in literals), "the scan read the module"
        assert [s for s in literals if _NONE_WRITTEN.search(s)] == []


# --------------------------------------------------------------------------------------
# Remembers lists the facts a seat saved to memory (#1401), and says what its knowledge
# record can hold (#1404)
# --------------------------------------------------------------------------------------

#: Words that belong to a transport, a parser or a traceback, never to the dock's copy.
_TECHNICAL = re.compile(
    r"Traceback|Error\b|Exception|JSONDecode|Expecting|line \d+ column|\.json|/|\\\\|"
    r"HTTP \d|Internal Server|Failed to fetch|pydantic|validation error",
    re.IGNORECASE,
)

_SAVE_CALL = ToolCallRequest(
    id="m1",
    name="record_memory_fact",
    arguments={"subject": "Kenny", "predicate": "favourite_colour", "object_value": "teal"},
)


@pytest.fixture
def saving_client(tmp_path: Path) -> Iterator[TestClient]:
    """A dashboard whose model, asked anything, saves one fact to memory and then answers."""
    app = create_ui_app(
        static_dir=tmp_path,
        storage_dir=tmp_path / "sessions",
        workspace_dir=tmp_path / "workspace",
        llm=MockLLMConnector(default_response="Noted.", tool_calls=(_SAVE_CALL,)),
    )
    with TestClient(app) as started:
        yield started


def _knowledge(client: TestClient, room_id: str, seat: str) -> dict[str, Any]:
    answered = client.get(f"/api/rooms/{room_id}/knowledge", params={"agent_id": seat})
    assert answered.status_code == 200, answered.text
    return cast("dict[str, Any]", answered.json())


def _memory_file(seat: str) -> Path:
    from uclone_x.core.agent_home import AgentHome

    return AgentHome.for_username(seat).memory_path


def _record(
    seat: str, subject: str, predicate: str, value: str, session: str, room: str | None = None
) -> str:
    from uclone_x.core.provenance import Provenance
    from uclone_x.memory.store import default_cross_session_memory

    fact = default_cross_session_memory(seat).record_fact(
        subject=subject,
        predicate=predicate,
        object_value=value,
        provenance=Provenance.primary(provider=f"agent.{seat}", model="memory"),
        source_session_id=session,
        source_room_id=room,
    )
    return fact.fact_id


class TestRemembersListsTheClonesFacts:
    """One clone-wide list, each fact marked `learned_here` (#1401, #1638 step 3)."""

    def test_a_fact_saved_in_a_room_turn_is_listed_as_learned_here(
        self, saving_client: TestClient
    ) -> None:
        """The whole path: a real room turn, the seat's real `record_memory_fact`, the read.

        Only the model's reply is scripted. The seat is composed by the room's own resolver
        with the memory it gives every seat, the tool runs, and the fact is read back from
        where the tool saved it. `sage` names no shipped persona, so it has every tool, as a
        new clone does.

        Killed by: src/uclone_x/ui/room_dock.py :: **known,
        Becomes: **{"facts": [], "facts_reason": None},
        """
        room_id = _create(saving_client, ["sage"])
        saving_client.post(f"/api/rooms/{room_id}/messages", json={"content": "I like teal"})
        _wait_for_rows(saving_client, room_id, 2)

        body = _knowledge(saving_client, room_id, "sage")

        assert [f["statement"] for f in body["facts"]] == ["Kenny favourite colour teal"]
        (fact,) = body["facts"]
        assert fact["learned_here"] is True
        assert fact["origin"] == "saved"
        assert (fact["subject"], fact["predicate"], fact["object_value"]) == (
            "Kenny",
            "favourite_colour",
            "teal",
        )
        assert fact["fact_id"].startswith("mem_")
        assert fact["source_turn_id"], "the turn it was learned in is on the fact"
        assert body["facts_reason"] is None
        assert "saved_facts" not in body and "remembers" not in body

    def test_another_clones_facts_are_not_listed(self, client: TestClient) -> None:
        """Each clone's memory is its own (P7); `critic` must not show what `scout` saved.

        Killed by: src/uclone_x/ui/room_dock.py :: facts = read_saved_facts(seat.id)
        Becomes: facts = read_saved_facts("scout")
        """
        room_id = _create(client, ["scout", "critic"])
        _record("scout", "Kenny", "favourite_colour", "teal", "elsewhere")
        _record("critic", "Build", "status", "green", "elsewhere")

        scout = _knowledge(client, room_id, "scout")
        critic = _knowledge(client, room_id, "critic")

        assert [f["statement"] for f in scout["facts"]] == ["Kenny favourite colour teal"]
        assert [f["statement"] for f in critic["facts"]] == ["Build status green"]
        assert critic["facts"][0]["learned_here"] is False

    def test_learned_here_is_this_conversation_and_legacy_rows_go_by_session(
        self, client: TestClient
    ) -> None:
        """A fact names its conversation; one saved before #1716 goes by the seat's session.

        Killed by: src/uclone_x/ui/knowledge.py :: here = fact.source_room_id == room_id
        Becomes: here = fact.source_room_id != room_id
        Killed by: src/uclone_x/ui/knowledge.py :: here = session_id is not None and fact.source_session_id == session_id
        Becomes: here = session_id is not None
        """
        room_id = _create(client, ["scout"])
        room = client.get(f"/api/rooms/{room_id}").json()
        session_id = next(p["session_id"] for p in room["participants"] if p["id"] == "scout")
        _record("scout", "Kenny", "lives_in", "Seoul", "s-1", room=room_id)
        _record("scout", "Kenny", "works_at", "UClone", "s-2", room="another-room")
        _record("scout", "Kenny", "likes", "tea", session_id)
        _record("scout", "Kenny", "drives", "a bike", "a-chat")

        facts = _knowledge(client, room_id, "scout")["facts"]

        assert {f["statement"]: f["learned_here"] for f in facts} == {
            "Kenny lives in Seoul": True,
            "Kenny works at UClone": False,
            "Kenny likes tea": True,
            "Kenny drives a bike": False,
        }

    def test_facts_are_listed_for_a_seat_nobody_has_spoken_to_here(
        self, client: TestClient
    ) -> None:
        """A seat nobody has spoken to here still knows what it learned elsewhere.

        Killed by: src/uclone_x/ui/room_dock.py :: **known,
        Becomes: **{},
        """
        room_id = _create(client, ["scout"])
        _record("scout", "Kenny", "lives_in", "Seoul", "a-chat")

        body = _knowledge(client, room_id, "scout")

        assert body["status"] == "ok"
        assert [f["statement"] for f in body["facts"]] == ["Kenny lives in Seoul"]

    def test_a_retracted_fact_is_not_listed(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/memory/store.py :: if not fact.retracted]
        Becomes: if True]
        """
        from uclone_x.core.provenance import Provenance
        from uclone_x.memory.store import default_cross_session_memory

        room_id = _create(client, ["scout"])
        gone = _record("scout", "Kenny", "lives_in", "Busan", "a-chat")
        default_cross_session_memory("scout").retract_fact(
            gone, "moved", Provenance.primary(provider="agent.scout", model="memory")
        )
        _record("scout", "Kenny", "works_at", "UClone", "a-chat")

        body = _knowledge(client, room_id, "scout")

        assert [f["statement"] for f in body["facts"]] == ["Kenny works at UClone"]

    def test_with_no_facts_the_list_says_what_it_covers(self, client: TestClient) -> None:
        """An empty list is 'none listed', never 'the clone knows nothing' (P6)."""
        room_id = _create(client, ["scout"])

        body = _knowledge(client, room_id, "scout")

        assert body["facts"] == []
        assert body["facts_reason"] == "No facts are listed for scout."
        assert not _absence_claim(body["facts_reason"])

    def test_a_seat_whose_id_names_no_agent_home_lists_none_and_claims_no_read(self) -> None:
        """No memory can be saved under such an id, so there are no facts that failed to read.

        `memory_for` refuses the id as `read_saved_facts` does, so the answer is the empty
        list's sentence, not "could not be read" (#1429 review).

        Killed by: src/uclone_x/ui/room_dock.py :: except AgentHomeError:  # no home, so nothing saved to read
        Becomes: except ZeroDivisionError:  # no home, so nothing saved to read
        """
        from uclone_x.ui.room_dock import _known_facts  # pyright: ignore[reportPrivateUsage]

        seat = Participant(
            id="CON", kind=ParticipantKind.AGENT, display_name="Con", session_id="s-1"
        )

        answer, facts = _known_facts("room-1", seat)

        assert answer == {"facts": [], "facts_reason": "No facts are listed for Con."}
        assert facts == []

    def test_an_unreadable_memory_is_said_plainly_and_left_where_it_is(
        self, client: TestClient
    ) -> None:
        """A damaged document: a plain sentence, no parser words, no path, nothing moved.

        The failure is real -- the store's own parser meets bytes that are not JSON -- so
        the check reads whatever that failure would have put in front of the reader.

        Killed by: src/uclone_x/ui/room_dock.py :: "facts": None,
        Becomes: "facts": [],
        Killed by: src/uclone_x/ui/room_dock.py :: "worked_out": None if facts is None else worked_out_list(facts, axioms),
        Becomes: "worked_out": worked_out_list(facts or [], axioms),
        """
        room_id = _create(client, ["scout"])
        path = _memory_file("scout")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"facts": [ {"subject": ', encoding="utf-8")

        body = _knowledge(client, room_id, "scout")

        assert body["facts"] is None
        # Nothing to work out from, which is not "nothing follows" (P6).
        assert body["worked_out"] is None
        reason = body["facts_reason"]
        assert reason == "What scout knows could not be read, so it cannot be shown."
        assert not _TECHNICAL.search(reason), reason
        assert not _absence_claim(reason), reason
        assert path.read_text(encoding="utf-8") == '{"facts": [ {"subject": '
        assert not list(path.parent.glob("memory.json.unreadable-*")), "a read moved it"


# --------------------------------------------------------------------------------------
# Correct and Forget: PATCH / DELETE /api/agents/{agent_id}/memory/{fact_id} (#1638 step 3)
# --------------------------------------------------------------------------------------


def _file_facts(seat: str) -> dict[str, dict[str, Any]]:
    """Every fact in the clone's memory document, retracted ones included, read from disk."""
    import json

    document = json.loads(_memory_file(seat).read_text(encoding="utf-8"))
    return {f["fact_id"]: f for f in document["facts"]}


def _assert_plain_refusal(detail: Any, *internals: str) -> None:
    """A refusal a person reads: one plain sentence with no id, path or exception text."""
    assert isinstance(detail, str), detail
    assert not _TECHNICAL.search(detail), detail
    for internal in internals:
        assert internal not in detail, (internal, detail)
    assert "mem_" not in detail and "KeyError" not in detail, detail


class TestForgetAFact:
    def test_forget_retracts_durably_and_leaves_the_list_and_the_prompt(
        self, client: TestClient
    ) -> None:
        """Forget is a retraction on disk with the user's reason, never a hard delete.

        Killed by: src/uclone_x/ui/room_dock.py :: lambda: store.retract_fact(fact_id, "forgotten by the user", _user_provenance()),
        Becomes: lambda: store.get_fact(fact_id),
        """
        from uclone_x.memory.store import default_cross_session_memory

        room_id = _create(client, ["scout"])
        kept = _record("scout", "Kenny", "works_at", "UClone", "a-chat")
        gone = _record("scout", "Kenny", "lives_in", "Busan", "a-chat")

        answered = client.delete(f"/api/agents/scout/memory/{gone}")

        assert answered.status_code == 200, answered.text
        assert answered.json()["fact"]["retracted"] is True
        on_disk = _file_facts("scout")
        assert on_disk[gone]["retracted"] is True
        assert on_disk[gone]["retraction_reason"] == "forgotten by the user"
        assert on_disk[kept]["retracted"] is False
        facts = _knowledge(client, room_id, "scout")["facts"]
        assert [f["fact_id"] for f in facts] == [kept]
        # A store loaded afresh -- the next process, the next prompt -- does not use it.
        prompt = default_cross_session_memory("scout").format_prompt_section()
        assert "Busan" not in prompt and "UClone" in prompt

    def test_forget_reaches_a_running_clones_store(self, saving_client: TestClient) -> None:
        """The edit goes through the store the seat uses, so a running clone stops using it.

        Killed by: src/uclone_x/ui/room_dock.py :: return stack.session_manager().memory_for(agent_id)
        Becomes: return CrossSessionMemory(storage_path=stack.session_manager().memory_for(agent_id).storage_path)
        """
        room_id = _create(saving_client, ["sage"])
        saving_client.post(f"/api/rooms/{room_id}/messages", json={"content": "I like teal"})
        _wait_for_rows(saving_client, room_id, 2)
        (fact,) = _knowledge(saving_client, room_id, "sage")["facts"]
        live = _stack(saving_client).session_manager().memory_for("sage")
        assert live.get_fact(fact["fact_id"]) is not None

        answered = saving_client.delete(f"/api/agents/sage/memory/{fact['fact_id']}")

        assert answered.status_code == 200, answered.text
        held = live.get_fact(fact["fact_id"])
        assert held is not None and held.retracted
        assert "teal" not in live.format_prompt_section()

    def test_an_unknown_fact_is_a_plain_404(self, client: TestClient) -> None:
        _create(client, ["scout"])
        _record("scout", "Kenny", "works_at", "UClone", "a-chat")

        refused = client.delete("/api/agents/scout/memory/mem_000000000000")

        assert refused.status_code == 404
        detail = refused.json()["detail"]
        assert detail == (
            "This fact is no longer in the clone's memory. It may have been forgotten or "
            "corrected already."
        )
        _assert_plain_refusal(detail, "mem_000000000000", "scout")

    def test_forgetting_twice_is_a_plain_404_and_changes_nothing(self, client: TestClient) -> None:
        """The store answers a second retraction quietly; the route checks the fact is active.

        Killed by: src/uclone_x/ui/room_dock.py :: if not facts or all(fact.fact_id != fact_id for fact in facts):
        Becomes: if not facts:
        """
        _create(client, ["scout"])
        gone = _record("scout", "Kenny", "lives_in", "Busan", "a-chat")
        _record("scout", "Kenny", "works_at", "UClone", "a-chat")
        assert client.delete(f"/api/agents/scout/memory/{gone}").status_code == 200
        before = _file_facts("scout")

        refused = client.delete(f"/api/agents/scout/memory/{gone}")

        assert refused.status_code == 404
        _assert_plain_refusal(refused.json()["detail"], gone)
        assert _file_facts("scout") == before

    def test_a_clone_with_no_memory_or_no_home_is_a_plain_404_and_creates_nothing(
        self, client: TestClient
    ) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: except AgentHomeError:  # an id no clone can have, so no fact of its
        Becomes: except ZeroDivisionError:  # an id no clone can have, so no fact of its
        """
        _create(client, ["scout"])

        nothing_saved = client.delete("/api/agents/scout/memory/mem_000000000000")
        no_home = client.delete("/api/agents/NOT A NAME/memory/mem_000000000000")

        assert nothing_saved.status_code == 404
        assert no_home.status_code == 404
        _assert_plain_refusal(nothing_saved.json()["detail"], "scout")
        _assert_plain_refusal(no_home.json()["detail"], "NOT A NAME")
        assert not _memory_file("scout").exists()

    def test_an_unreadable_memory_refuses_plainly_and_is_left_where_it_is(
        self, client: TestClient
    ) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: raise _memory_edit_refusal(409, "memory_unreadable") from None
        Becomes: raise _memory_edit_refusal(404, "fact_not_found") from None
        """
        _create(client, ["scout"])
        path = _memory_file("scout")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"facts": [ {"subject": ', encoding="utf-8")

        refused = client.delete("/api/agents/scout/memory/mem_000000000000")

        assert refused.status_code == 409
        detail = refused.json()["detail"]
        assert detail == "The clone's memory could not be read, so nothing was changed."
        _assert_plain_refusal(detail, str(path), "JSONDecodeError")
        assert path.read_text(encoding="utf-8") == '{"facts": [ {"subject": '

    def test_a_failed_save_refuses_plainly(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: raise _memory_edit_refusal(500, "save_failed") from None
        Becomes: raise
        """
        from uclone_x.memory.store import CrossSessionMemory

        _create(client, ["scout"])
        gone = _record("scout", "Kenny", "lives_in", "Busan", "a-chat")

        def refuse(self: CrossSessionMemory) -> None:
            raise PermissionError(13, "Permission denied", "/secret/home/scout/memory.json")

        monkeypatch.setattr(CrossSessionMemory, "save", refuse)
        refused = client.delete(f"/api/agents/scout/memory/{gone}")

        assert refused.status_code == 500
        detail = refused.json()["detail"]
        assert detail == "The change could not be saved, so it may not last after a restart."
        _assert_plain_refusal(detail, "/secret", "Permission denied", gone)


class TestCorrectAFact:
    def test_correct_records_a_corrected_fact_and_keeps_the_old_one_retracted(
        self, client: TestClient
    ) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: lambda: store.correct_fact(fact_id, value, _user_provenance())
        Becomes: lambda: store.retract_fact(fact_id, "corrected by the user", _user_provenance())
        """
        room_id = _create(client, ["scout"])
        old = _record("scout", "Kenny", "lives_in", "Busan", "s-1", room=room_id)

        answered = client.patch(f"/api/agents/scout/memory/{old}", json={"value": " Seoul "})

        assert answered.status_code == 200, answered.text
        new = answered.json()["fact"]
        assert new["origin"] == "corrected"
        assert new["object_value"] == "Seoul"
        assert new["confidence"] == 1.0
        assert new["contradicts_fact_id"] == old
        on_disk = _file_facts("scout")
        assert on_disk[old]["retracted"] is True
        assert on_disk[old]["retraction_reason"] == "corrected by the user"
        assert on_disk[new["fact_id"]]["origin"] == "corrected"
        (listed,) = _knowledge(client, room_id, "scout")["facts"]
        assert listed["statement"] == "Kenny lives in Seoul"
        assert listed["origin"] == "corrected"
        assert listed["learned_here"] is True, "a correction stays where it was learned"

    def test_a_blank_value_is_refused_plainly_and_changes_nothing(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: if not isinstance(value, str) or not value.strip():
        Becomes: if not isinstance(value, str):
        """
        _create(client, ["scout"])
        old = _record("scout", "Kenny", "lives_in", "Busan", "a-chat")
        before = _file_facts("scout")

        for body in ({"value": "   "}, {"value": 7}, {}):
            refused = client.patch(f"/api/agents/scout/memory/{old}", json=body)
            assert refused.status_code == 400, body
            detail = refused.json()["detail"]
            assert detail == "Write what it should say instead."
            _assert_plain_refusal(detail, old)
        assert _file_facts("scout") == before

    def test_correcting_an_unknown_fact_is_a_plain_404(self, client: TestClient) -> None:
        _create(client, ["scout"])
        _record("scout", "Kenny", "lives_in", "Busan", "a-chat")

        refused = client.patch("/api/agents/scout/memory/mem_000000000000", json={"value": "x"})

        assert refused.status_code == 404
        _assert_plain_refusal(refused.json()["detail"], "mem_000000000000")


class TestAnEmptyKnowledgeRecordIsNotPresentedAsNothingLearned:
    """#1404: nothing in a room turn writes to a clone's rules engine."""

    def test_an_empty_record_gives_no_reason_of_its_own(self, client: TestClient) -> None:
        """The Core adds no sentence to an empty list; the head says only that none are listed.

        Killed by: src/uclone_x/ui/room_dock.py :: "reason": None,
        Becomes: "reason": f"{seat.display_name} remembers nothing from this conversation.",
        """
        room_id = _create(client, ["scout"])
        client.post(f"/api/rooms/{room_id}/messages", json={"content": "hello"})
        _wait_for_rows(client, room_id, 2)

        body = _knowledge(client, room_id, "scout")

        assert body["status"] == "ok" and body["triples"] == []
        assert body["reason"] is None


class TestRoomDockTurnSummaryRoutes:
    """Covers #1491 GET /api/rooms/{room_id}/turns/{seq}."""

    def test_read_turn_summary_200(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: return summary.model_dump(mode="json")
        Becomes: return {}
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
            tool_uses=(_use("t1", written_path="file.txt"),),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["seq"] == 1
        assert body["turn_id"] == "t1"
        assert len(body["documents"]) == 1
        assert body["documents"][0]["path"] == "file.txt"

    def test_read_turn_summary_404(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: "code": "turn_not_found",
        Becomes: "code": "turn_found",
        """
        room_id = _create(client, ["scout"])
        resp = client.get(f"/api/rooms/{room_id}/turns/999")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "turn_not_found"


def _absence_claim(text: str) -> bool:
    """The frontend guard's claims that nothing was remembered or learned, for Core copy."""
    return bool(
        re.search(
            r"\b(?:remembers?|remembered|knows|learned|learnt) nothing\b"
            r"|\b(?:has|have|had|did|does|do)(?: not|n't) (?:remember(?:ed)?|learn(?:ed|t)?|saved?)\b"
            r"|\bnothing (?:(?:is|was|has been|had been|were) )?(?:remembered|learned|learnt|saved)\b"
            r"|\bnever (?:saved|remembered|learned)\b",
            text,
            re.IGNORECASE,
        )
    )


class TestRoomDockTurnTraceRoutes:
    """Covers #1490 GET /api/rooms/{room_id}/turns/{seq}/trace and step detail routes."""

    def test_trace_turn_not_found_returns_404(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: "message": f"Turn {seq} was not found in conversation {room_id}.",
        Becomes: "message": f"Turn {seq} was found in conversation {room_id}.",
        """
        room_id = _create(client, ["scout"])
        resp = client.get(f"/api/rooms/{room_id}/turns/999/trace")
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "turn_not_found"
        assert (
            resp.json()["detail"]["message"] == f"Turn 999 was not found in conversation {room_id}."
        )

    def test_trace_not_an_agent_turn(self, client: TestClient) -> None:
        """A person's row -- no turn id -- from the seated human.

        Killed by: src/uclone_x/ui/room_dock.py :: return participant.kind == ParticipantKind.AGENT
        Becomes: return True
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "user", turn_id=None, tools_recorded=False),),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is None
        assert body["reason"]["code"] == "not_an_agent_turn"

    def test_trace_turn_not_linked(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: return _TRACE_REASONS["turn_not_linked"]
        Becomes: return None
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id=None, tools_recorded=False),),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is None
        assert body["reason"]["code"] == "turn_not_linked"

    def test_trace_turn_not_saved(self, client: TestClient) -> None:
        """A row whose save failed, and whose seat has no saved session, says so.

        Killed by: src/uclone_x/ui/room_dock.py :: return _TRACE_REASONS["turn_not_saved" if unsaved else "session_not_found"]
        Becomes: return _TRACE_REASONS["session_not_found"]
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(
                    1, "scout", turn_id="t1", tools_recorded=False, persist_error="disk full"
                ),
            ),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is None
        assert body["reason"]["code"] == "turn_not_saved"

    def test_trace_session_not_found(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: if session_state is None:
        Becomes: if False:
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is None
        assert body["reason"]["code"] == "session_not_found"

    def test_trace_log_missing(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: if log_path is None or not log_path.exists():
        Becomes: if log_path is None:
        """
        room_id = _create(client, ["scout"])
        room = client.get(f"/api/rooms/{room_id}").json()
        session_id = next(p["session_id"] for p in room["participants"] if p["id"] == "scout")
        stack = _stack(client)
        store = stack.session_manager().core_store
        store.save(SessionState(session_id=session_id, agent_id="scout"))
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is None
        assert body["reason"]["code"] == "log_missing"

    def test_trace_log_unreadable(self, client: TestClient) -> None:
        """Should-fix (c): the reader gets a plain sentence and a stable code, no path or
        exception text; the route is not gated on developer mode.

        Killed by: src/uclone_x/ui/room_dock.py :: return {**_TRACE_REASONS["log_unreadable"], "detail": _log_failure_kind(exc)}
        Becomes: return {"code": "log_unreadable", "message": str(exc)}
        Killed by: src/uclone_x/ui/room_dock.py :: return "malformed_log"
        Becomes: return "unexpected"
        """
        room_id = _create(client, ["scout"])
        room = client.get(f"/api/rooms/{room_id}").json()
        session_id = next(p["session_id"] for p in room["participants"] if p["id"] == "scout")
        stack = _stack(client)
        store = stack.session_manager().core_store
        store.save(SessionState(session_id=session_id, agent_id="scout"))
        log_path = store.event_log_path(session_id)
        assert log_path is not None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("{bad json\n", encoding="utf-8")
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
        )
        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is None
        assert body["reason"] == {
            "code": "log_unreadable",
            "message": "This seat's session log could not be read, so the turn cannot be traced.",
            "detail": "malformed_log",
        }
        step = client.get(f"/api/rooms/{room_id}/turns/1/trace/steps/1")
        assert step.status_code == 200
        assert step.json()["reason"] == body["reason"]
        for leaked in (str(log_path.parent), log_path.name, "JSON", "Expecting", "bad json"):
            assert leaked not in resp.text and leaked not in step.text, leaked

    def test_trace_success_and_step_detail(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: return {**head, "trace": outcome}
        Becomes: return {**head, "trace": None}
        """
        room_id = _create(client, ["scout"])
        room = client.get(f"/api/rooms/{room_id}").json()
        session_id = next(p["session_id"] for p in room["participants"] if p["id"] == "scout")
        stack = _stack(client)
        store = stack.session_manager().core_store
        store.save(SessionState(session_id=session_id, agent_id="scout"))
        log_path = store.event_log_path(session_id)
        assert log_path is not None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        header = {"schema": "uclone_x.log.v0", "version": "0.1.0-dev", "session_id": session_id}
        events = [
            header,
            {
                "type": "TURN_START",
                "turn_index": 1,
                "caller_turn_id": "t1",
                "at": "2026-09-23T10:00:00Z",
            },
            {"type": "MODEL_RESPONSE", "turn_index": 1, "step": 1, "content": "hello trace"},
            {
                "type": "TURN_END",
                "turn_index": 1,
                "outcome": "completed",
                "at": "2026-09-23T10:00:01Z",
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
        )

        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"] is not None
        assert body["trace"]["turn_index"] == 1
        assert len(body["trace"]["steps"]) == 1
        assert body["trace"]["steps"][0]["response"]["content"] == "hello trace"

        # Step detail route
        step_resp = client.get(f"/api/rooms/{room_id}/turns/1/trace/steps/1")
        assert step_resp.status_code == 200
        step_body = step_resp.json()
        assert step_body["step"] == 1
        assert step_body["response"]["content"] == "hello trace"

        # Step not found
        step_404 = client.get(f"/api/rooms/{room_id}/turns/1/trace/steps/99")
        assert step_404.status_code == 404
        assert step_404.json()["detail"]["code"] == "step_not_found"
        assert step_body["reason"] is None

    @staticmethod
    def _scout_session(client: TestClient, room_id: str) -> str:
        room = client.get(f"/api/rooms/{room_id}").json()
        return str(next(p["session_id"] for p in room["participants"] if p["id"] == "scout"))

    def _real_turns(self, client: TestClient, room_id: str, count: int = 3) -> tuple[Any, Path]:
        """`count` one-step turns by the real agent loop, in the room's own session store."""
        session_id = self._scout_session(client, room_id)
        store = _stack(client).session_manager().core_store
        agent = BaseAgent(
            config=AgentConfig(agent_id="scout", name="Scout"),
            llm=MockLLMConnector(responses=[f"reply {n}" for n in range(1, count + 1)]),
            context=AgentContext(session_id=session_id, agent_id="scout"),
            store=store,
        )

        async def run() -> None:
            await agent.start()
            for n in range(1, count + 1):
                assert (await agent.execute_turn(f"turn {n}", caller_turn_id=f"t{n}")).is_completed
            agent.persist_session()

        asyncio.run(run())
        log_path = store.event_log_path(session_id)
        assert log_path is not None and log_path.exists()
        _seed(
            client,
            room_id,
            transcript=tuple(
                _utterance(n, "scout", turn_id=f"t{n}", tools_recorded=True)
                for n in range(1, count + 1)
            ),
        )
        return store, log_path

    @staticmethod
    def _drop_request(log_path: Path, request: int) -> None:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        kept = [
            line
            for line in lines
            if not (
                json.loads(line).get("type") == "REQUEST_CONTEXT"
                and json.loads(line).get("request") == request
            )
        ]
        assert len(kept) == len(lines) - 1
        log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    def test_a_gap_before_the_turn_is_a_stated_reason_on_its_step(self, client: TestClient) -> None:
        """Blocker 2, through the route: a request missing before the turn is not a 500.

        Killed by: src/uclone_x/agent/request_record.py :: on_error(step, err)
        Becomes: raise
        """
        room_id = _create(client, ["scout"])
        _, log_path = self._real_turns(client, room_id)
        self._drop_request(log_path, 1)
        tolerant = TestClient(client.app, raise_server_exceptions=False)

        resp = tolerant.get(f"/api/rooms/{room_id}/turns/2/trace")
        assert resp.status_code == 200, resp.text
        (step,) = resp.json()["trace"]["steps"]
        assert step["request_status"] == "unavailable"
        assert step["request_reason"] == "request 2 extends 1, last seen None"
        assert step["request_code"] == "chain_broken"
        assert step["response"]["content"] == "reply 2"

        detail = tolerant.get(f"/api/rooms/{room_id}/turns/2/trace/steps/1")
        assert detail.status_code == 200, detail.text
        assert detail.json()["reason"] is None
        assert detail.json()["request"] is None
        assert detail.json()["request_reason"] == "request 2 extends 1, last seen None"
        assert detail.json()["request_code"] == "chain_broken"

    def test_a_gap_after_the_turn_is_not_read(self, client: TestClient) -> None:
        """Blocker 2: a gap after the traced turn, and the request built on it, cost it nothing.

        Request 3 is gone, so request 4 extends a request the log does not hold; the
        traced turn 2 is step 1 of its turn, as request 4 is of its own.

        Also killed by one compound mutant that restores the #1490 defect (a later gap
        aborting the fold), in `src/uclone_x/agent/request_record.py`: drop the truncation
        `requests = requests[: chosen[-1] + 1]`, and make the unselected-request `continue`
        under `if id(event) not in chosen_ids:` raise `broken` when it is set.
        """
        room_id = _create(client, ["scout"])
        _, log_path = self._real_turns(client, room_id, count=4)
        self._drop_request(log_path, 3)

        resp = client.get(f"/api/rooms/{room_id}/turns/2/trace")
        assert resp.status_code == 200
        (step,) = resp.json()["trace"]["steps"]
        assert (step["request_status"], step["verified"]) == ("ok", True)
        detail = client.get(f"/api/rooms/{room_id}/turns/2/trace/steps/1").json()
        assert detail["verified"] is True and detail["reason"] is None

    def test_a_turn_whose_row_save_failed_is_still_traced_when_its_record_is_there(
        self, client: TestClient
    ) -> None:
        """`persist_error` is the row's save; the seat's log may hold the turn all the same.

        Killed by: src/uclone_x/ui/room_dock.py :: _TRACE_REASONS["turn_not_saved" if unsaved else "turn_not_linked"]
        Becomes: _TRACE_REASONS["turn_not_saved"]
        """
        room_id = _create(client, ["scout"])
        self._real_turns(client, room_id)
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(1, "scout", turn_id="t1", tools_recorded=True, persist_error="disk"),
                _utterance(2, "scout", turn_id="t9", tools_recorded=True),
                _utterance(3, "scout", turn_id="t8", tools_recorded=True, persist_error="disk"),
            ),
        )

        assert client.get(f"/api/rooms/{room_id}/turns/1/trace").json()["trace"] is not None
        unlinked = client.get(f"/api/rooms/{room_id}/turns/2/trace").json()
        assert unlinked["reason"]["code"] == "turn_not_linked"
        unsaved = client.get(f"/api/rooms/{room_id}/turns/3/trace").json()
        assert unsaved["reason"]["code"] == "turn_not_saved"
        step = client.get(f"/api/rooms/{room_id}/turns/2/trace/steps/1")
        assert step.status_code == 200
        assert step.json()["reason"]["code"] == "turn_not_linked"

    def test_the_step_route_gives_the_trace_routes_reasons(self, client: TestClient) -> None:
        """Should-fix (a): only a step the turn lacks is `step_not_found`.

        Killed by: src/uclone_x/ui/room_dock.py :: return {**empty, "reason": outcome}
        Becomes: raise HTTPException(status_code=404, detail={"code": "step_not_found"})
        Killed by: src/uclone_x/ui/room_dock.py :: return {**empty, "reason": reason}
        Becomes: raise HTTPException(status_code=404, detail={"code": "step_not_found"})
        """
        room_id = _create(client, ["scout"])
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(1, "user", turn_id=None, tools_recorded=False),
                _utterance(2, "scout", turn_id=None, tools_recorded=False),
                _utterance(3, "scout", turn_id="t1", tools_recorded=True),
            ),
        )
        for seq, code in (
            (1, "not_an_agent_turn"),
            (2, "turn_not_linked"),
            (3, "session_not_found"),
        ):
            trace = client.get(f"/api/rooms/{room_id}/turns/{seq}/trace")
            step = client.get(f"/api/rooms/{room_id}/turns/{seq}/trace/steps/1")
            assert step.status_code == 200, (seq, step.text)
            assert trace.json()["reason"]["code"] == code
            assert step.json()["reason"] == trace.json()["reason"]
            assert step.json()["request"] is None and step.json()["response"] is None
        missing = client.get(f"/api/rooms/{room_id}/turns/404/trace/steps/1")
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "turn_not_found"

    def test_an_agent_that_left_the_room_is_still_an_agent(self, client: TestClient) -> None:
        """Should-fix (b): the row answers for itself once its seat is gone.

        Killed by: src/uclone_x/ui/room_dock.py :: linked = message.turn_id is not None or
        Becomes: linked = False or
        """
        room_id = _create(client, ["scout", "helper"])
        _stack(client).service.remove_participant(room_id, "helper")
        _seed(
            client,
            room_id,
            transcript=(
                _utterance(1, "helper", turn_id="t1", tools_recorded=True),
                _utterance(2, "someone", turn_id=None, tools_recorded=False),
            ),
        )

        left = client.get(f"/api/rooms/{room_id}/turns/1/trace").json()
        assert left["participant_id"] == "helper"
        assert left["reason"]["code"] == "session_not_found"
        person = client.get(f"/api/rooms/{room_id}/turns/2/trace").json()
        assert person["reason"]["code"] == "not_an_agent_turn"

    def test_an_unknown_event_type_is_named_by_kind(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/room_dock.py :: return "unknown_event_type"
        Becomes: return "unexpected"
        """
        room_id = _create(client, ["scout"])
        session_id = self._scout_session(client, room_id)
        store = _stack(client).session_manager().core_store
        store.save(SessionState(session_id=session_id, agent_id="scout"))
        log_path = store.event_log_path(session_id)
        assert log_path is not None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        header = {"schema": "uclone_x.log.v0", "version": "0.1.0-dev", "session_id": session_id}
        log_path.write_text(
            json.dumps(header) + "\n" + json.dumps({"type": "NOT_AN_EVENT_TYPE"}) + "\n",
            encoding="utf-8",
        )
        _seed(
            client,
            room_id,
            transcript=(_utterance(1, "scout", turn_id="t1", tools_recorded=True),),
        )

        resp = client.get(f"/api/rooms/{room_id}/turns/1/trace")
        assert resp.json()["reason"]["detail"] == "unknown_event_type"
        assert "NOT_AN_EVENT_TYPE" not in resp.text and str(log_path.parent) not in resp.text

    def test_an_unreadable_session_or_a_failing_trace_is_a_stated_reason(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P6: nothing leaves the routes as a bare 500, and no exception text leaves them.

        Killed by: src/uclone_x/ui/room_dock.py :: return "reason", _TRACE_REASONS["trace_failed"]
        Becomes: raise
        Killed by: src/uclone_x/ui/room_dock.py :: return _TRACE_REASONS["session_unreadable"]
        Becomes: raise
        """
        import uclone_x.ui.room_dock as room_dock

        room_id = _create(client, ["scout"])
        _, log_path = self._real_turns(client, room_id)
        tolerant = TestClient(client.app, raise_server_exceptions=False)
        secret = f"boom at {log_path}"

        def failing(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(secret)

        monkeypatch.setattr(room_dock, "trace_turn", failing)
        monkeypatch.setattr(room_dock, "trace_step", failing)
        for url in ("trace", "trace/steps/1"):
            resp = tolerant.get(f"/api/rooms/{room_id}/turns/1/{url}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["reason"]["code"] == "trace_failed"
            assert "boom" not in resp.text

        store = _stack(client).session_manager().core_store
        monkeypatch.setattr(store, "load", failing)
        for url in ("trace", "trace/steps/1"):
            resp = tolerant.get(f"/api/rooms/{room_id}/turns/1/{url}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["reason"]["code"] == "session_unreadable"
            assert "boom" not in resp.text


_LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"


@pytest.mark.parametrize("language", ["en", "ko"])
def test_every_log_failure_kind_has_a_sentence_in_the_head(language: str) -> None:
    """The `log_unreadable` detail is a closed set of codes the head words (#1907).

    Equal, not contained: a kind the catalog lacks reads as a bare code on a Korean screen.

    Killed by: src/uclone_x/ui/room_dock.py :: "not_text", "read_failed", "unexpected"
    Becomes: "not_text", "read_failed", "renamed"
    """
    catalog = json.loads((_LOCALES / language / "dock.json").read_text(encoding="utf-8"))
    assert set(get_args(LogFailureKind)) == set(catalog["modelCalls"]["logFailures"])


@pytest.mark.parametrize("language", ["en", "ko"])
def test_every_topology_code_has_a_sentence_in_the_head(language: str) -> None:
    """The topology's reason and gap codes are closed sets the head words (#1911).

    Equal, not contained, as for the model-call codes (#1907).

    Killed by: src/uclone_x/ui/room_dock.py :: HistoryGapCode = Literal["before_record", "cleared", "rewound", "unsaved", "uncounted"]
    Becomes: HistoryGapCode = Literal["before_record", "cleared", "rewound", "unsaved", "renamed"]
    Killed by: src/uclone_x/ui/room_dock.py :: ToolCallGapCode = Literal["saved_without_tools", "unreported"]
    Becomes: ToolCallGapCode = Literal["saved_without_tools", "renamed"]
    Killed by: src/uclone_x/ui/room_dock.py :: TopologyReasonCode = Literal["no_seat"]
    Becomes: TopologyReasonCode = Literal["renamed"]
    """
    topology = json.loads((_LOCALES / language / "dock.json").read_text(encoding="utf-8"))[
        "topology"
    ]
    assert set(get_args(HistoryGapCode)) == set(topology["historyGapReasons"])
    assert set(get_args(ToolCallGapCode)) == set(topology["toolCallGapReasons"])
    assert set(get_args(TopologyReasonCode)) == set(topology["reasons"])
