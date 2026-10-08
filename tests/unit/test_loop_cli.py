"""Unit tests for ucx loop CLI command (Issue FR-Loop, P4, P6)."""

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.support.clones import make_clones
from uclone_x.agent import clone_builder
from uclone_x.agent.session import SessionStore
from uclone_x.cli.commands.loop import _run_loop_agent  # pyright: ignore[reportPrivateUsage]
from uclone_x.cli.main import app
from uclone_x.core.agent_home import seat_id_for
from uclone_x.llm.models import MessageRole
from uclone_x.room.models import ParticipantKind
from uclone_x.room.one_seat import ONE_SEAT_HUMAN_ID
from uclone_x.room.service import RoomService, participant_session_id
from uclone_x.room.store import RoomStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _loop_clones() -> None:  # pyright: ignore[reportUnusedFunction]
    """The clones these loops run as; a name no clone carries is refused (clone-data-scopes §3.4)."""
    make_clones("looper", "owner", "visitor")


def test_ucx_loop_help() -> None:
    result = runner.invoke(app, ["loop", "--help"])
    assert result.exit_code == 0
    assert "Execute recurring agent automation loops" in result.output
    assert "run" in result.output


def test_ucx_loop_run_help() -> None:
    result = runner.invoke(app, ["loop", "run", "--help"])
    assert result.exit_code == 0
    assert "--interval" in result.output
    assert "--max-runs" in result.output
    assert "--timeout" in result.output
    assert "--clean" in result.output
    assert "--until" in result.output


def test_ucx_loop_run_invalid_interval() -> None:
    result = runner.invoke(app, ["loop", "run", "--interval", "invalid_val", "some prompt"])
    assert result.exit_code == 2
    assert (
        "Invalid --interval argument" in result.output or "Invalid interval format" in result.output
    )


def test_ucx_loop_run_unrecognized_interval_in_text() -> None:
    result = runner.invoke(app, ["loop", "run", "just do something without any interval"])
    assert result.exit_code == 2
    assert "Could not determine interval" in result.output


def _loop_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    agent_name: str,
    session_id: str | None = None,
) -> int:
    """One tick of `ucx loop run` on the mock provider, with sessions under `tmp_path`."""
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))

    def _no_binder(llm: object) -> None:
        return None

    # No binder: a local connector's binder would embed against a model on this machine.
    monkeypatch.setattr(clone_builder, "connector_tool_binder", _no_binder)
    return asyncio.run(
        _run_loop_agent(
            "tick",
            3600.0,
            max_runs=1,
            provider="mock",
            workspace_dir=tmp_path,
            agent_name=agent_name,
            session_id=session_id,
        )
    )


def _seat(handle: str) -> str:
    """The id `handle`'s clone is seated by: a seat is keyed by clone id (§4 step 3)."""
    return seat_id_for(handle)


def _user_turns(session_id: str) -> list[str]:
    state = SessionStore().load(session_id)
    assert state is not None, f"no session {session_id!r} was saved"
    return [m.content or "" for m in state.messages if m.role == MessageRole.USER]


def test_a_loop_is_a_one_seat_room_with_a_seat_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A loop started without `--session-id` stores a room seating a person and the clone,
    and the clone's turns are saved in that seat's session, not `loop_<clone>` (§5.9).

    Killed by: src/uclone_x/cli/commands/loop.py :: session_id=room.session_id,
    Becomes: session_id=f"loop_{agent_name}",
    """
    assert _loop_once(monkeypatch, tmp_path, agent_name="looper") == 0

    rooms = RoomStore().list_room_ids()
    assert len(rooms) == 1, rooms
    room = RoomStore().load(rooms[0])
    assert room is not None
    assert [(p.id, p.kind) for p in room.participants] == [
        (ONE_SEAT_HUMAN_ID, ParticipantKind.HUMAN),
        (_seat("looper"), ParticipantKind.AGENT),
    ]
    assert room.head == "loop"  # the app will not post into it (#1885)
    assert _user_turns(participant_session_id(rooms[0], _seat("looper"))) == ["tick"]
    assert SessionStore().load("loop_looper") is None


def test_a_loop_tick_is_in_the_room_transcript(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each tick lands in the room's transcript as the person's prompt and the clone's reply,
    so the app shows the loop's turns, not a roster alone (#1837).

    Killed by: src/uclone_x/cli/commands/loop.py :: turn=loop_tick_turn(job, result, session_set_aside=set_aside),
    Becomes: turn=None,
    """
    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="nightly") == 0
    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="nightly") == 0

    state = RoomStore().load("nightly")
    assert state is not None
    spoken = [(m.sender_id, m.content) for m in state.transcript if m.kind == "utterance"]
    assert [sender for sender, _ in spoken] == [ONE_SEAT_HUMAN_ID, _seat("looper")] * 2
    assert [content for sender, content in spoken if sender == ONE_SEAT_HUMAN_ID] == [
        "tick",
        "tick",
    ]
    assert state.file_record.turns_landed == 2


def test_a_loop_says_it_kept_a_record_it_could_not_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A loop over a newer build's seat record says it kept it, on stderr and on the row (#1877).

    Before, `ucx loop` set the record aside and only logged it.

    Killed by: src/uclone_x/cli/commands/loop.py :: set_aside = report_set_aside(store, room.session_id, err_console)
    Becomes: set_aside = False
    """
    from uclone_x.agent.session import SessionState
    from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE

    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    seat = participant_session_id("nightly", _seat("looper"))
    document = json.loads(SessionState.seed(seat, _seat("looper")).model_dump_json())
    document["a_field_from_a_newer_build"] = True
    path = SessionStore().session_path(seat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="nightly") == 0

    err = " ".join(capsys.readouterr().err.split())
    assert err.count(SESSION_SET_ASIDE_NOTICE) == 1, err
    state = RoomStore().load("nightly")
    assert state is not None
    rows = [m for m in state.transcript if m.sender_id == _seat("looper") and m.kind == "utterance"]
    assert [m.session_set_aside for m in rows] == [True]


def test_a_loop_says_on_exit_that_its_last_save_kept_a_record_aside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The save on the way out sets the record aside: the loop says so once, then exits (#1921).

    The tick's own save fails here, so the unreadable record is still in place when the loop
    stops, and only the save on exit moves it. The notice is the one `ucx run` prints.

    Killed by: src/uclone_x/cli/commands/loop.py :: report_set_aside(store, room.session_id, err_console)  # the save on the way out
    Becomes: pass
    """
    from uclone_x.agent.base import BaseAgent
    from uclone_x.agent.session import SessionState
    from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE, set_aside_copies

    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    seat = participant_session_id("nightly", _seat("looper"))
    document = json.loads(SessionState.seed(seat, _seat("looper")).model_dump_json())
    document["a_field_from_a_newer_build"] = True
    path = SessionStore().session_path(seat)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    persist = BaseAgent.persist_session
    calls: list[int] = []

    def _first_save_fails(self: BaseAgent, *args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("the tick's save failed")
        return persist(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BaseAgent, "persist_session", _first_save_fails)

    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="nightly") == 0

    assert len(calls) == 2  # the tick's save, then the one on the way out
    assert len(set_aside_copies(path)) == 1
    err = " ".join(capsys.readouterr().err.split())
    assert err.count(SESSION_SET_ASIDE_NOTICE) == 1, err


def test_a_loop_resumes_the_room_session_id_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--session-id` names the room; a second loop in it continues the seat's session.

    Killed by: src/uclone_x/cli/commands/loop.py :: room = open_head_room(clone_id, session_id, head="loop", store=rooms)
    Becomes: room = open_head_room(clone_id, None, head="loop", store=rooms)
    """
    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="nightly") == 0
    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="nightly") == 0

    assert RoomStore().list_room_ids() == ("nightly",)
    assert _user_turns(participant_session_id("nightly", _seat("looper"))) == ["tick", "tick"]


def test_a_loop_in_another_clones_room_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Naming a room another clone holds exits 2 before any turn, naming the room.

    Killed by: src/uclone_x/cli/commands/loop.py ::
        except (PathTraversalError, RoomError) as exc:
    Becomes: except PathTraversalError as exc:
    """
    assert _loop_once(monkeypatch, tmp_path, agent_name="owner", session_id="theirs") == 0

    assert _loop_once(monkeypatch, tmp_path, agent_name="visitor", session_id="theirs") == 2
    assert SessionStore().load(participant_session_id("theirs", _seat("visitor"))) is None


def test_a_loop_in_a_room_with_other_clones_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A room seating another clone beside this one is not a loop's: exit 2, no turn.

    Killed by: src/uclone_x/room/one_seat.py :: if not is_one_seat(state.participants):
    Becomes: if False:
    """
    rooms = RoomService(RoomStore())
    # The loop's own head, so the one-seat guard is the only refusal left to make.
    rooms.create(
        title="both",
        room_id="room_multi",
        head="loop",
        seats=(
            (ONE_SEAT_HUMAN_ID, ParticipantKind.HUMAN),
            (_seat("looper"), ParticipantKind.AGENT),
            ("champion", ParticipantKind.AGENT),
        ),
    )

    assert _loop_once(monkeypatch, tmp_path, agent_name="looper", session_id="room_multi") == 2
    assert SessionStore().load(participant_session_id("room_multi", _seat("looper"))) is None
