"""The rows a failed turn and a failed save land on, as the head's tests read them (#1408).

A row's `error` and `persist_error` are the raw cause -- `f"{type(exc).__name__}: {exc}"`
or the agent's own failure text -- and the head used to print them on the row line, so a
reader who does not read code saw class names and the absolute path of a file under their
home directory. The head now states each in a fixed plain sentence and leaves the cause on
the field for the log's reader.

A test of that head is only as good as its input. Fixture strings written for the purpose
tend to look clean already ("provider unavailable"), and a check that the row does not
render them then passes against a head that renders them. So the head's test reads its rows
from `frontend/src/test/room-failed-rows.json`, and this module writes that file from real
failures and fails the moment the two disagree:

* a turn whose agent raised a real `FileNotFoundError` on a missing file, landing through
  the orchestrator's own `except Exception` branch;
* a turn the Core refused on a spent token budget, through a real `BaseAgent`;
* a turn whose provider failed on a real read of a missing file, which `BaseAgent` returns
  as the result's `error` rather than raising;
* a reply whose seat session could not be written, because the session directory is a
  file -- a real failing write in the real `SessionStore`.

`tmp_path` is replaced by `/home/reader/.uclone` rather than by a token without a slash, so
the head's "no path" check still has a path to catch.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import TurnResult
from uclone_x.agent.session import SessionState, SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, StreamChunk
from uclone_x.room.models import Participant, ParticipantKind, RoomMessage
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry

ROOM = "room_failed_rows"
SEAT = "scout"
#: What `tmp_path` becomes in the fixture: still a path, so the head's check has one to catch.
READER_HOME = "/home/reader/.uclone"
FIXTURE = (
    Path(__file__).resolve().parents[2] / "frontend" / "src" / "test" / "room-failed-rows.json"
)
#: The row fields the head reads to draw a failure. Timestamps and ids are left out: they
#: differ on every run and the head's failure copy reads none of them.
FIELDS = (
    "seq",
    "sender_id",
    "kind",
    "content",
    "completed",
    "error",
    "refusal",
    "persist_error",
    "knowledge_persist_error",
)


def _seated_room(tmp_path: Path) -> RoomStore:
    store = RoomStore(tmp_path / "rooms")
    service = RoomService(store)
    service.create("Failures", room_id=ROOM)
    service.add_participant(ROOM, "user", kind=ParticipantKind.HUMAN)
    service.add_participant(ROOM, SEAT)
    return store


def _host(
    sessions_dir: Path, llm: MockLLMConnector, budget: TokenBudgetManager | None = None
) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=llm,
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(sessions_dir),
        budget=budget,
    )


class _ReadsAMissingFile(MockLLMConnector):
    """A provider whose every call reads a file that is not there."""

    def __init__(self, missing: Path) -> None:
        super().__init__(default_model="mock-gpt-4o")
        self.missing = missing

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.missing.read_text(encoding="utf-8")
        return await super().generate(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.missing.read_text(encoding="utf-8")
        async for chunk in super().stream(request):
            yield chunk


class _RaisingAgent:
    """An agent whose turn raises: the orchestrator's own `except Exception` branch."""

    def __init__(self, missing: Path) -> None:
        self.missing = missing

    async def execute_turn(self, prompt: str, *, stream_callback: Any = None) -> TurnResult:
        self.missing.read_text(encoding="utf-8")
        raise AssertionError("unreachable: the read above raises")

    def checkpoint_turn(self, session_id: str | None = None) -> SessionState:
        return SessionState(session_id=session_id or "x", agent_id=SEAT)

    def roll_back_turn(self, checkpoint: SessionState, *, reason: str) -> int:
        return 0

    def persist_session(self, session_id: str | None = None) -> None:
        return None


class _OneAgentResolver:
    def __init__(self, agent: Any) -> None:
        self.agent = agent

    async def resolve(self, participant: Participant) -> Any:
        return self.agent


async def _last_row(tmp_path: Path, resolver: Any) -> RoomMessage:
    orchestrator = RoomOrchestrator(
        store=_seated_room(tmp_path), selectors=(MentionSelector(),), resolver=resolver
    )
    state = await orchestrator.post(ROOM, "user", "@scout go")
    row = state.transcript[-1]
    assert row.sender_id == SEAT, state.transcript
    return row


def _as_head_reads_it(row: RoomMessage, tmp_path: Path) -> dict[str, Any]:
    dumped = json.loads(row.model_dump_json())
    text = json.dumps({k: dumped.get(k) for k in FIELDS})
    # Both spellings: macOS reports `tmp_path` under `/private/var` and as `/var`.
    for root in sorted({str(tmp_path.resolve()), str(tmp_path)}, key=len, reverse=True):
        text = text.replace(json.dumps(root)[1:-1], READER_HOME)
    return json.loads(text)


async def _failed_rows(tmp_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    case = tmp_path / "raised"
    case.mkdir()
    row = await _last_row(case, _OneAgentResolver(_RaisingAgent(case / "notes" / "today.md")))
    rows["raised"] = _as_head_reads_it(row, case)

    case = tmp_path / "refused"
    case.mkdir()
    budget = TokenBudgetManager(default_max_tokens=0)
    host = _host(case / "sessions", MockLLMConnector(default_model="mock-gpt-4o"), budget)
    row = await _last_row(case, RoomAgentResolver(host))
    rows["refused"] = _as_head_reads_it(row, case)

    case = tmp_path / "provider"
    case.mkdir()
    host = _host(case / "sessions", _ReadsAMissingFile(case / "models" / "weights.bin"))
    row = await _last_row(case, RoomAgentResolver(host))
    rows["provider"] = _as_head_reads_it(row, case)

    case = tmp_path / "unsaved"
    case.mkdir()
    sessions = case / "sessions"
    host = _host(sessions, MockLLMConnector(default_model="mock-gpt-4o", default_response="done"))
    # After the store is built, so its constructor's mkdir succeeds and the turn's write does
    # not: the directory it writes into is now a file.
    shutil.rmtree(sessions)
    sessions.write_text("not a directory", encoding="utf-8")
    row = await _last_row(case, RoomAgentResolver(host))
    rows["unsaved"] = _as_head_reads_it(row, case)

    return rows


def _looks_raw(text: str | None) -> bool:
    """True when `text` is what the head must not print: a path or an exception class."""
    return bool(text) and bool(
        "/" in str(text) or re.search(r"[A-Za-z]*(Error|Exception)\b", str(text))
    )


@pytest.mark.asyncio
async def test_the_heads_failed_row_fixture_is_what_real_failures_land(tmp_path: Path) -> None:
    """`frontend/src/test/room-failed-rows.json` is this run's output, and stays so.

    Each row is also checked for the property the head's test depends on: the raw cause is
    really raw. A failure that happened to land a clean sentence would make the head's
    "not rendered" assertions vacuous, so it fails here rather than passing there.

    Killed by: src/uclone_x/room/orchestrator.py :: error = f"{type(exc).__name__}: {exc}"
    Becomes: error = "the turn failed"
    """
    rows = await _failed_rows(tmp_path)

    assert rows["raised"]["error"] and rows["raised"]["refusal"] is None
    assert rows["refused"]["refusal"] == "budget_exceeded", rows["refused"]
    assert rows["provider"]["error"] and rows["provider"]["refusal"] is None
    assert rows["unsaved"]["error"] is None and rows["unsaved"]["content"] == "done"
    for name, field in (
        ("raised", "error"),
        ("provider", "error"),
        ("unsaved", "persist_error"),
    ):
        assert _looks_raw(rows[name][field]), (
            f"the {name} row's {field} is {rows[name][field]!r}: no path and no class name, "
            "so the head's test that it is not rendered would prove nothing"
        )
        assert READER_HOME in rows[name][field] or "Error" in rows[name][field]

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["rows"] == rows, (
        f"{FIXTURE} no longer matches what these failures land, so the head's test reads rows "
        "the Core does not write. Replace its `rows` with:\n" + json.dumps(rows, indent=2)
    )
