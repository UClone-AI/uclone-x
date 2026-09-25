"""A room's story: every seat works on it, a tool call moves it, and the room's deletion
leaves it (#1555).

The seats are fakes that record what the orchestrator handed their turn, so a passing test
says the room passed its story, not that a method was called.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.models import ToolExecutionRecord, TurnResult
from uclone_x.agent.session import SessionState
from uclone_x.llm import MockLLMConnector
from uclone_x.room.models import (
    Participant,
    ParticipantKind,
    RoomPolicy,
    RoomState,
    SelectionVerdict,
    SpeakerDecision,
    SpeakerRequest,
)
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.store import RoomStore
from uclone_x.story import OPEN_STORY_KEY
from uclone_x.story.library import StoryLibrary
from uclone_x.tools.models import ToolResultStatus
from uclone_x.ui.app import create_ui_app

ALICE = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")


def _seat(agent_id: str) -> Participant:
    return Participant(
        id=agent_id,
        kind=ParticipantKind.AGENT,
        display_name=agent_id.title(),
        session_id=f"sess_room__r1__{agent_id}",
        ontology_namespace=f"https://uclone-x.ai/ontology/r1/{agent_id}",
    )


class _Seat:
    """Records the conversation and story each turn was given."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.session_id = f"sess_room__r1__{agent_id}"
        self.seen: list[tuple[str | None, str | None]] = []
        self.tool_executions: tuple[ToolExecutionRecord, ...] = ()

    async def execute_turn(
        self,
        prompt: str,
        *,
        room_id: str | None = None,
        story_id: str | None = None,
        **kwargs: Any,
    ) -> TurnResult:
        self.seen.append((room_id, story_id))
        return TurnResult(
            turn_index=len(self.seen),
            content=f"{self.agent_id} wrote",
            provenance=None,
            tool_executions=self.tool_executions,
        )

    def checkpoint_turn(self, session_id: str | None = None) -> SessionState:
        return SessionState(session_id=session_id or self.session_id, agent_id=self.agent_id)

    def roll_back_turn(self, checkpoint: Any, *, reason: str) -> int:
        return 0

    def persist_session(self, session_id: str | None = None) -> SessionState:
        return SessionState(session_id=session_id or self.session_id, agent_id=self.agent_id)


class _Resolver:
    def __init__(self, seats: dict[str, _Seat]) -> None:
        self._seats = seats

    async def resolve(self, participant: Participant) -> Any:
        return self._seats[participant.id]


class _Speakers:
    """Gives the floor to each named seat once, in order, then abstains."""

    name = "scripted"

    def __init__(self, *ids: str) -> None:
        self._ids = list(ids)

    async def select(self, request: SpeakerRequest) -> SpeakerDecision:
        if self._ids:
            return SpeakerDecision(
                verdict=SelectionVerdict.SPEAK, speaker_id=self._ids.pop(0), selector=self.name
            )
        return SpeakerDecision(verdict=SelectionVerdict.ABSTAIN, selector=self.name)


def _room(
    tmp_path: Path, story_id: str | None
) -> tuple[RoomStore, dict[str, _Seat], RoomOrchestrator]:
    store = RoomStore(tmp_path / "rooms")
    store.save(
        RoomState(
            room_id="r1",
            participants=(ALICE, _seat("scout"), _seat("critic")),
            policy=RoomPolicy(max_agent_turns_per_human_message=2),
            story_id=story_id,
        )
    )
    seats = {"scout": _Seat("scout"), "critic": _Seat("critic")}
    orch = RoomOrchestrator(
        store=store, selectors=[_Speakers("scout", "critic")], resolver=_Resolver(seats)
    )
    return store, seats, orch


class TestEverySeatWorksOnTheRoomsStory:
    @pytest.mark.asyncio
    async def test_two_seats_in_one_room_resolve_the_same_story_root(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: story_id=state.story_id,
        Becomes: story_id=None,
        """
        story = StoryLibrary(tmp_path).create("Tide", "r1").story_id
        _, seats, orch = _room(tmp_path, story)

        await orch.post("r1", "alice", "write chapter one")

        assert seats["scout"].seen == [("r1", story)]
        assert seats["critic"].seen == [("r1", story)]
        seen = [sid for s in seats.values() for _, sid in s.seen]
        roots = {StoryLibrary(tmp_path).root(sid) for sid in seen if sid is not None}
        assert roots == {(tmp_path / "stories" / story).resolve()}

    @pytest.mark.asyncio
    async def test_seats_hold_the_lease_as_one_conversation(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: room_id=room_id,
        Becomes: room_id=None,
        """
        _, seats, orch = _room(tmp_path, None)

        await orch.post("r1", "alice", "hello")

        assert {c for s in seats.values() for c, _ in s.seen} == {"r1"}

    @pytest.mark.asyncio
    async def test_a_story_opened_by_one_seat_is_the_rooms_for_the_next(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: "story_id": story_after(executions, state.story_id),
        Becomes: "story_id": state.story_id,
        """
        store, seats, orch = _room(tmp_path, None)
        seats["scout"].tool_executions = (
            ToolExecutionRecord(
                tool_name="story_library",
                output={OPEN_STORY_KEY: "tide-0001", "path": "stories/tide-0001/story.yaml"},
                status=ToolResultStatus.SUCCESS,
                writes_files=True,
                opens_story=True,
            ),
        )

        await orch.post("r1", "alice", "start a story")

        assert seats["critic"].seen == [("r1", "tide-0001")]
        landed = store.load("r1")
        assert landed is not None and landed.story_id == "tide-0001"


# --------------------------------------------------------------------------------------
# Deleting the room
# --------------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_ui_app(
        static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=MockLLMConnector()
    )
    with TestClient(app) as started:
        yield started


class TestDeletingTheRoom:
    def test_the_story_files_stay_and_the_lease_is_given_back(self, client: TestClient) -> None:
        """Killed by: src/uclone_x/ui/rooms.py :: _release_story(stack.session_manager().workspace_dir, story_id, room_id)
        Becomes: pass
        """
        created = client.post("/api/rooms", json={"title": "Novel", "agent_ids": ["scout"]})
        assert created.status_code in (200, 201), created.text
        room_id = created.json()["room_id"]
        stack = cast(Any, client.app).state.room_stack
        library = StoryLibrary(stack.session_manager().workspace_dir)
        story = library.create("Tide", room_id).story_id
        library.write_file(
            story, "chapter-01.md", "It began.", conversation_id=room_id, expected_digest=None
        )
        state = stack.store.load(room_id)
        stack.store.save(state.model_copy(update={"story_id": story}))

        assert client.delete(f"/api/rooms/{room_id}").status_code == 204

        assert client.get(f"/api/rooms/{room_id}").status_code == 404
        assert library.read_file(story, "chapter-01.md").text == "It began."
        assert library.load(story).lease is None
