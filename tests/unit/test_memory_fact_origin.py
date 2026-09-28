"""A memory fact says how, and in which conversation and turn, it was learned (#1638 step 2).

Design: the clone knowledge graph design, §3.2. `origin` is one of `told`, `found`,
`saved`, `corrected`; `source_room_id` and `source_turn_id` name the conversation and the
room turn. `record_memory_fact` fills them from the turn it runs in, and a `memory.json`
written before the fields existed still loads, every row as `saved`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import RecordMemoryFactTool
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry


def _provenance() -> Provenance:
    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="agent.test", model="memory"),
        served_by=ServiceRef(provider="agent.test", model="memory"),
    )


def _fact(**overrides: object) -> MemoryFact:
    fields: dict[str, object] = {
        "subject": "user",
        "predicate": "prefers",
        "object_value": "short answers",
        "provenance": _provenance(),
        "source_session_id": "sess_1",
    }
    fields.update(overrides)
    return MemoryFact(**fields)  # pyright: ignore[reportArgumentType]


class TestTheFields:
    @pytest.mark.parametrize("origin", ["told", "found", "saved", "corrected"])
    def test_each_origin_the_design_names_is_accepted(self, origin: str) -> None:
        fact = _fact(origin=origin, source_room_id="room_a", source_turn_id="turn_12")
        assert (fact.origin, fact.source_room_id, fact.source_turn_id) == (
            origin,
            "room_a",
            "turn_12",
        )

    def test_an_origin_outside_the_four_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _fact(origin="inferred")

    def test_a_fact_given_none_of_them_is_saved_with_no_room_or_turn(self) -> None:
        fact = _fact()
        assert (fact.origin, fact.source_room_id, fact.source_turn_id) == ("saved", None, None)

    def test_the_fields_survive_a_save_and_a_reload(self, tmp_path: Path) -> None:
        """A fact recorded with all three reads back with all three from the file.

        Killed by: src/uclone_x/memory/store.py :: source_turn_id=source_turn_id,
        Becomes: source_turn_id=None,
        """
        path = tmp_path / "memory.json"
        store = CrossSessionMemory(storage_path=path)
        recorded = store.record_fact(
            subject="user",
            predicate="works_at",
            object_value="Hanbit",
            provenance=_provenance(),
            source_session_id="sess_1",
            origin="told",
            source_room_id="room_b",
            source_turn_id="turn_3",
        )

        reloaded = CrossSessionMemory(storage_path=path).list_facts()

        assert [(f.fact_id, f.origin, f.source_room_id, f.source_turn_id) for f in reloaded] == [
            (recorded.fact_id, "told", "room_b", "turn_3")
        ]


class TestRecordMemoryFactSetsThemFromItsTurn:
    def test_the_tool_stores_the_room_and_turn_of_its_context(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/memory/tools.py :: source_room_id=context.room_id,
        Becomes: source_room_id=None,
        """
        store = CrossSessionMemory(storage_path=tmp_path / "memory.json")
        tool = RecordMemoryFactTool(store)
        context = ToolContext(
            agent_id="scout", session_id="sess_1", room_id="room_a", turn_id="turn_12"
        )

        result = asyncio.run(
            tool.execute(
                {"subject": "user", "predicate": "prefers", "object_value": "short answers"},
                context,
            )
        )

        assert result.success, result.error
        assert [(f.origin, f.source_room_id, f.source_turn_id) for f in store.list_facts()] == [
            ("saved", "room_a", "turn_12")
        ]

    def test_a_room_turn_reaches_the_fact_through_execute_turn(self, tmp_path: Path) -> None:
        """The whole path: the room's turn id and room id, given to `execute_turn`, land on
        the fact the model's own `record_memory_fact` call saves in that turn.

        Killed by: src/uclone_x/agent/turn_executor.py :: self._turn_caller_turn_id = caller_turn_id
        Becomes: self._turn_caller_turn_id = None
        """
        store = CrossSessionMemory(storage_path=tmp_path / "memory.json")
        llm = MockLLMConnector(
            responses=["", "I'll remember that."],
            tool_calls=[
                ToolCallRequest(
                    id="c1",
                    name="record_memory_fact",
                    arguments={
                        "subject": "user",
                        "predicate": "favourite colour",
                        "object_value": "teal",
                    },
                )
            ],
        )
        host = HostDependencies(
            bus=EventBus(),
            llm=llm,
            tools=ToolRegistry(),
            tracer=TelemetryTracer(),
            store=SessionStore(),
            memory=store,
        )
        agent = compose_agent(config=AgentConfig(agent_id="scout", name="Scout"), host=host)

        result = asyncio.run(
            agent.execute_turn(
                "Remember that my favourite colour is teal.",
                caller_turn_id="turn_7",
                room_id="room_b",
            )
        )

        saves = [r for r in result.tool_executions if r.tool_name == "record_memory_fact"]
        assert [r.status for r in saves] == ["success"], [r.error for r in saves]
        assert [(f.origin, f.source_room_id, f.source_turn_id) for f in store.list_facts()] == [
            ("saved", "room_b", "turn_7")
        ]

    def test_a_turn_no_room_started_records_no_room_or_turn(self, tmp_path: Path) -> None:
        store = CrossSessionMemory(storage_path=tmp_path / "memory.json")
        tool = RecordMemoryFactTool(store)

        result = asyncio.run(
            tool.execute(
                {"subject": "user", "predicate": "prefers", "object_value": "tea"},
                ToolContext(agent_id="scout", session_id="sess_1"),
            )
        )

        assert result.success, result.error
        assert [(f.origin, f.source_room_id, f.source_turn_id) for f in store.list_facts()] == [
            ("saved", None, None)
        ]


class TestARowWrittenBeforeTheFieldsLoadsAsSaved:
    def test_an_old_document_loads_every_row_as_saved_and_keeps_it(self, tmp_path: Path) -> None:
        """A `memory.json` whose rows have no `origin`, `source_room_id` or `source_turn_id`.

        The rows are what `model_dump(mode="json")` wrote before this change. They load with
        no failure, as `saved` with no room or turn, and a later save keeps every one of them
        with its original fields.

        Killed by: src/uclone_x/memory/models.py :: default="saved",
        Becomes: default="told",
        """
        path = tmp_path / "memory.json"
        old_row = {
            "fact_id": "mem_000000000001",
            "subject": "user",
            "predicate": "prefers",
            "object_value": "short answers",
            "confidence": 0.9,
            "provenance": _provenance().model_dump(mode="json"),
            "source_session_id": "sess_old",
            "created_at": "2026-09-01T00:00:00Z",
            "updated_at": "2026-09-01T00:00:00Z",
            "retracted": False,
            "retraction_reason": None,
            "retracted_at": None,
            "contradicts_fact_id": None,
            "tags": ["ui"],
            "metadata": {"note": "kept"},
        }
        path.write_text(json.dumps({"version": "1.0.0", "facts": [old_row]}), encoding="utf-8")

        store = CrossSessionMemory(storage_path=path)

        assert store.load_failure is None
        [fact] = store.list_facts()
        assert (fact.origin, fact.source_room_id, fact.source_turn_id) == ("saved", None, None)

        store.record_fact(
            subject="project_x",
            predicate="uses",
            object_value="Postgres 16",
            provenance=_provenance(),
            source_session_id="sess_new",
        )
        rows = {
            row["fact_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))["facts"]
        }
        kept = rows["mem_000000000001"]
        for key, value in old_row.items():
            assert kept[key] == value, key
        assert kept["origin"] == "saved"
        assert len(rows) == 2
