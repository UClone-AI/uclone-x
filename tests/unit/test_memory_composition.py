"""Cross-session memory reaches a composed agent, and an unreadable store says so.

`compose_agent` is the only path every head builds an agent through. Until #1097 it
had no `memory` field to pass, so `BaseAgent`'s memory-tool registration — which runs
only when a store is given — could not fire in any head, and `CrossSessionMemory` was
instantiated nowhere outside `tests/`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.models import AgentConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.agent_home import (
    AGENT_ID_PREFIX,
    AGENTS_DIR_ENV_VAR,
    AgentHomeError,
    default_agents_root,
)
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse
from uclone_x.memory.store import (
    CrossSessionMemory,
    default_cross_session_memory,
)
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry


def _host(**overrides: object) -> HostDependencies:
    base: dict[str, object] = {
        "bus": EventBus(),
        "llm": MockLLMConnector(),
        "tools": ToolRegistry(),
        "tracer": TelemetryTracer(),
        "store": SessionStore(),
    }
    base.update(overrides)
    return HostDependencies(**base)  # pyright: ignore[reportArgumentType]


def test_compose_agent_passes_memory_through_to_the_agent(tmp_path: Path) -> None:
    """A host that wires memory gets an agent that has it, and its three tools.

    Killed by: src/uclone_x/agent/composition.py :: memory=host.memory,
    Becomes: memory=None,
    """
    memory = CrossSessionMemory(storage_path=tmp_path / "mem.json")
    agent = compose_agent(config=AgentConfig(agent_id="a1", name="A1"), host=_host(memory=memory))

    assert agent.memory is memory
    assert agent.tools is not None
    for name in ("record_memory_fact", "retract_memory_fact", "query_memory_facts"):
        assert agent.tools.get(name) is not None, name


def test_compose_agent_without_memory_registers_no_memory_tools() -> None:
    agent = compose_agent(config=AgentConfig(agent_id="a2", name="A2"), host=_host())

    assert agent.memory is None
    assert agent.tools is not None
    assert agent.tools.get("record_memory_fact") is None


def test_default_memory_location_honours_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A headless run must not be forced to write into the invoking user's home.

    Killed by: src/uclone_x/core/agent_home.py :: override = os.environ.get(AGENTS_DIR_ENV_VAR)
    Becomes: override = None
    """
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path))

    assert default_agents_root() == tmp_path
    store = default_cross_session_memory("agent-one")
    assert store.storage_path == tmp_path / "agent-one" / "memory.json"


def test_a_username_no_directory_can_carry_is_refused_not_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The four names that used to share one memory file now each raise.

    `"".join(c if c.isalnum() or c in "-_" else "_" for c in agent_id)` folded every
    character it disliked into `_`, so `a.b`, `a/b`, `a b` and `a_b` all resolved to
    `a_b.json`. Four agents read and overwrote one document, and nothing anywhere said
    so. Only `a_b` is a name; the other three are refused (P6).

    Killed by: src/uclone_x/core/agent_home.py :: refuse_an_unusable_username(username)
    Becomes: pass
    """
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path))

    for rejected in ("a.b", "a/b", "a b", "Scout", "-lead", "trail_"):
        with pytest.raises(AgentHomeError) as excinfo:
            default_cross_session_memory(rejected)
        assert rejected in str(excinfo.value)

    assert default_cross_session_memory("a_b").storage_path == tmp_path / "a_b" / "memory.json"


def test_an_agents_home_records_an_id_that_survives_a_second_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The id is minted once and then read, never re-minted.

    An agent whose identifier changes is a different agent to everything that recorded
    the old one, so the second call through must return the first call's value.

    Killed by: src/uclone_x/memory/store.py :: home.agent_id()
    Becomes: pass
    """
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path))

    default_cross_session_memory("archivist")
    id_path = tmp_path / "archivist" / "id"
    first = id_path.read_text(encoding="utf-8").strip()
    assert first.startswith(AGENT_ID_PREFIX)

    default_cross_session_memory("archivist")
    assert id_path.read_text(encoding="utf-8").strip() == first


def test_an_unreadable_store_is_quarantined_and_reported(tmp_path: Path) -> None:
    """An unreadable memory is not an empty memory, and must not be overwritten.

    Killed by: src/uclone_x/memory/store.py :: os.replace(self._storage_path, quarantine)
    Becomes: pass
    """
    path = tmp_path / "mem.json"
    path.write_text("{not json at all", encoding="utf-8")

    memory = CrossSessionMemory(storage_path=path)

    assert memory.load_failure is not None
    assert not path.exists()
    assert list(tmp_path.glob("mem.json.unreadable-*"))
    section = memory.format_prompt_section()
    assert "UNAVAILABLE" in section


def test_a_readable_store_reports_no_failure(tmp_path: Path) -> None:
    path = tmp_path / "mem.json"
    path.write_text(json.dumps({"version": "1.0.0", "facts": []}), encoding="utf-8")

    memory = CrossSessionMemory(storage_path=path)

    assert memory.load_failure is None
    assert memory.format_prompt_section() == ""
    assert path.exists()


def test_two_agents_sharing_one_registry_keep_their_own_memories(tmp_path: Path) -> None:
    """The UI gives all four heads one `ToolRegistry`, and memory tools are not shareable.

    Every other tool is stateless with respect to the agent holding it, so one instance in a
    shared registry is correct. A memory tool is bound to *one* store at construction. With
    the tools resolved from the registry, the second agent's registration is skipped as a
    duplicate and both agents execute the first agent's instance — the critic records into
    the champion's file, and nothing in either agent reports an error.

    Killed by: src/uclone_x/agent/base.py :: local = self._agent_local_tools.get(name)
    Becomes: local = None
    """
    registry = ToolRegistry()
    first_memory = CrossSessionMemory(storage_path=tmp_path / "first.json")
    second_memory = CrossSessionMemory(storage_path=tmp_path / "second.json")

    first = compose_agent(
        config=AgentConfig(agent_id="first", name="First"),
        host=_host(tools=registry, memory=first_memory),
    )
    second = compose_agent(
        config=AgentConfig(agent_id="second", name="Second"),
        host=_host(tools=registry, memory=second_memory),
    )

    record = asyncio.run(
        second.execute_tool_call(
            "record_memory_fact",
            {
                "subject": "second agent",
                "predicate": "wrote",
                "object_value": "into its own store",
            },
        )
    )

    assert record.status == "success", record.error
    # The shared registry holds one advertised copy — the first agent's — which is what
    # makes the resolution order load-bearing rather than incidental.
    assert first.tools is registry and second.tools is registry
    assert [fact.subject for fact in second_memory.list_facts()] == ["second agent"]
    assert first_memory.list_facts() == []


def test_an_agent_without_memory_cannot_reach_another_agents_store(tmp_path: Path) -> None:
    """The dangerous half is the agent composed with no store at all.

    `ui/rooms.py` builds `HostDependencies` over the chat manager's registry and passes no
    `memory=`. With the registry consulted as a fallback, that agent's `record_memory_fact`
    resolves to whichever agent registered one first and writes into *its* file, reporting
    `success`. An agent given no memory has no memory: the call has to fail, and the tool
    must not be advertised to it either.

    Killed by: src/uclone_x/agent/base.py :: if isinstance(candidate, AGENT_BOUND_TOOL_TYPES):
    Becomes: if False:
    """
    registry = ToolRegistry()
    owner_memory = CrossSessionMemory(storage_path=tmp_path / "owner.json")

    compose_agent(
        config=AgentConfig(agent_id="owner", name="Owner"),
        host=_host(tools=registry, memory=owner_memory),
    )
    memoryless = compose_agent(
        config=AgentConfig(agent_id="roomer", name="Roomer"), host=_host(tools=registry)
    )

    with pytest.raises(KeyError, match="not registered"):
        asyncio.run(
            memoryless.execute_tool_call(
                "record_memory_fact",
                {"subject": "room", "predicate": "wrote", "object_value": "somewhere"},
            )
        )

    assert owner_memory.list_facts() == []


class _ToolCapturingLLM(MockLLMConnector):
    """Records the tool definitions each turn advertises."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.advertised: list[tuple[str, ...]] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.advertised.append(tuple(tool.name for tool in request.tools))
        return await super().generate(request)


def test_a_memory_tool_is_advertised_only_to_the_agent_it_belongs_to(tmp_path: Path) -> None:
    """A tool whose every call answers "not registered" must not be offered.

    It is advertised out of the shared registry, which holds the other agent's copy. Left
    in the list, a memoryless agent is told it can record facts and is refused when it
    tries — the model's next move is to report the task impossible for the wrong reason.

    Killed by: src/uclone_x/agent/base.py :: if isinstance(t, AGENT_BOUND_TOOL_TYPES) and t.name not in self._agent_local_tools:
    Becomes: if False and t.name not in self._agent_local_tools:
    """
    registry = ToolRegistry()
    owner_llm, roomer_llm = _ToolCapturingLLM(), _ToolCapturingLLM()

    owner = compose_agent(
        config=AgentConfig(agent_id="owner", name="Owner"),
        host=_host(
            tools=registry,
            llm=owner_llm,
            memory=CrossSessionMemory(storage_path=tmp_path / "owner.json"),
        ),
    )
    roomer = compose_agent(
        config=AgentConfig(agent_id="roomer", name="Roomer"),
        host=_host(tools=registry, llm=roomer_llm),
    )

    asyncio.run(owner.execute_turn("hello"))
    asyncio.run(roomer.execute_turn("hello"))

    assert "record_memory_fact" in owner_llm.advertised[0]
    assert "record_memory_fact" not in roomer_llm.advertised[0]


class _ForeignQueryParams(BaseModel):
    """Parameters for a tool that merely shares a memory tool's name."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    q: str = Field(description="Anything at all.")


class _ForeignQueryTool(BaseTool[_ForeignQueryParams]):
    """A shared implementation registered under `query_memory_facts`.

    Stands in for a third-party or MCP tool that happens to pick the name. Nothing about it
    is bound to one agent: one instance serves every agent in the registry, exactly like
    `file_read` does.
    """

    name: str = "query_memory_facts"
    description: str = "Not the memory tool; just a tool with the same name."

    async def run(self, params: _ForeignQueryParams, context: ToolContext) -> str:
        return f"foreign:{params.q}"


def test_a_shared_tool_that_merely_shares_the_name_still_resolves() -> None:
    """The guard is about the instance's owner, not about three reserved strings.

    Matched on the name, the refusal fires for any tool called `query_memory_facts` —
    including one registered by an MCP server or a plugin, which is a shared implementation
    owned by nobody. Every agent without its own memory store would be told that a
    registered, working tool is not registered, which is the same false report the guard
    exists to prevent, pointed the other way (P6).

    Killed by: src/uclone_x/agent/base.py :: if isinstance(candidate, AGENT_BOUND_TOOL_TYPES):
    Becomes: if candidate is not None and candidate.name.endswith("memory_facts"):
    """
    registry = ToolRegistry()
    registry.register(_ForeignQueryTool())
    llm = _ToolCapturingLLM()

    agent = compose_agent(
        config=AgentConfig(agent_id="roomer", name="Roomer"), host=_host(tools=registry, llm=llm)
    )

    result = asyncio.run(agent.execute_tool_call("query_memory_facts", {"q": "anything"}))
    assert result.status == "success", result.error
    assert result.output == "foreign:anything"

    asyncio.run(agent.execute_turn("hello"))
    assert "query_memory_facts" in llm.advertised[0]


# --------------------------------------------------------------------------------------
# A save the model asked for either lands or says it did not (#1375)
# --------------------------------------------------------------------------------------


def _agent_with_memory(tmp_path: Path) -> tuple[BaseAgent, CrossSessionMemory]:
    memory = CrossSessionMemory(storage_path=tmp_path / "mem.json")
    agent = compose_agent(
        config=AgentConfig(agent_id="rememberer", name="Rememberer"),
        host=_host(memory=memory),
    )
    return agent, memory


def test_a_save_with_the_tags_a_local_model_sends_is_recorded(tmp_path: Path) -> None:
    """The arguments qwen3:8b sent, verbatim, land as a fact.

    Measured against the local model on 2026-09-23: asked to write a note and remember a
    colour, it called `record_memory_fact` with `tags` as a JSON array in three runs out of
    three. The parameters were validated strictly, and strict validation accepts only a
    Python tuple for `tuple[str, ...]` -- which no JSON decoder produces -- so every such
    call failed while the advertised schema said `array`. Nothing was saved, and the room's
    Remembers list stayed empty.

    Killed by: src/uclone_x/memory/tools.py :: _MODEL_ARGUMENTS = ConfigDict(frozen=True, extra="forbid")
    Becomes: _MODEL_ARGUMENTS = ConfigDict(frozen=True, extra="forbid", strict=True)
    """
    agent, memory = _agent_with_memory(tmp_path)

    record = asyncio.run(
        agent.execute_tool_call(
            "record_memory_fact",
            {
                "subject": "user",
                "predicate": "favorite_color",
                "object_value": "teal",
                "confidence": 1,
                "tags": ["preferences"],
            },
        )
    )

    assert record.status == "success", record.error
    facts = memory.list_facts()
    assert [(f.subject, f.object_value, f.tags) for f in facts] == [
        ("user", "teal", ("preferences",))
    ]


def test_a_save_refused_on_its_arguments_says_nothing_was_saved(tmp_path: Path) -> None:
    """The model reads the result; the result has to say the save did not happen.

    The refusal used to be pydantic's report alone, which names a field and an input type
    and never says that the fact is not in memory. The model went on to tell the user it
    had saved the colour.

    Killed by: src/uclone_x/tools/base.py :: {self.not_run_note}
    Becomes: {''}
    """
    agent, memory = _agent_with_memory(tmp_path)

    record = asyncio.run(
        agent.execute_tool_call(
            "record_memory_fact", {"subject": "user", "predicate": "favorite_color"}
        )
    )

    assert record.status == "error"
    assert "Nothing was saved to memory." in str(record.error)
    assert "object_value" in str(record.error)
    assert memory.list_facts() == []


def test_a_save_refused_on_an_empty_value_says_nothing_was_saved(tmp_path: Path) -> None:
    """The store's own refusal carries the same sentence, and its reason.

    Killed by: src/uclone_x/memory/tools.py :: raise ValueError(f"Nothing was saved to memory: {exc}") from exc
    Becomes: raise
    """
    agent, memory = _agent_with_memory(tmp_path)

    record = asyncio.run(
        agent.execute_tool_call(
            "record_memory_fact",
            {"subject": "user", "predicate": "favorite_color", "object_value": "   "},
        )
    )

    assert record.status == "error"
    assert "Nothing was saved to memory: object_value cannot be empty" in str(record.error)
    assert memory.list_facts() == []
