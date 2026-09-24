"""A sub-agent gets no memory tools, unless its parent shares its memory read-only (#1431).

Owner ruling on #1431: a sub-agent is a new, throwaway agent, so it is not given the
memory tools. Amended on the same issue: the parent may choose, per delegation, to let the
child read its memory. The child then gets `query_memory_facts` against the parent's
store and nothing that writes to it: the store rewrites its whole document with no lock,
so a second writer would overwrite the parent's.

Before this, a child inherited its parent's whole resolved list, memory tools included, while
being built with no store: permitted three tools it did not have, and reported on
`/api/agents` as having them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    BASE_MEMORY_TOOLS,
    AgentConfig,
    AgentContext,
    PersonaDefinition,
)
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import ToolCallRequest
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import QueryMemoryFactsTool, ReadOnlyMemory
from uclone_x.skills.auditor import SkillRegistry
from uclone_x.tools.builtin.filesystem import FileReadTool
from uclone_x.tools.builtin.subagent import SubagentDelegationParams, SubagentDelegationTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import AgentSessionManager, create_ui_app

RECORD = "record_memory_fact"
QUERY = "query_memory_facts"
RETRACT = "retract_memory_fact"


def _lead(
    tmp_path: Path,
    *,
    memory: bool = True,
    allowed_tools: tuple[str, ...] = ("web_search",),
    llm: MockLLMConnector | None = None,
    tools: ToolRegistry | None = None,
    agent_id: str = "lead",
    operator_tools: tuple[str, ...] = (),
    skills: SkillRegistry | None = None,
) -> BaseAgent:
    """A persona agent with sub-agents on, and -- unless told otherwise -- its own store.

    `operator_tools` is an operator's list, which is taken as written in place of the
    persona's (and so carries no base set).
    """
    agent = BaseAgent(
        config=AgentConfig(
            agent_id=agent_id,
            name=agent_id,
            enable_subagent_tools=True,
            allowed_tools=operator_tools,
        ),
        llm=llm or MockLLMConnector(),
        tools=tools if tools is not None else ToolRegistry(),
        context=AgentContext(
            session_id=f"sess_{agent_id}", agent_id=agent_id, workspace_root=tmp_path
        ),
        memory=CrossSessionMemory(storage_path=tmp_path / f"{agent_id}-memory.json")
        if memory
        else None,
        skills=skills,
    )
    agent.define_persona(
        PersonaDefinition(
            name="lead",
            role="Lead",
            system_prompt="Lead.",
            allowed_tools=allowed_tools,
            enable_subagent_tools=True,
        )
    )
    agent.persona = "lead"
    return agent


async def _remember(agent: BaseAgent) -> None:
    record = await agent.execute_tool_call(
        RECORD, {"subject": "user", "predicate": "favourite colour", "object_value": "teal"}
    )
    assert record.status is ToolResultStatus.SUCCESS, record.error


def _held(agent: BaseAgent) -> set[str]:
    return {tool.name for tool in agent.available_tools()}


class TestByDefaultAChildHasNoMemory:
    @pytest.mark.asyncio
    async def test_the_child_is_not_permitted_the_memory_tools_its_parent_keeps(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/base.py :: allowed_tools=child_allowed,
        Becomes: allowed_tools=self._config.allowed_tools,
        """
        parent = _lead(tmp_path)
        child = await parent.spawn_subagent(role="helper", goal="help")

        assert set(BASE_MEMORY_TOOLS) <= set(parent.config.allowed_tools)
        assert set(BASE_MEMORY_TOOLS) <= _held(parent)
        assert not set(BASE_MEMORY_TOOLS) & set(child.config.allowed_tools)
        assert not set(BASE_MEMORY_TOOLS) & _held(child)
        # The rest of the parent's list is still inherited.
        assert "web_search" in child.config.allowed_tools

    @pytest.mark.asyncio
    async def test_a_forced_memory_call_is_refused_as_not_permitted(self, tmp_path: Path) -> None:
        """The child's own list refuses it, not the accident of having no store behind it.

        Before the fix the call was permitted and failed only as "not found".

        Killed by: src/uclone_x/agent/base.py :: if name not in BASE_MEMORY_TOOLS
        Becomes: if True
        """
        llm = MockLLMConnector(
            responses=["", "done"],
            tool_calls=[
                ToolCallRequest(
                    id="c1",
                    name=RECORD,
                    arguments={"subject": "a", "predicate": "b", "object_value": "c"},
                )
            ],
        )
        parent = _lead(tmp_path, llm=llm)
        child = await parent.spawn_subagent(role="helper", goal="help")

        result = await child.run_turn("Remember that a b c.")

        calls = [r for r in result.tool_executions if r.tool_name == RECORD]
        assert [r.status for r in calls] == [ToolResultStatus.ERROR]
        assert calls[0].error is not None and "allowed_tools" in calls[0].error
        assert parent.memory is not None and parent.memory.list_facts() == []

    @pytest.mark.asyncio
    async def test_a_parent_permitted_only_memory_tools_gives_a_child_no_tools(
        self, tmp_path: Path
    ) -> None:
        """Taking the memory names out must not leave an empty list, which permits everything.

        Killed by: src/uclone_x/agent/base.py :: child_tools = ToolRegistry()
        Becomes: pass
        """
        registry = ToolRegistry(tools=[FileReadTool()])
        parent = _lead(tmp_path, tools=registry, operator_tools=BASE_MEMORY_TOOLS)
        assert parent.config.allowed_tools == BASE_MEMORY_TOOLS

        child = await parent.spawn_subagent(role="helper", goal="help")

        assert child.config.allowed_tools == ()
        assert _held(child) == set()

    @pytest.mark.asyncio
    async def test_a_parent_permitted_only_memory_tools_with_skills_gives_a_child_no_tools(
        self, tmp_path: Path
    ) -> None:
        """Skills reach a child through its host, and a child with skills registers
        `load_skill` itself -- which its empty list would then permit.

        Killed by: src/uclone_x/agent/base.py :: host_fields["skills"] = None
        Becomes: pass
        """
        parent = _lead(
            tmp_path,
            tools=ToolRegistry(tools=[FileReadTool()]),
            operator_tools=BASE_MEMORY_TOOLS,
            skills=SkillRegistry(),
        )
        assert "load_skill" not in _held(parent)

        child = await parent.spawn_subagent(role="helper", goal="help")

        assert _held(child) == set()
        with pytest.raises(KeyError):
            await child.execute_tool_call("load_skill", {"name": "anything"})


class TestSharingTheParentsMemory:
    @pytest.mark.asyncio
    async def test_the_child_gets_query_only_and_reads_the_parents_facts(
        self, tmp_path: Path
    ) -> None:
        """Killed by: src/uclone_x/agent/base.py :: or (shared_memory is not None and name == QueryMemoryFactsTool.name)
        Becomes: or (shared_memory is not None and name in BASE_MEMORY_TOOLS)
        """
        parent = _lead(tmp_path)
        await _remember(parent)

        child = await parent.spawn_subagent(role="helper", goal="help", share_parent_memory=True)

        assert set(BASE_MEMORY_TOOLS) & set(child.config.allowed_tools) == {QUERY}
        assert set(BASE_MEMORY_TOOLS) & _held(child) == {QUERY}
        recalled = await child.execute_tool_call(QUERY, {"subject": "user"})
        assert recalled.status is ToolResultStatus.SUCCESS, recalled.error
        assert "teal" in str(recalled.output)

    @pytest.mark.asyncio
    async def test_the_childs_query_tool_holds_no_path_to_a_write(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: reader = QueryMemoryFactsTool(ReadOnlyMemory(shared_memory))
        Becomes: reader = QueryMemoryFactsTool(shared_memory)
        """
        parent = _lead(tmp_path)
        child = await parent.spawn_subagent(role="helper", goal="help", share_parent_memory=True)

        reader = child._resolve_tool(QUERY)  # pyright: ignore[reportPrivateUsage]
        assert isinstance(reader, QueryMemoryFactsTool)
        bound = reader._memory  # pyright: ignore[reportPrivateUsage]
        assert isinstance(bound, ReadOnlyMemory)
        for write in ("record_fact", "retract_fact", "save"):
            assert not hasattr(bound, write), write
        # The child has no store of its own either, so no record or retract is bound to it.
        assert child.memory is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("name", "arguments"),
        [
            (RECORD, {"subject": "user", "predicate": "favourite colour", "object_value": "red"}),
            (RETRACT, {"fact_id": "__FACT__", "reason": "child says so"}),
        ],
    )
    async def test_a_forced_write_is_refused_and_the_parents_document_is_unchanged(
        self, tmp_path: Path, name: str, arguments: dict[str, Any]
    ) -> None:
        parent = _lead(tmp_path)
        await _remember(parent)
        assert parent.memory is not None and parent.memory.storage_path is not None
        document = parent.memory.storage_path
        before = document.read_bytes()
        fact_id = parent.memory.list_facts()[0].fact_id
        arguments = {k: (fact_id if v == "__FACT__" else v) for k, v in arguments.items()}

        child = await parent.spawn_subagent(role="helper", goal="help", share_parent_memory=True)

        # Directly, and as a model would ask for it mid-turn.
        with pytest.raises((KeyError, PermissionError)):
            await child.execute_tool_call(name, arguments)
        child._llm = MockLLMConnector(  # pyright: ignore[reportPrivateUsage]
            responses=["", "done"],
            tool_calls=[ToolCallRequest(id="w1", name=name, arguments=arguments)],
        )
        result = await child.run_turn("Change the parent's memory.")
        calls = [r for r in result.tool_executions if r.tool_name == name]
        assert [r.status for r in calls] == [ToolResultStatus.ERROR]
        assert calls[0].error is not None and "allowed_tools" in calls[0].error

        assert document.read_bytes() == before
        assert [f.object_value for f in parent.memory.list_facts()] == ["teal"]

    @pytest.mark.asyncio
    async def test_a_parent_without_a_store_shares_nothing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: return "you have no memory of your own to share"
        Becomes: return None
        """
        parent = _lead(tmp_path, memory=False)
        assert QUERY in parent.config.allowed_tools  # permitted, with nothing behind it

        child = await parent.spawn_subagent(role="helper", goal="help", share_parent_memory=True)

        assert parent.memory_share_refusal() == "you have no memory of your own to share"
        assert not set(BASE_MEMORY_TOOLS) & set(child.config.allowed_tools)
        assert not set(BASE_MEMORY_TOOLS) & _held(child)

    @pytest.mark.asyncio
    async def test_a_parent_not_permitted_to_query_shares_nothing(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: if allowed and QueryMemoryFactsTool.name not in allowed:
        Becomes: if False:
        """
        parent = _lead(tmp_path, operator_tools=("web_search",))
        assert QUERY not in parent.config.allowed_tools

        child = await parent.spawn_subagent(role="helper", goal="help", share_parent_memory=True)

        assert parent.memory_share_refusal() is not None
        assert QUERY not in _held(child)

    @pytest.mark.asyncio
    async def test_spawning_shares_nothing_unless_asked(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: share_parent_memory: bool = False,
        Becomes: share_parent_memory: bool = True,
        """
        parent = _lead(tmp_path)
        child = await parent.spawn_subagent(role="helper", goal="help")
        assert QUERY not in _held(child)


class TestTheDelegationTool:
    def test_the_flag_is_off_unless_the_model_turns_it_on(self) -> None:
        """Killed by: src/uclone_x/tools/builtin/subagent.py :: default=False,
        Becomes: default=True,
        """
        params = SubagentDelegationParams.model_validate({"role": "r", "goal": "g", "prompt": "p"})
        assert params.share_parent_memory is False

    async def _delegate(
        self, parent: BaseAgent, tmp_path: Path, **extra: Any
    ) -> tuple[Any, list[BaseAgent]]:
        spawned: list[BaseAgent] = []
        original = parent.spawn_subagent

        async def spy(**kwargs: Any) -> BaseAgent:
            child = await original(**kwargs)
            spawned.append(child)
            return child

        parent.spawn_subagent = spy  # type: ignore[method-assign]
        result = await SubagentDelegationTool().execute(
            params={"role": "helper", "goal": "help", "prompt": "Help.", **extra},
            context=ToolContext(
                agent_id=parent.agent_id,
                session_id="sess_lead",
                trace_id="t",
                workspace_root=tmp_path,
                agent_delegate=parent,
            ),
        )
        return result, spawned

    @pytest.mark.asyncio
    async def test_sharing_reaches_the_child_and_the_result_says_so(self, tmp_path: Path) -> None:
        parent = _lead(tmp_path)
        result, spawned = await self._delegate(parent, tmp_path, share_parent_memory=True)

        assert result.success, result.error
        assert result.output["parent_memory"] == (
            "The subagent could read your memory, but not change it."
        )
        assert set(BASE_MEMORY_TOOLS) & _held(spawned[0]) == {QUERY}

    @pytest.mark.asyncio
    async def test_without_a_store_the_result_says_nothing_was_shared(self, tmp_path: Path) -> None:
        parent = _lead(tmp_path, memory=False)
        result, spawned = await self._delegate(parent, tmp_path, share_parent_memory=True)

        assert result.success, result.error
        assert result.output["parent_memory"] == (
            "The subagent was not given your memory: you have no memory of your own to share."
        )
        assert not set(BASE_MEMORY_TOOLS) & _held(spawned[0])

    @pytest.mark.asyncio
    async def test_by_default_nothing_is_shared_or_reported(self, tmp_path: Path) -> None:
        parent = _lead(tmp_path)
        result, spawned = await self._delegate(parent, tmp_path)

        assert result.success, result.error
        assert "parent_memory" not in result.output
        assert not set(BASE_MEMORY_TOOLS) & _held(spawned[0])


class TestTheTopologyReportsHeldTools:
    """`/api/agents` reports `capabilities` as the tools an agent has, not its permissions."""

    @pytest.mark.asyncio
    async def test_a_sub_agent_and_an_agent_with_no_store(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/ui/app.py :: "capabilities": capabilities,
        Becomes: "capabilities": allowed_tools,
        """
        registry = ToolRegistry(tools=[FileReadTool()])
        session_mgr = AgentSessionManager(storage_dir=tmp_path / "sessions", tools=registry)
        parent = _lead(tmp_path, tools=registry, allowed_tools=("file_read",))
        child = await parent.spawn_subagent(role="helper", goal="help")
        storeless = _lead(
            tmp_path, tools=registry, allowed_tools=("file_read",), memory=False, agent_id="bare"
        )
        agents = session_mgr._agents  # pyright: ignore[reportPrivateUsage]
        for agent in (parent, child, storeless):
            agents[f"{agent.agent_id}:{agent.context.session_id}"] = agent

        client = TestClient(
            create_ui_app(
                static_dir=tmp_path / "static",
                storage_dir=tmp_path / "sessions",
                llm=MockLLMConnector(),
                session_manager=session_mgr,
            )
        )
        response = client.get("/api/agents")
        assert response.status_code == 200, response.text
        rows = {row["id"]: row for row in response.json()["agents"]}

        assert set(rows[parent.agent_id]["capabilities"]) == {"file_read", *BASE_MEMORY_TOOLS}
        assert rows[child.agent_id]["capabilities"] == ["file_read"]
        assert rows["bare"]["capabilities"] == ["file_read"]
        # The permission list is still reported, under its own name.
        assert set(BASE_MEMORY_TOOLS) <= set(rows["bare"]["allowed_tools"])
        nodes = {node["id"]: node for node in response.json()["topology"]["nodes"]}
        assert nodes["bare"]["capabilities"] == ["file_read"]
