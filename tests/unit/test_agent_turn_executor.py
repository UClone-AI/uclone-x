"""`agent/turn_executor.py` on its own: the turn loop behind `BaseAgent.execute_turn` (#1736).

The loop moved out of `agent/base.py` unchanged, and the behavioural tests that drive it
through `BaseAgent.execute_turn` stay where they were (`test_turn_step_logging.py`,
`test_turn_evidence_persistence.py`, `test_agent_base.py`, ...). This file pins what the
move itself introduced: the executor holds no copy of agent state, the agent methods it
calls back are the agent's current ones, and what the turn writes lands on the agent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, ToolExecutionRecord
from uclone_x.core.provenance import Provenance
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_USAGE = TokenUsage(provider="scripted", model="scripted", input_tokens=0, output_tokens=0)


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=text,
        tool_calls=(),
        usage=_USAGE,
        provenance=Provenance.primary("scripted", "scripted"),
    )


def _call(call_id: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name="context_probe", arguments={}),),
        usage=_USAGE,
        provenance=Provenance.primary("scripted", "scripted"),
    )


class _ScriptedLLM(BaseLLMConnector):
    """Replays a fixed script, then repeats its last reply, and records what it was sent."""

    def __init__(self, *responses: ModelResponse) -> None:
        super().__init__()
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


class _NoParams(BaseModel):
    pass


class _ContextProbe(BaseTool[_NoParams]):
    """Records the `ToolContext` each call was given."""

    name = "context_probe"
    description = "Records the context it runs in"

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[ToolContext] = []

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        self.contexts.append(context)
        return {"seen": True}


def _agent(llm: _ScriptedLLM, probe: _ContextProbe | None = None) -> BaseAgent:
    registry = ToolRegistry()
    if probe is not None:
        registry.register(probe)
    return BaseAgent(
        config=AgentConfig(
            agent_id="turn_executor",
            name="Turn executor",
            llm_config=AgentLLMConfig(model_name="scripted"),
        ),
        llm=llm,
        tools=registry,
    )


@pytest.mark.asyncio
async def test_a_connector_swapped_after_construction_is_the_one_the_turn_calls() -> None:
    """The turn reads `_llm` when it runs, not when the agent was built.

    Callers replace the connector on a live agent (the harness ladder assigns agent
    attributes after construction). A scope that captured it would keep calling the old
    one, and answer from a model nobody configured any more.

    Killed by: src/uclone_x/agent/base.py :: llm=lambda: self._llm,
    Becomes: llm=lambda _l=self._llm: _l,
    """
    built_with = _ScriptedLLM(_answer("from the old connector"))
    swapped_in = _ScriptedLLM(_answer("from the new connector"))
    agent = _agent(built_with)
    await agent.start()

    agent._llm = swapped_in  # pyright: ignore[reportPrivateUsage]
    result = await agent.execute_turn("hello")

    assert result.content == "from the new connector"
    assert built_with.requests == []


@pytest.mark.asyncio
async def test_a_tool_round_replaced_on_the_instance_is_the_one_the_turn_runs() -> None:
    """The loop calls `_execute_tools` back through the agent, so a patched one is used.

    Tests and evals replace `agent._execute_tools` on the instance to observe or stub a
    step's tool round. Were the executor to hold the method it was built with, the
    replacement would be silently bypassed and the real tools would run instead.

    Killed by: src/uclone_x/agent/base.py :: execute_tools=lambda: self._execute_tools,
    Becomes: execute_tools=lambda _e=self._execute_tools: _e,
    """
    probe = _ContextProbe()
    agent = _agent(_ScriptedLLM(_call("tc_1"), _answer("done")), probe)
    await agent.start()
    rounds: list[tuple[ToolCallRequest, ...]] = []
    original: Callable[..., Awaitable[tuple[list[ChatMessage], list[ToolExecutionRecord]]]]
    original = agent._execute_tools  # pyright: ignore[reportPrivateUsage]

    async def observed(
        tool_calls: tuple[ToolCallRequest, ...] | list[ToolCallRequest], **kwargs: Any
    ) -> tuple[list[ChatMessage], list[ToolExecutionRecord]]:
        rounds.append(tuple(tool_calls))
        return await original(tool_calls, **kwargs)

    agent._execute_tools = observed  # type: ignore[method-assign]
    result = await agent.execute_turn("use the tool")

    assert [[call.id for call in calls] for calls in rounds] == [["tc_1"]]
    assert result.content == "done"


@pytest.mark.asyncio
async def test_the_turn_counter_the_loop_advances_is_the_agents() -> None:
    """Each turn's `+= 1` goes through the scope's setter to the agent's own counter.

    The executor has no counter of its own; a setter that dropped the write would leave
    every turn numbered as the first, and tool contexts carry that number as `turn_index`.

    Killed by: src/uclone_x/agent/base.py :: set_turn_counter=lambda value: setattr(self, "_turn_counter", value),
    Becomes: set_turn_counter=lambda value: None,
    """
    probe = _ContextProbe()
    agent = _agent(_ScriptedLLM(_call("tc_1"), _answer("done")), probe)
    await agent.start()

    await agent.execute_turn("first")
    agent._llm = _ScriptedLLM(_call("tc_2"), _answer("done"))  # pyright: ignore[reportPrivateUsage]
    await agent.execute_turn("second")

    assert agent.turn_counter == 2
    assert [context.turn_index for context in probe.contexts] == [1, 2]


@pytest.mark.asyncio
async def test_a_tool_is_handed_the_agent_itself_as_its_delegate() -> None:
    """`agent_delegate` is the agent, not the executor that now builds the context.

    The move turned `agent_delegate=self` into a read through the scope, because `self`
    in the loop is now the executor. Tools that spawn sub-agents or deduct steps call
    agent methods (`consume_steps`, ...) on the delegate; the executor has none of them.

    Killed by: src/uclone_x/agent/base.py :: agent_delegate=lambda: self,
    Becomes: agent_delegate=lambda: None,
    """
    probe = _ContextProbe()
    agent = _agent(_ScriptedLLM(_call("tc_1"), _answer("done")), probe)
    await agent.start()

    await agent.execute_turn("use the tool")

    assert len(probe.contexts) == 1
    assert probe.contexts[0].agent_delegate is agent


@pytest.mark.asyncio
async def test_the_public_entry_point_passes_the_story_through_to_the_tools() -> None:
    """`BaseAgent.execute_turn` hands every keyword to the executor, `story_id` included.

    A delegator that dropped one would still return a turn; the loss would show only in
    what the turn's tool calls were told, so this reads it there.

    Killed by: src/uclone_x/agent/base.py :: story_id=story_id,
    Becomes: story_id=None,
    """
    probe = _ContextProbe()
    agent = _agent(_ScriptedLLM(_call("tc_1"), _answer("done")), probe)
    await agent.start()

    await agent.execute_turn("use the tool", room_id="room-1", story_id="story-1")

    assert [(c.room_id, c.story_id) for c in probe.contexts] == [("room-1", "story-1")]


@pytest.mark.asyncio
async def test_a_turn_given_a_workspace_runs_its_tools_there_and_only_that_turn(
    tmp_path: Path,
) -> None:
    """A conversation's workspace binds the turn's tools; the next turn without one does not.

    The room passes its workspace on every turn (clone-data-scopes §3.6), so the sandbox
    bound is the conversation's, and a turn with none falls back to the agent's own.

    Killed by: src/uclone_x/agent/turn_executor.py :: self._scope.set_turn_workspace_root(workspace_root)
    Becomes: self._scope.set_turn_workspace_root(None)
    """
    default, other = tmp_path / "default", tmp_path / "other"
    default.mkdir()
    other.mkdir()
    probe = _ContextProbe()
    llm = _ScriptedLLM(_call("c1"), _answer("one"), _call("c2"), _answer("two"))
    registry = ToolRegistry()
    registry.register(probe)
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="turn_executor",
            name="Turn executor",
            llm_config=AgentLLMConfig(model_name="scripted"),
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(
            session_id="sess_ws", agent_id="turn_executor", workspace_root=default
        ),
    )
    await agent.start()

    await agent.execute_turn("in the other folder", workspace_root=other)
    await agent.execute_turn("in the default one")

    assert [c.workspace_root for c in probe.contexts] == [other.resolve(), default.resolve()]
