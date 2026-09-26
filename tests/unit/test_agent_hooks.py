"""Unit tests for agent lifecycle and tool execution hook system (Issue #328)."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.hooks import (
    BaseHook,
    FailurePolicy,
    HookAction,
    HookContext,
    HookDecision,
    HookEvent,
    HookRunner,
    ScriptHook,
)
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus, EventType
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.tools import LocalTool, ToolRegistry
from uclone_x.tools.models import ToolContext, ToolResult

# --------------------------------------------------------------------------------------
# Test helpers & Mocks
# --------------------------------------------------------------------------------------


class MockLLMProvider(BaseLLMConnector):
    """Mock LLM provider returning canned responses or tool calls."""

    def __init__(
        self,
        canned_content: str = "Mock response",
        canned_tool_calls: Sequence[ToolCallRequest] = (),
    ) -> None:
        super().__init__()
        self.canned_content = canned_content
        self.canned_tool_calls = tuple(canned_tool_calls)
        self.calls: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.calls.append(request)
        # Canned tool calls are spent once. The agent takes agent steps until the model
        # stops asking for tools (P4, amended 2026-09-05); a stub that repeats its call
        # forever describes a model that never answers.
        tools_already_ran = any(m.role is MessageRole.TOOL for m in request.messages)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=self.canned_content,
            tool_calls=() if tools_already_ran else self.canned_tool_calls,
            usage=TokenUsage(input_tokens=10, output_tokens=10, provider="mock"),
            provenance=Provenance.primary("mock"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta_content=self.canned_content)


class EchoTool(LocalTool):
    """Tool that returns its arguments as output."""

    def __init__(self) -> None:
        super().__init__(name="echo_tool", description="Echoes input")

    async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(success=True, output=params, provenance=Provenance.primary("echo_tool"))


# --------------------------------------------------------------------------------------
# Python Hook Unit Tests
# --------------------------------------------------------------------------------------


class BlockingToolHook(BaseHook):
    """Hook that blocks tool execution."""

    def __init__(self, block_reason: str = "Blocked by safety policy") -> None:
        super().__init__(name="BlockingToolHook")
        self.block_reason = block_reason

    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.BLOCK, reason=self.block_reason)


class BlockingTurnHook(BaseHook):
    """Hook that blocks turn execution."""

    def __init__(self, block_reason: str = "Blocked turn by safety policy") -> None:
        super().__init__(name="BlockingTurnHook")
        self.block_reason = block_reason

    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.BLOCK, reason=self.block_reason)


class ModifyingHook(BaseHook):
    """Hook that modifies payload at various lifecycle stages."""

    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(
            action=HookAction.MODIFY,
            modified_payload={"input": "modified turn input"},
        )

    async def on_post_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(
            action=HookAction.MODIFY,
            modified_payload={"content": "modified assistant response"},
        )

    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        args = dict(context.payload.get("arguments", {}))
        args["sanitized"] = True
        return HookDecision(
            action=HookAction.MODIFY,
            modified_payload={"arguments": args},
        )

    async def on_post_tool_use(self, context: HookContext) -> HookDecision:
        out: object = context.payload.get("output")
        if isinstance(out, dict):
            new_out: dict[str, Any] = {str(k): v for k, v in cast(dict[Any, Any], out).items()}
            new_out["post_processed"] = True
            return HookDecision(
                action=HookAction.MODIFY,
                modified_payload={"output": new_out},
            )
        return HookDecision(action=HookAction.ALLOW)


class ErrorTrackingHook(BaseHook):
    """Hook that records error events."""

    def __init__(self) -> None:
        super().__init__(name="ErrorTrackingHook")
        self.recorded_errors: list[dict[str, Any]] = []

    async def on_error(self, context: HookContext) -> HookDecision:
        self.recorded_errors.append(dict(context.payload))
        return HookDecision(action=HookAction.ALLOW)


# --------------------------------------------------------------------------------------
# Tests: Models & BaseHook Dispatch
# --------------------------------------------------------------------------------------


def test_hook_models_serialization() -> None:
    """Validate HookDecision and HookContext strict serialization and enum resolution."""
    decision = HookDecision(action=HookAction.BLOCK, reason="Security rule violation")
    assert decision.action == HookAction.BLOCK
    assert decision.reason == "Security rule violation"
    assert decision.modified_payload is None

    # Test case-insensitive resolution via _missing_
    assert HookAction("allow") == HookAction.ALLOW
    assert HookAction("BLOCK") == HookAction.BLOCK
    assert HookEvent("pre_turn") == HookEvent.PRE_TURN
    assert HookEvent("PRE_TOOL_USE") == HookEvent.PRE_TOOL_USE
    assert FailurePolicy("FAIL_CLOSED") == FailurePolicy.FAIL_CLOSED

    ctx = HookContext(
        agent_id="test_agent",
        event_type=HookEvent.PRE_TOOL_USE,
        payload={"arg1": "val1"},
    )
    assert ctx.agent_id == "test_agent"
    assert ctx.payload == {"arg1": "val1"}


@pytest.mark.asyncio
async def test_base_hook_default_allow() -> None:
    """Default BaseHook methods must all return ALLOW decisions."""
    hook = BaseHook(name="DefaultHook")
    ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)

    assert (await hook.on_pre_turn(ctx)).action == HookAction.ALLOW
    assert (await hook.on_post_turn(ctx)).action == HookAction.ALLOW
    assert (await hook.on_pre_tool_use(ctx)).action == HookAction.ALLOW
    assert (await hook.on_post_tool_use(ctx)).action == HookAction.ALLOW
    assert (await hook.on_error(ctx)).action == HookAction.ALLOW
    assert (await hook.dispatch(ctx)).action == HookAction.ALLOW


# --------------------------------------------------------------------------------------
# Tests: HookRunner Orchestration
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_runner_short_circuit_on_block() -> None:
    """HookRunner must short-circuit and stop executing subsequent hooks once BLOCK is returned."""
    executed_second = False

    class SecondHook(BaseHook):
        async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
            nonlocal executed_second
            executed_second = True
            return HookDecision(action=HookAction.ALLOW)

    runner = HookRunner(hooks=[BlockingToolHook("Policy block"), SecondHook()])
    ctx = HookContext(
        agent_id="a1",
        event_type=HookEvent.PRE_TOOL_USE,
        payload={"tool_name": "bash_run"},
    )

    decision = await runner.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
    assert decision.action == HookAction.BLOCK
    assert decision.reason == "Policy block"
    assert not executed_second


@pytest.mark.asyncio
async def test_hook_runner_aggregate_modify() -> None:
    """HookRunner must accumulate multiple MODIFY payloads in sequence."""

    class HookA(BaseHook):
        async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
            args = dict(context.payload.get("arguments", {}))
            args["a"] = 1
            return HookDecision(action=HookAction.MODIFY, modified_payload={"arguments": args})

    class HookB(BaseHook):
        async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
            args = dict(context.payload.get("arguments", {}))
            args["b"] = 2
            return HookDecision(action=HookAction.MODIFY, modified_payload={"arguments": args})

    runner = HookRunner(hooks=[HookA(), HookB()])
    ctx = HookContext(
        agent_id="a1",
        event_type=HookEvent.PRE_TOOL_USE,
        payload={"arguments": {"orig": 0}},
    )

    decision = await runner.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
    assert decision.action == HookAction.MODIFY
    assert decision.modified_payload is not None
    assert decision.modified_payload["arguments"] == {"orig": 0, "a": 1, "b": 2}


@pytest.mark.asyncio
async def test_hook_runner_ask_after_modify_asks_about_the_rewritten_call() -> None:
    """#1601: an ASK after a MODIFY carries the rewrite, so the approved call is the one asked about.

    The call's fixed keys stay as the agent built them.

    Killed by: src/uclone_x/agent/hooks/runner.py :: if decision.action == HookAction.ASK and modified:
    Becomes: if False:

    Killed by: src/uclone_x/agent/hooks/runner.py :: asked = _with_fixed_keys(asked, fixed)
    Becomes: asked = asked
    """

    class Rewrite(BaseHook):
        async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
            args = dict(context.payload.get("arguments", {}))
            args["a"] = 1
            return HookDecision(
                action=HookAction.MODIFY,
                modified_payload={"arguments": args, "tool_name": "other_tool"},
            )

    class Ask(BaseHook):
        async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
            return HookDecision(action=HookAction.ASK, reason="check with the person")

    runner = HookRunner(hooks=[Rewrite(), Ask()])
    ctx = HookContext(
        agent_id="a1",
        event_type=HookEvent.PRE_TOOL_USE,
        payload={"tool_name": "echo", "arguments": {"orig": 0}},
    )

    decision = await runner.run_hooks(HookEvent.PRE_TOOL_USE, ctx)
    assert decision.action == HookAction.ASK
    assert decision.reason == "check with the person"
    assert decision.modified_payload is not None
    assert decision.modified_payload["arguments"] == {"orig": 0, "a": 1}
    assert decision.modified_payload["tool_name"] == "echo"


@pytest.mark.asyncio
async def test_hook_runner_exception_handling_fail_open_vs_closed() -> None:
    """HookRunner must handle hook exceptions per failure policy."""

    class CrashingHook(BaseHook):
        async def on_pre_turn(self, context: HookContext) -> HookDecision:
            raise RuntimeError("Hook crashed unexpectedly")

    # Fail Open
    runner_open = HookRunner(
        hooks=[CrashingHook(name="CrashOpen", failure_policy=FailurePolicy.FAIL_OPEN)]
    )
    ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)
    res_open = await runner_open.run_hooks(HookEvent.PRE_TURN, ctx)
    assert res_open.action == HookAction.ALLOW

    # Fail Closed
    runner_closed = HookRunner(
        hooks=[CrashingHook(name="CrashClosed", failure_policy=FailurePolicy.FAIL_CLOSED)]
    )
    res_closed = await runner_closed.run_hooks(HookEvent.PRE_TURN, ctx)
    assert res_closed.action == HookAction.BLOCK
    assert "Hook crashed unexpectedly" in (res_closed.reason or "")


@pytest.mark.asyncio
async def test_hook_runner_bus_event_publishing() -> None:
    """HookRunner publishes HOOK_EXECUTED and HOOK_FAILED events to EventBus."""
    bus = EventBus()
    sub = bus.subscribe("agent.hooks")
    runner = HookRunner(
        hooks=[BlockingToolHook("Blocked test")],
        bus=bus,
    )
    ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TOOL_USE)
    await runner.run_hooks(HookEvent.PRE_TOOL_USE, ctx)

    event = await sub.get()
    assert event.type == EventType.HOOK_EXECUTED
    assert event.payload.get("action") == "block"
    assert event.payload.get("reason") == "Blocked test"
    sub.close()


# --------------------------------------------------------------------------------------
# Tests: ScriptHook Execution (Exit Codes, Timeout, JSON IPC)
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_script_hook_exit_0_allow() -> None:
    """ScriptHook with exit 0 and ALLOW json returns ALLOW decision."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import sys, json
data = json.load(sys.stdin)
print(json.dumps({"action": "allow", "reason": "Passed security check"}))
"""
        )
        script_path = f.name

    try:
        hook = ScriptHook(script_path=script_path)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TOOL_USE, payload={"tool": "t1"})
        decision = await hook.dispatch(ctx)
        assert decision.action == HookAction.ALLOW
        assert decision.reason == "Passed security check"
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_exit_0_modify() -> None:
    """ScriptHook with exit 0 and MODIFY json returns modified payload."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import sys, json
data = json.load(sys.stdin)
args = data.get("payload", {}).get("arguments", {})
args["injected"] = 42
print(json.dumps({"action": "modify", "modified_payload": {"arguments": args}}))
"""
        )
        script_path = f.name

    try:
        hook = ScriptHook(script_path=script_path)
        ctx = HookContext(
            agent_id="a1",
            event_type=HookEvent.PRE_TOOL_USE,
            payload={"arguments": {"orig": 1}},
        )
        decision = await hook.dispatch(ctx)
        assert decision.action == HookAction.MODIFY
        assert decision.modified_payload is not None
        assert decision.modified_payload["arguments"] == {"orig": 1, "injected": 42}
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_exit_2_blocks_with_stderr() -> None:
    """ScriptHook with exit 2 returns BLOCK with stderr reason."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import sys
sys.stderr.write("Blocked by external safety policy script\\n")
sys.exit(2)
"""
        )
        script_path = f.name

    try:
        hook = ScriptHook(script_path=script_path)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TOOL_USE)
        decision = await hook.dispatch(ctx)
        assert decision.action == HookAction.BLOCK
        assert "Blocked by external safety policy script" in (decision.reason or "")
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_nonzero_exit_fail_open_vs_fail_closed() -> None:
    """ScriptHook with non-zero exit (1) respects FAIL_OPEN and FAIL_CLOSED policies."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import sys
sys.stderr.write("Unexpected internal crash in hook\\n")
sys.exit(1)
"""
        )
        script_path = f.name

    try:
        # FAIL_OPEN
        hook_open = ScriptHook(script_path=script_path, failure_policy=FailurePolicy.FAIL_OPEN)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)
        res_open = await hook_open.dispatch(ctx)
        assert res_open.action == HookAction.ALLOW

        # FAIL_CLOSED
        hook_closed = ScriptHook(script_path=script_path, failure_policy=FailurePolicy.FAIL_CLOSED)
        res_closed = await hook_closed.dispatch(ctx)
        assert res_closed.action == HookAction.BLOCK
        assert "Unexpected internal crash in hook" in (res_closed.reason or "")
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_timeout_fail_open_vs_fail_closed() -> None:
    """ScriptHook timeout respects FAIL_OPEN and FAIL_CLOSED policies."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import time
time.sleep(5)
"""
        )
        script_path = f.name

    try:
        # FAIL_OPEN with small timeout
        hook_open = ScriptHook(
            script_path=script_path,
            failure_policy=FailurePolicy.FAIL_OPEN,
            timeout_seconds=0.1,
        )
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)
        res_open = await hook_open.dispatch(ctx)
        assert res_open.action == HookAction.ALLOW
        assert "timed out" in (res_open.reason or "")

        # FAIL_CLOSED with small timeout
        hook_closed = ScriptHook(
            script_path=script_path,
            failure_policy=FailurePolicy.FAIL_CLOSED,
            timeout_seconds=0.1,
        )
        res_closed = await hook_closed.dispatch(ctx)
        assert res_closed.action == HookAction.BLOCK
        assert "timed out" in (res_closed.reason or "")
    finally:
        os.remove(script_path)


# --------------------------------------------------------------------------------------
# Tests: Full BaseAgent Lifecycle and Tool Execution Interception
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_base_agent_tool_interception_blocked() -> None:
    """BaseAgent tool execution blocked by hook synthesizes tool message and records error without crashing."""
    registry = ToolRegistry()
    registry.register(EchoTool())

    llm = MockLLMProvider(
        canned_content="Invoking echo tool",
        canned_tool_calls=[
            ToolCallRequest(
                id="call_123",
                name="echo_tool",
                arguments={"msg": "hello"},
            )
        ],
    )

    blocking_hook = BlockingToolHook(block_reason="Unauthorized tool usage")
    config = AgentConfig(
        agent_id="test_hook_agent",
        name="HookAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )

    agent = BaseAgent(
        config=config,
        llm=llm,
        tools=registry,
        hooks=[blocking_hook],
    )

    turn_result = await agent.execute_turn("Execute echo tool")

    assert turn_result.is_completed is True
    assert len(turn_result.tool_executions) == 1
    rec = turn_result.tool_executions[0]
    assert rec.status == "error"
    assert rec.error is not None
    assert "Tool execution blocked by hook: Unauthorized tool usage" in rec.error

    # Verify synthesized tool response message in agent history
    tool_messages = [m for m in agent.history if m.role == MessageRole.TOOL]
    assert len(tool_messages) == 1
    assert "Tool execution blocked by hook: Unauthorized tool usage" in (
        tool_messages[0].content or ""
    )
    assert tool_messages[0].tool_call_id == "call_123"


@pytest.mark.asyncio
async def test_base_agent_tool_interception_modified() -> None:
    """BaseAgent tool execution modified by hooks applies argument and output modifications."""
    registry = ToolRegistry()
    registry.register(EchoTool())

    llm = MockLLMProvider(
        canned_content="Done",
        canned_tool_calls=[
            ToolCallRequest(
                id="call_mod",
                name="echo_tool",
                arguments={"original_param": "raw_input"},
            )
        ],
    )

    mod_hook = ModifyingHook()
    config = AgentConfig(
        agent_id="test_mod_agent",
        name="ModAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
        hooks=(mod_hook,),
    )

    agent = BaseAgent(
        config=config,
        llm=llm,
        tools=registry,
    )

    turn_result = await agent.execute_turn("Execute mod tool")
    assert turn_result.is_completed is True
    assert len(turn_result.tool_executions) == 1
    rec = turn_result.tool_executions[0]
    assert rec.status == "success"

    # Pre-hook injected "sanitized": True, Tool echoed arguments, Post-hook injected "post_processed": True
    assert isinstance(rec.output, dict)
    assert rec.output.get("sanitized") is True
    assert rec.output.get("post_processed") is True


@pytest.mark.asyncio
async def test_base_agent_pre_turn_and_post_turn_hooks() -> None:
    """BaseAgent executes PRE_TURN and POST_TURN hooks modifying turn prompt and response."""
    llm = MockLLMProvider(canned_content="Original Assistant Response")
    mod_hook = ModifyingHook()

    config = AgentConfig(
        agent_id="test_turn_mod_agent",
        name="TurnModAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(config=config, llm=llm, hooks=[mod_hook])

    turn_result = await agent.execute_turn("User prompt")
    assert turn_result.is_completed is True
    # PRE_TURN hook modified input to "modified turn input"
    user_msg = [m for m in agent.history if m.role == MessageRole.USER][0]
    assert user_msg.content == "modified turn input"

    # POST_TURN hook modified output to "modified assistant response"
    assert turn_result.content == "modified assistant response"
    assistant_msg = [m for m in agent.history if m.role == MessageRole.ASSISTANT][0]
    assert assistant_msg.content == "modified assistant response"


@pytest.mark.asyncio
async def test_base_agent_pre_turn_blocked() -> None:
    """BaseAgent aborts turn when PRE_TURN hook returns BLOCK."""
    llm = MockLLMProvider(canned_content="Should not be called")
    block_turn_hook = BlockingTurnHook(block_reason="Turn disallowed by security policy")

    config = AgentConfig(
        agent_id="test_turn_block_agent",
        name="TurnBlockAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(config=config, llm=llm, hooks=[block_turn_hook])

    turn_result = await agent.execute_turn("User prompt")
    assert turn_result.is_completed is False
    assert turn_result.error == "Turn disallowed by security policy"
    assert (
        "Turn execution blocked by hook: Turn disallowed by security policy" in turn_result.content
    )
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_base_agent_on_error_hook_on_exception() -> None:
    """BaseAgent triggers ON_ERROR hook when an unexpected exception occurs during turn execution."""

    class FailingLLMProvider(BaseLLMConnector):
        @property
        def provider_name(self) -> str:
            return "failing"

        async def generate(self, request: LLMRequest) -> ModelResponse:
            raise RuntimeError("LLM provider internal failure")

        async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
            raise RuntimeError("LLM provider internal failure")
            yield StreamChunk(delta_content="")  # type: ignore[unreachable]

    err_hook = ErrorTrackingHook()
    config = AgentConfig(
        agent_id="test_err_agent",
        name="ErrAgent",
        llm_config=AgentLLMConfig(model_name="failing-model"),
    )
    agent = BaseAgent(config=config, llm=FailingLLMProvider(), hooks=[err_hook])

    result = await agent.execute_turn("Crash turn")
    assert result.is_completed is False
    assert result.error == "LLM provider internal failure"

    assert len(err_hook.recorded_errors) == 1
    assert "LLM provider internal failure" in err_hook.recorded_errors[0]["error"]


@pytest.mark.asyncio
async def test_script_hook_sh_execution() -> None:
    """ScriptHook executes .sh shell script with json input."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(
            """#!/bin/bash
echo '{"action": "allow", "reason": "Shell check passed"}'
"""
        )
        script_path = f.name

    try:
        hook = ScriptHook(script_path=script_path)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TOOL_USE)
        decision = await hook.dispatch(ctx)
        assert decision.action == HookAction.ALLOW
        assert decision.reason == "Shell check passed"
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_empty_stdout_exit_0() -> None:
    """ScriptHook with empty stdout on exit 0 defaults to ALLOW."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import sys
# Empty output
sys.exit(0)
"""
        )
        script_path = f.name

    try:
        hook = ScriptHook(script_path=script_path)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TOOL_USE)
        decision = await hook.dispatch(ctx)
        assert decision.action == HookAction.ALLOW
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_malformed_json_fail_open_vs_fail_closed() -> None:
    """ScriptHook with malformed JSON on exit 0 handles FAIL_OPEN vs FAIL_CLOSED."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """print("THIS IS NOT JSON")
"""
        )
        script_path = f.name

    try:
        # FAIL_OPEN
        hook_open = ScriptHook(script_path=script_path, failure_policy=FailurePolicy.FAIL_OPEN)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)
        res_open = await hook_open.dispatch(ctx)
        assert res_open.action == HookAction.ALLOW

        # FAIL_CLOSED
        hook_closed = ScriptHook(script_path=script_path, failure_policy=FailurePolicy.FAIL_CLOSED)
        res_closed = await hook_closed.dispatch(ctx)
        assert res_closed.action == HookAction.BLOCK
        assert "malformed JSON" in (res_closed.reason or "")
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_nondict_json_fail_open_vs_fail_closed() -> None:
    """ScriptHook with non-dict JSON output handles FAIL_OPEN vs FAIL_CLOSED."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """import json
print(json.dumps([1, 2, 3]))
"""
        )
        script_path = f.name

    try:
        # FAIL_OPEN
        hook_open = ScriptHook(script_path=script_path, failure_policy=FailurePolicy.FAIL_OPEN)
        ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)
        res_open = await hook_open.dispatch(ctx)
        assert res_open.action == HookAction.ALLOW

        # FAIL_CLOSED
        hook_closed = ScriptHook(script_path=script_path, failure_policy=FailurePolicy.FAIL_CLOSED)
        res_closed = await hook_closed.dispatch(ctx)
        assert res_closed.action == HookAction.BLOCK
        assert "expected object" in (res_closed.reason or "")
    finally:
        os.remove(script_path)


@pytest.mark.asyncio
async def test_script_hook_nonexistent_binary_fail_open_vs_closed() -> None:
    """ScriptHook with nonexistent executable handles FAIL_OPEN vs FAIL_CLOSED."""
    hook_open = ScriptHook(
        script_path="/nonexistent/path/to/hook.sh", failure_policy=FailurePolicy.FAIL_OPEN
    )
    ctx = HookContext(agent_id="a1", event_type=HookEvent.PRE_TURN)
    res_open = await hook_open.dispatch(ctx)
    assert res_open.action == HookAction.ALLOW

    hook_closed = ScriptHook(
        script_path="/nonexistent/path/to/hook.sh", failure_policy=FailurePolicy.FAIL_CLOSED
    )
    res_closed = await hook_closed.dispatch(ctx)
    assert res_closed.action == HookAction.BLOCK
    assert "fail_closed" in (res_closed.reason or "")


@pytest.mark.asyncio
async def test_base_agent_post_tool_block_and_multiple_tools() -> None:
    """BaseAgent handles POST_TOOL_USE blocking and multiple concurrent tool calls."""
    registry = ToolRegistry()
    registry.register(EchoTool())

    class PostToolBlockHook(BaseHook):
        async def on_post_tool_use(self, context: HookContext) -> HookDecision:
            if context.payload.get("tool_name") == "echo_tool":
                args = context.payload.get("arguments", {})
                if args.get("val") == "secret":
                    return HookDecision(
                        action=HookAction.BLOCK, reason="Output contains secret data"
                    )
            return HookDecision(action=HookAction.ALLOW)

    llm = MockLLMProvider(
        canned_content="Invoking tools",
        canned_tool_calls=[
            ToolCallRequest(id="c1", name="echo_tool", arguments={"val": "public"}),
            ToolCallRequest(id="c2", name="echo_tool", arguments={"val": "secret"}),
        ],
    )

    config = AgentConfig(
        agent_id="test_multi_agent",
        name="MultiAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(config=config, llm=llm, tools=registry, hooks=[PostToolBlockHook()])

    turn_result = await agent.execute_turn("Run tools")
    assert turn_result.is_completed is True
    assert len(turn_result.tool_executions) == 2

    rec1 = turn_result.tool_executions[0]
    assert rec1.status == "success"
    assert rec1.output == {"val": "public"}

    rec2 = turn_result.tool_executions[1]
    assert rec2.status == "error"
    assert "Output contains secret data" in (rec2.error or "")


@pytest.mark.asyncio
async def test_base_agent_hook_runner_injection() -> None:
    """BaseAgent can be initialized with an explicit HookRunner instance."""
    hook_runner = HookRunner(hooks=[BlockingToolHook("Runner injected block")])
    config = AgentConfig(
        agent_id="test_runner_agent",
        name="RunnerAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(config=config, hook_runner=hook_runner)
    assert agent.hook_runner is hook_runner
    assert len(agent.hook_runner.hooks) == 1


# --------------------------------------------------------------------------------------
# Issue #665 (follow-up) — the PRE_TOOL_USE payload is not a re-frozen field
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_script_hook_receives_nested_tool_arguments(tmp_path: Path) -> None:
    """A `PRE_TOOL_USE` script hook runs for a tool call carrying nested arguments.

    `HookContext.payload` is a plain `dict[str, Any]` (`agent/hooks/models.py`), not an
    `ImmutableMapping`, so nothing re-freezes what `_execute_single_tool` puts in it and
    nothing deep-unwraps it on the way out. `ScriptRunner._run_script` then does
    `json.dumps(context.model_dump())`, which raised
    `TypeError: Object of type mappingproxy is not JSON serializable` on the shallow
    `dict(tc.arguments)`.

    **The failure was quiet, which is why this test asserts the hook ran rather than
    asserting no exception.** The raise happens inside `hook.dispatch`, and `HookRunner`
    converts it under the hook's failure policy — fail-closed by design on the pre-tool
    path — into `BLOCK`. So the observable symptom is a *refused tool call* with a
    plausible reason, not a crash: a test asserting only that `execute_turn` returned
    would pass against the broken code. Both halves are asserted here: the script saw
    `PRE_TOOL_USE`, and the tool it guards actually executed.

    Killed by: src/uclone_x/agent/base.py :: "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    seen_path = tmp_path / "events_seen.txt"
    script_path = tmp_path / "pre_tool_hook.py"
    script_path.write_text(
        "import sys, json\n"
        "data = json.load(sys.stdin)\n"
        f"open({str(seen_path)!r}, 'a').write(str(data['event_type']).upper() + '\\n')\n"
        'print(json.dumps({"action": "allow"}))\n',
        encoding="utf-8",
    )

    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = MockLLMProvider(
        canned_content="Invoking echo tool",
        canned_tool_calls=[
            ToolCallRequest(
                id="call_nested",
                name="echo_tool",
                arguments={
                    "path": "README.md",
                    "options": {"encoding": "utf-8"},
                    "ranges": [{"start": 1, "end": 20}],
                },
            )
        ],
    )
    config = AgentConfig(
        agent_id="test_script_hook_agent",
        name="ScriptHookAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(
        config=config,
        llm=llm,
        tools=registry,
        # FAIL_CLOSED is the default and the point: on the pre-tool path a hook that
        # raises denies the call, so the defect surfaced as a refusal, not an error.
        hooks=[ScriptHook(script_path=str(script_path), failure_policy=FailurePolicy.FAIL_CLOSED)],
    )

    turn_result = await agent.execute_turn("Echo with nested arguments")

    seen = seen_path.read_text(encoding="utf-8").split() if seen_path.exists() else []
    assert "PRE_TOOL_USE" in seen, f"pre-tool hook never ran; script saw {seen}"
    assert len(turn_result.tool_executions) == 1
    rec = turn_result.tool_executions[0]
    assert rec.status == "success", f"tool call was refused: {rec.error}"


@pytest.mark.asyncio
async def test_blocked_tool_call_records_nested_arguments() -> None:
    """A hook-blocked call with nested arguments still produces its refusal record.

    `ToolExecutionRecord.arguments` is `ImmutableJsonMapping`. The shallow
    `dict(tc.arguments)` here was cleared as benign during review of #673 on the reasoning
    that the field re-freezes what it is given; it does not — `JsonValue` validation runs
    *before* `AfterValidator(freeze_mapping)`, so a nested `MappingProxyType` is rejected,
    and the BLOCK branch raised `ValidationError` out of `_execute_single_tool` instead of
    returning the refusal it exists to return.

    Killed by: src/uclone_x/agent/base.py :: arguments=cast(dict[str, Any], unwrap_immutable(tc.arguments)),
    """
    nested_arguments: dict[str, Any] = {
        "msg": "hello",
        "options": {"encoding": "utf-8"},
        "ranges": [{"start": 1, "end": 20}],
    }
    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = MockLLMProvider(
        canned_content="Invoking echo tool",
        canned_tool_calls=[
            ToolCallRequest(id="call_blocked", name="echo_tool", arguments=nested_arguments)
        ],
    )
    config = AgentConfig(
        agent_id="test_block_nested_agent",
        name="BlockNestedAgent",
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    agent = BaseAgent(
        config=config,
        llm=llm,
        tools=registry,
        hooks=[BlockingToolHook(block_reason="Unauthorized tool usage")],
    )

    turn_result = await agent.execute_turn("Echo with nested arguments")

    assert turn_result.is_completed is True
    assert len(turn_result.tool_executions) == 1
    rec = turn_result.tool_executions[0]
    assert rec.status == "error"
    assert "Unauthorized tool usage" in (rec.error or "")
    assert unwrap_immutable(rec.arguments) == nested_arguments
