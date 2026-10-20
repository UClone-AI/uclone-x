"""Tests for in-turn tool guardrails: loop prevention, repeated refusals (#2300) and context
budget headroom (#1953)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, ToolExecutionRecord
from uclone_x.agent.turn_executor import repeated_refusals
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="agent", model="dummy"),
    served_by=ServiceRef(provider="agent", model="dummy"),
    attempts=(),
)
_USAGE = TokenUsage(provider="dummy", model="dummy", input_tokens=10, output_tokens=10)


class _ImageParams(BaseModel):
    prompt: str = Field(default="", description="image prompt")


class _MockGenerateImageTool(BaseTool[_ImageParams]):
    name = "generate_image"
    description = "generates an image"

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def run(self, params: _ImageParams, context: ToolContext) -> dict[str, Any]:
        self.call_count += 1
        ws = context.workspace_root or Path.cwd()
        img_path = ws / "artifacts" / "images" / f"img_real_{self.call_count}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"PNGDATA")
        rel = f"artifacts/images/img_real_{self.call_count}.png"
        return {
            "status": "success",
            "path": rel,
            "relative_url": f"/api/artifacts/content?path={rel}",
            "markdown_gallery": f"![Image](/api/artifacts/content?path={rel})",
        }


class _ScriptedLLM(BaseLLMConnector):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.recorded_requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.recorded_requests.append(request)
        return self._responses.pop(0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        yield StreamChunk(
            delta_content=resp.content,
            finish_reason=resp.finish_reason,
            tool_calls=resp.tool_calls,
        )


@pytest.mark.asyncio
async def test_duplicate_generate_image_in_same_turn_is_guarded(tmp_path: Path) -> None:
    """When a model attempts to call generate_image again in the same turn, it is intercepted and nudged."""
    # Step 1: Model calls generate_image
    resp_step1 = ModelResponse(
        content=None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCallRequest(
                id="call_img_1",
                name="generate_image",
                arguments={"prompt": "cat in garden"},
            ),
        ),
        usage=_USAGE,
        provenance=_PROV,
    )
    # Step 2: Model erroneously calls generate_image AGAIN
    resp_step2 = ModelResponse(
        content=None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCallRequest(
                id="call_img_2",
                name="generate_image",
                arguments={"prompt": "cat in garden 2"},
            ),
        ),
        usage=_USAGE,
        provenance=_PROV,
    )
    # Step 3: Model receives guard notice and concludes response
    resp_step3 = ModelResponse(
        content="Here is your image: ![Image](/api/artifacts/content?path=artifacts/images/img_real_1.png)",
        finish_reason=FinishReason.STOP,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )

    llm = _ScriptedLLM([resp_step1, resp_step2, resp_step3])
    img_tool = _MockGenerateImageTool()
    registry = ToolRegistry()
    registry.register(img_tool)

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="artist",
            name="Artist",
            llm_config=AgentLLMConfig(model_name="dummy"),
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(
            agent_id="artist",
            session_id="sess_guard_test",
            workspace_root=tmp_path,
        ),
    )

    result = await agent.execute_turn("Draw a cat")
    assert result.is_completed is True
    assert result.stop_reason == "model_stopped_after_nudge"
    # generate_image was executed only once; second call was intercepted
    assert img_tool.call_count == 1
    # On step 3 request, tools were stripped to force final text response
    assert len(llm.recorded_requests) == 3
    assert len(llm.recorded_requests[2].tools) == 0
    assert "img_real_1.png" in result.content


@pytest.mark.asyncio
async def test_context_window_headroom_guard_intercepts_overflow(tmp_path: Path) -> None:
    """When remaining context budget is below safety headroom, further tool calls are guarded."""
    resp_step1 = ModelResponse(
        content=None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCallRequest(
                id="call_img_1",
                name="generate_image",
                arguments={"prompt": "scenery"},
            ),
        ),
        usage=_USAGE,
        provenance=_PROV,
    )
    resp_step2 = ModelResponse(
        content=None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCallRequest(
                id="call_img_2",
                name="generate_image",
                arguments={"prompt": "scenery 2"},
            ),
        ),
        usage=_USAGE,
        provenance=_PROV,
    )
    resp_step3 = ModelResponse(
        content="Final scenery: ![Image](/api/artifacts/content?path=artifacts/images/img_real_1.png)",
        finish_reason=FinishReason.STOP,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )

    llm = _ScriptedLLM([resp_step1, resp_step2, resp_step3])
    img_tool = _MockGenerateImageTool()
    registry = ToolRegistry()
    registry.register(img_tool)

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="artist",
            name="Artist",
            llm_config=AgentLLMConfig(model_name="dummy", context_limit=4096),
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(
            agent_id="artist",
            session_id="sess_headroom_test",
            workspace_root=tmp_path,
        ),
    )

    result = await agent.execute_turn("Draw scenery")
    assert result.is_completed is True
    assert result.stop_reason == "model_stopped_after_nudge"
    assert img_tool.call_count == 1
    assert "img_real_1.png" in result.content


class _RefusingParams(BaseModel):
    scene: str = Field(default="s1", description="scene")
    text: str = Field(default="", description="the scene's text")


class _RefusingTool(BaseTool[_RefusingParams]):
    """Refuses every write the same way, whatever the scene or the text."""

    name = "write_scene"
    description = "writes a scene"
    retry_key_arguments = frozenset({"scene"})

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def run(self, params: _RefusingParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        self.call_count += 1
        raise ValueError("Nothing was written: the story waits for the person.")


class _UrlParams(BaseModel):
    url: str = Field(description="url")


class _TimingOutTool(BaseTool[_UrlParams]):
    """Fails every fetch with the same transport error, whatever the URL."""

    name = "web_fetch"
    description = "fetches a page"

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def run(self, params: _UrlParams, context: ToolContext) -> dict[str, Any]:
        del context
        self.urls.append(params.url)
        raise ValueError("ReadTimeout: timed out")


class _NoParams(BaseModel):
    pass


class _EchoTool(BaseTool[_NoParams]):
    name = "echo"
    description = "answers"

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        self.call_count += 1
        return {"ok": True}


def _step(*calls: tuple[str, dict[str, Any]]) -> ModelResponse:
    return ModelResponse(
        content=None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(
            ToolCallRequest(id=f"call_{name}_{n}_{id(arguments)}", name=name, arguments=arguments)
            for n, (name, arguments) in enumerate(calls)
        ),
        usage=_USAGE,
        provenance=_PROV,
    )


_FINAL = ModelResponse(
    content="The story is waiting for your answer.",
    finish_reason=FinishReason.STOP,
    tool_calls=(),
    usage=_USAGE,
    provenance=_PROV,
)


def _guarded_agent(tmp_path: Path, llm: _ScriptedLLM, *tools: BaseTool[Any]) -> BaseAgent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="writer", name="Writer", llm_config=AgentLLMConfig(model_name="dummy")
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(
            agent_id="writer", session_id="sess_refusals", workspace_root=tmp_path
        ),
    )


@pytest.mark.asyncio
async def test_one_scene_written_again_with_new_text_is_not_run_a_third_time(
    tmp_path: Path,
) -> None:
    """#2300: a live Writer wrote one scene 155 times, each with new prose, all refused.

    Killed by: src/uclone_x/agent/turn_executor.py :: keys: object = getattr(self._tool_invoker.resolve(name), "retry_key_arguments", None)
    Becomes: keys: object = None
    """
    write = _RefusingTool()
    llm = _ScriptedLLM(
        [
            _step(("write_scene", {"scene": "s1", "text": "one"})),
            _step(("write_scene", {"scene": "s1", "text": "two"})),
            _step(("write_scene", {"scene": "s1", "text": "three"})),
            _FINAL,
        ]
    )

    result = await _guarded_agent(tmp_path, llm, write, _EchoTool()).execute_turn("네")

    assert write.call_count == 2
    assert len(llm.recorded_requests) == 4
    assert {t.name for t in llm.recorded_requests[3].tools} >= {"write_scene", "echo"}
    assert result.content == "The story is waiting for your answer."


@pytest.mark.asyncio
async def test_two_timeouts_on_two_urls_do_not_stop_a_third_url_or_any_tool(
    tmp_path: Path,
) -> None:
    """#2302 review: a call per tool name lost a research clone the web after two timeouts.

    Killed by: src/uclone_x/agent/turn_executor.py :: kept = {k: v for k, v in arguments.items() if keys is None or k in keys}
    Becomes: kept = {}
    """
    fetch, echo = _TimingOutTool(), _EchoTool()
    llm = _ScriptedLLM(
        [
            _step(("web_fetch", {"url": "https://a.example"})),
            _step(("web_fetch", {"url": "https://b.example"})),
            _step(("web_fetch", {"url": "https://c.example"}), ("echo", {})),
            _FINAL,
        ]
    )

    await _guarded_agent(tmp_path, llm, fetch, echo).execute_turn("look these up")

    assert fetch.urls == ["https://a.example", "https://b.example", "https://c.example"]
    assert echo.call_count == 1


@pytest.mark.asyncio
async def test_only_the_repeated_call_is_skipped_and_the_rest_of_its_step_runs(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: run = [tc for tc in tool_calls if tc.id not in repeated]
    Becomes: run = []
    """
    write, echo = _RefusingTool(), _EchoTool()
    llm = _ScriptedLLM(
        [
            _step(("write_scene", {"scene": "s1"})),
            _step(("write_scene", {"scene": "s1"})),
            _step(("write_scene", {"scene": "s1"}), ("echo", {})),
            _FINAL,
        ]
    )

    await _guarded_agent(tmp_path, llm, write, echo).execute_turn("네")

    assert write.call_count == 2
    assert echo.call_count == 1
    third = [m for m in llm.recorded_requests[3].messages if m.role.value == "tool"][-2:]
    assert "was refused 2 times" in (third[0].content or "")
    assert "ok" in (third[1].content or "")


def test_the_limit_counts_the_same_call_refused_the_same_way() -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: if times >= REPEATED_REFUSAL_LIMIT:
    Becomes: if times > REPEATED_REFUSAL_LIMIT:
    """

    def refused(error: str = "Waits for the person.") -> ToolExecutionRecord:
        return ToolExecutionRecord(
            tool_name="write_scene",
            arguments={"scene": "s1"},
            status=ToolResultStatus.ERROR,
            error=error,
        )

    call = ToolCallRequest(id="c", name="write_scene", arguments={"scene": "s1"})
    elsewhere = ToolCallRequest(id="e", name="write_scene", arguments={"scene": "s2"})
    other = ToolCallRequest(id="d", name="story_start", arguments={"scene": "s1"})

    assert repeated_refusals([call], [refused()]) == {}
    assert repeated_refusals([call], [refused(), refused()]) == {"c": 2}
    assert repeated_refusals([elsewhere, other], [refused(), refused()]) == {}
    ok = ToolExecutionRecord(tool_name="write_scene", arguments={"scene": "s1"})
    assert repeated_refusals([call], [refused(), ok]) == {}


def test_two_different_refusals_are_not_a_repeat() -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: (record.tool_name, call_shape(record.arguments, keys_of(record.tool_name)), error)
    Becomes: (record.tool_name, call_shape(record.arguments, keys_of(record.tool_name)), "")
    """
    first, second = (
        ToolExecutionRecord(
            tool_name="write_scene",
            arguments={"scene": "s1"},
            status=ToolResultStatus.ERROR,
            error=e,
        )
        for e in ("Bad scene id.", "Waits for the person.")
    )
    call = ToolCallRequest(id="c", name="write_scene", arguments={"scene": "s1"})

    assert repeated_refusals([call], [first, second]) == {}


@pytest.mark.asyncio
async def test_a_repeat_in_the_step_does_not_let_a_second_image_past_the_guard(
    tmp_path: Path,
) -> None:
    """#2302 re-review: with a repeated refused call in the step, the step's other calls
    skipped the guardrail and a second image was drawn.

    Killed by: src/uclone_x/agent/turn_executor.py :: guard_notice = self._evaluate_tool_guardrail(tool_calls, tool_executions, req)
    Becomes: guard_notice = None if repeated_refusals(tool_calls, tool_executions, self._retry_keys_of) else self._evaluate_tool_guardrail(tool_calls, tool_executions, req)
    """
    write, image = _RefusingTool(), _MockGenerateImageTool()
    llm = _ScriptedLLM(
        [
            _step(("generate_image", {"prompt": "a"})),
            _step(("write_scene", {"scene": "s1"})),
            _step(("write_scene", {"scene": "s1"})),
            _step(("write_scene", {"scene": "s1"}), ("generate_image", {"prompt": "b"})),
            _FINAL,
            _FINAL,
        ]
    )

    await _guarded_agent(tmp_path, llm, write, image).execute_turn("draw, then write")

    assert image.call_count == 1
    assert write.call_count == 2


@pytest.mark.asyncio
async def test_a_repeat_in_the_step_does_not_let_a_call_past_the_overflow_guard(
    tmp_path: Path,
) -> None:
    """#2302 re-review, measured: the same early return skipped the context-overflow guard,
    so a step at the edge of the window ran its other calls. The window is set so that the
    fourth request, the one with the repeat, just reaches the guard's threshold.

    Killed by: src/uclone_x/agent/turn_executor.py :: guard_notice = self._evaluate_tool_guardrail(tool_calls, tool_executions, req)
    Becomes: guard_notice = None if repeated_refusals(tool_calls, tool_executions, self._retry_keys_of) else self._evaluate_tool_guardrail(tool_calls, tool_executions, req)
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    def steps() -> list[ModelResponse]:
        return [
            _step(("write_scene", {"scene": "s1"})),
            _step(("write_scene", {"scene": "s1"})),
            _step(("write_scene", {"scene": "s1"}), ("echo", {})),
            _FINAL,
            _FINAL,
        ]

    def agent(
        workspace: Path, llm: _ScriptedLLM, window: int | None, *tools: BaseTool[Any]
    ) -> BaseAgent:
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        config = AgentLLMConfig(model_name="dummy", context_limit=window, max_tokens=64)
        return BaseAgent(
            config=AgentConfig(agent_id="writer", name="Writer", llm_config=config),
            llm=llm,
            tools=registry,
            context=AgentContext(agent_id="writer", session_id="s", workspace_root=workspace),
        )

    # The same agent with no window, in a folder of the same length, gives the requests'
    # sizes; the third request (the step with the repeat) then sits exactly at the guard's
    # threshold, and the two before it under it.
    probe = _ScriptedLLM(steps())
    await agent(tmp_path / "a", probe, None, _RefusingTool(), _EchoTool()).execute_turn("네")
    third = probe.recorded_requests[2]
    window = estimate_request_tokens(LLMRequest(messages=third.messages, tools=third.tools)) + 64

    write, echo = _RefusingTool(), _EchoTool()
    llm = _ScriptedLLM(steps())
    real = agent(tmp_path / "b", llm, window, write, echo)

    await real.execute_turn("네")

    assert write.call_count == 2
    # The overflow guard held the whole step: the other call did not run. (Its notices
    # then do not fit either, and the step is refused; that is the window's own limit.)
    assert echo.call_count == 0
