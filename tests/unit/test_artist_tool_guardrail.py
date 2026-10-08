"""Tests for in-turn tool guardrails: loop prevention and context budget headroom (#1953)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
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
from uclone_x.tools.models import ToolContext
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
