"""Tests for Tier 1 in-turn artifact hallucination nudge and recovery."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

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
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="agent", model="dummy"),
    served_by=ServiceRef(provider="agent", model="dummy"),
    attempts=(),
)
_USAGE = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)


class _ImageParams(BaseModel):
    prompt: str = Field(description="image prompt")


class _MockGenerateImageTool(BaseTool[_ImageParams]):
    name = "generate_image"
    description = "generates an image"

    def run(self, params: _ImageParams, context: ToolContext) -> ToolResult:
        ws = context.workspace_root or Path.cwd()
        img_path = ws / "artifacts" / "images" / "img_real.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"PNGDATA")
        rel = "artifacts/images/img_real.png"
        return ToolResult(
            success=True,
            output=f"/api/artifacts/content?path={rel}",
            artifacts=(rel,),
            provenance=_PROV,
        )


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
async def test_artifact_hallucination_triggers_nudge_and_recovers(tmp_path: Path) -> None:
    """When model hallucinates an image URL without calling generate_image, nudge fires and model recovers."""
    # Step 0: Model hallucinates a nonexistent image link without calling generate_image
    resp_step0 = ModelResponse(
        content="Here is your image: ![Hallucinated](/api/artifacts/content?path=artifacts/images/img_fake123.png)",
        finish_reason=FinishReason.STOP,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )
    # Step 1 (after nudge): Model calls generate_image
    resp_step1 = ModelResponse(
        content=None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCallRequest(
                id="call_img_1",
                name="generate_image",
                arguments={"prompt": "cat painting"},
            ),
        ),
        usage=_USAGE,
        provenance=_PROV,
    )
    # Step 2: Model outputs the real image link
    resp_step2 = ModelResponse(
        content="Here is your real image: ![Cat](/api/artifacts/content?path=artifacts/images/img_real.png)",
        finish_reason=FinishReason.STOP,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )

    llm = _ScriptedLLM([resp_step0, resp_step1, resp_step2])
    registry = ToolRegistry()
    registry.register(_MockGenerateImageTool())

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
            session_id="sess_artist",
            workspace_root=tmp_path,
        ),
    )

    result = await agent.execute_turn("Make me an image")
    assert result.is_completed

    # Check durable events for ARTIFACT_NUDGE
    artifact_nudges = [e for e in agent.pending_durable_events if e.get("type") == "ARTIFACT_NUDGE"]
    assert len(artifact_nudges) == 1
    assert artifact_nudges[0]["missing_artifact"] == "artifacts/images/img_fake123.png"

    # Final content contains the real image
    assert "artifacts/images/img_real.png" in result.content


@pytest.mark.asyncio
async def test_artifact_hallucination_persists_sanitized_by_tier2_hook(tmp_path: Path) -> None:
    """If model ignores nudge and hallucinates again, Tier 2 hook sanitizes it to prevent 404."""
    resp_step0 = ModelResponse(
        content="![Fake0](/api/artifacts/content?path=artifacts/images/img_fake0.png)",
        finish_reason=FinishReason.STOP,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )
    resp_step1 = ModelResponse(
        content="Still giving you: ![Fake1](/api/artifacts/content?path=artifacts/images/img_fake1.png)",
        finish_reason=FinishReason.STOP,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )

    llm = _ScriptedLLM([resp_step0, resp_step1])
    registry = ToolRegistry()
    registry.register(_MockGenerateImageTool())

    agent = BaseAgent(
        config=AgentConfig(
            agent_id="stubborn_artist",
            name="Stubborn Artist",
            llm_config=AgentLLMConfig(model_name="dummy"),
        ),
        llm=llm,
        tools=registry,
        context=AgentContext(
            agent_id="stubborn_artist",
            session_id="sess_stubborn",
            workspace_root=tmp_path,
        ),
    )

    result = await agent.execute_turn("Make me an image")
    assert result.is_completed

    # Tier 1 fired once
    artifact_nudges = [e for e in agent.pending_durable_events if e.get("type") == "ARTIFACT_NUDGE"]
    assert len(artifact_nudges) == 1

    # Tier 2 hook sanitized the second hallucinated image so no 404 is rendered
    assert "img_fake1.png" in result.content
    assert (
        "> ⚠️ *[이미지 생성 도구가 실행되지 않아 이미지가 표시되지 않습니다: img_fake1.png]*"
        in result.content
    )
    assert "/api/artifacts/content" not in result.content
