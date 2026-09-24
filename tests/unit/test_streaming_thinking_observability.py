# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
"""Unit tests for thinking and reasoning stream observability in UClone-X."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    StreamChunk,
    TokenUsage,
)


def parse_sse_events(raw_text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse raw SSE text into list of (event_type, json_data)."""
    events: list[tuple[str, dict[str, Any]]] = []
    blocks = raw_text.strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        event_name = "message"
        data_str = ""
        for line in lines:
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                data_str = line[6:].strip()
        if data_str:
            events.append((event_name, json.loads(data_str)))
    return events


class ThinkingStreamingMockLLM(MockLLMConnector):
    """Mock LLM that streams delta_thinking before delta_content."""

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        # Emulate reasoning phase
        thinking_steps = ["Let me think", " about the problem", " carefully."]
        for step in thinking_steps:
            yield StreamChunk(delta_thinking=step)

        # Emulate response phase
        content_steps = ["Here is", " the final", " answer."]
        for i, step in enumerate(content_steps):
            is_last = i == len(content_steps) - 1
            usage = (
                TokenUsage(
                    provider="mock",
                    model="mock-thinking",
                    input_tokens=15,
                    output_tokens=len(thinking_steps) + len(content_steps),
                    total_tokens=15 + len(thinking_steps) + len(content_steps),
                )
                if is_last
                else None
            )
            yield StreamChunk(
                delta_content=step,
                finish_reason=FinishReason.STOP if is_last else None,
                usage=usage,
            )


@pytest.mark.asyncio
async def test_ollama_stream_yields_thinking_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test OllamaConnector.stream yields StreamChunk with delta_thinking and logs progress."""
    caplog.set_level(logging.INFO)

    lines = [
        json.dumps({"message": {"thinking": "Considering user input..."}}),
        json.dumps({"message": {"thinking": " Formulating strategy."}}),
        json.dumps({"message": {"content": "Hello!"}}),
        json.dumps(
            {
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 5,
            }
        ),
    ]

    class FakeResponse:
        status_code = 200

        async def aiter_lines(self) -> AsyncIterator[str]:
            for line in lines:
                yield line

        async def aread(self) -> bytes:
            return b""

    class FakeStreamContext:
        async def __aenter__(self) -> FakeResponse:
            return FakeResponse()

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    class FakeClient:
        def stream(self, *args: Any, **kwargs: Any) -> FakeStreamContext:
            return FakeStreamContext()

        async def aclose(self) -> None:
            pass

    connector = OllamaConnector(base_url="http://fake-ollama:11434")
    monkeypatch.setattr(connector, "_get_client", lambda: FakeClient())

    req = LLMRequest(model="qwen3:8b", messages=())
    chunks: list[StreamChunk] = []
    async for chunk in connector.stream(req):
        chunks.append(chunk)

    assert len(chunks) == 4
    assert chunks[0].delta_thinking == "Considering user input..."
    assert chunks[1].delta_thinking == " Formulating strategy."
    assert chunks[2].delta_content == "Hello!"
    assert chunks[3].finish_reason == FinishReason.STOP
    assert chunks[3].usage is not None
    assert chunks[3].usage.total_tokens == 15

    # Verify log messages recorded
    logs = caplog.text
    assert "Model qwen3:8b started reasoning/thinking" in logs
    assert "Model qwen3:8b finished thinking (2 tokens), generating response" in logs
    assert "Stream finished: model=qwen3:8b" in logs


@pytest.mark.asyncio
async def test_base_agent_invoke_model_streams_thinking(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Test BaseAgent._invoke_model properly receives and logs thinking stream."""
    caplog.set_level(logging.INFO)

    cfg = AgentConfig(
        agent_id="test-thinking-agent",
        name="Test Thinking Agent",
        llm_config=AgentLLMConfig(model_name="mock-thinking"),
    )
    llm = ThinkingStreamingMockLLM()
    from uclone_x.agent.session import SessionStore

    agent = BaseAgent(config=cfg, llm=llm, store=SessionStore(storage_dir=tmp_path / "sessions"))

    captured_events: list[tuple[str, dict[str, Any]]] = []

    def stream_cb(event_type: str, data: dict[str, Any]) -> None:
        captured_events.append((event_type, data))

    res = await agent.execute_turn("Test query", stream_callback=stream_cb)

    assert res.is_completed is True
    assert res.content == "Here is the final answer."

    # Verify captured status events containing thinking details
    thinking_details = [
        data.get("detail")
        for evt, data in captured_events
        if evt == "status" and data.get("status") == "thinking"
    ]
    assert "Let me think" in thinking_details
    assert " about the problem" in thinking_details
    assert " carefully." in thinking_details

    # Verify captured token events
    tokens: list[str] = [
        str(data.get("content") or "") for evt, data in captured_events if evt == "token"
    ]
    assert "".join(tokens) == "Here is the final answer."

    # Verify agent logs recorded thinking start and finish
    logs = caplog.text
    assert "began thinking/reasoning" in logs
    assert "finished thinking (3 tokens); generating response" in logs
