"""Mock LLM provider connector for deterministic offline testing and fallbacks (Principle 5 & Principle 6)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.compactor import estimate_reply_tokens, estimate_request_tokens
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
)


class MockLLMConnector(BaseLLMConnector):
    """Mock LLM provider returning deterministic responses, tool calls, and token stream chunks."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        default_response: str | None = None,
        tool_calls: Sequence[ToolCallRequest] | None = None,
        api_key: str | None = "mock-key",
        base_url: str | None = "http://mock-llm.local",
        default_model: str = "mock-model",
        latency_seconds: float = 0.0,
        streaming_chunk_delay: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        # Accepted and recorded although nothing here waits on a socket, so that
        # `create_llm_connector(provider=..., timeout=...)` means the same thing for every
        # provider name it maps. Without it a caller that names its own ceiling -- which
        # `build_agent_answerer` now does -- raises `TypeError` for `mock` alone, and the
        # failure arrives from the factory instead of from the suite's own precondition,
        # which is the message that actually names the flag to pass.
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)
        self._responses: list[str] = list(responses) if responses is not None else []
        self._default_response = default_response
        self._tool_calls: list[ToolCallRequest] = list(tool_calls) if tool_calls is not None else []
        self._call_count: int = 0
        self._default_model = default_model
        self.latency_seconds: float = max(0.0, latency_seconds)
        self.streaming_chunk_delay: float = max(0.0, streaming_chunk_delay)

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def call_count(self) -> int:
        return self._call_count

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if self.latency_seconds > 0:
            await asyncio.sleep(self.latency_seconds)
        self._call_count += 1
        model_name = (
            request.model.strip()
            if (request.model and request.model.strip() and request.model.strip() != "default")
            else self._default_model
        )
        service_ref = ServiceRef(provider="mock", model=model_name)

        if self._responses:
            content = self._responses.pop(0)
        elif self._default_response is not None:
            content = self._default_response
        else:
            last_msg = ""
            if request.messages:
                last_msg = request.messages[-1].content or ""
            content = (
                f"Agent processed message: '{last_msg}'. "
                "BaseAgent reasoning completed successfully."
            )

        # A real model asks for a tool once and then answers from its result. Returning
        # the same call on every invocation made this stub unusable the moment the agent
        # gained a step loop: it requested the same tool until the step ceiling stopped
        # it. A configured call is spent once *its own* result is in the request.
        #
        # Keyed on the configured `tool_call_id`s rather than on "any TOOL message is
        # present": a session hydrated from an earlier one carries that session's tool
        # results, and the broader test read those as this stub's calls having already
        # run — so the stub answered without ever asking for its tool, and the skill it
        # was configured to load was silently never loaded.
        spent_ids = {m.tool_call_id for m in request.messages if m.role is MessageRole.TOOL}
        tool_calls_to_return: tuple[ToolCallRequest, ...] = tuple(
            tc for tc in self._tool_calls if tc.id not in spent_ids
        )
        finish_reason = FinishReason.TOOL_CALLS if tool_calls_to_return else FinishReason.STOP

        # No provider counted these tokens, so they are the shared estimate and say so — the
        # rule for any count a provider did not report (design doc §6.7 [#939], #983). A
        # `PROVIDER` label here made a mock-backed dashboard show them as counted.
        in_tokens = estimate_request_tokens(request)
        out_tokens = estimate_reply_tokens(content, tool_calls_to_return)
        usage = TokenUsage(
            provider="mock",
            model=model_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            count_source=TokenCountSource.ESTIMATE,
        )

        provenance = Provenance(
            path=ExecutionPath.PRIMARY,
            requested=service_ref,
            served_by=service_ref,
        )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls_to_return,
            usage=usage,
            finish_reason=finish_reason,
            model_name=model_name,
            provenance=provenance,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        text = resp.content or ""
        words = text.split(" ") if text else [""]
        for i, word in enumerate(words):
            if self.streaming_chunk_delay > 0 and i > 0:
                await asyncio.sleep(self.streaming_chunk_delay)
            chunk_text = word if i == 0 else f" {word}"
            is_last = i == len(words) - 1
            yield StreamChunk(
                delta_content=chunk_text,
                finish_reason=resp.finish_reason if is_last else None,
                usage=resp.usage if is_last else None,
                tool_calls=resp.tool_calls if is_last and resp.tool_calls else (),
            )
