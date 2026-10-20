"""A provider's signature on a tool call survives everything between two requests (#2227).

Gemini 3 signs a function call (`thoughtSignature`) and refuses a request that replays the
call without it (400 "Function call is missing a thought_signature", measured live on
`gemini-3.8-flash` on 2026-10-10), so every tool turn's second request failed.
`ToolCallRequest.signature` carries it; these tests hold that nothing between the reply
and the next request drops it: the agent loop, the redaction every history write goes
through, a reload from the session log, and compaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.core.session_state import redact_message
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
    ToolCallRequest,
)
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry

SIGNATURE = "Eq0BCqoBAWkUfRN-opaque"
#: A credential shape in the arguments, so the history write redacts this call (#569).
SECRET_ARGS = {"note": "sk-1234567890123456789012345678901234567890"}


class _SigningLLM(MockLLMConnector):
    """Signs the call it makes, and refuses -- as Gemini 3 does -- a request that replays a
    call without the signature it was issued with."""

    def __init__(self, calls: int) -> None:
        super().__init__(default_response="done")
        self._calls = calls
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        for message in request.messages:
            for call in message.tool_calls:
                if call.signature != SIGNATURE:
                    raise AssertionError(f"call {call.id} replayed without its signature")
        self._tool_calls = []
        if self._calls:
            self._calls -= 1
            self._tool_calls = [
                ToolCallRequest(
                    id=f"c{len(self.requests)}",
                    name="note",
                    arguments=SECRET_ARGS,
                    signature=SIGNATURE,
                )
            ]
        return await super().generate(request)


def _agent(tmp_path: Path, store: SessionStore, llm: MockLLMConnector) -> BaseAgent:
    async def note(params: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output="noted",
            provenance=Provenance.primary(provider="local.test", model="note"),
        )

    registry = ToolRegistry()
    registry.register(LocalTool("note", "Notes.", handler=note, writes_files=False))
    return BaseAgent(
        config=AgentConfig(
            agent_id="signer",
            name="Signer",
            workspace_dir=tmp_path,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        tools=registry,
        store=store,
    )


def _replayed(request: LLMRequest) -> list[str | None]:
    return [c.signature for m in request.messages for c in m.tool_calls]


def test_redaction_keeps_the_signature_of_a_call_it_rewrites() -> None:
    """Every history write redacts the message; a call whose arguments held a credential
    shape was rebuilt field by field and lost its signature.

    Killed by: src/uclone_x/core/session_state.py :: ToolCallRequest.model_validate({**tc.model_dump(), "arguments": sanitized_args})
    Becomes: ToolCallRequest(id=tc.id, name=tc.name, arguments=sanitized_args)
    """
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(
            ToolCallRequest(id="c1", name="note", arguments=SECRET_ARGS, signature=SIGNATURE),
        ),
    )
    clean = redact_message(message)
    assert dict(clean.tool_calls[0].arguments) == {"note": "[REDACTED]"}
    assert clean.tool_calls[0].signature == SIGNATURE


@pytest.mark.asyncio
async def test_a_tool_turn_replays_the_signature_and_after_a_reload_too(tmp_path: Path) -> None:
    store = SessionStore(storage_dir=tmp_path / "sessions")
    llm = _SigningLLM(calls=1)
    agent = _agent(tmp_path, store, llm)
    await agent.execute_turn("note it")
    assert _replayed(llm.requests[-1]) == [SIGNATURE]
    agent.persist_session()

    later = _SigningLLM(calls=0)
    fresh = _agent(tmp_path, store, later)
    fresh.hydrate_session(agent.session_id)
    await fresh.execute_turn("and now?")
    assert _replayed(later.requests[0]) == [SIGNATURE]


@pytest.mark.asyncio
async def test_compaction_keeps_the_signature_of_a_kept_call() -> None:
    call = ToolCallRequest(id="c9", name="note", arguments={}, signature=SIGNATURE)
    old = [
        message
        for n in range(6)
        for message in (
            ChatMessage(role=MessageRole.USER, content=f"turn {n}"),
            ChatMessage(role=MessageRole.ASSISTANT, content=f"answer {n}"),
        )
    ]
    recent = [
        ChatMessage(role=MessageRole.USER, content="note it"),
        ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=(call,)),
        ChatMessage(role=MessageRole.TOOL, name="note", tool_call_id="c9", content="noted"),
    ]
    outcome = await ContextCompactor(keep_recent_turns=1).compact([*old, *recent])
    assert len(outcome.messages) < len(old) + len(recent)  # it did compact
    assert [c.signature for m in outcome.messages for c in m.tool_calls] == [SIGNATURE]


def _signed_history() -> LLMRequest:
    """A history as Gemini leaves it: a signed call, its result, and a follow-up."""
    return LLMRequest(
        model="m",
        messages=(
            ChatMessage(role=MessageRole.USER, content="note it"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="noting",
                tool_calls=(
                    ToolCallRequest(id="c1", name="note", arguments={}, signature=SIGNATURE),
                ),
            ),
            ChatMessage(role=MessageRole.TOOL, name="note", tool_call_id="c1", content="noted"),
            ChatMessage(role=MessageRole.USER, content="and now?"),
        ),
    )


def _signature_paths(value: object, path: str = "") -> list[str]:
    """Every place in a payload that holds the signature or a key spelt like one."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, inner in cast(dict[str, object], value).items():
            if "signature" in str(key).lower().replace("_", ""):
                found.append(f"{path}.{key}")
            found.extend(_signature_paths(inner, f"{path}.{key}"))
    elif isinstance(value, list):
        for n, inner in enumerate(cast(list[object], value)):
            found.extend(_signature_paths(inner, f"{path}[{n}]"))
    elif isinstance(value, str) and SIGNATURE in value:
        found.append(path)
    return found


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: OpenAIConnector(api_key="k", model="m"), id="openai"),
        pytest.param(lambda: OllamaConnector(model="m"), id="ollama"),
        pytest.param(lambda: AnthropicConnector(api_key="k", model="m"), id="anthropic"),
    ],
)
def test_a_gemini_signature_never_reaches_another_provider(make: Any) -> None:
    """A clone switched off Gemini replays a history that holds Gemini's signatures; no
    other provider's request carries one, under any spelling or at any depth (#2234).

    Killed by: src/uclone_x/llm/connectors/openai.py :: "id": tc.id,
    Becomes: "id": tc.id, "thoughtSignature": tc.signature,
    """
    payload = make()._build_payload(_signed_history())  # pyright: ignore[reportPrivateUsage]
    assert _signature_paths(payload) == []


def test_the_signature_probe_finds_the_one_gemini_sends() -> None:
    """The probe above is not vacuous: on Gemini's own request it finds the signature."""
    payload = GeminiConnector(model="m", api_key="k")._build_payload(  # pyright: ignore[reportPrivateUsage]
        _signed_history()
    )
    assert _signature_paths(payload) != []
