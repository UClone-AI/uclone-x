"""K + act as a product tools module, ``k_act`` (#2188, owner ruling 2026-10-04).

What the issue requires of the module, each pinned here:

1. no tool is declared to the provider; the base tools are described in the system text,
   and a tool host binding adds is described once, on the user message that bound it;
2. the host reads the reply's ``<tool_call>`` text, a block written over several lines
   included, and runs it through the same dispatch: bind on call, did-you-mean, the
   persona's range;
3. the reply and the tool result stay in history, so every request of an epoch extends
   the one before it, and the log-only rebuild is byte-identical to what was sent;
4. a reply whose call cannot be read is refused in plain words, with no fallback to
   native calling and nothing run; a reply with no call and no text is asked once more;
5. the default stays ``native`` (the pinned default bytes are `test_tools_module.py`'s);
6. the stream a person watches never shows ``<tool_call>`` text.

The scripted connector fails the test if a turn asks for more replies than it holds, and
the embedder answers by a fixed table for as many calls as it gets.
"""

from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.k_act import (
    TEXT_CALL_ACT,
    TEXT_CALL_INTRO,
    TEXT_CALL_UNREADABLE_MESSAGE,
    TEXT_TOOLS_ADDED_HEADER,
    TextCallStreamFilter,
    TextToolCallUnreadableError,
    text_call_messages,
    text_tool_calls,
    visible_reply_text,
)
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.request_record import rebuild_requests
from uclone_x.agent.session import SessionStore
from uclone_x.agent.tools_module import (
    TOOLS_MODULE_PROVIDER_MESSAGE,
    UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE,
    ToolsModuleUnsupportedError,
    select_tools_module,
)
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.reader import read_session_log
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry
from uclone_x.tools.tool_binder import ToolBinder

_AXES = ("alpha_image", "beta_mail", "gamma_calc")
_SESSION = "sess_k_act"


def _toward(name: str) -> tuple[float, ...]:
    vector = [0.9 if axis == name else 0.0 for axis in _AXES]
    return (*vector, math.sqrt(1.0 - 0.81))


_QUERIES = {
    "draw it": _toward("alpha_image"),
    "now mail it": _toward("beta_mail"),
    "and once more": _toward("alpha_image"),
}


class _Embedder:
    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimensions(self) -> int:
        return len(_AXES) + 1

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        out: list[tuple[float, ...]] = []
        for text in texts:
            if text in _QUERIES:
                out.append(_QUERIES[text])
            else:
                name = text.split(":", 1)[0]
                out.append((*(1.0 if axis == name else 0.0 for axis in _AXES), 0.0))
        return tuple(out)


class _Params(BaseModel):
    text: str = Field(default="")


#: Every tool run, in order, by name: the effect a dispatch test reads.
_RAN: list[str] = []


def _tool(tool_name: str) -> BaseTool[_Params]:
    class _Named(BaseTool[_Params]):
        name = tool_name
        description = f"The {tool_name} tool."

        def run(self, params: _Params, context: ToolContext) -> str:
            _RAN.append(tool_name)
            return f"{tool_name} ran on {params.text}"

    return _Named()


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("file_read", *_AXES):
        registry.register(_tool(name))
    return registry


def _ref() -> ServiceRef:
    return ServiceRef(provider="mock", model="mock-model")


def _response(content: str, calls: tuple[ToolCallRequest, ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=calls,
        usage=TokenUsage(provider="mock", model="mock-model", input_tokens=1, output_tokens=1),
        finish_reason=FinishReason.STOP,
        model_name="mock-model",
        provenance=Provenance(path=ExecutionPath.PRIMARY, requested=_ref(), served_by=_ref()),
    )


class _Scripted(MockLLMConnector):
    """Answers each request with the next text of the script, and keeps every request.

    An item is the reply's text, or ``(text, calls)`` for a provider that parsed calls.
    """

    def __init__(self, script: list[str | tuple[str, tuple[ToolCallRequest, ...]]]) -> None:
        super().__init__()
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    def _next(self, request: LLMRequest) -> tuple[str, tuple[ToolCallRequest, ...]]:
        self.requests.append(request)
        assert self.script, "the turn asked for more replies than the script holds"
        step = self.script.pop(0)
        return step if isinstance(step, tuple) else (step, ())

    async def generate(self, request: LLMRequest) -> ModelResponse:
        content, calls = self._next(request)
        return _response(content, calls)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        content, _ = self._next(request)
        # Cut every few characters, so a tag is split across chunks.
        for start in range(0, len(content), 5):
            yield StreamChunk(delta_content=content[start : start + 5])
        yield StreamChunk(
            usage=TokenUsage(provider="mock", model="mock-model", input_tokens=1, output_tokens=1),
            finish_reason=FinishReason.STOP,
            model="mock-model",
        )


def _block(name: str, text: str = "x") -> str:
    return '<tool_call>{"name": "' + name + '", "arguments": {"text": "' + text + '"}}</tool_call>'


def _agent(
    wire: _Scripted,
    store: SessionStore,
    module: str | None = "k_act",
    *,
    allowed: tuple[str, ...] = (),
    max_steps: int = 10,
) -> BaseAgent:
    return BaseAgent(
        config=AgentConfig(
            agent_id="texted",
            name="T",
            llm_config=AgentLLMConfig(model_name="mock-model"),
            tools_module=module,
            allowed_tools=allowed,
            max_steps=max_steps,
        ),
        llm=wire,
        tools=_registry(),
        context=AgentContext(session_id=_SESSION, agent_id="texted"),
        tool_binder=ToolBinder(_Embedder()),
        store=store,
    )


@pytest.fixture(autouse=True)
def clear_ran() -> None:
    _RAN.clear()


def _system(request: LLMRequest) -> str:
    first = request.messages[0]
    assert first.role is MessageRole.SYSTEM
    return first.content or ""


def _dump(request: LLMRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def _events(store: SessionStore) -> list[dict[str, Any]]:
    path = store.event_log_path(_SESSION)
    assert path is not None
    return [dict(event) for event in read_session_log(path)]


# --------------------------------------------------------------------------------------
# Selection and defaults
# --------------------------------------------------------------------------------------


def test_k_act_is_selectable_and_nothing_chosen_stays_native() -> None:
    """Killed by: src/uclone_x/agent/tools_module.py :: TOOLS_MODULES: Final[tuple[ToolsModuleName, ...]] = ("native", "pinned", "bound", "k_act")
    Becomes: TOOLS_MODULES: Final[tuple[ToolsModuleName, ...]] = ("native", "pinned", "bound")
    """
    assert select_tools_module("k_act", "ollama") == "k_act"
    for provider in ("ollama", "anthropic", "openai", "mock", ""):
        assert select_tools_module(None, provider) == "native"
    assert "k_act" in UNKNOWN_TOOLS_MODULE_SETTING_MESSAGE


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini", "vllm", "", "Gemini "])
def test_k_act_is_refused_on_a_provider_it_cannot_run_on(provider: str) -> None:
    """A hosted API refuses a tool result after a reply that declared no call, so the
    choice is refused when the seat is built, not at the first tool step (#2200 review).

    Killed by: src/uclone_x/agent/tools_module.py :: if known == "k_act" and provider.strip().lower() not in K_ACT_PROVIDERS:
    Becomes: if False:
    """
    with pytest.raises(ToolsModuleUnsupportedError) as caught:
        select_tools_module("k_act", provider)
    assert str(caught.value) == TOOLS_MODULE_PROVIDER_MESSAGE
    assert caught.value.provider == provider
    for fine in ("ollama", " Ollama ", "mock"):
        assert select_tools_module("k_act", fine) == "k_act"
    # The other modules run anywhere.
    assert select_tools_module("bound", provider) == "bound"


def test_the_provider_refusal_is_plain_copy() -> None:
    message = TOOLS_MODULE_PROVIDER_MESSAGE
    assert message[0].isupper() and message.endswith(".")
    assert "Ollama" in message and "choose another way" in message
    assert len(message.split()) <= 40


@pytest.mark.parametrize(
    "internal",
    ["k_act", "_", "tools_module", "Error", "provider", "openai", "anthropic", "vllm", "<", ".py"],
)
def test_the_provider_refusal_names_no_internals(internal: str) -> None:
    assert internal not in TOOLS_MODULE_PROVIDER_MESSAGE


class _Hosted(_Scripted):
    """The scripted connector, reporting itself as a hosted API."""

    @property
    def provider_name(self) -> str:
        return "openai"


def test_a_seat_on_a_hosted_provider_is_not_built_under_k_act(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: config.tools_module, str(getattr(llm, "provider_name", "") or "")
    Becomes: config.tools_module, "ollama"
    """
    with pytest.raises(ToolsModuleUnsupportedError):
        _agent(_Hosted([]), SessionStore(tmp_path))


#: What a person must never see when a head refuses the setting.
_INTERNALS = ("Traceback", ".py", "Error", "tools_module", "k_act", "uclone_x", "line ")


def _clone_on_k_act(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clone set to ``k_act``, on a host whose provider ``k_act`` does not run on.

    The heads run on the scripted ``mock`` provider, so the provider list is narrowed to
    Ollama alone: what is tested is each head's handling of the refusal.
    """
    from uclone_x.core.agent_home import create_clone

    monkeypatch.setattr("uclone_x.agent.tools_module.K_ACT_PROVIDERS", frozenset({"ollama"}))
    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("UCLONE_ROOM_DIR", str(tmp_path / "rooms"))
    create_clone(
        "texted",
        "handle: texted\nname: texted\nrole: Tester\n"
        "system_prompt: You test.\ntools_module: k_act\n",
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "texted", "--provider", "mock", "--prompt", "hi"],
        [
            "loop",
            "run",
            "tick",
            "--interval",
            "1h",
            "--max-runs",
            "1",
            "--provider",
            "mock",
            "--agent",
            "texted",
        ],
        ["a2a", "serve", "--agent-id", "texted"],
        ["acp", "serve", "--persona", "texted"],
    ],
    ids=["run", "loop", "a2a", "acp"],
)
def test_each_cli_head_refuses_k_act_on_its_provider_in_the_plain_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from typer.testing import CliRunner

    from uclone_x.cli import main

    _clone_on_k_act(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    serving = MagicMock()
    monkeypatch.setattr("uvicorn.run", serving)
    acp_server = MagicMock()
    acp_server.return_value.run_stdio = AsyncMock()
    monkeypatch.setattr("uclone_x.cli.commands.acp.one_seat_acp_server", acp_server)

    result = CliRunner().invoke(main.app, argv)

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert TOOLS_MODULE_PROVIDER_MESSAGE in " ".join(result.output.split())
    for internal in _INTERNALS:
        assert internal not in result.output, internal
    assert not serving.called
    assert not acp_server.called


def test_a_room_seat_with_k_act_on_its_provider_shows_the_plain_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio

    from uclone_x.room.models import ParticipantKind
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack, reader_facing_reason

    _clone_on_k_act(monkeypatch, tmp_path)
    mgr = AgentSessionManager(
        storage_dir=tmp_path / "sessions",
        llm=MockLLMConnector(),
        workspace_dir=tmp_path / "workspace",
    )
    stack = RoomStack(mgr)
    state = stack.service.create(title="Modules")
    stack.service.add_participant(
        state.room_id, participant_id="texted", kind=ParticipantKind.AGENT
    )
    seated = stack.store.load(state.room_id)
    assert seated is not None
    resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
    participant = next(p for p in seated.participants if p.id == "texted")

    with pytest.raises(ToolsModuleUnsupportedError) as caught:
        asyncio.run(resolver.resolve(participant))

    reason = reader_facing_reason(caught.value)
    assert reason == TOOLS_MODULE_PROVIDER_MESSAGE
    for internal in _INTERNALS:
        assert internal not in reason, internal


# --------------------------------------------------------------------------------------
# Reading a reply
# --------------------------------------------------------------------------------------


def test_a_reply_is_read_block_by_block_and_a_block_over_several_lines_is_one_call() -> None:
    """granite3.3:8b's T02 shape (#2198): the object pretty-printed inside the block."""
    pretty = (
        "<tool_call>\n{\n"
        '  "name": "alpha_image",\n'
        '  "arguments": {\n    "text": "a cat"\n  }\n}\n</tool_call>'
    )
    calls = text_tool_calls(f"Drawing it. {pretty}\n{_block('beta_mail')}")
    assert [(c.id, c.name, dict(c.arguments)) for c in calls] == [
        ("call_0", "alpha_image", {"text": "a cat"}),
        ("call_1", "beta_mail", {"text": "x"}),
    ]
    assert text_tool_calls("No call here, just words.") == ()
    as_text = '<tool_call>{"name": "beta_mail", "arguments": "{\\"text\\": \\"y\\"}"}</tool_call>'
    assert dict(text_tool_calls(as_text)[0].arguments) == {"text": "y"}


@pytest.mark.parametrize(
    "reply",
    [
        "<tool_call>{not json at all}</tool_call>",
        "<tool_call></tool_call>",
        '<tool_call>{"arguments": {}}</tool_call>',
        '<tool_call>{"name": "alpha_image", "arguments": [1, 2]}</tool_call>',
        f'{_block("alpha_image")} and <tool_call>{{"name": "beta_mail", "arguments": </tool_call>',
        '<tool_call>{"name": "alpha_image", "arguments": {"text": "cut off',
    ],
    ids=["not-json", "empty", "no-name", "list-args", "one-of-two-broken", "unclosed"],
)
def test_a_block_that_does_not_read_as_a_call_is_refused_whole(reply: str) -> None:
    """Killed by: src/uclone_x/agent/k_act.py :: if not isinstance(name, str) or not name.strip() or arguments is None:
    Becomes: if False:
    """
    with pytest.raises(TextToolCallUnreadableError):
        text_tool_calls(reply)


def test_the_visible_reply_and_the_stream_never_show_a_block() -> None:
    reply = f"Let me look. {_block('alpha_image')} Done soon."
    assert visible_reply_text(reply) == "Let me look.  Done soon."
    filt = TextCallStreamFilter()
    shown = "".join(filt.feed(reply[i : i + 3]) for i in range(0, len(reply), 3)) + filt.flush()
    assert shown == "Let me look.  Done soon."
    # Text that only looks like the start of a tag is shown once the stream ends.
    tail = TextCallStreamFilter()
    assert tail.feed("a < b and <to") == "a < b and "
    assert tail.flush() == "<to"
    # An unclosed block is never shown.
    cut = TextCallStreamFilter()
    assert cut.feed('ok <tool_call>{"name": "x"') == "ok "
    assert cut.flush() == ""


def test_a_reply_from_another_module_is_sent_as_text_under_k_act() -> None:
    """After a change of module, a native call in history has no block in its text."""
    native = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=None,
        tool_calls=(ToolCallRequest(id="call_0", name="alpha_image", arguments={"text": "x"}),),
    )
    written = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=_block("beta_mail"),
        tool_calls=(ToolCallRequest(id="call_0", name="beta_mail", arguments={"text": "x"}),),
    )
    sent = text_call_messages([native, written])
    assert [m.tool_calls for m in sent] == [(), ()]
    assert (
        sent[0].content
        == '<tool_call>{"name": "alpha_image", "arguments": {"text": "x"}}</tool_call>'
    )
    assert sent[1].content == _block("beta_mail")


# --------------------------------------------------------------------------------------
# A turn under k_act
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_is_declared_the_base_is_in_the_system_text_and_a_bound_tool_on_the_message(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/tool_invoker.py :: return () if self._tools_module == "k_act" else tuple(defs)
    Becomes: return tuple(defs)
    Killed by: src/uclone_x/agent/turn_executor.py :: turn_live.replace(last, announced, cause="text_tools_bound")
    Becomes: pass
    Killed by: src/uclone_x/agent/request_record.py :: messages = text_call_messages(messages)
    Becomes: pass
    """
    wire = _Scripted([_block("alpha_image", "a cat"), "Drawn."])
    agent = _agent(wire, SessionStore(tmp_path))
    result = await agent.execute_turn("draw it")

    assert result.error is None, result.error
    assert result.content == "Drawn."
    assert _RAN == ["alpha_image"]
    assert [r.tools for r in wire.requests] == [(), ()]
    system = _system(wire.requests[0])
    assert f"{TEXT_CALL_INTRO} {TEXT_CALL_ACT}" in system
    assert "- file_read: The file_read tool." in system
    assert "alpha_image" not in system  # a catalog tool, bound on the message instead
    user = wire.requests[0].messages[-1]
    assert user.role is MessageRole.USER
    assert user.content == (
        f"draw it\n\n{TEXT_TOOLS_ADDED_HEADER}\n- alpha_image: The alpha_image tool.\n"
        '  args schema: {"properties":{"text":{"default":"","type":"string"}},"type":"object"}'
    )
    # The second request repeats the reply as written, as text only, then the result.
    reply, result_msg = wire.requests[1].messages[-2:]
    assert (reply.role, reply.content, reply.tool_calls) == (
        MessageRole.ASSISTANT,
        _block("alpha_image", "a cat"),
        (),
    )
    assert result_msg.role is MessageRole.TOOL
    assert result_msg.content == "alpha_image ran on a cat"


@pytest.mark.asyncio
async def test_two_turns_and_a_restart_extend_one_epoch_and_rebuild_byte_for_byte(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: text_calls=tools_module == "k_act",
    Becomes: text_calls=False,
    """
    store = SessionStore(tmp_path)
    first = _Scripted([_block("alpha_image"), "drawn", _block("beta_mail"), "mailed"])
    agent = _agent(first, store)
    await agent.start()
    assert (await agent.execute_turn("draw it")).error is None
    assert (await agent.execute_turn("now mail it")).error is None
    agent.persist_session(_SESSION)

    second = _Scripted([_block("alpha_image"), "again"])
    restarted = _agent(second, store)
    assert restarted.hydrate_session(_SESSION) is not None
    await restarted.start()
    assert (await restarted.execute_turn("and once more")).error is None
    restarted.persist_session(_SESSION)

    sent = [*first.requests, *second.requests]
    assert _RAN == ["alpha_image", "beta_mail", "alpha_image"]
    # Append-only: each request's conversation starts with the whole of the one before.
    for before, after in zip(sent, sent[1:], strict=False):
        assert list(after.messages[: len(before.messages)]) == list(before.messages)
    # A tool announced once is not announced again after the restart.
    announced = [
        m.content.count(TEXT_TOOLS_ADDED_HEADER)
        for m in second.requests[-1].messages
        if m.role is MessageRole.USER and m.content
    ]
    assert announced == [1, 1, 0]

    state = store.load(_SESSION)
    assert state is not None
    assert len(state.context_epochs) == 1  # a restart that only extends opens none
    rebuilt = rebuild_requests(store, state, _events(store))
    assert len(rebuilt) == len(sent) == 6
    for number, (request, original) in enumerate(zip(rebuilt, sent, strict=True)):
        assert request.verified, number
        assert request.tools_module == "k_act", number
        assert _dump(request.request) == _dump(original.model_copy(update={"context_window": None}))
    assert {s.tools_module for s in state.context_snapshots} == {"k_act"}


@pytest.mark.asyncio
async def test_a_catalog_tool_named_in_text_is_bound_on_the_call_and_announced_on_its_result(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: self._announce_bound_on_call(tool_messages, step_executions, shown_tool_names)
    Becomes: pass
    """
    wire = _Scripted([_block("gamma_calc"), "Calculated."])
    agent = _agent(wire, SessionStore(tmp_path))
    result = await agent.execute_turn("draw it")

    assert result.error is None
    assert _RAN == ["gamma_calc"]
    tool_result = wire.requests[1].messages[-1]
    assert tool_result.role is MessageRole.TOOL
    assert tool_result.content == (
        f"gamma_calc ran on x\n\n{TEXT_TOOLS_ADDED_HEADER}\n- gamma_calc: The gamma_calc tool.\n"
        '  args schema: {"properties":{"text":{"default":"","type":"string"}},"type":"object"}'
    )


@pytest.mark.asyncio
async def test_an_unknown_name_gets_did_you_mean_and_a_name_outside_the_range_is_refused(
    tmp_path: Path,
) -> None:
    wire = _Scripted([f"{_block('alpha_imag')}{_block('gamma_calc')}", "Sorry."])
    agent = _agent(wire, SessionStore(tmp_path), allowed=("file_read", "alpha_image", "beta_mail"))
    result = await agent.execute_turn("draw it")

    assert result.error is None
    assert _RAN == []
    unknown, outside = wire.requests[1].messages[-2:]
    assert unknown.role is MessageRole.TOOL and "alpha_image" in (unknown.content or "")
    assert outside.role is MessageRole.TOOL and "gamma_calc ran" not in (outside.content or "")


@pytest.mark.asyncio
async def test_a_provider_that_parsed_the_call_itself_runs_it_once_from_the_text(
    tmp_path: Path,
) -> None:
    parsed = (ToolCallRequest(id="call_0", name="alpha_image", arguments={"text": "x"}),)
    wire = _Scripted([("", parsed), "Drawn."])
    agent = _agent(wire, SessionStore(tmp_path))
    assert (await agent.execute_turn("draw it")).error is None
    assert _RAN == ["alpha_image"]
    reply = wire.requests[1].messages[-2]
    assert (reply.content, reply.tool_calls) == (
        '<tool_call>{"name": "alpha_image", "arguments": {"text": "x"}}</tool_call>',
        (),
    )


# --------------------------------------------------------------------------------------
# A reply that cannot be read, and a reply with nothing in it
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unreadable_call_refuses_the_turn_plainly_runs_nothing_and_falls_back_to_nothing(
    tmp_path: Path,
) -> None:
    """Owner ruling 2026-10-04: no same-turn fallback to native; a plain sentence.

    Killed by: src/uclone_x/agent/turn_executor.py :: resp = text_call_response(resp)
    Becomes: pass
    Killed by: src/uclone_x/agent/turn_executor.py :: return "tool_call_unreadable", TEXT_CALL_UNREADABLE_MESSAGE, None
    Becomes: return stop_reason, str(exc), None
    """
    garbled = '<tool_call>{"name": "alpha_image", "arguments": {text: oops}}</tool_call>'
    wire = _Scripted([garbled])
    agent = _agent(wire, SessionStore(tmp_path))
    result = await agent.execute_turn("draw it")

    assert result.error == TEXT_CALL_UNREADABLE_MESSAGE
    assert result.stop_reason == "tool_call_unreadable"
    assert not result.is_completed
    assert _RAN == []
    assert len(wire.requests) == 1  # nothing asked again, natively or otherwise
    assert all(r.tools == () for r in wire.requests)
    assert not any(m.role is MessageRole.ASSISTANT for m in agent.history), (
        "the unread reply must not enter history"
    )


def test_the_unreadable_sentence_is_plain_copy() -> None:
    """Plain copy: whole sentences a person can act on, ending with what to do."""
    message = TEXT_CALL_UNREADABLE_MESSAGE
    assert message.endswith(".")
    assert message[0].isupper()
    assert "nothing was run" in message
    assert "again" in message
    assert len(message.split()) <= 40


@pytest.mark.parametrize(
    "internal",
    ["<", ">", "tool_call", "JSON", "json", "parse", "Error", "_", "k_act", "native", "{"],
)
def test_the_unreadable_sentence_names_no_internals(internal: str) -> None:
    assert internal not in TEXT_CALL_UNREADABLE_MESSAGE
    assert str(TextToolCallUnreadableError()) == TEXT_CALL_UNREADABLE_MESSAGE


@pytest.mark.asyncio
async def test_a_reply_with_no_call_and_no_text_is_asked_once_more_then_ends(
    tmp_path: Path,
) -> None:
    wire = _Scripted(["", "Here you go."])
    agent = _agent(wire, SessionStore(tmp_path))
    result = await agent.execute_turn("draw it")

    assert result.error is None
    assert result.content == "Here you go."
    assert _RAN == []
    assert "Your last reply was empty." in (wire.requests[1].messages[-1].content or "")
    assert all(r.tools == () for r in wire.requests)

    twice = _Scripted(["", ""])
    quiet = _agent(twice, SessionStore(tmp_path / "quiet"))
    ended = await quiet.execute_turn("draw it")
    assert len(twice.requests) == 2
    assert ended.content == ""
    assert ended.stop_reason == "model_stopped_after_nudge"


# --------------------------------------------------------------------------------------
# What a person watching the stream sees
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_answer_ending_in_a_cut_off_tag_is_shown_without_it(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: shown = self._tool_invoker.visible_reply(resp_content)
    Becomes: shown = resp_content
    Killed by: src/uclone_x/agent/k_act.py :: return "" if self._inside else _without_cut_tag(held)
    Becomes: return "" if self._inside else held
    """
    wire = _Scripted(["Here it is. <tool_call"])
    agent = _agent(wire, SessionStore(tmp_path))
    tokens: list[str] = []

    async def watch(event: str, data: dict[str, Any]) -> None:
        if event == "token":
            tokens.append(str(data.get("content") or ""))

    result = await agent.execute_turn("draw it", stream_callback=watch)

    assert result.error is None
    assert result.content == "Here it is."
    assert "<tool" not in "".join(tokens)
    assert _RAN == []


@pytest.mark.asyncio
async def test_a_turn_out_of_steps_returns_no_call_text(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: content=self._tool_invoker.visible_reply(resp_content),
    Becomes: content=resp_content,
    """
    wire = _Scripted([f"Trying. {_block('alpha_image')}", f"Again. {_block('alpha_image', 'y')}"])
    agent = _agent(wire, SessionStore(tmp_path), max_steps=2)
    result = await agent.execute_turn("draw it")

    assert result.stop_reason == "step_budget_exceeded"
    assert result.content == "Again."


@pytest.mark.asyncio
async def test_the_stream_shows_the_answer_and_never_the_call_text(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: stream_callback = TextCallStream(stream_callback)
    Becomes: pass
    """
    wire = _Scripted([f"One moment. {_block('alpha_image')}", "The picture is ready."])
    agent = _agent(wire, SessionStore(tmp_path))
    tokens: list[str] = []

    async def watch(event: str, data: dict[str, Any]) -> None:
        if event == "token":
            tokens.append(str(data.get("content") or ""))

    result = await agent.execute_turn("draw it", stream_callback=watch)

    assert result.content == "The picture is ready."
    streamed = "".join(tokens)
    assert "tool_call" not in streamed and "<" not in streamed and "alpha_image" not in streamed
    assert streamed == "One moment. The picture is ready."
    assert _RAN == ["alpha_image"]
