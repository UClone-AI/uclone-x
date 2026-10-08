"""Image parts on `ChatMessage`: the seam every consumer of a message is typed against (#2107).

A picture travels beside a message's text (`ChatMessage.images`), never inside it. Its
bytes are kept once per session, as a context body named by their digest -- the way a
long tool result's full text is (#1848) -- and are never part of a message's logged body,
its saved record, or its digest. These tests hold the seam: the model, the token count,
redaction, the session log and a reload.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.prompt_assembler import restored_anchor_provenance
from uclone_x.agent.request_record import LogReader
from uclone_x.agent.session import SessionStore
from uclone_x.agent.session_lifecycle import _LiveSession  # pyright: ignore[reportPrivateUsage]
from uclone_x.core.provenance import Provenance
from uclone_x.core.session_log import SessionLogKind, logged_message
from uclone_x.core.session_state import redact_message
from uclone_x.llm.compactor import ContextCompactor, estimate_message_tokens
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    IMAGE_NOT_SEEN_NOTE,
    IMAGE_TOKEN_ESTIMATE,
    ChatMessage,
    ImagePart,
    LLMRequest,
    MessageRole,
    ModelResponse,
    ToolCallRequest,
)
from uclone_x.llm.protocols import request_for_model
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry

#: Not a real picture: the seam carries bytes, it does not decode them.
_RAW = b"\xff\xd8\xff\xe0 a screenshot's bytes \xff\xd9"
_B64 = base64.b64encode(_RAW).decode("ascii")


def _image() -> ImagePart:
    return ImagePart.from_bytes(_RAW, "image/jpeg", width=640, height=400)


# ======================================================================================
# The model
# ======================================================================================


def test_the_bytes_are_never_dumped_and_the_digest_names_them() -> None:
    """Killed by: src/uclone_x/llm/models.py :: exclude=True,
    Becomes: exclude=False,
    """
    message = ChatMessage(
        role=MessageRole.TOOL, content="ok", tool_call_id="c1", images=(_image(),)
    )
    dumped = message.model_dump_json()
    assert _B64 not in dumped
    (part,) = json.loads(dumped)["images"]
    assert part["digest"] == _image().digest
    assert ChatMessage.model_validate_json(dumped).images[0].data is None
    # A copy in memory keeps them: that is what a request is built from.
    assert message.model_copy().images[0].data == _B64


def test_a_message_without_images_dumps_as_before() -> None:
    message = ChatMessage(role=MessageRole.USER, content="hi")
    assert "images" not in message.model_dump()


def test_bytes_that_do_not_match_their_digest_are_refused() -> None:
    """Killed by: src/uclone_x/llm/models.py :: if self.data is not None and image_digest(self.data) != self.digest:
    Becomes: if False:
    """
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/jpeg", digest=_image().digest, data=_B64 + "AAAA")


def test_an_assistant_message_cannot_carry_an_image() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role=MessageRole.ASSISTANT, content="x", images=(_image(),))


# ======================================================================================
# Consumers of a message
# ======================================================================================


def test_an_image_counts_toward_the_token_estimate() -> None:
    """Killed by: src/uclone_x/llm/compactor.py :: total += 4 + IMAGE_TOKEN_ESTIMATE * len(msg.images)
    Becomes: total += 4
    """
    plain = ChatMessage(role=MessageRole.TOOL, content="ok", tool_call_id="c1")
    pictured = plain.model_copy(update={"images": (_image(), _image())})
    assert estimate_message_tokens([pictured]) - estimate_message_tokens([plain]) == (
        2 * IMAGE_TOKEN_ESTIMATE
    )


def test_redacting_a_message_keeps_its_image() -> None:
    """Killed by: src/uclone_x/core/session_state.py :: images=message.images,
    Becomes: images=(),
    """
    message = ChatMessage(
        role=MessageRole.TOOL,
        content="token sk-ant-api03-" + "a" * 40,
        tool_call_id="c1",
        images=(_image(),),
    )
    clean = redact_message(message)
    assert clean.content != message.content
    assert clean.images == message.images
    assert clean.images[0].data == _B64


def test_the_logged_body_holds_the_digest_not_the_bytes() -> None:
    message = ChatMessage(
        role=MessageRole.TOOL, content="ok", tool_call_id="c1", images=(_image(),)
    )
    item = logged_message(message)
    assert _B64 not in item.body
    assert _image().digest in item.body
    # The same message with or without its bytes in memory is the same entry.
    bare = ChatMessage.model_validate_json(message.model_dump_json())
    assert logged_message(bare).digest == item.digest
    assert item.images == ((_image().digest, _B64),)


# ======================================================================================
# A tool's picture through a session, and back after a reload
# ======================================================================================


class _SeeingLLM(MockLLMConnector):
    """Calls `look` on its first request, then answers; says whether it takes images."""

    def __init__(self, calls: Sequence[str], *, sees: bool = True) -> None:
        super().__init__(default_response="done")
        self._calls = list(calls)
        self.sees = sees
        self.asked = 0
        self.requests: list[LLMRequest] = []

    async def accepts_images(self, model: str | None = None) -> bool:
        self.asked += 1
        return self.sees

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        self._tool_calls = (
            [ToolCallRequest(id=f"c{len(self.requests)}", name=self._calls.pop(0), arguments={})]
            if self._calls
            else []
        )
        return await super().generate(request)


class _PictureTool(LocalTool):
    returns_images: ClassVar[bool] = True


def _tools(seen: list[bool]) -> list[LocalTool]:
    async def look(params: dict[str, Any], context: ToolContext) -> ToolResult:
        seen.append(context.accepts_images)
        return ToolResult(
            success=True,
            output="A screenshot of the page.",
            images=(_image(),) if context.accepts_images else (),
            provenance=Provenance.primary(provider="local.test", model="look"),
        )

    async def plain(params: dict[str, Any], context: ToolContext) -> ToolResult:
        seen.append(context.accepts_images)
        return ToolResult(
            success=True,
            output="text",
            provenance=Provenance.primary(provider="local.test", model="plain"),
        )

    return [
        _PictureTool("look", "Looks.", handler=look, writes_files=False),
        LocalTool("plain", "Reads.", handler=plain, writes_files=False),
    ]


def _agent(
    tmp_path: Path, store: SessionStore, llm: MockLLMConnector, seen: list[bool]
) -> BaseAgent:
    registry = ToolRegistry()
    for tool in _tools(seen):
        registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="seer",
            name="Seer",
            workspace_dir=tmp_path,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm,
        tools=registry,
        store=store,
    )


def _tool_images(request: LLMRequest) -> list[ImagePart]:
    return [part for m in request.messages if m.role == MessageRole.TOOL for part in m.images]


@pytest.mark.asyncio
async def test_a_tool_picture_reaches_the_next_request_with_its_bytes(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/tool_execution.py :: images=res.images if res.success else (),
    Becomes: images=(),
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    llm = _SeeingLLM(["look"])
    await _agent(tmp_path, store, llm, seen).execute_turn("look at it")

    assert seen == [True]
    (part,) = _tool_images(llm.requests[-1])
    assert part.data == _B64


@pytest.mark.asyncio
async def test_a_tool_that_returns_no_pictures_does_not_ask_the_provider(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: tool_returns_images(self._tool_invoker.resolve(tc.name)) for tc in tool_calls
    Becomes: True for tc in tool_calls
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    llm = _SeeingLLM(["plain"])
    await _agent(tmp_path, store, llm, seen).execute_turn("read it")
    assert seen == [False]
    assert llm.asked == 0


@pytest.mark.asyncio
async def test_a_model_that_cannot_see_gets_no_picture(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: accepts_images=await self._model_accepts_images(tool_calls),
    Becomes: accepts_images=True,
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    llm = _SeeingLLM(["look"], sees=False)
    await _agent(tmp_path, store, llm, seen).execute_turn("look at it")
    assert seen == [False]
    assert _tool_images(llm.requests[-1]) == []


@pytest.mark.asyncio
async def test_the_bytes_are_kept_once_outside_the_record_and_come_back_on_reload(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: self.pending_bodies[digest] = data
    Becomes: pass
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    agent = _agent(tmp_path, store, _SeeingLLM(["look"]), seen)
    await agent.execute_turn("look at it")
    agent.persist_session()
    sid = agent.session_id

    record = store.session_path(sid).read_text(encoding="utf-8")
    assert _B64 not in record
    assert store.load_context_body(sid, _image().digest) == _B64
    body_files = [p.read_text(encoding="utf-8") for p in store.context_body_dir(sid).iterdir()]
    assert sum(_B64 in text for text in body_files) == 1

    later = _SeeingLLM(["plain"])
    fresh = _agent(tmp_path, store, later, seen)
    fresh.hydrate_session(sid)
    (part,) = [p for m in fresh.history for p in m.images]
    assert part.data == _B64
    await fresh.execute_turn("and now?")
    assert [p.data for p in _tool_images(later.requests[0])] == [_B64]


@pytest.mark.asyncio
async def test_a_lost_picture_is_sent_without_its_bytes_not_refused(tmp_path: Path) -> None:
    """A picture whose body is gone or altered reloads as unavailable; the turn goes on.

    Killed by: src/uclone_x/core/session_log.py :: if data is None or image_digest(data) != part.digest:
    Becomes: if data is None:
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    agent = _agent(tmp_path, store, _SeeingLLM(["look"]), seen)
    await agent.execute_turn("look at it")
    agent.persist_session()
    sid = agent.session_id
    (store.context_body_dir(sid) / _image().digest).write_text("altered", encoding="utf-8")

    later = _SeeingLLM([])
    fresh = _agent(tmp_path, store, later, seen)
    fresh.hydrate_session(sid)
    await fresh.execute_turn("and now?")
    (part,) = _tool_images(later.requests[0])
    assert part.data is None
    assert part.digest == _image().digest


@pytest.mark.asyncio
async def test_session_store_load_restores_image_bytes_onto_reconstructed_messages(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/session.py :: with_image_data(message, partial(self.load_context_body, session_id))
    Becomes: message
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    agent = _agent(tmp_path, store, _SeeingLLM(["look"]), seen)
    await agent.execute_turn("look at it")
    agent.persist_session()
    sid = agent.session_id

    loaded = store.load(sid)
    assert loaded is not None
    (part,) = [p for m in loaded.messages if m.role == MessageRole.TOOL for p in m.images]
    assert part.data == _B64


@pytest.mark.asyncio
async def test_live_session_from_state_restores_image_bytes_when_state_messages_lack_bytes(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: self.decoded[item.digest] = with_image_data(message, self._load_any)
    Becomes: self.decoded[item.digest] = message
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    agent = _agent(tmp_path, store, _SeeingLLM(["look"]), seen)
    await agent.execute_turn("look at it")
    agent.persist_session()
    sid = agent.session_id

    loaded = store.load(sid)
    assert loaded is not None
    stripped_messages = tuple(
        m.model_copy(
            update={"images": tuple(p.model_copy(update={"data": None}) for p in m.images)}
        )
        for m in loaded.messages
    )
    state_without_bytes = loaded.model_copy(update={"messages": stripped_messages})
    live = _LiveSession.from_state(
        state_without_bytes,
        anchor_provenance=restored_anchor_provenance(loaded.anchor_provenance),
        load_body=lambda digest: store.load_context_body(sid, digest),
    )
    (part,) = [p for m in live.messages if m.role == MessageRole.TOOL for p in m.images]
    assert part.data == _B64


@pytest.mark.asyncio
async def test_log_reader_message_of_restores_image_bytes(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/request_record.py :: self._decoded[digest] = with_image_data(message, self._load)
    Becomes: self._decoded[digest] = message
    """
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    agent = _agent(tmp_path, store, _SeeingLLM(["look"]), seen)
    await agent.execute_turn("look at it")
    agent.persist_session()
    sid = agent.session_id
    state = store.load(sid)
    assert state is not None

    reader = LogReader(
        state.session_log,
        lambda digest: store.load_context_body(sid, digest),
        purpose="rebuild",
    )
    (tool_entry,) = [e for e in state.session_log if e.kind == SessionLogKind.TOOL_RESULT]
    restored = reader.message_of(tool_entry.id)
    assert restored.images[0].data == _B64


@pytest.mark.asyncio
async def test_a_deleted_image_body_is_sent_without_its_bytes_not_refused(
    tmp_path: Path,
) -> None:
    store = SessionStore(storage_dir=tmp_path / "sessions")
    seen: list[bool] = []
    agent = _agent(tmp_path, store, _SeeingLLM(["look"]), seen)
    await agent.execute_turn("look at it")
    agent.persist_session()
    sid = agent.session_id
    body_file = store.context_body_dir(sid) / _image().digest
    assert body_file.is_file()
    body_file.unlink()

    loaded = store.load(sid)
    assert loaded is not None
    (tool_msg,) = [m for m in loaded.messages if m.role == MessageRole.TOOL]
    assert tool_msg.images[0].data is None

    later = _SeeingLLM([])
    fresh = _agent(tmp_path, store, later, seen)
    fresh.hydrate_session(sid)
    await fresh.execute_turn("and now?")
    (part,) = _tool_images(later.requests[0])
    assert part.data is None
    assert part.digest == _image().digest


# ======================================================================================
# A conversation that moves to a model that cannot see (#2123)
# ======================================================================================


class _PerModelLLM(_SeeingLLM):
    """Sees only with the models in `seeing`, as a catalogue entry says per model."""

    def __init__(self, calls: Sequence[str], *, seeing: frozenset[str]) -> None:
        super().__init__(calls)
        self.seeing = seeing

    async def accepts_images(self, model: str | None = None) -> bool:
        self.asked += 1
        return model in self.seeing


def _all_images(request: LLMRequest) -> list[ImagePart]:
    return [part for m in request.messages for part in m.images]


def _look_result(messages: Sequence[ChatMessage]) -> ChatMessage:
    (message,) = [m for m in messages if m.role == MessageRole.TOOL and m.name == "look"]
    return message


async def _switched_to_blind(tmp_path: Path) -> tuple[BaseAgent, _PerModelLLM, SessionStore]:
    """A turn whose `look` gave a picture, then a turn on a model that cannot see."""
    store = SessionStore(storage_dir=tmp_path / "sessions")
    llm = _PerModelLLM(["look"], seeing=frozenset({"mock-model"}))
    agent = _agent(tmp_path, store, llm, [])
    await agent.execute_turn("look at it")
    assert [p.data for p in _tool_images(llm.requests[-1])] == [_B64]
    agent.hot_reload_llm(model_name="blind-model")
    await agent.execute_turn("and now?")
    return agent, llm, store


@pytest.mark.asyncio
async def test_a_model_switched_to_that_cannot_see_is_sent_a_note_not_the_picture(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/turn_executor.py :: req = await request_for_model(llm, req)
    Becomes: req = req
    """
    _, llm, _ = await _switched_to_blind(tmp_path)
    blind = llm.requests[-1]
    assert blind.model == "blind-model"
    assert _all_images(blind) == []
    assert (
        _look_result(blind.messages).content
        == f"A screenshot of the page.\n\n{IMAGE_NOT_SEEN_NOTE}"
    )


@pytest.mark.asyncio
async def test_switching_back_to_a_model_that_sees_sends_the_picture_again(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/llm/protocols.py :: request.model or None
    Becomes: None or None
    """
    agent, llm, _ = await _switched_to_blind(tmp_path)
    agent.hot_reload_llm(model_name="mock-model")
    await agent.execute_turn("look again")
    seeing = llm.requests[-1]
    assert [p.data for p in _all_images(seeing)] == [_B64]
    assert IMAGE_NOT_SEEN_NOTE not in (_look_result(seeing.messages).content or "")


@pytest.mark.asyncio
async def test_the_stored_session_keeps_its_picture_after_a_blind_turn(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/llm/models.py :: messages.append(message.model_copy(update={"content": "\n\n".join(text), "images": ()}))
    Becomes: messages.append(message)
    """
    agent, llm, store = await _switched_to_blind(tmp_path)
    # Killed in the other direction too: the note is what the request carries.
    assert _all_images(llm.requests[-1]) == []

    kept = _look_result(agent.history)
    assert [p.data for p in kept.images] == [_B64]
    assert kept.content == "A screenshot of the page."

    agent.persist_session()
    sid = agent.session_id
    assert IMAGE_NOT_SEEN_NOTE not in store.session_path(sid).read_text(encoding="utf-8")
    fresh = _agent(tmp_path, store, _SeeingLLM([]), [])
    fresh.hydrate_session(sid)
    reloaded = _look_result(fresh.history)
    assert [p.data for p in reloaded.images] == [_B64]
    assert reloaded.content == "A screenshot of the page."


@pytest.mark.asyncio
async def test_a_summarizer_that_cannot_see_is_sent_a_note_not_the_picture() -> None:
    """Killed by: src/uclone_x/llm/compactor.py :: summary_request = await request_for_model(self.summarizer, summary_request)
    Becomes: summary_request = summary_request
    """
    summarizer = _SeeingLLM([], sees=False)
    compactor = ContextCompactor(summarizer=summarizer)
    await compactor._generate_llm_summary(  # pyright: ignore[reportPrivateUsage]
        [ChatMessage(role=MessageRole.USER, images=(_image(),))]
    )
    (request,) = summarizer.requests
    assert _all_images(request) == []
    assert IMAGE_NOT_SEEN_NOTE in [m.content for m in request.messages]


class _ProviderWithoutProbe:
    """A connector or provider that does not implement ImageInputProbe."""


@pytest.mark.asyncio
async def test_a_provider_without_image_input_probe_strips_images() -> None:
    """Killed by: src/uclone_x/llm/protocols.py :: if isinstance(provider, ImageInputProbe) and await provider.accepts_images(
    Becomes: if not isinstance(provider, ImageInputProbe) or await provider.accepts_images(
    """
    request = LLMRequest(
        model="blind",
        messages=(ChatMessage(role=MessageRole.USER, content="look", images=(_image(),)),),
    )
    result = await request_for_model(_ProviderWithoutProbe(), request)
    assert _all_images(result) == []
    assert IMAGE_NOT_SEEN_NOTE in (result.messages[0].content or "")
