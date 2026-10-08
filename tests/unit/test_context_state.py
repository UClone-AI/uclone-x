"""The context state: per epoch, which log entry each request showed, in which form (#1443).

The four rules of the request-layering design, §5.8, each shown on the requests a real
turn sends and records:

1. within an epoch the context only appends;
2. an entry already shown in the epoch is sent as a back-reference, not again;
3. forms drop only at a compaction, at a turn boundary, which opens a new epoch;
4. a stub names the `tr_` handle `tool_result_read` reads back.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from tests.support.memory_seed import seed_fact
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks import BaseHook, HookAction, HookContext, HookDecision
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, AgentState
from uclone_x.agent.request_record import (
    EpochRecordError,
    RequestRecordError,
    rebuild_epoch_conversations,
    rebuild_requests,
)
from uclone_x.agent.session import SessionState, SessionStore, content_digest
from uclone_x.agent.session_lifecycle import (
    AnchorWriter,
    _LiveSession,  # pyright: ignore[reportPrivateUsage]
)
from uclone_x.agent.turn_executor import TurnExecutor
from uclone_x.core.context_state import (
    BACK_REFERENCE_MIN_CHARS,
    EPOCH_UNDECLARED,
    ContextEntry,
    ContextEpoch,
    ContextForm,
    advance,
    back_reference_text,
    compacted_entries,
    derive_compacted_forms,
    message_form,
    opening_entries,
    render_conversation,
    render_entries,
)
from uclone_x.core.provenance import Provenance
from uclone_x.core.session_log import (
    RETRIEVAL_TOOLS,
    SUBAGENT_TOOLS,
    SessionLogEntry,
    SessionLogKind,
    SessionLogProvenance,
    history_entry_ids,
    is_kept_text,
    logged_message,
    logged_text,
    new_entry,
    stored_result_entry,
)
from uclone_x.core.tool_results import (
    excerpt_tool_result,
    handle_in,
    result_handle,
    stored_result_stub,
)
from uclone_x.errors import FormTextMismatchError
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
    RenderedFrom,
    ToolCallRequest,
)
from uclone_x.log.reader import read_session_log
from uclone_x.memory.models import MemoryFact
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import QueryMemoryFactsTool
from uclone_x.tools.builtin.filesystem import FileSearchTool
from uclone_x.tools.builtin.subagent import SubagentDelegationTool
from uclone_x.tools.builtin.tool_results import ToolResultReadTool
from uclone_x.tools.builtin.web import WebSearchTool
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry

# ======================================================================================
# Helpers
# ======================================================================================


class _ScriptedLLM(MockLLMConnector):
    """Asks for the tool calls of step N on its N-th request, then answers; keeps requests."""

    def __init__(self, steps: Sequence[Sequence[ToolCallRequest]]) -> None:
        super().__init__(default_response="done")
        self.steps = [list(s) for s in steps]
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        self._tool_calls = self.steps.pop(0) if self.steps else []
        return await super().generate(request)


def _returning(name: str, output: str) -> LocalTool:
    async def handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output=output,
            provenance=Provenance.primary(provider="local.test", model=name),
        )

    return LocalTool(name, f"Returns a fixed {name} result.", handler=handler, writes_files=False)


def _call(call_id: str, name: str, **arguments: Any) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


def _agent(
    tmp_path: Path,
    llm: MockLLMConnector,
    tools: Sequence[Any],
    *,
    sid: str = "sess_1443",
    compaction_threshold_tokens: int | None = None,
    compactor: ContextCompactor | None = None,
    memory: CrossSessionMemory | None = None,
) -> tuple[BaseAgent, SessionStore]:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    store = SessionStore(tmp_path / "store")
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="ctx",
            name="Ctx",
            workspace_dir=tmp_path,
            llm_config=(
                AgentLLMConfig(model_name="mock-model")
                if compaction_threshold_tokens is None
                else AgentLLMConfig(
                    model_name="mock-model",
                    compaction_threshold_tokens=compaction_threshold_tokens,
                )
            ),
        ),
        llm=llm,
        tools=registry,
        store=store,
        compactor=compactor,
        memory=memory,
        context=AgentContext(session_id=sid, agent_id="ctx", current_state=AgentState.IDLE),
    )
    return agent, store


def _conversation(request: LLMRequest) -> list[ChatMessage]:
    """A request's messages without the system turn and the turn context around them."""
    return [m for m in request.messages if m.role is not MessageRole.SYSTEM or m.compaction_ledger]


def _tool(request: LLMRequest, call_id: str) -> ChatMessage:
    (message,) = [m for m in request.messages if m.tool_call_id == call_id]
    return message


def _request_contexts(store: SessionStore, sid: str) -> list[dict[str, Any]]:
    log = store.event_log_path(sid)
    assert log is not None
    return [dict(e) for e in read_session_log(log) if e.get("type") == "REQUEST_CONTEXT"]


def _entry(entry: str, form: ContextForm = ContextForm.FULL) -> ContextEntry:
    return ContextEntry(entry=entry, form=form)


BODY = "\n".join(f"row {i:04d} of the report" for i in range(200))  # 4,599 characters


# ======================================================================================
# Forms
# ======================================================================================


def test_a_form_is_the_one_recorded_where_it_was_decided_not_read_from_the_text() -> None:
    """A tool result's form is what was recorded on it where it was shortened (#1854).

    Text alone decides nothing: a result that begins exactly like a stub's or an
    excerpt's header, with no form recorded, is `full`.

    Killed by: src/uclone_x/core/context_state.py :: if message.role == MessageRole.TOOL and message.form is not None:
    Becomes: if False:
    """
    handle = "tr_0123456789abcdef"

    def tool(content: str, form: Literal["excerpt", "stub"] | None = None) -> ChatMessage:
        return ChatMessage(
            role=MessageRole.TOOL, content=content, tool_call_id="c", name="t", form=form
        )

    stub_text = stored_result_stub(handle, BODY, keep_chars=40)
    excerpt_text = excerpt_tool_result(BODY, handle, cap_bytes=1_000)
    assert message_form(tool(stub_text, "stub")) is ContextForm.STUB
    assert message_form(tool(excerpt_text, "excerpt")) is ContextForm.EXCERPT
    assert message_form(tool(stub_text)) is ContextForm.FULL
    assert message_form(tool(excerpt_text)) is ContextForm.FULL
    assert message_form(tool("[Tool Output Truncated (path=truncate, 9 chars total):\nx")) is (
        ContextForm.FULL
    )
    assert message_form(tool(BODY)) is ContextForm.FULL
    ledger = ChatMessage(role=MessageRole.SYSTEM, content="summary", compaction_ledger=True)
    assert message_form(ledger) is ContextForm.SUMMARY
    with pytest.raises(ValidationError):
        ChatMessage(role=MessageRole.USER, content="x", form="stub")


@pytest.mark.asyncio
async def test_a_file_that_begins_like_a_stub_header_is_recorded_full(tmp_path: Path) -> None:
    """A `file_read` whose first line is a stub's header -- a file quoting one -- is shown
    whole, so the context state records it `full` (#1854): the form is recorded where it
    is decided, and nothing here shortened it.

    Killed by: src/uclone_x/agent/base.py :: if handle is None:
    Becomes: if False:
    """
    quoted = stored_result_stub("tr_0123456789abcdef", BODY, keep_chars=40)
    llm = _ScriptedLLM([[_call("c1", "file_read")]])
    agent, store = _agent(tmp_path, llm, [_returning("file_read", quoted)])
    await agent.start()
    assert (await agent.execute_turn("read the notes")).is_completed
    agent.persist_session()

    assert _tool(llm.requests[-1], "c1").content == quoted
    state = store.load("sess_1443")
    assert state is not None
    (epoch,) = state.context_epochs
    results = [
        x for x in epoch.entries if state.session_log[int(x.entry[1:])].kind.value == "tool_result"
    ]
    assert [x.form for x in results] == [ContextForm.FULL]


def _assert_epochs_render_from_the_log(
    store: SessionStore, state: SessionState, requests: Sequence[LLMRequest]
) -> None:
    """Each epoch, rendered from the session log alone, is the conversation of a request
    that was sent -- in order, the last epoch the last request's (#1848)."""
    log = store.event_log_path(state.session_id)
    assert log is not None
    rebuilt = rebuild_requests(store, state, [dict(e) for e in read_session_log(log)])
    assert [r.request.messages for r in rebuilt] == [r.messages for r in requests]
    sent = [list(r.layers.conversation) for r in rebuilt if r.layers is not None]
    assert len(sent) == len(requests)
    rendered = rebuild_epoch_conversations(store, state)
    assert len(rendered) == len(state.context_epochs) >= 1
    after = -1
    for conversation in rendered:
        matches = [i for i, c in enumerate(sent) if i > after and c == conversation]
        assert matches, "an epoch's rendering is not a request that was sent"
        after = matches[-1]
    assert rendered[-1] == sent[-1]


# ======================================================================================
# Rule 1 -- within an epoch the context only appends
# ======================================================================================


def test_a_request_that_extends_the_epoch_grows_it_and_one_that_does_not_opens_another() -> None:
    """Killed by: src/uclone_x/core/context_state.py :: if shown[: len(current.entries)] == current.entries:
    Becomes: if True:
    """
    first = advance((), [_entry("e1")], turn=1, step=1)
    assert [(e.number, e.opened_by) for e in first] == [(0, ("start",))]

    grown = advance(first, [_entry("e1"), _entry("e2")], turn=1, step=2)
    assert len(grown) == 1 and [x.entry for x in grown[0].entries] == ["e1", "e2"]
    assert advance(grown, [_entry("e1"), _entry("e2")], turn=1, step=3) == grown

    rewritten = advance(grown, [_entry("e1"), _entry("e3")], turn=2, step=1)
    assert [(e.number, e.turn, e.step, e.opened_by) for e in rewritten] == [
        (0, 1, 1, ("start",)),
        (1, 2, 1, (EPOCH_UNDECLARED,)),
    ]
    assert rewritten[0] == grown[0]  # an epoch that closed is never rewritten
    declared = advance(grown, [_entry("e3")], turn=2, step=1, opened_by=["rollback"])
    assert declared[-1].opened_by == ("rollback",)


def test_a_form_change_inside_the_listed_prefix_is_a_new_epoch() -> None:
    """Killed by: src/uclone_x/core/context_state.py :: if shown[: len(current.entries)] == current.entries:
    Becomes: if [x.entry for x in shown[: len(current.entries)]] == [x.entry for x in current.entries]:
    """
    epochs = advance((), [_entry("e1", ContextForm.EXCERPT)], turn=1, step=1)
    after = advance(epochs, [_entry("e1", ContextForm.STUB)], turn=2, step=1)
    assert len(after) == 2


@pytest.mark.asyncio
async def test_every_request_of_a_turn_extends_the_one_before(tmp_path: Path) -> None:
    """Three steps, one turn: one epoch, and each request's conversation begins with the
    previous one's, byte for byte. The recorded delta says the same: nothing kept is
    re-recorded.

    Before #1443 a pass between steps shrank the first result once the step crossed the
    threshold, so the third request rewrote what the second had shown.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: session.record_shown(list(layers.shown), step=step)
    Becomes: pass
    """
    llm = _ScriptedLLM([[_call("c1", "small")], [_call("c2", "dump")]])
    agent, store = _agent(
        tmp_path,
        llm,
        [_returning("small", "o" * 2_000), _returning("dump", "y" * 6_000)],
        compaction_threshold_tokens=1_200,
    )
    await agent.start()
    assert (await agent.execute_turn("go")).is_completed
    agent.persist_session()

    assert len(llm.requests) == 3
    for before, after in zip(llm.requests, llm.requests[1:], strict=False):
        sent, then = _conversation(before), _conversation(after)
        assert then[: len(sent)] == sent
    assert _tool(llm.requests[2], "c1").content == "o" * 2_000

    contexts = _request_contexts(store, "sess_1443")
    for before, after in zip(contexts, contexts[1:], strict=False):
        assert after["kept_entry_count"] == before["kept_entry_count"] + len(
            before["appended_entries"]
        )
    state = store.load("sess_1443")
    assert state is not None
    (epoch,) = state.context_epochs
    assert epoch.opened_by == ("start",)
    assert [x.entry for x in epoch.entries] == [
        e.id for e in state.session_log[1 : 1 + len(epoch.entries)]
    ]


@pytest.mark.asyncio
async def test_a_rollback_opens_an_epoch_that_says_so(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: self.declare_new_epoch("rollback")
    Becomes: pass
    """
    llm = _ScriptedLLM([])
    agent, _store = _agent(tmp_path, llm, [])
    await agent.start()
    checkpoint = agent.checkpoint_turn()
    assert (await agent.execute_turn("first try")).is_completed
    agent.roll_back_turn(checkpoint, reason="test")
    assert (await agent.execute_turn("second try")).is_completed

    epochs = agent.get_session().context_epochs
    assert [e.opened_by for e in epochs] == [("start",), ("rollback",)]


# ======================================================================================
# Rule 2 -- an entry already shown is sent as a back-reference
# ======================================================================================


def test_a_repeated_result_refers_back_to_the_first() -> None:
    """Killed by: src/uclone_x/core/context_state.py :: earlier = first_with.get(content)
    Becomes: earlier = None
    """
    long = "z" * BACK_REFERENCE_MIN_CHARS
    messages = [
        ChatMessage(role=MessageRole.USER, content=long),
        ChatMessage(role=MessageRole.TOOL, content=long, tool_call_id="c1", name="read"),
        ChatMessage(role=MessageRole.TOOL, content="short", tool_call_id="c2", name="read"),
        ChatMessage(role=MessageRole.TOOL, content=long, tool_call_id="c3", name="read"),
        ChatMessage(role=MessageRole.TOOL, content="short", tool_call_id="c4", name="read"),
    ]
    rendered, refers = render_conversation(messages)
    assert refers == [None, None, None, 1, None]
    assert rendered[3].content == back_reference_text(messages[1])
    assert "`c1`" in (rendered[3].content or "")
    assert (rendered[3].tool_call_id, rendered[3].name) == ("c3", "read")
    # A user message is never replaced, and a short result is cheaper to repeat.
    assert [m.content for i, m in enumerate(rendered) if i != 3] == [
        m.content for i, m in enumerate(messages) if i != 3
    ]
    # Rendering is append-safe: a later message never changes an earlier one's rendering.
    assert render_conversation(messages[:4])[0] == rendered[:4]


def test_a_result_just_under_the_minimum_is_sent_again() -> None:
    """Killed by: src/uclone_x/core/context_state.py :: and len(content) >= BACK_REFERENCE_MIN_CHARS
    Becomes: and len(content) >= 0
    """
    short = "z" * (BACK_REFERENCE_MIN_CHARS - 1)
    messages = [
        ChatMessage(role=MessageRole.TOOL, content=short, tool_call_id=f"c{n}", name="read")
        for n in range(2)
    ]
    assert render_conversation(messages) == (messages, [None, None])


@pytest.mark.asyncio
async def test_a_repeated_read_is_sent_once_and_recorded_as_the_same_entry(
    tmp_path: Path,
) -> None:
    """The same file read three times in one turn: the later requests carry it once, and
    the context state records each repeat as a back-reference to the first entry.

    Killed by: src/uclone_x/core/context_state.py :: base if earlier is None else base.model_copy(update={"same_as": bases[earlier].entry})
    Becomes: base
    """
    llm = _ScriptedLLM([[_call(f"c{n}", "read")] for n in range(3)])
    agent, store = _agent(tmp_path, llm, [_returning("read", BODY)])
    await agent.start()
    assert (await agent.execute_turn("read it")).is_completed
    agent.persist_session()

    last = llm.requests[-1]
    assert _tool(last, "c0").content == BODY
    for repeat in ("c1", "c2"):
        assert _tool(last, repeat).content == back_reference_text(_tool(last, "c0"))
    state = store.load("sess_1443")
    assert state is not None
    (epoch,) = state.context_epochs
    results = [
        x for x in epoch.entries if state.session_log[int(x.entry[1:])].kind.value == "tool_result"
    ]
    assert [x.same_as for x in results] == [None, results[0].entry, results[0].entry]
    # What was sent is what the record rebuilds.
    log = store.event_log_path("sess_1443")
    assert log is not None
    rebuilt = rebuild_requests(store, state, [dict(e) for e in read_session_log(log)])
    assert [r.request.messages for r in rebuilt] == [r.messages for r in llm.requests]
    _assert_epochs_render_from_the_log(store, state, llm.requests)


# ======================================================================================
# Rules 3 and 4 -- forms drop only at a compaction; a stub names what reads it back
# ======================================================================================


@pytest.mark.asyncio
async def test_a_compaction_at_a_turn_start_drops_forms_in_a_new_epoch(tmp_path: Path) -> None:
    """Turn 1 shows a long result as an excerpt. Turn 2 starts with a compaction, which
    turns it into a stub; the stub's first request opens a new epoch the compaction is
    named in, and the model reads the whole result back through the handle the stub names.

    Killed by: src/uclone_x/agent/compaction_driver.py :: live.declare_new_epoch("compaction")
    Becomes: pass
    """
    body = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path,
        llm,
        [_returning("dump", body), ToolResultReadTool()],
        compaction_threshold_tokens=3_000,
        compactor=compactor,
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    excerpt = _tool(llm.requests[-1], "c1").content
    assert excerpt is not None and message_form(_tool(llm.requests[-1], "c1")) is (
        ContextForm.EXCERPT
    )
    handle = handle_in(excerpt)
    assert handle is not None
    llm.steps.append([_call("c2", "tool_result_read", handle=handle, offset=0, length=200_000)])
    turn_one_requests = len(llm.requests)

    assert (await agent.execute_turn("read all of it")).is_completed
    agent.persist_session()

    stub_request = llm.requests[turn_one_requests]
    stub = _tool(stub_request, "c1")
    assert message_form(stub) is ContextForm.STUB
    assert stub.content is not None and f'tool_result_read(handle="{handle}"' in stub.content

    state = store.load(sid)
    assert state is not None
    epochs = state.context_epochs
    assert [(e.turn, e.step, e.opened_by) for e in epochs] == [
        (1, 1, ("start",)),
        (2, 1, ("compaction",)),
    ]

    def form_of(epoch: ContextEpoch, call_id: str) -> ContextForm:
        for shown in epoch.entries:
            body_of = store.load_context_body(sid, state.session_log[int(shown.entry[1:])].digest)
            message = ChatMessage.model_validate_json(body_of or "{}")
            if message.tool_call_id == call_id:
                return shown.form
        raise AssertionError(call_id)

    assert form_of(epochs[0], "c1") is ContextForm.EXCERPT
    assert form_of(epochs[1], "c1") is ContextForm.STUB
    # Rule 4: the handle reads the whole result back, in the turn that followed.
    page = _tool(llm.requests[-1], "c2").content
    assert page is not None and page.startswith(f"[Stored tool result {handle}: characters 0 ")
    shown = page.split("\n", 1)[1]
    assert len(shown) > 1_000 and body.startswith(shown)
    assert agent._result_bodies(sid).read(handle) == body  # pyright: ignore[reportPrivateUsage]
    # Both epochs -- the excerpt's and the stub's -- render from the log as they were sent.
    _assert_epochs_render_from_the_log(store, state, llm.requests)


class _RewriteAnswer(BaseHook):
    """A post-turn hook that rewrites the turn's final answer."""

    async def on_post_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.MODIFY, modified_payload={"content": "checked"})


@pytest.mark.asyncio
async def test_a_rewritten_final_answer_does_not_label_the_next_epoch(tmp_path: Path) -> None:
    """A post-turn hook that rewrites the final answer changes a message no request has
    shown yet, so it breaks nothing and declares nothing: the epoch the next turn's
    compaction opens says `compaction` alone, not a stale `post_turn_hook` (#1854).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if self._shows_any((entry,)):
    Becomes: if True:
    """
    body = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path,
        llm,
        [_returning("dump", body), ToolResultReadTool()],
        compaction_threshold_tokens=3_000,
        compactor=compactor,
    )
    agent.hook_runner.register_hooks([_RewriteAnswer()])
    await agent.start()
    assert (await agent.execute_turn("dump it")).content == "checked"
    assert (await agent.execute_turn("and now?")).is_completed
    agent.persist_session()

    assert any(m.content == "checked" for m in llm.requests[-1].messages)
    state = store.load(sid)
    assert state is not None
    assert [e.opened_by for e in state.context_epochs] == [("start",), ("compaction",)]


@pytest.mark.asyncio
async def test_a_sanitized_answer_is_written_back_and_declares_nothing_no_request_showed(
    tmp_path: Path,
) -> None:
    """An answer that links an image the workspace does not hold is rewritten, in the
    history and the log, to the directive the head words (#1971, item 5). The answer is
    the turn's last message, which no request has shown, so the rewrite declares nothing:
    the next request extends the epoch with the sanitized answer.

    Killed by: src/uclone_x/agent/turn_executor.py :: self._rewrite_answer(assistant_msg_idx, sanitized, "artifact_sanitized")
    Becomes: pass
    """
    gone = "![Gone](/api/artifacts/content?path=artifacts/images/img_gone.png)"
    llm = _ScriptedLLM([])
    llm._default_response = f"Here it is: {gone}"  # pyright: ignore[reportPrivateUsage]
    agent, store = _agent(tmp_path, llm, [])
    await agent.start()
    result = await agent.execute_turn("draw it")
    assert result.is_completed and gone not in (result.content or "")
    agent.persist_session()

    state = store.load("sess_1443")
    assert state is not None
    answer = state.messages[-1]
    assert answer.role is MessageRole.ASSISTANT
    assert gone not in (answer.content or "") and 'missing-image{file="img_gone.png"}' in (
        answer.content or ""
    )
    llm._default_response = "ok"  # pyright: ignore[reportPrivateUsage]
    assert (await agent.execute_turn("thanks")).is_completed
    agent.persist_session()
    state = store.load("sess_1443")
    assert state is not None
    assert [e.opened_by for e in state.context_epochs] == [("start",)]
    assert any(m.content == answer.content for m in llm.requests[-1].messages)


# ======================================================================================
# The record: persisted, validated, restored
# ======================================================================================


def test_epochs_are_numbered_by_position() -> None:
    """Killed by: src/uclone_x/core/session_state.py :: if epoch.number != position:
    Becomes: if False:
    """
    epoch = ContextEpoch(number=1, turn=1, step=1, opened_by=("start",), entries=())
    with pytest.raises(ValidationError, match="numbered 1"):
        SessionState(session_id="s", agent_id="a", context_epochs=(epoch,))


def test_a_record_from_before_the_context_state_reads_back_with_none() -> None:
    state = SessionState(session_id="s", agent_id="a")
    raw = json.loads(state.model_dump_json())
    del raw["context_epochs"]
    assert SessionState.model_validate_json(json.dumps(raw)).context_epochs == ()


@pytest.mark.asyncio
async def test_the_epochs_survive_a_restart_and_a_restored_session_extends_them(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: context_epochs=list(state.context_epochs),
    Becomes: context_epochs=[],
    """
    llm = _ScriptedLLM([])
    agent, store = _agent(tmp_path, llm, [])
    await agent.start()
    assert (await agent.execute_turn("one")).is_completed
    agent.persist_session()
    saved = store.load("sess_1443")
    assert saved is not None and len(saved.context_epochs) == 1

    again, _ = _agent(tmp_path, llm, [])
    again.hydrate_session("sess_1443")
    await again.start()
    assert (await again.execute_turn("two")).is_completed
    # The restored history renders as it did, so turn 2 extends the epoch turn 1 opened.
    (epoch,) = again.get_session().context_epochs
    assert (epoch.turn, epoch.opened_by) == (1, ("start",))  # turn 1's, not a new one
    before = saved.context_epochs[0].entries
    assert epoch.entries[: len(before)] == before
    assert len(epoch.entries) > len(before)


def test_a_history_maps_to_the_latest_entries_of_each_digest() -> None:
    """Killed by: src/uclone_x/core/session_log.py :: digest: by_digest.get(digest, [])[-count:][::-1] for digest, count in wanted.items()
    Becomes: digest: by_digest.get(digest, [])[:count][::-1] for digest, count in wanted.items()
    """
    a, b = "a" * 64, "b" * 64

    def entry(n: int, digest: str) -> SessionLogEntry:
        return SessionLogEntry(
            id=f"e{n}",
            kind=SessionLogKind.UTTERANCE,
            turn=0,
            digest=digest,
            size=1,
            provenance=SessionLogProvenance.RECORDED,
        )

    log = [entry(0, a), entry(1, b), entry(2, a), entry(3, a)]
    assert history_entry_ids(log, [a, b, a]) == ["e2", "e1", "e3"]
    assert history_entry_ids(log, [b, b]) == ["e1", None]


# ======================================================================================
# Records that hold a form as its text (#1848): refused, never shown as that text
# ======================================================================================

#: What an error's copy must not carry: ids, handles, digests, exception names.
_INTERNALS = ("tr_", "e0", "e1", "e2", "digest", "Error", "#18", "form", "excerpt", "stub")


def _as_recorded(message: ChatMessage) -> ChatMessage:
    """`message` as a saved record holds it: a form with no text (#1848)."""
    return message if message.form is None else message.model_copy(update={"content": None})


def _old_form_record(tmp_path: Path, *, recorded: bool) -> tuple[SessionStore, SessionState]:
    """A saved session whose log keeps an excerpt as its text: `recorded` says whether
    the message still records what it was cut from (the #1972 build) or not (main)."""
    store = SessionStore(tmp_path / "store")
    handle = result_handle(BODY)
    kept = logged_text(SessionLogKind.TOOL_RESULT, BODY, blob=handle)
    store.save_context_body("s", kept.digest, kept.body)
    excerpt = ChatMessage(
        role=MessageRole.TOOL,
        content=excerpt_tool_result(BODY, handle, cap_bytes=1_000),
        name="t",
        tool_call_id="c1",
        form="excerpt",
        rendered_from=RenderedFrom(handle=handle, limit=1_000, readable=True),
    )
    # The body as a build that logged a form's text wrote it: the whole message, text in.
    old_body = (
        excerpt.model_dump_json()
        if recorded
        else excerpt.model_copy(update={"rendered_from": None}).model_dump_json()
    )
    digest = content_digest(old_body)
    store.save_context_body("s", digest, old_body)
    provenance = SessionLogProvenance.RECORDED
    log = (
        new_entry(0, kept, turn=1, provenance=provenance),
        SessionLogEntry(
            id="e1",
            kind=SessionLogKind.TOOL_RESULT,
            digest=digest,
            size=len(old_body),
            turn=1,
            provenance=provenance,
            blob=handle,
        ),
    )
    epoch = ContextEpoch(
        number=0, turn=1, step=1, opened_by=("start",), entries=(_entry("e1", ContextForm.EXCERPT),)
    )
    # The record holds the form as a form; only the log body holds its text.
    recorded_form = excerpt.model_copy(update={"content": None})
    state = SessionState(
        session_id="s",
        agent_id="a",
        messages=(recorded_form,),
        session_log=log,
        context_epochs=(epoch,),
    )
    return store, state


def _pre_1854_record(
    tmp_path: Path, form: ContextForm | None, *, entries: tuple[str, ...] = ()
) -> tuple[SessionStore, SessionState]:
    """A saved session holding a tool message with no form, whose text is a stub's, and
    an epoch that shows its entry in `form` -- or no epoch, for `None`: before #1854 the
    form was read from text. `entries` is what the record names as each message's log
    entry; a build before #1848 named none."""
    store = SessionStore(tmp_path / "store")
    message = ChatMessage(
        role=MessageRole.TOOL,
        content=stored_result_stub(result_handle(BODY), BODY, keep_chars=40),
        name="t",
        tool_call_id="c1",
    )
    item = logged_message(message)
    store.save_context_body("s", item.digest, item.body)
    log = (new_entry(0, item, turn=1, provenance=SessionLogProvenance.RECORDED),)
    epochs = (
        ()
        if form is None
        else (
            ContextEpoch(
                number=0, turn=1, step=1, opened_by=("start",), entries=(_entry("e0", form),)
            ),
        )
    )
    state = SessionState(
        session_id="s",
        agent_id="a",
        messages=(message,),
        session_log=log,
        context_epochs=epochs,
        history_entries=entries,
    )
    return store, state


@pytest.mark.parametrize("form", [ContextForm.STUB, None], ids=["epoch_shows_a_stub", "no_epochs"])
def test_a_record_from_before_forms_were_recorded_is_set_aside_not_labelled_full(
    tmp_path: Path, form: ContextForm | None
) -> None:
    """A stub saved before #1854 is a message with no form -- shown by its epoch as a
    `stub`, or in a record with no epochs at all. Restored, it would be labelled `full`,
    which it is not. Every record written since #1848 names each message's log entry, and
    this one, like every record before it, names none; so it is refused by that structure
    alone and set aside, and the person is told so in the plain words every set-aside
    record gets (#1974 item 5; no users, so it is not converted).

    Killed by: src/uclone_x/core/session_state.py :: if state.messages and state.session_log and not state.history_entries:
    Becomes: if False:
    """
    from uclone_x.core.session_state import recorded_before_history_entries
    from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE

    store, state = _pre_1854_record(tmp_path, form)
    path = store.session_path("s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(), encoding="utf-8")
    written = path.read_bytes()

    assert store.load("s") is None
    assert recorded_before_history_entries(SessionState.model_validate_json(written))
    agent, agents_store = _agent(tmp_path, _ScriptedLLM([]), [], sid="s")
    assert agent.hydrate_session("s") is None
    agent.persist_session("s")
    (aside,) = [p for p in path.parent.iterdir() if ".unreadable-" in p.name]
    assert aside.read_bytes() == written
    assert agents_store.take_set_aside("s") is True
    assert SESSION_SET_ASIDE_NOTICE
    for internal in (
        *_INTERNALS,
        "1848",
        "1854",
        "history_entries",
        "epoch",
        "digest",
        "ValidationError",
    ):
        assert internal not in SESSION_SET_ASIDE_NOTICE


def test_output_that_quotes_a_stub_header_loads_and_is_labelled_full(tmp_path: Path) -> None:
    """A tool whose own output begins like a stub -- a file quoting one -- is a whole
    message its epoch shows as `full`. The refusal reads no text, so the record loads and
    the message stays `full` (#1854, #1974).
    """
    store, state = _pre_1854_record(tmp_path, ContextForm.FULL, entries=("e0",))
    store.save(state)

    loaded = store.load("s")
    assert loaded is not None
    assert loaded.messages == state.messages
    assert message_form(loaded.messages[0]) is ContextForm.FULL


@pytest.mark.parametrize("recorded", [True, False], ids=["with_source", "text_only"])
def test_a_log_entry_that_holds_a_forms_text_is_refused_in_plain_words(
    tmp_path: Path, recorded: bool
) -> None:
    """A log written before forms were logged as records holds an excerpt's text. The
    rebuild refuses it as unreadable, in the plain words every unreadable record gets,
    rather than showing the text it holds -- whether or not the text happens to be the
    rendering (#1848). No id, handle or exception name reaches the copy.

    Killed by: src/uclone_x/agent/request_record.py :: message.rendered_from is None or message.content is not None
    Becomes: message.rendered_from is None
    """
    store, state = _old_form_record(tmp_path, recorded=recorded)

    with pytest.raises(EpochRecordError) as raised:
        rebuild_epoch_conversations(store, state)

    assert raised.value.code == "unreadable"
    assert str(raised.value) == (
        "Part of this conversation's record could not be read, so it cannot be rebuilt."
    )
    assert not any(token in str(raised.value) for token in _INTERNALS)
    assert "e1" in str(raised.value.detail)


def test_a_record_whose_messages_hold_a_form_with_no_source_is_not_loaded(
    tmp_path: Path,
) -> None:
    """A record from before `rendered_from` holds a form it cannot render. It fails
    validation, so the store reads it as absent and sets it aside on the next save, as it
    does any record it cannot read (#1844) -- it is never shown as the text it held.

    Killed by: src/uclone_x/core/session_state.py :: if message.form is not None and message.rendered_from is None:
    Becomes: if False:
    """
    store, state = _old_form_record(tmp_path, recorded=True)
    saved = store.save(state)
    raw = json.loads(store.session_path("s").read_text(encoding="utf-8"))
    raw["messages"][0].pop("rendered_from")
    store.session_path("s").write_text(json.dumps(raw), encoding="utf-8")

    assert store.load("s") is None
    with pytest.raises(ValidationError, match="records no result"):
        saved.with_messages([saved.messages[0].model_copy(update={"rendered_from": None})])


def test_a_form_that_records_no_source_is_never_logged() -> None:
    """Killed by: src/uclone_x/core/session_log.py :: if message.rendered_from is None:
    Becomes: if False:
    """
    orphan = ChatMessage(
        role=MessageRole.TOOL, content="[cut]", name="t", tool_call_id="c1", form="excerpt"
    )
    with pytest.raises(ValueError, match="cannot keep it as a form"):
        logged_message(orphan)


# ======================================================================================
# A compaction's forms, derived from the history before it (#1848)
# ======================================================================================


def test_a_compactions_forms_are_derived_from_what_each_message_replaces() -> None:
    """A kept message keeps the form its entry was shown in -- here an excerpt with no
    form on it, recorded as one by an earlier epoch -- the ledger is `summary`, and a
    pruned message takes the smaller form the compactor gave it. A pruned message that
    would rank above the form it replaces is listed as rising and keeps that form.

    Killed by: src/uclone_x/core/context_state.py :: if FORM_ORDER.index(pruned) < FORM_ORDER.index(previous.form):
    Becomes: if False:
    """
    user = ChatMessage(role=MessageRole.USER, content="go")
    legacy = ChatMessage(
        role=MessageRole.TOOL, content="head ... tail", name="t", tool_call_id="c1"
    )
    full = ChatMessage(role=MessageRole.TOOL, content=BODY, name="t", tool_call_id="c2")
    stub = ChatMessage(
        role=MessageRole.TOOL, content="[stub]", name="t", tool_call_id="c3", form="stub"
    )
    before = [
        (_entry("e0"), user),
        (_entry("e1", ContextForm.EXCERPT), legacy),
        (_entry("e2"), full),
        (_entry("e3", ContextForm.STUB), stub),
    ]
    ledger = ChatMessage(role=MessageRole.SYSTEM, content="ledger", compaction_ledger=True)
    pruned = full.model_copy(update={"content": "[stored]", "form": "stub"})
    widened = stub.model_copy(update={"content": "[stub", "form": "excerpt"})

    derived = derive_compacted_forms(
        before, [ledger, user, legacy, pruned, widened], [None, 0, 1, 2, 3]
    )

    assert derived.forms == (
        ContextForm.SUMMARY,
        ContextForm.FULL,
        ContextForm.EXCERPT,
        ContextForm.STUB,
        ContextForm.STUB,
    )
    assert derived.rising == (4,)
    # The pruned message is the entry it replaced, dropped a form; the ledger is its own.
    assert derived.shows == (None, "e0", "e1", "e2", "e3")


def test_a_kept_message_is_its_entry_carried_over_not_a_rewrite() -> None:
    """A message the compaction returned unchanged is not compared on the ladder: an
    excerpt with no form on it, which reads as `full`, is not a rise from the `excerpt`
    its entry was recorded in.

    Killed by: src/uclone_x/core/context_state.py :: if message == source:
    Becomes: if False:
    """
    legacy = ChatMessage(
        role=MessageRole.TOOL, content="head ... tail", name="t", tool_call_id="c1"
    )

    derived = derive_compacted_forms([(_entry("e1", ContextForm.EXCERPT), legacy)], [legacy], [0])

    assert derived == derived.__class__(forms=(ContextForm.EXCERPT,), shows=("e1",), rising=())


def test_origins_that_do_not_match_the_compacted_messages_are_refused() -> None:
    """Killed by: src/uclone_x/core/context_state.py :: if not 0 <= origin < len(before):
    Becomes: if False:
    """
    user = ChatMessage(role=MessageRole.USER, content="go")
    with pytest.raises(ValueError, match="origins"):
        derive_compacted_forms([(_entry("e0"), user)], [user], [])
    with pytest.raises(ValueError, match="no ledger"):
        derive_compacted_forms([(_entry("e0"), user)], [user], [None])
    with pytest.raises(ValueError, match="names input"):
        derive_compacted_forms([(_entry("e0"), user)], [user], [1])


async def test_the_compactor_says_where_each_message_it_returns_came_from() -> None:
    """Killed by: src/uclone_x/llm/compactor.py :: origins=(*anchor_at, *kept_ledgers, None, *dialog_at[cut:]),
    Becomes: origins=(),
    """
    compactor = ContextCompactor(keep_recent_turns=1)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="anchor"),
        ChatMessage(role=MessageRole.USER, content="one"),
        ChatMessage(role=MessageRole.ASSISTANT, content="first"),
        ChatMessage(role=MessageRole.USER, content="two"),
        ChatMessage(role=MessageRole.ASSISTANT, content="second"),
    ]

    outcome = await compactor.compact(messages)

    assert outcome.origins == (0, None, 3, 4)
    for message, origin in zip(outcome.messages, outcome.origins, strict=True):
        if origin is None:
            assert message.compaction_ledger
        else:
            assert message == messages[origin]


@pytest.mark.parametrize("keep_recent_turns", [50, 1], ids=["prune_only", "summary"])
async def test_each_ledger_a_compaction_keeps_names_the_ledger_it_is(
    keep_recent_turns: int,
) -> None:
    """Prior ledgers beyond the cap are superseded, the newest kept; each kept ledger's
    origin is that ledger, not one of the superseded ones. Adapted from the #1884
    review's probe, on both of the compactor's paths.

    Killed by: src/uclone_x/llm/compactor.py :: return list(ledger_at[len(ledger_at) - len(retained) :])
    Becomes: return list(ledger_at[: len(retained)])
    """
    compactor = ContextCompactor(keep_recent_turns=keep_recent_turns, max_ledgers=2)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="anchor"),
        *(
            ChatMessage(role=MessageRole.SYSTEM, content=f"ledger {n}", compaction_ledger=True)
            for n in range(4)
        ),
    ]
    for n in range(3):
        messages += [
            ChatMessage(role=MessageRole.USER, content=f"ask {n}"),
            ChatMessage(role=MessageRole.ASSISTANT, content=f"answer {n}"),
        ]

    outcome = await compactor.compact(messages)
    # Each case reaches its own path: only the summary path writes a new ledger.
    assert (None in outcome.origins) is (keep_recent_turns == 1)

    kept = [
        (message.content, messages[origin].content)
        for message, origin in zip(outcome.messages, outcome.origins, strict=True)
        if origin is not None and message.compaction_ledger
    ]
    assert kept, "the pass kept no prior ledger, so it checks nothing"
    assert all(content == source for content, source in kept)
    assert kept[-1][0] == "ledger 3"


class _NoBodiesCompactor(ContextCompactor):
    """A compactor that holds no session's bodies, whatever the driver hands it."""

    @property
    def result_bodies(self) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
        return None

    @result_bodies.setter
    def result_bodies(self, value: object) -> None:
        del value


class _TruncatingCompactor(_NoBodiesCompactor):
    """A custom compactor that truncates a form's text as if it were plain output."""

    def prune_tool_message(self, msg: ChatMessage) -> ChatMessage:
        return super().prune_tool_message(msg.model_copy(update={"rendered_from": None}))


@pytest.mark.asyncio
async def test_a_compaction_keeps_a_stub_it_would_widen_into_an_excerpt(tmp_path: Path) -> None:
    """A custom compactor that truncates a long stub's text to a head and tail records it
    as an `excerpt` -- a form above the stub's, and a cut that can drop the handle the stub
    names (Rule 4). The built-in compactor leaves a form it cannot re-stub as it is (#1848),
    but the driver does not rely on that: derived from the history before it, the new epoch
    refuses the rise, the stub is kept as it was, and the requests still rebuild from the
    log.

    Killed by: src/uclone_x/agent/compaction_driver.py :: if derived is not None and derived.rising:
    Becomes: if False:
    """
    sid = "sess_1443"
    store = SessionStore(tmp_path / "store")
    handle = result_handle(BODY)
    kept = logged_text(SessionLogKind.TOOL_RESULT, BODY, blob=handle)
    store.save_context_body(sid, kept.digest, kept.body)
    stub = ChatMessage(
        role=MessageRole.TOOL,
        content=stored_result_stub(handle, BODY, keep_chars=2_000),
        name="t",
        tool_call_id="c1",
        form="stub",
        rendered_from=RenderedFrom(handle=handle, limit=2_000, readable=True),
    )
    history = [
        ChatMessage(role=MessageRole.USER, content="read it"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(_call("c1", "t"),)),
        stub,
        ChatMessage(role=MessageRole.ASSISTANT, content="done"),
    ]
    # A record holds a form as what it records, not as its text, and names the log
    # entry of each message (#1848).
    recorded = tuple(_as_recorded(message) for message in history)
    logged = [logged_message(message) for message in recorded]
    for item in logged:
        store.save_context_body(sid, item.digest, item.body)
    provenance = SessionLogProvenance.RECORDED
    store.save(
        SessionState(
            session_id=sid,
            agent_id="ctx",
            messages=recorded,
            session_log=tuple(
                new_entry(position, item, turn=0, provenance=provenance)
                for position, item in enumerate([kept, *logged])
            ),
            history_entries=tuple(f"e{position}" for position in range(1, len(logged) + 1)),
        )
    )
    llm = _ScriptedLLM([])
    compactor = _TruncatingCompactor(keep_recent_turns=50, max_tool_output_chars=200)
    agent, _ = _agent(tmp_path, llm, [], compaction_threshold_tokens=300, compactor=compactor)
    assert compactor.prune_tool_message(stub).form == "excerpt"  # what the rise would be
    agent.hydrate_session(sid)
    await agent.start()
    assert (await agent.execute_turn("and now?")).is_completed
    agent.persist_session()

    assert _tool(llm.requests[-1], "c1") == stub
    state = store.load(sid)
    assert state is not None
    assert "compaction" in state.context_epochs[-1].opened_by
    forms = [
        shown.form
        for shown in state.context_epochs[-1].entries
        if state.session_log[int(shown.entry[1:])].kind is SessionLogKind.TOOL_RESULT
    ]
    assert forms == [ContextForm.STUB]
    _assert_epochs_render_from_the_log(store, state, llm.requests)


# ======================================================================================
# A dropped form is the same entry, shown through a rendering in the log (#1848)
# ======================================================================================


@pytest.mark.asyncio
async def test_a_compacted_result_is_its_entry_in_a_smaller_form_not_a_new_entry(
    tmp_path: Path,
) -> None:
    """The compaction that turns an excerpt into a stub drops the entry's form; it does
    not add an entry. The new epoch lists the excerpt's entry, in `stub`, and names the
    log entry holding the stub's text as its rendering; rendered from that log body, both
    epochs are the requests that were sent.

    Killed by: src/uclone_x/core/context_state.py :: else ContextEntry(entry=shows, form=form, rendering=body)
    Becomes: else ContextEntry(entry=body, form=form, rendering=None)
    """
    body = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path,
        llm,
        [_returning("dump", body), ToolResultReadTool()],
        compaction_threshold_tokens=3_000,
        compactor=compactor,
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    handle = handle_in(_tool(llm.requests[-1], "c1").content or "")
    llm.steps.append([_call("c2", "tool_result_read", handle=handle, offset=0, length=200)])
    assert (await agent.execute_turn("read some of it")).is_completed
    agent.persist_session()

    state = store.load(sid)
    assert state is not None
    first, second = state.context_epochs
    assert second.opened_by == ("compaction",)

    def message(entry: str) -> ChatMessage:
        logged = store.load_context_body(sid, state.session_log[int(entry[1:])].digest)
        assert logged is not None
        return ChatMessage.model_validate_json(logged)

    def result(epoch: ContextEpoch) -> ContextEntry:
        (shown,) = [s for s in epoch.entries if message(s.entry).tool_call_id == "c1"]
        return shown

    excerpt, stub = result(first), result(second)
    assert (excerpt.form, excerpt.rendering) == (ContextForm.EXCERPT, None)
    assert stub.entry == excerpt.entry
    assert stub.form is ContextForm.STUB
    assert stub.rendering is not None and stub.rendering != stub.entry
    # The log holds the stub as what it was cut from; the request rendered its text.
    sent = _tool(llm.requests[-1], "c1")
    assert message(stub.rendering).content is None
    assert message(stub.rendering) == sent.model_copy(update={"content": None})
    assert message_form(message(stub.rendering)) is ContextForm.STUB
    # Written with the rendering, read back with it.
    assert ContextEpoch.model_validate(second.model_dump()) == second
    _assert_epochs_render_from_the_log(store, state, llm.requests)


def _shown_by_call(
    store: SessionStore, state: SessionState, epoch: ContextEpoch
) -> dict[str, ContextEntry]:
    """Per tool call id, the entry an epoch shows that call's result as."""
    found: dict[str, ContextEntry] = {}
    for shown in epoch.entries:
        body = store.load_context_body(
            state.session_id, state.session_log[int(shown.entry[1:])].digest
        )
        message = ChatMessage.model_validate_json(body or "{}")
        if message.tool_call_id is not None and message.role is MessageRole.TOOL:
            found[message.tool_call_id] = shown
    return found


async def _compact_twice(tmp_path: Path) -> tuple[BaseAgent, SessionStore, _ScriptedLLM]:
    """Three turns: two long results, each an excerpt, and a compaction at the start of
    turns 2 and 3. The second compaction keeps the first one's stub as it was."""
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path,
        llm,
        [
            _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(1_500))),
            _returning("dump2", "\n".join(f"row {i:05d}: other" for i in range(1_500))),
            ToolResultReadTool(),
        ],
        compaction_threshold_tokens=3_000,
        compactor=compactor,
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    llm.steps.append([_call("c2", "dump2")])
    assert (await agent.execute_turn("and the other")).is_completed
    assert (await agent.execute_turn("thanks")).is_completed
    agent.persist_session()
    return agent, store, llm


@pytest.mark.asyncio
async def test_an_entry_keeps_its_id_across_repeated_compactions(tmp_path: Path) -> None:
    """A stub a first compaction made is still its entry after a second compaction keeps
    it: the second derives what the history showed from the renderings the epoch in force
    recorded, so it does not read the stub's body as an entry of its own.

    Killed by: src/uclone_x/agent/compaction_driver.py :: live.logged_history(), opening_entries(live.context_epochs, live.compacted_entries)
    Becomes: live.logged_history(), dict(live.compacted_entries)
    """
    _agent_, store, llm = await _compact_twice(tmp_path)
    state = store.load("sess_1443")
    assert state is not None
    epochs = state.context_epochs
    assert [e.opened_by for e in epochs] == [("start",), ("compaction",), ("compaction",)]
    first, second, third = (_shown_by_call(store, state, e) for e in epochs)

    assert first["c1"].form is ContextForm.EXCERPT
    assert second["c1"].form is ContextForm.STUB and second["c1"].rendering is not None
    # The second compaction keeps the stub: same entry, same form, same rendering.
    assert third["c1"] == second["c1"]
    assert third["c1"].entry == first["c1"].entry
    # And it drops the second result the way the first dropped the first.
    assert second["c2"].form is ContextForm.EXCERPT
    assert third["c2"].entry == second["c2"].entry
    assert third["c2"].form is ContextForm.STUB and third["c2"].rendering is not None
    _assert_epochs_render_from_the_log(store, state, llm.requests)


@pytest.mark.asyncio
async def test_a_compacted_message_is_one_the_epoch_shows(tmp_path: Path) -> None:
    """After a compaction, the stub in the history is shown by the epoch through its
    rendering, so rewriting it would break Rule 1 and must declare a new epoch (#1854).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: bodies = {shown.body for shown in epoch}
    Becomes: bodies = {shown.entry for shown in epoch}
    """
    agent, _store, _llm = await _compact_twice(tmp_path)
    lifecycle = agent._session_lifecycle  # pyright: ignore[reportPrivateUsage]
    live = lifecycle._live_session("sess_1443")  # pyright: ignore[reportPrivateUsage]
    (index,) = [
        i
        for i, m in enumerate(live.messages)
        if m.tool_call_id == "c1" and m.role is MessageRole.TOOL
    ]
    assert message_form(live.messages[index]) is ContextForm.STUB
    assert live.shown_in_epoch(index)


def test_a_rebuild_with_a_dangling_rendering_raises(tmp_path: Path) -> None:
    """An epoch whose entry names a rendering the log does not have cannot be rebuilt:
    the rebuild says so rather than showing the entry's own body in its place.

    Killed by: src/uclone_x/agent/request_record.py :: if not 0 <= position < len(self._log):
    Becomes: if not 0 <= position:
    """
    store = SessionStore(tmp_path / "store")
    message = ChatMessage(role=MessageRole.USER, content="hello")
    logged = logged_message(message)
    store.save_context_body("s", logged.digest, logged.body)
    entry = new_entry(0, logged, turn=1, provenance=SessionLogProvenance.RECORDED)
    epoch = ContextEpoch(
        number=0,
        turn=1,
        step=1,
        opened_by=("start",),
        entries=(ContextEntry(entry="e0", form=ContextForm.STUB, rendering="e7"),),
    )
    state = SessionState(
        session_id="s",
        agent_id="a",
        messages=(message,),
        session_log=(entry,),
        context_epochs=(epoch,),
    )

    with pytest.raises(RequestRecordError) as raised:
        rebuild_epoch_conversations(store, state)
    assert "no log entry e7" in str(raised.value.detail)
    # The same entry without the dangling rendering renders: what raised is the rendering.
    plain = epoch.model_copy(update={"entries": (_entry("e0"),)})
    assert rebuild_epoch_conversations(
        store, state.model_copy(update={"context_epochs": (plain,)})
    ) == [[message]]


def test_a_rebuild_renders_an_excerpt_from_its_kept_result_and_needs_that_result(
    tmp_path: Path,
) -> None:
    """The log-only rebuild shows an excerpt rendered from the full result its handle
    names in the session log, with the cap it records, not the text logged for it. When
    the log holds no such result, the rebuild says so instead of showing the logged text.

    Killed by: src/uclone_x/agent/request_record.py :: detail=f"no log entry for the kept result {handle}",
    Becomes: detail=f"no log entry {handle}",
    """
    store = SessionStore(tmp_path / "store")
    full = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    handle = result_handle(full)
    kept = logged_text(SessionLogKind.TOOL_RESULT, full, blob=handle)
    store.save_context_body("s", kept.digest, kept.body)
    source = RenderedFrom(handle=handle, limit=4_000, readable=False)
    message = ChatMessage(
        role=MessageRole.TOOL,
        content="not the excerpt",
        name="dump",
        tool_call_id="c1",
        form="excerpt",
        rendered_from=source,
    )
    logged = logged_message(message)
    store.save_context_body("s", logged.digest, logged.body)
    provenance = SessionLogProvenance.RECORDED
    log = (
        new_entry(0, kept, turn=1, provenance=provenance),
        new_entry(1, logged, turn=1, provenance=provenance),
    )
    epoch = ContextEpoch(
        number=0,
        turn=1,
        step=1,
        opened_by=("start",),
        entries=(ContextEntry(entry="e1", form=ContextForm.EXCERPT),),
    )
    state = SessionState(
        session_id="s",
        agent_id="a",
        messages=(_as_recorded(message),),
        session_log=log,
        context_epochs=(epoch,),
    )

    ((shown,),) = rebuild_epoch_conversations(store, state)
    expected = excerpt_tool_result(full, handle, cap_bytes=4_000, readable=False)
    assert shown == message.model_copy(update={"content": expected})

    # The same log with the kept result's entry replaced by another message's.
    without = (log[1].model_copy(update={"id": "e0"}), log[1])
    with pytest.raises(RequestRecordError) as raised:
        rebuild_epoch_conversations(store, state.model_copy(update={"session_log": without}))
    assert raised.value.code == "log_entry_missing"
    assert raised.value.detail == f"no log entry for the kept result {handle}"


def _excerpt_record(tmp_path: Path) -> tuple[SessionStore, SessionState, str, str]:
    """A saved session whose one epoch shows an excerpt of a kept result: the store, the
    record, the kept result's handle, and the digest its body is stored under."""
    store = SessionStore(tmp_path / "store")
    full = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    handle = result_handle(full)
    kept = logged_text(SessionLogKind.TOOL_RESULT, full, blob=handle)
    store.save_context_body("s", kept.digest, kept.body)
    message = ChatMessage(
        role=MessageRole.TOOL,
        content=excerpt_tool_result(full, handle, cap_bytes=4_000),
        name="dump",
        tool_call_id="c1",
        form="excerpt",
        rendered_from=RenderedFrom(handle=handle, limit=4_000, readable=True),
    )
    logged = logged_message(message)
    store.save_context_body("s", logged.digest, logged.body)
    provenance = SessionLogProvenance.RECORDED
    state = SessionState(
        session_id="s",
        agent_id="a",
        messages=(_as_recorded(message),),
        session_log=(
            new_entry(0, kept, turn=1, provenance=provenance),
            new_entry(1, logged, turn=1, provenance=provenance),
        ),
        context_epochs=(
            ContextEpoch(
                number=0,
                turn=1,
                step=1,
                opened_by=("start",),
                entries=(ContextEntry(entry="e1", form=ContextForm.EXCERPT),),
            ),
        ),
        history_entries=("e1",),
    )
    return store, state, handle, kept.digest


@pytest.mark.parametrize(
    ("damage", "code", "copy"),
    [
        (
            "deleted",
            "body_missing",
            "Part of this conversation's record is missing, so it cannot be rebuilt.",
        ),
        (
            "rewritten",
            "unreadable",
            "Part of this conversation's record could not be read, so it cannot be rebuilt.",
        ),
    ],
)
def test_a_rebuild_whose_kept_result_is_lost_or_altered_is_refused_in_plain_words(
    tmp_path: Path, damage: str, code: str, copy: str
) -> None:
    """The rebuild renders an excerpt from the kept result its handle names, so a kept
    result whose body is gone, or no longer the text the handle was made from, stops the
    rebuild (#1969, item 1): it is not shown as some other text. The copy is the plain
    rebuild wording, with no handle, digest or exception name; the specifics are on the
    error's `code` and `detail`.

    Killed by: src/uclone_x/agent/request_record.py :: code="body_missing",  # a kept result's body
    Becomes: code="log_entry_missing",  # a kept result's body
    Killed by: src/uclone_x/agent/request_record.py :: if result_handle(text) != handle:
    Becomes: if False:
    """
    store, state, handle, digest = _excerpt_record(tmp_path)
    assert len(rebuild_epoch_conversations(store, state)[0]) == 1
    path = store.context_body_dir("s") / digest
    if damage == "deleted":
        path.unlink()
    else:
        path.write_text("some other text entirely", encoding="utf-8")

    with pytest.raises(EpochRecordError) as raised:
        rebuild_epoch_conversations(store, state)

    assert raised.value.code == code
    assert str(raised.value) == copy
    for internal in (handle, digest, *_INTERNALS, "/"):
        assert internal not in str(raised.value)
    assert (digest if damage == "deleted" else handle) in str(raised.value.detail)


def test_a_kept_result_that_will_not_decode_is_refused_in_plain_words(tmp_path: Path) -> None:
    """A kept result's body file that is there but will not read -- bad bytes here -- is
    refused as `unreadable` in the rebuild's plain words, not raised as the decoder's
    text; the digest and the exception's name are on `detail` only (#1974).

    Killed by: src/uclone_x/agent/request_record.py :: except (OSError, UnicodeDecodeError) as exc:
    Becomes: except (OSError,) as exc:
    """
    store, state, handle, digest = _excerpt_record(tmp_path)
    (store.context_body_dir("s") / digest).write_bytes(b"\xff\xfe\xfa not text")

    with pytest.raises(EpochRecordError) as raised:
        rebuild_epoch_conversations(store, state)

    assert raised.value.code == "unreadable"
    assert str(raised.value) == (
        "Part of this conversation's record could not be read, so it cannot be rebuilt."
    )
    for internal in (handle, digest, *_INTERNALS, "/", "utf", "decode"):
        assert internal not in str(raised.value)
    assert digest in str(raised.value.detail)
    assert "UnicodeDecodeError" in str(raised.value.detail)


def test_a_record_whose_kept_result_is_lost_still_loads_saves_and_deletes(
    tmp_path: Path,
) -> None:
    """Loading, restoring, saving and deleting a record read no kept result: a form is
    held as what it records (#1848), so a body that is gone or will not decode never
    fails them (#1974, items 1 and 7). The loss is refused when the history is next
    shown, not when it is restored.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: messages=self.recorded,
    Becomes: messages=self.messages,
    """
    store, state, _handle, digest = _excerpt_record(tmp_path)
    store.save(state)
    body = store.context_body_dir("s") / digest
    body.write_bytes(b"\xff\xfe not text")
    loaded = store.load("s")
    assert loaded is not None and loaded.messages == state.messages
    store.save(loaded)
    body.unlink()

    agent, _ = _agent(tmp_path, _ScriptedLLM([]), [], sid="s")
    assert agent.hydrate_session("s") is not None
    agent.persist_session("s")
    saved = store.load("s")
    assert saved is not None
    assert [m.content for m in saved.messages] == [None]
    live = agent._active_session  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(EpochRecordError):
        _ = live.messages
    assert store.delete("s")


@pytest.mark.asyncio
async def test_a_delete_drops_the_history_rendered_before_it(tmp_path: Path) -> None:
    """A delete drops the full results not yet written, and the history rendered from
    them: shown again, the excerpt is refused as missing rather than served from the
    rendering made before the delete (#1974, item 3).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: self.view = None  # nor served from the view rendered before the delete (#1974)
    Becomes: pass
    """
    output = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    agent, _ = _agent(tmp_path, llm, [_returning("dump", output), ToolResultReadTool()])
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    live = agent._active_session  # pyright: ignore[reportPrivateUsage]
    assert any(m.form == "excerpt" for m in live.messages)

    agent.delete_session()

    with pytest.raises(EpochRecordError) as raised:
        _ = live.messages
    assert raised.value.code == "body_missing"


@pytest.mark.asyncio
async def test_a_saved_and_restored_session_sends_what_its_log_rebuilds(
    tmp_path: Path,
) -> None:
    """Save, load, next request: the record holds each form with no text, the restored
    session renders it from the kept result, and every request sent -- before and after
    the restart -- is byte for byte what the log-only rebuild makes of the record (#1848).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: messages=self.recorded,
    Becomes: messages=self.messages,
    """
    output = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    tools = [_returning("dump", output), ToolResultReadTool()]
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    agent, store = _agent(tmp_path, llm, tools)
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    agent.persist_session()
    saved = store.load("sess_1443")
    assert saved is not None
    forms = [m for m in saved.messages if m.form is not None]
    assert forms and all(m.content is None and m.rendered_from for m in forms)

    restarted, _ = _agent(tmp_path, llm, tools)
    assert restarted.hydrate_session("sess_1443") is not None
    await restarted.start()
    assert (await restarted.execute_turn("and now?")).is_completed
    restarted.persist_session()
    state = store.load("sess_1443")
    assert state is not None

    excerpt = _tool(llm.requests[-1], "c1")
    assert excerpt.form == "excerpt" and excerpt.content
    assert excerpt.content == _tool(llm.requests[0 + 1], "c1").content
    _assert_epochs_render_from_the_log(store, state, llm.requests)


def test_a_record_that_holds_a_forms_text_is_set_aside_not_loaded(tmp_path: Path) -> None:
    """A record written before forms were recorded holds an excerpt's text -- here cut to
    a share, 1763 bytes where its `rendered_from` renders 3763. Nothing in the record is
    rendered or trusted as that text: the record fails validation, so loading it finds
    nothing and the next save sets it aside unchanged, and no state can be built that
    holds a form's text (#1848; no users, so it is not converted).

    Killed by: src/uclone_x/core/session_state.py :: if message.form is not None and message.content is not None:
    Becomes: if False:
    """
    store, state, handle, _ = _excerpt_record(tmp_path)
    full = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    (message,) = state.messages
    assert message.content is None
    shared = excerpt_tool_result(full, handle, cap_bytes=2_000)
    store.save(state)
    path = store.session_path("s")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["messages"] = [message.model_dump(mode="json")]
    raw["messages"][0]["content"] = shared
    path.write_text(json.dumps(raw), encoding="utf-8")
    written = path.read_bytes()

    assert store.load("s") is None
    with pytest.raises(ValidationError, match="holds text"):
        state.with_messages([message.model_copy(update={"content": shared})])

    agent, _ = _agent(tmp_path, _ScriptedLLM([]), [], sid="s")
    assert agent.hydrate_session("s") is None
    agent.persist_session("s")
    (aside,) = [p for p in path.parent.iterdir() if ".unreadable-" in p.name]
    assert aside.read_bytes() == written


def test_a_history_handed_in_with_a_forms_text_is_held_as_its_record_or_refused_plainly(
    tmp_path: Path,
) -> None:
    """A caller that hands `load_history` an excerpt with its text: text that is the
    excerpt's rendering is held as the form, with no text, and saved that way; text that
    is not -- cut to a share the form does not record -- is refused in plain words, with
    the handle on `detail` only, and the session is left as it was (#1848, #1974).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: return message.model_copy(update={"content": None})
    Becomes: return message
    Killed by: src/uclone_x/agent/session_lifecycle.py :: if shown != clean.content:
    Becomes: if False:
    """
    store, state, handle, _ = _excerpt_record(tmp_path)
    full = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    store.save(state)
    agent, _ = _agent(tmp_path, _ScriptedLLM([]), [], sid="s")
    assert agent.hydrate_session("s") is not None
    (message,) = state.messages
    rendered = excerpt_tool_result(full, handle, cap_bytes=4_000)

    agent.load_history([message.model_copy(update={"content": rendered})])
    agent.persist_session("s")
    saved = store.load("s")
    assert saved is not None
    assert [m.content for m in saved.messages] == [None]
    before = agent.get_session("s")

    cut = message.model_copy(update={"content": excerpt_tool_result(full, handle, cap_bytes=2_000)})
    with pytest.raises(FormTextMismatchError) as raised:
        agent.load_history([cut])

    assert str(raised.value) == (
        "Part of this conversation could not be saved as it was written, so nothing was changed."
    )
    assert raised.value.reason_code == "form_text_mismatch"
    for internal in (handle, *_INTERNALS):
        assert internal not in str(raised.value)
    assert handle in raised.value.detail
    assert agent.get_session("s") == before


@pytest.mark.asyncio
async def test_a_restart_between_a_compaction_and_the_next_request_keeps_the_entries(
    tmp_path: Path,
) -> None:
    """A compaction's record holds the entries it derived, so a session restored before
    the next request shows the compacted result as the entry it was, through the same
    rendering, instead of logging the stub as an entry of its own.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: compacted_entries={shown.body: shown for shown in state.compacted_entries},
    Becomes: compacted_entries={},
    """
    body = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    tools = [_returning("dump", body), ToolResultReadTool()]
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path, llm, tools, compaction_threshold_tokens=3_000, compactor=compactor
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    agent.persist_session()
    await agent.compact_session()

    saved = store.load(sid)
    assert saved is not None
    (pending,) = [e for e in saved.compacted_entries if e.rendering is not None]
    rendering = pending.rendering
    assert rendering is not None and int(rendering[1:]) < len(saved.session_log)

    again, _ = _agent(tmp_path, llm, tools, compaction_threshold_tokens=3_000)
    again.hydrate_session(sid)
    await again.start()
    assert (await again.execute_turn("and now?")).is_completed
    again.persist_session()

    state = store.load(sid)
    assert state is not None
    first, *_, last = state.context_epochs
    before, after = _shown_by_call(store, state, first), _shown_by_call(store, state, last)
    assert before["c1"].form is ContextForm.EXCERPT
    assert after["c1"] == pending
    assert after["c1"].entry == before["c1"].entry
    # The request recorded them, so the record no longer carries them.
    assert state.compacted_entries == ()
    _assert_epochs_render_from_the_log(store, state, llm.requests)


class _EpochTaggingLLM(_ScriptedLLM):
    """Also notes, per request, the epoch the session recorded it in.

    The request's entries are recorded before it is sent (`record_shown`), so at
    `generate` the session's last epoch is the request's own.
    """

    def __init__(self, steps: Sequence[Sequence[ToolCallRequest]]) -> None:
        super().__init__(steps)
        self.agent: BaseAgent | None = None
        self.epoch_of: list[int] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        assert self.agent is not None
        self.epoch_of.append(self.agent.get_session().context_epochs[-1].number)
        return await super().generate(request)


def _wire(messages: Sequence[ChatMessage]) -> list[str]:
    return [m.model_dump_json() for m in messages]


def _logged_forms(store: SessionStore, state: SessionState) -> list[dict[str, Any]]:
    """Every logged body of `state` that is a message in a smaller form, as JSON."""
    forms: list[dict[str, Any]] = []
    for entry in state.session_log:
        text = store.load_context_body(state.session_id, entry.digest)
        assert text is not None, entry.id
        if text.startswith("{"):
            data: Any = json.loads(text)
            if isinstance(data, dict):
                record = cast(dict[str, Any], data)
                if record.get("form") is not None:
                    forms.append(record)
    return forms


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("over_cap", "keep_recent_turns"),
    [(True, 50), (False, 50), (True, 1)],
    ids=["over-cap", "under-cap", "summary"],
)
async def test_every_request_is_its_own_epochs_rendering_byte_for_byte(
    tmp_path: Path, over_cap: bool, keep_recent_turns: int
) -> None:
    """Over a session of up to five epochs -- opened by the start, two compactions, a
    rollback, and a compaction followed by a restart before the next request -- each
    request's conversation, as each message's JSON dump, is a prefix of the conversation its own epoch renders from the log alone, and the
    last request of each epoch is that rendering exactly. So the log and the context
    state reproduce what each request sent, and within an epoch the context only
    appended (Rule 1).

    Compared per request against its own epoch, not against any epoch: a request that
    re-sent what an earlier epoch showed would still match some epoch.

    Run with tool results over the cap -- each an `excerpt` at ingest and a `stub` after
    a compaction -- and with results short enough to stay whole. The log holds no text
    for an `excerpt` or `stub`, only what it was cut from (#1848), so both the request
    and the rebuild render each one from the kept result's body, and they agree byte for
    byte. Run once more with compactions that summarize all but the last turn, so the
    ledger's `summary` form is sent and rebuilt too.

    Killed by: src/uclone_x/agent/request_record.py :: return reader.render(epoch.entries, what=f"epoch {epoch.number}")
    Becomes: return reader.render(epoch.entries[:-1], what=f"epoch {epoch.number}")
    Killed by: src/uclone_x/core/context_state.py :: return message.model_copy(update={"content": text})
    Becomes: return message
    """
    sid = "sess_1443"
    lines = 1_500 if over_cap else 12
    llm = _EpochTaggingLLM([[_call("c1", "dump")]])
    tools = [
        _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(lines))),
        _returning("dump2", "\n".join(f"row {i:05d}: other" for i in range(lines))),
        ToolResultReadTool(),
    ]
    compactor = ContextCompactor(keep_recent_turns=keep_recent_turns)
    agent, store = _agent(
        tmp_path, llm, tools, compaction_threshold_tokens=3_000, compactor=compactor
    )
    llm.agent = agent
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    llm.steps.append([_call("c2", "dump2")])
    assert (await agent.execute_turn("and the other")).is_completed
    assert (await agent.execute_turn("thanks")).is_completed
    checkpoint = agent.checkpoint_turn()
    assert (await agent.execute_turn("undo me")).is_completed
    agent.roll_back_turn(checkpoint, reason="test")
    assert (await agent.execute_turn("instead")).is_completed
    agent.persist_session()
    await agent.compact_session()

    again, _ = _agent(tmp_path, llm, tools, compaction_threshold_tokens=3_000)
    again.hydrate_session(sid)
    llm.agent = again
    await again.start()
    assert (await again.execute_turn("and now?")).is_completed
    again.persist_session()

    state = store.load(sid)
    assert state is not None
    # Short results give the turn-start compactions nothing to shorten, so only the
    # rollback opens an epoch of its own. The last compaction shortens nothing either
    # unless it summarizes, and then the restart's request shows the very entries the
    # rollback's epoch did, adopted by id (#1848), and only appends: the same epoch.
    # (Adopted by text, a message equal to an earlier one was read as that one's entry,
    # and the restart opened an epoch for a history it did not change.)
    compacted = [("compaction",), ("compaction",)] if over_cap else []
    summarized = [("compaction", "restored")] if keep_recent_turns == 1 else []
    assert [e.opened_by for e in state.context_epochs] == [
        ("start",),
        *compacted,
        ("rollback",),
        *summarized,
    ]
    sent = [_wire(_conversation(r)) for r in llm.requests]
    rendered = [_wire(c) for c in rebuild_epoch_conversations(store, state)]
    assert len(sent) == len(llm.epoch_of) > len(rendered)
    assert sorted(set(llm.epoch_of)) == list(range(len(rendered)))
    for index, (conversation, number) in enumerate(zip(sent, llm.epoch_of, strict=True)):
        epoch = rendered[number]
        assert epoch[: len(conversation)] == conversation, f"request {index}, epoch {number}"
        if index + 1 == len(sent) or llm.epoch_of[index + 1] != number:
            assert conversation == epoch, f"request {index} is not its epoch's last rendering"
    forms = {m.form for r in llm.requests for m in _conversation(r) if m.rendered_from}
    logged = _logged_forms(store, state)
    # The log holds each form as what it was cut from, and never its text.
    assert all(data.get("content") is None and data["rendered_from"] for data in logged)
    if keep_recent_turns == 1:
        # The summaries were sent, and the rebuild renders them in the summary form.
        assert "excerpt" in forms
        summaries = [
            m
            for r in llm.requests
            for m in _conversation(r)
            if message_form(m) is ContextForm.SUMMARY
        ]
        assert summaries
        assert any(
            message_form(m) is ContextForm.SUMMARY
            for c in rebuild_epoch_conversations(store, state)
            for m in c
        )
    elif over_cap:
        # Both forms were sent, rendered from the kept results.
        assert forms == {"excerpt", "stub"}
        assert {data["form"] for data in logged} == {"excerpt", "stub"}
    else:
        assert forms == set() and logged == []
    # The same holds for the requests as the record rebuilds them.
    _assert_epochs_render_from_the_log(store, state, llm.requests)


@pytest.mark.asyncio
async def test_within_an_epoch_each_request_extends_the_last_byte_for_byte(
    tmp_path: Path,
) -> None:
    """Within an epoch every request the provider gets -- the system turn and the
    conversation, as each message's JSON -- begins with the request before it, byte for
    byte, so a cached prefix stays valid (§5.8, Rule 1). That holds across the steps of a
    turn and across turns. A writer that hands the history an over-cap result's
    `excerpt` with text other than its kept body renders to is refused where it writes
    (#1971, item 3): the log keeps the form, not the text, so that text would never have
    been sent, and the writer is told so loudly instead (#1848).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if shown != clean.content:
    Becomes: if False:
    """
    llm = _EpochTaggingLLM([[_call("c1", "dump")], [_call("c2", "dump2")]])
    tools = [
        _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(1_500))),
        _returning("dump2", "\n".join(f"row {i:05d}: other" for i in range(1_500))),
        ToolResultReadTool(),
    ]
    agent, _store = _agent(tmp_path, llm, tools)
    llm.agent = agent
    await agent.start()
    assert (await agent.execute_turn("dump both")).is_completed
    live = agent._active_session  # pyright: ignore[reportPrivateUsage]
    history = live.messages
    at = [i for i, m in enumerate(history) if m.rendered_from is not None]
    assert len(at) == 2
    for index in at:
        other = history[index].model_copy(update={"content": "left by a writer"})
        with pytest.raises(FormTextMismatchError):
            live.replace(index, other, cause="test")
        with pytest.raises(FormTextMismatchError):
            live.append(other)
    assert live.messages == history
    assert (await agent.execute_turn("and now?")).is_completed
    assert (await agent.execute_turn("thanks")).is_completed

    # One epoch: nothing here may show the history differently.
    assert set(llm.epoch_of) == {0} and len(llm.requests) == 5
    wire = [_wire(r.messages) for r in llm.requests]
    assert wire[0][0] == _wire(llm.requests[0].messages[:1])[0]
    assert llm.requests[0].messages[0].role is MessageRole.SYSTEM
    for index, (before, after) in enumerate(zip(wire, wire[1:], strict=False)):
        assert after[: len(before)] == before, f"request {index + 1} rewrote request {index}"
    last = llm.requests[-1]
    for call_id in ("c1", "c2"):
        shown = _tool(last, call_id)
        assert shown.form == "excerpt" and shown.rendered_from is not None
        assert "left by a writer" not in (shown.content or "")


@pytest.mark.asyncio
async def test_a_request_whose_kept_result_is_gone_is_refused_in_plain_words(
    tmp_path: Path,
) -> None:
    """The request renders an `excerpt` from its kept result, so when that result is no
    longer in the store the turn stops, rather than sending text nothing can be read back
    from. The person is told plainly, with no handle, digest, path or exception name; the
    specifics are on the error's `detail` and `code`.

    The body is lost between turns, after the first turn read it: the next request reads
    its kept results again rather than serving them from memory (#1971, item 4).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: purpose="send",
    Becomes: purpose="rebuild",
    Killed by: src/uclone_x/agent/session_lifecycle.py :: live.results.clear()
    Becomes: pass
    """
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    output = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    agent, store = _agent(tmp_path, llm, [_returning("dump", output), ToolResultReadTool()])
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    agent.persist_session()
    handle = result_handle(output)
    state = store.load("sess_1443")
    assert state is not None
    kept = stored_result_entry(state.session_log, handle)
    assert kept is not None
    # The last request read the result and kept it (#1971).
    live = agent._active_session  # pyright: ignore[reportPrivateUsage]
    assert live.results.get(handle) == output
    (store.context_body_dir("sess_1443") / kept.digest).unlink()

    result = await agent.execute_turn("and now?")
    plain = "Part of this conversation's record is missing, so it cannot continue."
    assert not result.is_completed
    assert result.error == plain
    for internal in (handle, kept.digest, "tr_", "store", "Error", "/", "rebuilt"):
        assert internal not in (result.error or "")

    with pytest.raises(EpochRecordError) as raised:
        agent._prepare_turn_messages()  # pyright: ignore[reportPrivateUsage]
    assert raised.value.code == "body_missing"
    assert raised.value.detail == f"no context body {kept.digest}"
    assert str(raised.value) == plain


@pytest.mark.asyncio
async def test_a_kept_result_removed_during_a_turn_is_refused_at_its_next_request(
    tmp_path: Path,
) -> None:
    """A kept result read by one step of a turn and removed before the next is not sent
    from memory: each request reads its kept results again, so the next step stops with
    the plain copy instead of sending text the store no longer holds (#1971, item 7).

    Killed by: src/uclone_x/agent/session_lifecycle.py :: live.results.clear()
    Becomes: pass
    """
    output = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    handle = result_handle(output)
    lost: list[Path] = []

    async def lose(params: dict[str, Any], context: ToolContext) -> ToolResult:
        for path in lost:
            path.unlink()
        return ToolResult(
            success=True,
            output="ok",
            provenance=Provenance.primary(provider="local.test", model="lose"),
        )

    tools = [
        _returning("dump", output),
        LocalTool("lose", "Removes a kept result.", handler=lose, writes_files=False),
        ToolResultReadTool(),
    ]
    llm = _ScriptedLLM([[_call("c1", "dump")], [], [_call("c2", "lose")]])
    agent, store = _agent(tmp_path, llm, tools)
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    agent.persist_session()
    state = store.load("sess_1443")
    assert state is not None
    kept = stored_result_entry(state.session_log, handle)
    assert kept is not None
    lost.append(store.context_body_dir("sess_1443") / kept.digest)

    result = await agent.execute_turn("and now?")

    assert len(llm.requests) == 3  # the step after the removal was never sent
    assert not result.is_completed
    assert result.error == "Part of this conversation's record is missing, so it cannot continue."
    for internal in (handle, kept.digest, "tr_", "Error", "/"):
        assert internal not in (result.error or "")


@pytest.mark.asyncio
async def test_an_epoch_a_compaction_opens_is_named_for_it_after_a_restart(
    tmp_path: Path,
) -> None:
    """The compaction's cause is saved with its record, so a session restored before the
    next request opens its epoch as `compaction` and `restored`, not `restored` alone.
    Loading does not save `restored`: a record saved again before the request carries
    only the compaction, which the request then records and clears.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: epoch_causes=list(state.epoch_causes),
    Becomes: epoch_causes=[],
    Killed by: src/uclone_x/agent/session_lifecycle.py :: epoch_causes=tuple(c for c in self.epoch_causes if c != EPOCH_RESTORED),
    Becomes: epoch_causes=tuple(c for c in self.epoch_causes if True),
    """
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    tools = [_returning("dump", BODY), ToolResultReadTool()]
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path, llm, tools, compaction_threshold_tokens=3_000, compactor=compactor
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    agent.persist_session()
    await agent.compact_session()
    saved = store.load(sid)
    assert saved is not None and saved.epoch_causes == ("compaction",)

    again, _ = _agent(tmp_path, llm, tools, compaction_threshold_tokens=3_000)
    again.hydrate_session(sid)
    again.persist_session()
    resaved = store.load(sid)
    assert resaved is not None and resaved.epoch_causes == ("compaction",)
    await again.start()
    assert (await again.execute_turn("and now?")).is_completed
    again.persist_session()

    state = store.load(sid)
    assert state is not None
    assert state.context_epochs[-1].opened_by == ("compaction", "restored")
    assert state.epoch_causes == ()


# ======================================================================================
# The epoch in force alone (#1848)
# ======================================================================================


@pytest.mark.asyncio
async def test_a_request_is_read_from_the_last_epoch_not_an_earlier_one(tmp_path: Path) -> None:
    """A request's entries come from the epoch in force alone: an earlier epoch's record
    is never read. A saved record whose earlier epoch says a rendering shows some other
    entry still restores into a request that extends the last epoch, which is what the
    record says the history shows now.

    Killed by: src/uclone_x/core/context_state.py :: for shown in epochs[-1].entries:
    Becomes: for shown in (e for epoch in epochs for e in epoch.entries):
    """
    _agent_, store, llm = await _compact_twice(tmp_path)
    state = store.load("sess_1443")
    assert state is not None
    first, second, third = state.context_epochs
    stub = _shown_by_call(store, state, second)["c1"]
    assert stub.rendering is not None and _shown_by_call(store, state, third)["c1"] == stub
    # The earlier epoch, rewritten to say that rendering shows the first user message.
    other = first.entries[0].entry
    assert other != stub.entry
    rewritten = second.model_copy(
        update={
            "entries": tuple(
                shown.model_copy(update={"entry": other}) if shown == stub else shown
                for shown in second.entries
            )
        }
    )
    store.save(state.model_copy(update={"context_epochs": (first, rewritten, third)}))

    tools = [
        _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(1_500))),
        _returning("dump2", "\n".join(f"row {i:05d}: other" for i in range(1_500))),
        ToolResultReadTool(),
    ]
    again, _ = _agent(tmp_path, llm, tools)
    assert again.hydrate_session("sess_1443") is not None
    await again.start()
    assert (await again.execute_turn("and now?")).is_completed
    again.persist_session()

    after = store.load("sess_1443")
    assert after is not None
    *_, last = after.context_epochs
    assert len(after.context_epochs) == 3, "the restored request opened an epoch"
    assert last.entries[: len(third.entries)] == third.entries
    assert _shown_by_call(store, after, last)["c1"] == stub


@pytest.mark.asyncio
async def test_a_rollback_shows_the_history_as_the_checkpoints_epoch_did(tmp_path: Path) -> None:
    """A rollback puts the checkpoint's history back, so the next epoch opens from the
    epoch in force at the checkpoint: a stub whose rendering the undone turn's epoch no
    longer lists is still the entry it was, shown through that rendering, not a new
    entry of its own. Held across a save and a restart before the next request.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: self.compacted_entries = opening_entries(
    Becomes: self.compacted_entries = {} and opening_entries(
    Killed by: src/uclone_x/agent/session_lifecycle.py :: if list(log) == self.session_log[: len(log)]:
    Becomes: if False:
    """
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    tools = [
        _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(1_500))),
        _returning("dump2", "\n".join(f"row {i:05d}: other" for i in range(1_500))),
        ToolResultReadTool(),
    ]
    compactor = ContextCompactor(keep_recent_turns=50)
    agent, store = _agent(
        tmp_path, llm, tools, compaction_threshold_tokens=3_000, compactor=compactor
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    llm.steps.append([_call("c2", "dump2")])
    assert (await agent.execute_turn("and the other")).is_completed
    checkpoint = agent.checkpoint_turn()
    stub = {
        m.tool_call_id: shown
        for shown, m in zip(
            checkpoint.context_epochs[-1].entries,
            _conversation(llm.requests[-1]),
            strict=True,
        )
    }["c1"]
    assert stub.form is ContextForm.STUB and stub.rendering is not None

    # The undone turn summarizes everything but its own turn, so its epoch lists the
    # ledger and not the stub.
    compactor.keep_recent_turns = 1
    await agent.compact_session()
    assert (await agent.execute_turn("undo me")).is_completed
    undone = agent.get_session().context_epochs[-1]
    assert stub.rendering not in {shown.body for shown in undone.entries}
    agent.roll_back_turn(checkpoint, reason="test")
    compactor.keep_recent_turns = 50
    agent.persist_session()

    again, _ = _agent(tmp_path, llm, tools, compaction_threshold_tokens=3_000, compactor=compactor)
    again.hydrate_session(sid)
    await again.start()
    assert (await again.execute_turn("instead")).is_completed
    again.persist_session()

    state = store.load(sid)
    assert state is not None
    shown = _shown_by_call(store, state, state.context_epochs[-1])
    assert (shown["c1"].entry, shown["c1"].rendering) == (stub.entry, stub.rendering)
    _assert_epochs_render_from_the_log(store, state, llm.requests)


@pytest.mark.asyncio
async def test_a_rollback_puts_back_the_checkpoints_entries_by_identity(tmp_path: Path) -> None:
    """The undone turn compacted, so the history it leaves holds the first result as a
    stub where the checkpoint held it as an excerpt. The rollback puts back the very
    entries the checkpoint named -- the excerpt's entry, not a new one logged for its
    text -- and logs nothing (#1974, item 12): the log only grows, so the checkpoint's
    log is a prefix of this one, entry for entry, and its history is its entries. The
    event says the prefix was rewritten, decided by entry, not by comparing text.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if list(log) == self.session_log[: len(log)]:
    Becomes: if False:
    Killed by: src/uclone_x/agent/session_lifecycle.py :: prefix_rewritten = entries[:kept] != live.entries_of(checkpoint)
    Becomes: prefix_rewritten = False
    """
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    tools = [
        _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(1_500))),
        ToolResultReadTool(),
    ]
    agent, store = _agent(
        tmp_path,
        llm,
        tools,
        compaction_threshold_tokens=3_000,
        compactor=ContextCompactor(keep_recent_turns=50),
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    checkpoint = agent.checkpoint_turn()
    assert checkpoint.history_entries
    assert (await agent.execute_turn("undo me")).is_completed
    undone = agent.get_session()
    assert [e.opened_by for e in undone.context_epochs][-1] == ("compaction",)
    logged = len(undone.session_log)

    agent.roll_back_turn(checkpoint, reason="test")

    live = agent._active_session  # pyright: ignore[reportPrivateUsage]
    assert live.entry_ids() == list(checkpoint.history_entries)
    assert len(live.session_log) == logged
    assert live.recorded == checkpoint.messages
    agent.persist_session()
    log = store.event_log_path(sid)
    assert log is not None
    (event,) = [e for e in read_session_log(log) if e.get("type") == "TURN_ROLLED_BACK"]
    assert event["prefix_rewritten"] is True


@pytest.mark.asyncio
async def test_the_history_is_the_last_epoch_and_what_was_logged_since(tmp_path: Path) -> None:
    """The session holds no list of its history (#1848): it is the entries the last
    request showed -- the anchor, then the last epoch's -- and every message logged
    since. A record names each message's entry, and a restart adopts those entries. A
    request that showed anything other than the history's last entries fails there,
    rather than being recorded as the epoch the history is then read from.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if start < 0 or entries[start:] != [each.body for each in shown]:
    Becomes: if False:
    Killed by: src/uclone_x/agent/session_lifecycle.py :: entries.extend(shown.body for shown in self.context_epochs[-1].entries)
    Becomes: entries.extend(shown.entry for shown in self.context_epochs[-1].entries)
    """
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    tools = [
        _returning("dump", "\n".join(f"line {i:05d}: payload" for i in range(1_500))),
        ToolResultReadTool(),
    ]
    agent, store = _agent(
        tmp_path,
        llm,
        tools,
        compaction_threshold_tokens=3_000,
        compactor=ContextCompactor(keep_recent_turns=50),
    )
    await agent.start()
    assert (await agent.execute_turn("dump it")).is_completed
    # The second turn compacts at its start, so its epoch shows a stub through a
    # rendering: an entry whose body is not the entry itself.
    assert (await agent.execute_turn("and again")).is_completed
    live = agent._active_session  # pyright: ignore[reportPrivateUsage]
    assert not hasattr(live, "history")
    epoch = live.context_epochs[-1]
    assert any(shown.rendering is not None for shown in epoch.entries)
    since = [e.id for e in live.session_log[live.mark :] if not is_kept_text(e)]
    assert since, "the turn's answer is logged after its last request"
    assert live.entry_ids() == [*live.head, *(shown.body for shown in epoch.entries), *since]
    assert [m.content for m in live.messages][-1] == "done"

    agent.persist_session()
    state = store.load("sess_1443")
    assert state is not None
    assert state.history_entries == tuple(live.entry_ids())
    again, _ = _agent(tmp_path, llm, tools, compaction_threshold_tokens=3_000)
    again.hydrate_session("sess_1443")
    restored = again._active_session  # pyright: ignore[reportPrivateUsage]
    assert restored.entry_ids() == list(state.history_entries)
    assert restored.messages == live.messages

    with pytest.raises(RuntimeError):
        live.record_shown(list(epoch.entries[:-1]), step=1)


def test_a_record_whose_message_is_not_its_entrys_body_is_refused_plainly() -> None:
    """A record names the log entry of each message; one whose message is not that
    entry's body is refused when adopted, in the plain words every unreadable record
    gets, with the entry named on `detail` only -- never matched to some other entry.

    Killed by: src/uclone_x/agent/session_lifecycle.py :: if self._entry(entry_id).digest != item.digest:
    Becomes: if False:
    """
    first, second = (
        ChatMessage(role=MessageRole.USER, content="one"),
        ChatMessage(role=MessageRole.USER, content="two"),
    )
    log = tuple(
        new_entry(i, logged_message(m), turn=1, provenance=SessionLogProvenance.RECORDED)
        for i, m in enumerate((first, second))
    )
    good = SessionState(
        session_id="s", agent_id="a", messages=(first,), session_log=log, history_entries=("e0",)
    )
    assert _LiveSession.from_state(
        good, anchor_provenance=AnchorWriter.CALLER, load_body=lambda _digest: None
    ).entry_ids() == ["e0"]
    wrong = good.model_copy(update={"history_entries": ("e1",)})

    with pytest.raises(EpochRecordError) as raised:
        _LiveSession.from_state(
            wrong, anchor_provenance=AnchorWriter.CALLER, load_body=lambda _digest: None
        )

    assert str(raised.value) == (
        "Part of this conversation's record could not be read, so it cannot continue."
    )
    assert raised.value.code == "unreadable"
    assert "e1" in str(raised.value.detail)
    for internal in _INTERNALS:
        assert internal not in str(raised.value)


def _record_naming(tmp_path: Path, entries: Sequence[str]) -> tuple[SessionStore, Path, bytes]:
    """A saved record of two messages, "one" and "two", logged as `e0` and `e1` beside a
    kept text `e2`, that names `entries` as its messages' log entries -- written as a
    document, since the model refuses to build most of the records this names (#1985).
    It is at revision 2, as a record a session has saved is."""
    one, two = (
        ChatMessage(role=MessageRole.USER, content="one"),
        ChatMessage(role=MessageRole.USER, content="two"),
    )
    log = (
        new_entry(0, logged_message(one), turn=1, provenance=SessionLogProvenance.RECORDED),
        new_entry(1, logged_message(two), turn=1, provenance=SessionLogProvenance.RECORDED),
        new_entry(
            2,
            logged_text(SessionLogKind.TOOL_RESULT, BODY, blob=result_handle(BODY)),
            turn=1,
            provenance=SessionLogProvenance.RECORDED,
        ),
    )
    document = json.loads(
        SessionState(
            session_id="s",
            agent_id="a",
            messages=(one, two),
            session_log=log,
            history_entries=("e0", "e1"),
            revision=2,
        ).model_dump_json()
    )
    document["history_entries"] = list(entries)
    document["messages"] = [one.model_dump(mode="json"), two.model_dump(mode="json")]
    store = SessionStore(tmp_path / "store")
    for m in (one, two):
        item = logged_message(m)
        store.save_context_body("s", item.digest, item.body)
    path = store.session_path("s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return store, path, path.read_bytes()


def _refused_on_load_and_set_aside(
    tmp_path: Path, entries: Sequence[str], caplog: pytest.LogCaptureFixture
) -> str:
    """Load the record naming `entries` the way a resume does, and save over it; what the
    store logged as the cause. The record is read as absent, the resume starts afresh,
    the save keeps the record aside unchanged, and the person is told so once (#1844)."""
    store, path, written = _record_naming(tmp_path, entries)
    with caplog.at_level(logging.WARNING, logger="uclone_x.agent.session"):
        assert store.load("s") is None
    cause = caplog.text
    agent, agents_store = _agent(tmp_path, _ScriptedLLM([]), [], sid="s")
    assert agent.hydrate_session("s") is None
    agent.persist_session("s")
    (aside,) = [p for p in path.parent.iterdir() if ".unreadable-" in p.name]
    assert aside.read_bytes() == written
    assert agents_store.take_set_aside("s") is True
    return cause


def test_a_record_whose_message_is_not_its_entry_is_set_aside_not_refused_on_every_resume(
    tmp_path: Path,
) -> None:
    """A record whose entries are well formed but whose message is not the body of the
    entry it names was read by the store and refused only where it was adopted -- on
    every resume, and never set aside, so the session could never be opened again. The
    store now reads it as unreadable, so the resume starts afresh, the next save keeps
    the record aside unchanged, and the session after that loads (#1985).

    Killed by: src/uclone_x/agent/session.py :: mismatch = history_entries_not_their_messages(state)
    Becomes: mismatch = None
    """
    store, path, written = _record_naming(tmp_path, ("e1", "e0"))

    assert store.load("s") is None
    agent, agents_store = _agent(tmp_path, _ScriptedLLM([]), [], sid="s")
    assert agent.hydrate_session("s") is None
    assert agent.hydrate_session("s") is None  # not an error on the second resume either
    agent.persist_session("s")

    (aside,) = [p for p in path.parent.iterdir() if ".unreadable-" in p.name]
    assert aside.read_bytes() == written
    assert agents_store.take_set_aside("s") is True
    assert agent.hydrate_session("s") is not None  # the session opens again


def test_a_record_naming_fewer_entries_than_messages_is_refused_on_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: src/uclone_x/core/session_state.py :: if self.messages and len(entries) != len(self.messages):
    Becomes: if False:
    """
    cause = _refused_on_load_and_set_aside(tmp_path, ("e0",), caplog)

    assert "names 1 log entries for 2 messages" in cause


def test_a_record_naming_an_entry_not_in_its_log_is_refused_on_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: src/uclone_x/core/session_state.py :: if index is None or index >= len(self.session_log):
    Becomes: if False:
    """
    cause = _refused_on_load_and_set_aside(tmp_path, ("e0", "e7"), caplog)

    assert "'e7', not in the log" in cause


def test_a_record_naming_a_kept_text_as_a_message_is_refused_on_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refused by structure, before any message is hashed: the cause says it is a kept
    text, not that the message differs from its body.

    Killed by: src/uclone_x/core/session_state.py :: if is_kept_text(self.session_log[index]):
    Becomes: if False:
    """
    cause = _refused_on_load_and_set_aside(tmp_path, ("e0", "e2"), caplog)

    assert "'e2', a kept text, not a message" in cause


def test_a_state_with_messages_replaced_names_their_entries_so_it_loads(tmp_path: Path) -> None:
    """`with_messages` names each new message's log entry when its log holds all of them.
    Dropping them, it made a state that saved with a log and no entries, which the store
    then set aside on the next load as a record from before #1848 (#1985).

    Killed by: src/uclone_x/core/session_state.py :: history_entries=_logged_entries(self.session_log, replacement),
    Becomes: history_entries=(),
    """
    store, _path, _written = _record_naming(tmp_path, ("e0", "e1"))
    loaded = store.load("s")
    assert loaded is not None

    kept = store.save(loaded.with_messages(loaded.messages[:1]))

    assert kept.history_entries == ("e0",)
    reloaded = store.load("s")
    assert reloaded is not None
    assert reloaded.messages == loaded.messages[:1]
    # A message the log does not hold names none; the live session logs it on adoption.
    novel = loaded.with_messages([ChatMessage(role=MessageRole.USER, content="three")])
    assert novel.history_entries == ()


def test_a_record_without_epoch_causes_is_written_as_before() -> None:
    """`epoch_causes` is left out of a record that has none, so a record an older build
    reads has no key it does not know (#1844); one that has them keeps them.

    Killed by: src/uclone_x/core/session_state.py :: data.pop("epoch_causes", None)
    Becomes: pass
    """
    state = SessionState(session_id="s", agent_id="a")
    assert "epoch_causes" not in state.model_dump(mode="json")
    assert "epoch_causes" not in state.model_dump_json()
    # A record written without the key, as before the field, reads back with none.
    assert SessionState.model_validate_json(state.model_dump_json()).epoch_causes == ()
    pending = SessionState(session_id="s", agent_id="a", epoch_causes=("compaction",))
    assert SessionState.model_validate_json(pending.model_dump_json()) == pending
    assert pending.with_plan(None).epoch_causes == ("compaction",)


def test_a_record_without_compacted_entries_is_written_as_before() -> None:
    """`compacted_entries` is left out of a record that has none, so a record an older
    build reads has no key it does not know (#1844); one that has them keeps them.

    Killed by: src/uclone_x/core/session_state.py :: data.pop("compacted_entries", None)
    Becomes: pass
    """
    state = SessionState(session_id="s", agent_id="a")
    assert "compacted_entries" not in state.model_dump(mode="json")
    assert "compacted_entries" not in state.model_dump_json()
    shown = ContextEntry(entry="e1", form=ContextForm.STUB, rendering="e4")
    pending = SessionState(session_id="s", agent_id="a", compacted_entries=(shown,))
    assert SessionState.model_validate_json(pending.model_dump_json()) == pending
    assert pending.with_plan(None).compacted_entries == (shown,)


def test_an_entry_renders_from_its_rendering_body_not_its_own() -> None:
    """An entry a compaction dropped to a stub reads the stub's log body; its own body --
    the excerpt the history held before -- is not what the request shows.

    Killed by: src/uclone_x/core/context_state.py :: message = message_of(shown.body)
    Becomes: message = message_of(shown.entry)
    """
    handle = result_handle(BODY)
    excerpt = ChatMessage(
        role=MessageRole.TOOL,
        name="t",
        tool_call_id="c1",
        form="excerpt",
        rendered_from=RenderedFrom(handle=handle, limit=1_000, readable=True),
    )
    stub = excerpt.model_copy(
        update={
            "form": "stub",
            "rendered_from": RenderedFrom(handle=handle, limit=40, readable=True),
        }
    )
    bodies = {"e1": excerpt, "e4": stub}
    results = {handle: BODY}

    shown = ContextEntry(entry="e1", form=ContextForm.STUB, rendering="e4")

    assert shown.body == "e4"
    assert render_entries([shown], bodies.__getitem__, results.__getitem__) == [
        stub.model_copy(update={"content": stored_result_stub(handle, BODY, keep_chars=40)})
    ]
    assert render_entries(
        [_entry("e1", ContextForm.EXCERPT)], bodies.__getitem__, results.__getitem__
    ) == [
        excerpt.model_copy(update={"content": excerpt_tool_result(BODY, handle, cap_bytes=1_000)})
    ]


def test_an_entry_without_a_rendering_is_written_as_before() -> None:
    """`rendering` is left out of an entry that has none, so a record an older build
    reads has no field it does not know (#1844); one that has it keeps it.

    Killed by: src/uclone_x/core/context_state.py :: data.pop("rendering", None)
    Becomes: pass
    """
    assert _entry("e1").model_dump() == {"entry": "e1", "form": "full", "same_as": None}
    assert "rendering" not in _entry("e1").model_dump_json()
    rendered = ContextEntry(entry="e1", form=ContextForm.STUB, rendering="e4")
    assert rendered.model_dump()["rendering"] == "e4"
    assert ContextEntry.model_validate_json(rendered.model_dump_json()) == rendered


def test_a_compactions_entries_name_the_entry_each_message_shows() -> None:
    """Per compacted message, the entry it shows: a pruned message is the entry it
    replaced, rendered by its own body; a kept one and the ledger are their own entry.

    Killed by: src/uclone_x/core/context_state.py :: shows.append(None if pruned is ContextForm.FULL else previous.entry)
    Becomes: shows.append(None)
    """
    user = ChatMessage(role=MessageRole.USER, content="go")
    full = ChatMessage(role=MessageRole.TOOL, content=BODY, name="t", tool_call_id="c2")
    ledger = ChatMessage(role=MessageRole.SYSTEM, content="ledger", compaction_ledger=True)
    pruned = full.model_copy(update={"content": "[stored]", "form": "stub"})
    derived = derive_compacted_forms(
        [(_entry("e0"), user), (_entry("e2"), full)], [ledger, user, pruned], [None, 0, 1]
    )

    entries = compacted_entries(["e5", "e0", "e6"], derived)

    assert entries == {
        "e5": ContextEntry(entry="e5", form=ContextForm.SUMMARY),
        "e0": ContextEntry(entry="e0", form=ContextForm.FULL),
        "e6": ContextEntry(entry="e2", form=ContextForm.STUB, rendering="e6"),
    }
    with pytest.raises(ValueError):
        compacted_entries(["e5", "e0"], derived)


def test_a_rendering_carries_over_and_is_not_its_own_bodys_form() -> None:
    """A later epoch shows a rendering as the entry an earlier one recorded it for. The
    rendering's form is the entry's, not a form of the rendering's own body: an epoch that
    listed it is no record that the body is a stub when it is next its own entry.

    Killed by: src/uclone_x/core/context_state.py :: renders.setdefault(shown.rendering, shown.model_copy(update={"same_as": None}))
    Becomes: renders.setdefault(shown.rendering, shown)
    """
    shown = ContextEntry(entry="e1", form=ContextForm.STUB, rendering="e4", same_as="e0")
    epoch = ContextEpoch(
        number=1, turn=2, step=1, opened_by=("compaction",), entries=(_entry("e0"), shown)
    )

    assert opening_entries([epoch], {}) == {"e4": shown.model_copy(update={"same_as": None})}


# ======================================================================================
# Kinds: retrieval, sub-agent and memory entries (#1849)
# ======================================================================================


def test_the_kind_tool_names_are_the_tools_names() -> None:
    """The names `log_kind` sorts by are the names the tools register under, so renaming
    a tool cannot silently turn its results back into plain `tool_result` entries."""
    assert {SubagentDelegationTool.name} == SUBAGENT_TOOLS
    assert {QueryMemoryFactsTool.name, WebSearchTool.name, FileSearchTool.name} == RETRIEVAL_TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "kind"),
    [
        ("web_search", SessionLogKind.RETRIEVAL),
        ("delegate_subagent", SessionLogKind.SUBAGENT),
        ("read", SessionLogKind.TOOL_RESULT),
    ],
)
async def test_retrieval_and_subagent_results_are_logged_as_their_kind_and_back_referenced(
    tmp_path: Path, tool_name: str, kind: SessionLogKind
) -> None:
    """A retrieval hit and a sub-agent's result are their own kinds of entry, and Rule 2's
    back-reference covers them as it covers any tool result.

    Killed by: src/uclone_x/core/session_log.py :: if tool_name in SUBAGENT_TOOLS:
    Becomes: if tool_name in ():
    Killed by: src/uclone_x/core/session_log.py :: if tool_name in RETRIEVAL_TOOLS:
    Becomes: if tool_name in ():
    """
    llm = _ScriptedLLM([[_call(f"c{n}", tool_name)] for n in range(2)])
    agent, store = _agent(tmp_path, llm, [_returning(tool_name, BODY)])
    await agent.start()
    assert (await agent.execute_turn("look it up")).is_completed
    agent.persist_session()

    last = llm.requests[-1]
    assert _tool(last, "c1").content == back_reference_text(_tool(last, "c0"))
    state = store.load("sess_1443")
    assert state is not None
    (epoch,) = state.context_epochs
    results = [x for x in epoch.entries if state.session_log[int(x.entry[1:])].kind is kind]
    assert [x.same_as for x in results] == [None, results[0].entry]
    _assert_epochs_render_from_the_log(store, state, llm.requests)


def _memory_with_a_fact(object_value: str = "postgres 16") -> CrossSessionMemory:
    memory = CrossSessionMemory(max_facts_in_prompt=5)
    seed_fact(
        memory,
        MemoryFact(
            fact_id="mem_db",
            subject="service",
            predicate="database",
            object_value=object_value,
            provenance=Provenance.primary(provider="agent.test", model="memory"),
            source_session_id="sess_a",
            confidence=0.9,
            created_at="2026-09-20T00:00:00+00:00",
        ),
    )
    return memory


async def _run_recalling_turn(
    tmp_path: Path, object_value: str = "postgres 16"
) -> tuple[_ScriptedLLM, SessionStore]:
    llm = _ScriptedLLM([[_call("c0", "read")]])
    agent, store = _agent(
        tmp_path, llm, [_returning("read", "short")], memory=_memory_with_a_fact(object_value)
    )
    await agent.start()
    assert (await agent.execute_turn("which database does the service use?")).is_completed
    agent.persist_session()
    return llm, store


@pytest.mark.asyncio
async def test_recalled_memory_is_logged_once_per_turn_and_read_back_by_its_handle(
    tmp_path: Path,
) -> None:
    """The section a turn recalled is a `memory` entry whose `tr_` handle
    `tool_result_read` reads back after the turn.

    Killed by: src/uclone_x/agent/turn_executor.py :: turn_live.log_entry(logged_text(SessionLogKind.MEMORY, section, blob=handle))
    Becomes: None
    Killed by: src/uclone_x/agent/turn_executor.py :: handle = result_handle(redact_credentials(section))
    Becomes: handle = None
    """
    llm, store = await _run_recalling_turn(tmp_path)

    tail = llm.requests[0].messages[-1].content or ""
    assert "postgres 16" in tail
    state = store.load("sess_1443")
    assert state is not None
    (memory_entry,) = [e for e in state.session_log if e.kind is SessionLogKind.MEMORY]
    assert memory_entry.turn == 1
    assert memory_entry.provenance is SessionLogProvenance.RECORDED
    section = store.load_context_body("sess_1443", memory_entry.digest)
    assert section is not None and "- service: database -> postgres 16" in section
    assert section in tail
    handle = memory_entry.blob
    assert handle is not None and handle.startswith("tr_")
    # Read back from the store alone, by an agent that holds only what was saved.
    fresh, _ = _agent(tmp_path, _ScriptedLLM([]), [])
    fresh.hydrate_session("sess_1443")
    bodies = fresh._result_bodies("sess_1443")  # pyright: ignore[reportPrivateUsage]
    assert bodies.read(handle) == section
    page = await ToolResultReadTool().execute(
        {"handle": handle, "offset": 0, "length": 10_000},
        ToolContext(agent_id="ctx", session_id="sess_1443", trace_id="t", stored_results=bodies),
    )
    assert page.success and isinstance(page.output, str) and section in page.output
    # Its entry is not a message of the conversation: no epoch shows it.
    assert all(x.entry != memory_entry.id for e in state.context_epochs for x in e.entries)
    _assert_epochs_render_from_the_log(store, state, llm.requests)


@pytest.mark.asyncio
async def test_a_recalled_memory_handle_names_the_redacted_section(tmp_path: Path) -> None:
    """A credential in a recalled fact is redacted before the handle is taken, so the
    handle names the body the log keeps and reads back (#1848).

    Killed by: src/uclone_x/agent/turn_executor.py :: handle = result_handle(redact_credentials(section))
    Becomes: handle = result_handle(section)
    """
    secret = "sk-1234567890123456789012345678901234567890"
    _llm, store = await _run_recalling_turn(tmp_path, object_value=f"postgres 16 with key {secret}")
    state = store.load("sess_1443")
    assert state is not None
    (memory_entry,) = [e for e in state.session_log if e.kind is SessionLogKind.MEMORY]
    body = store.load_context_body("sess_1443", memory_entry.digest)
    assert body is not None and secret not in body and "postgres 16" in body
    assert memory_entry.blob is not None
    assert memory_entry.blob == result_handle(body)
    fresh, _ = _agent(tmp_path, _ScriptedLLM([]), [])
    fresh.hydrate_session("sess_1443")
    bodies = fresh._result_bodies("sess_1443")  # pyright: ignore[reportPrivateUsage]
    assert bodies.read(memory_entry.blob) == body


@pytest.mark.asyncio
async def test_logging_recalled_memory_leaves_every_request_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requests of a turn that logs its recalled memory are byte-for-byte the requests
    of the same turn without the logging: the prefix, the tail and the conversation."""
    logged, _ = await _run_recalling_turn(tmp_path / "logged")

    def no_log(self: TurnExecutor, turn_live: object) -> None:
        return None

    monkeypatch.setattr(TurnExecutor, "_log_recalled_memory", no_log)
    silent, _ = await _run_recalling_turn(tmp_path / "silent")

    assert len(logged.requests) == len(silent.requests) == 2
    for with_log, without in zip(logged.requests, silent.requests, strict=True):
        # The two runs differ only in their workspace, which the system turn names.
        sent = with_log.model_dump_json().replace(str(tmp_path / "logged"), "<ws>")
        assert sent == without.model_dump_json().replace(str(tmp_path / "silent"), "<ws>")


def test_a_logged_text_body_and_its_digest_carry_no_credential() -> None:
    """A text logged outside the history, such as recalled memory, is stored redacted, and
    its digest is that of the redacted body, so no credential reaches the body store.

    Killed by: src/uclone_x/core/session_log.py :: body = redact_credentials(text)
    Becomes: body = text
    """
    secret = "abcdefghijklmnopqrstuvwx0123"
    text = f"- deploy: config -> api_key={secret}"

    rendered = logged_text(SessionLogKind.MEMORY, text, blob=None)

    assert secret not in rendered.body
    assert rendered.body.startswith("- deploy: config -> api_key=")
    assert rendered.digest == hashlib.sha256(rendered.body.encode("utf-8")).hexdigest()
    assert rendered.digest != hashlib.sha256(text.encode("utf-8")).hexdigest()
