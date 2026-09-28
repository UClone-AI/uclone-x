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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.hooks import BaseHook, HookAction, HookContext, HookDecision
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig, AgentState
from uclone_x.agent.request_record import (
    RequestRecordError,
    rebuild_epoch_conversations,
    rebuild_requests,
)
from uclone_x.agent.session import SessionState, SessionStore
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
    recorded_forms,
    recorded_renderings,
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
    logged_message,
    logged_text,
    new_entry,
)
from uclone_x.core.tool_results import (
    artifacts_dir_for,
    excerpt_tool_result,
    handle_in,
    load_tool_result,
    store_tool_result,
    stored_result_stub,
)
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
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

    Killed by: src/uclone_x/agent/base.py :: if content == msg.content:
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
        assert after["kept_message_count"] == before["kept_message_count"] + len(
            before["appended_messages"]
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
    """Killed by: src/uclone_x/agent/session_lifecycle.py :: live.declare_new_epoch("rollback")
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
    compactor = ContextCompactor(keep_recent_turns=50, workspace_root=tmp_path, session_id=sid)
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
    assert load_tool_result(artifacts_dir_for(tmp_path), sid, handle) == body
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

    Killed by: src/uclone_x/agent/turn_executor.py :: if self._active_session.shown_in_epoch(index):
    Becomes: if True:
    """
    body = "\n".join(f"line {i:05d}: payload" for i in range(1_500))
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    compactor = ContextCompactor(keep_recent_turns=50, workspace_root=tmp_path, session_id=sid)
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
            turn=None,
            digest=digest,
            size=1,
            provenance=SessionLogProvenance.RECORDED,
        )

    log = [entry(0, a), entry(1, b), entry(2, a), entry(3, a)]
    assert history_entry_ids(log, [a, b, a]) == ["e2", "e1", "e3"]
    assert history_entry_ids(log, [b, b]) == ["e1", None]


# ======================================================================================
# Records from before forms were recorded on messages (#1852 era; #1866)
# ======================================================================================


def _legacy_record(
    store: SessionStore, sid: str, *, forms: Sequence[ContextForm] | None = None
) -> tuple[SessionState, list[ChatMessage]]:
    """A saved session as the #1852 build left it: an excerpt and a stub written with no
    form on the message, and an epoch whose forms were read from their headers."""
    handle = "tr_0123456789abcdef"
    messages = [
        ChatMessage(role=MessageRole.USER, content="read both"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(_call("c1", "t"), _call("c2", "t"))),
        ChatMessage(
            role=MessageRole.TOOL,
            content=excerpt_tool_result(BODY, handle, cap_bytes=1_000),
            tool_call_id="c1",
            name="t",
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content=stored_result_stub(handle, BODY, keep_chars=40),
            tool_call_id="c2",
            name="t",
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="done"),
    ]
    log: list[SessionLogEntry] = []
    for position, message in enumerate(messages):
        item = logged_message(message)
        store.save_context_body(sid, item.digest, item.body)
        log.append(new_entry(position, item, turn=1, provenance=SessionLogProvenance.RECORDED))
    read_from_headers = forms or (
        ContextForm.FULL,
        ContextForm.FULL,
        ContextForm.EXCERPT,
        ContextForm.STUB,
        ContextForm.FULL,
    )
    epoch = ContextEpoch(
        number=0,
        turn=1,
        step=1,
        opened_by=("start",),
        entries=tuple(_entry(e.id, f) for e, f in zip(log, read_from_headers, strict=True)),
    )
    state = SessionState(
        session_id=sid,
        agent_id="ctx",
        messages=tuple(messages),
        turn_counter=1,
        session_log=tuple(log),
        context_epochs=(epoch,),
    )
    return store.save(state), messages


def test_an_epoch_recorded_from_headers_rebuilds_without_raising(tmp_path: Path) -> None:
    """An epoch of the #1852 build says `excerpt` and `stub` for tool results that carry
    no form; rebuilt from the log, it renders as it was sent instead of refusing the
    record (#1866). A message that does carry a form must still match its entry.

    Killed by: src/uclone_x/core/context_state.py :: if shown_form(message, shown.form) is not shown.form:
    Becomes: if message_form(message) is not shown.form:
    """
    store = SessionStore(tmp_path / "store")
    state, messages = _legacy_record(store, "sess_legacy")

    assert rebuild_epoch_conversations(store, state) == [messages]

    # A form on the message is still checked: a recorded excerpt listed as a stub is
    # a record that does not match its log, and is refused as one.
    shortened = messages[2].model_copy(update={"form": "excerpt"})
    item = logged_message(shortened)
    store.save_context_body("sess_legacy", item.digest, item.body)
    log = list(state.session_log)
    log[2] = log[2].model_copy(update={"digest": item.digest})
    epoch = state.context_epochs[0]
    wrong = list(epoch.entries)
    wrong[2] = _entry(wrong[2].entry, ContextForm.STUB)
    broken = state.model_copy(
        update={
            "session_log": tuple(log),
            "context_epochs": (epoch.model_copy(update={"entries": tuple(wrong)}),),
        }
    )
    with pytest.raises(RequestRecordError):
        rebuild_epoch_conversations(store, broken)


@pytest.mark.asyncio
async def test_a_restored_legacy_stub_keeps_the_form_its_epoch_recorded(tmp_path: Path) -> None:
    """A stub written before #1854 carries no form. The next request after a restart
    shows it as the `stub` its earlier epoch recorded, not as `full` (#1866): a carried
    entry's form is derived from the epochs before it, so it never rises.

    Killed by: src/uclone_x/core/context_state.py :: known.get(body) or ContextEntry(entry=body, form=shown_form(message, recorded.get(body)))
    Becomes: known.get(body) or ContextEntry(entry=body, form=shown_form(message, None))
    """
    store = SessionStore(tmp_path / "store")
    _, legacy = _legacy_record(store, "sess_1443")
    llm = _ScriptedLLM([])
    agent, _ = _agent(tmp_path, llm, [])
    agent.hydrate_session("sess_1443")
    await agent.start()
    assert (await agent.execute_turn("and now?")).is_completed
    agent.persist_session()

    state = store.load("sess_1443")
    assert state is not None
    last = state.context_epochs[-1]
    forms = {shown.entry: shown.form for shown in last.entries}
    assert (forms["e2"], forms["e3"]) == (ContextForm.EXCERPT, ContextForm.STUB)
    # The restored request only appended to the #1852 epoch: nothing it showed changed.
    assert [e.opened_by for e in state.context_epochs] == [("start",)]
    (rebuilt,) = rebuild_epoch_conversations(store, state)
    assert rebuilt[: len(legacy)] == legacy
    assert [m.content for m in _conversation(llm.requests[-1])[: len(legacy)]] == [
        m.content for m in legacy
    ]


@pytest.mark.asyncio
async def test_a_legacy_stub_that_its_stored_body_renders_to_is_recorded_as_a_stub(
    tmp_path: Path,
) -> None:
    """A stub written before #1854 with no epoch to say what it is: compaction re-renders
    the stub from the body its handle names, and when that is exactly the message's text,
    records it as a stub rather than leaving it `full` (#1866). The text is compared with
    a rendering, not parsed, so a file that only quotes a stub stays `full`.

    Killed by: src/uclone_x/llm/compactor.py :: if stub == msg.content and msg.form is None:
    Becomes: if False:
    """
    sid = "sess_legacy"
    compactor = ContextCompactor(workspace_root=tmp_path, session_id=sid)
    handle = store_tool_result(artifacts_dir_for(tmp_path), sid, BODY)
    stub_text = stored_result_stub(handle, BODY, keep_chars=compactor.max_tool_output_chars // 2)
    quoted = stored_result_stub("tr_0123456789abcdef", BODY, keep_chars=40)
    messages = [
        ChatMessage(role=MessageRole.USER, content="go"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(_call("c1", "t"), _call("c2", "t"))),
        ChatMessage(role=MessageRole.TOOL, content=stub_text, tool_call_id="c1", name="t"),
        ChatMessage(role=MessageRole.TOOL, content=quoted, tool_call_id="c2", name="t"),
    ]

    outcome = await compactor.compact(messages)

    by_call = {m.tool_call_id: m for m in outcome.messages if m.role is MessageRole.TOOL}
    assert by_call["c1"].content == stub_text
    assert message_form(by_call["c1"]) is ContextForm.STUB
    assert message_form(by_call["c2"]) is ContextForm.FULL


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


@pytest.mark.asyncio
async def test_a_compaction_keeps_a_stub_it_would_widen_into_an_excerpt(tmp_path: Path) -> None:
    """A compactor with no session truncates a long stub to a head and tail, which it
    records as an `excerpt` -- a form above the stub's, and a cut that can drop the handle
    the stub names (Rule 4). Derived from the history before it, the new epoch refuses the
    rise: the stub is kept as it was, and the requests still rebuild from the log.

    Killed by: src/uclone_x/agent/compaction_driver.py :: if derived is not None and derived.rising:
    Becomes: if False:
    """
    sid = "sess_1443"
    store = SessionStore(tmp_path / "store")
    stub = ChatMessage(
        role=MessageRole.TOOL,
        content=stored_result_stub("tr_0123456789abcdef", BODY, keep_chars=2_000),
        name="t",
        tool_call_id="c1",
        form="stub",
    )
    history = [
        ChatMessage(role=MessageRole.USER, content="read it"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(_call("c1", "t"),)),
        stub,
        ChatMessage(role=MessageRole.ASSISTANT, content="done"),
    ]
    store.save(SessionState(session_id=sid, agent_id="ctx", messages=tuple(history)))
    llm = _ScriptedLLM([])
    compactor = ContextCompactor(keep_recent_turns=50, max_tool_output_chars=200)
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
    compactor = ContextCompactor(keep_recent_turns=50, workspace_root=tmp_path, session_id=sid)
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
    assert message(stub.rendering) == _tool(llm.requests[-1], "c1")
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
    sid = "sess_1443"
    llm = _ScriptedLLM([[_call("c1", "dump")]])
    compactor = ContextCompactor(keep_recent_turns=50, workspace_root=tmp_path, session_id=sid)
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
    it: the second derives what the history showed from the renderings earlier epochs
    recorded, so it does not read the stub's body as an entry of its own.

    Killed by: src/uclone_x/agent/compaction_driver.py :: {**recorded_renderings(epochs), **live.compacted_entries},
    Becomes: {**live.compacted_entries},
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

    Killed by: src/uclone_x/agent/session_lifecycle.py :: return any(shown.body == entry for shown in self.context_epochs[-1].entries)
    Becomes: return any(shown.entry == entry for shown in self.context_epochs[-1].entries)
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

    Killed by: src/uclone_x/agent/request_record.py :: if not 0 <= position < len(state.session_log):
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
    compactor = ContextCompactor(keep_recent_turns=50, workspace_root=tmp_path, session_id=sid)
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
    compactor = ContextCompactor(keep_recent_turns=50, workspace_root=tmp_path, session_id=sid)
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
    excerpt = ChatMessage(
        role=MessageRole.TOOL, content="head ... tail", name="t", tool_call_id="c1", form="excerpt"
    )
    stub = excerpt.model_copy(update={"content": "[stored]", "form": "stub"})
    bodies = {"e1": excerpt, "e4": stub}

    shown = ContextEntry(entry="e1", form=ContextForm.STUB, rendering="e4")

    assert shown.body == "e4"
    assert render_entries([shown], bodies.__getitem__) == [stub]
    assert render_entries([_entry("e1", ContextForm.EXCERPT)], bodies.__getitem__) == [excerpt]


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

    Killed by: src/uclone_x/core/context_state.py :: if shown.rendering is None and shown.form in (ContextForm.EXCERPT, ContextForm.STUB):
    Becomes: if shown.form in (ContextForm.EXCERPT, ContextForm.STUB):
    """
    shown = ContextEntry(entry="e1", form=ContextForm.STUB, rendering="e4", same_as="e0")
    epoch = ContextEpoch(
        number=1, turn=2, step=1, opened_by=("compaction",), entries=(_entry("e0"), shown)
    )

    assert recorded_renderings([epoch]) == {"e4": shown.model_copy(update={"same_as": None})}
    assert recorded_forms([epoch]) == {}


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

    Killed by: src/uclone_x/core/session_log.py :: if message.name in SUBAGENT_TOOLS:
    Becomes: if message.name in ():
    Killed by: src/uclone_x/core/session_log.py :: if message.name in RETRIEVAL_TOOLS:
    Becomes: if message.name in ():
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


def _memory_with_a_fact() -> CrossSessionMemory:
    memory = CrossSessionMemory(max_facts_in_prompt=5)
    memory._facts["mem_db"] = MemoryFact(  # pyright: ignore[reportPrivateUsage]
        fact_id="mem_db",
        subject="service",
        predicate="database",
        object_value="postgres 16",
        provenance=Provenance.primary(provider="agent.test", model="memory"),
        source_session_id="sess_a",
        confidence=0.9,
        created_at="2026-09-20T00:00:00+00:00",
    )
    return memory


async def _run_recalling_turn(tmp_path: Path) -> tuple[_ScriptedLLM, SessionStore]:
    llm = _ScriptedLLM([[_call("c0", "read")]])
    agent, store = _agent(
        tmp_path, llm, [_returning("read", "short")], memory=_memory_with_a_fact()
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
    Killed by: src/uclone_x/agent/turn_executor.py :: handle = store_tool_result(
    Becomes: handle = None and store_tool_result(
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
    assert load_tool_result(artifacts_dir_for(tmp_path), "sess_1443", handle) == section
    page = await ToolResultReadTool().execute(
        {"handle": handle, "offset": 0, "length": 10_000},
        ToolContext(agent_id="ctx", session_id="sess_1443", trace_id="t", workspace_root=tmp_path),
    )
    assert page.success and isinstance(page.output, str) and section in page.output
    # Its entry is not a message of the conversation: no epoch shows it.
    assert all(x.entry != memory_entry.id for e in state.context_epochs for x in e.entries)
    _assert_epochs_render_from_the_log(store, state, llm.requests)


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
