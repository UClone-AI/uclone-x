"""Tool results at ingest: canonical JSON, a size cap with a stored body, and compaction
that checks between steps and cuts only on turn boundaries (#1422).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.session import SessionState, SessionStore, reap_orphaned_tool_artifacts
from uclone_x.core.provenance import Provenance
from uclone_x.core.tool_results import (
    STEP_EXCERPT_MIN_BYTES,
    STEP_NO_ROOM_MESSAGE,
    STEP_OVER_WINDOW_MESSAGE,
    STEP_REPLY_RESERVE_TOKENS,
    STORED_RESULT_PREFIX,
    TOOL_RESULT_CAP_BYTES,
    StoredResultNotFoundError,
    artifacts_dir_for,
    canonical_tool_text,
    handle_in,
    ingest_tool_text,
    load_tool_result,
    read_tool_result_page,
    result_handle,
    step_result_caps,
    stub_tool_result,
)
from uclone_x.llm.compactor import ContextCompactor
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ModelResponse,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.tools.builtin.subagent import SubagentDelegationTool
from uclone_x.tools.builtin.tool_results import ToolResultReadTool
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import LocalTool, ToolRegistry

# ======================================================================================
# Helpers
# ======================================================================================


class _ScriptedLLM(MockLLMConnector):
    """Asks for the tool calls of step N on its N-th request, then answers; keeps requests."""

    def __init__(self, steps: Sequence[Sequence[ToolCallRequest]]) -> None:
        super().__init__(default_response="done")
        self._steps = [list(s) for s in steps]
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        self._tool_calls = self._steps.pop(0) if self._steps else []
        return await super().generate(request)


def _returning(name: str, output: Any) -> LocalTool:
    async def handler(params: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            output=output,
            provenance=Provenance.primary(provider="local.test", model=name),
        )

    return LocalTool(name, f"Returns a fixed {name} result.", handler=handler, writes_files=False)


def _agent(
    workspace: Path,
    llm: MockLLMConnector,
    tools: Sequence[Any],
    *,
    llm_config: AgentLLMConfig | None = None,
    **config: Any,
) -> BaseAgent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="ingest",
            name="Ingest",
            workspace_dir=workspace,
            llm_config=llm_config or AgentLLMConfig(model_name="mock-model"),
            **config,
        ),
        llm=llm,
        tools=registry,
    )


def _tool_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    return [m for m in messages if m.role is MessageRole.TOOL]


def _big_text(size: int = 30_000) -> str:
    # Distinct lines, so any slice is locatable and a lost range is visible.
    lines = [f"line {i:05d}: {'é' if i % 7 == 0 else 'x'} payload" for i in range(size // 20)]
    return "\n".join(lines)


# ======================================================================================
# Canonical text
# ======================================================================================


def test_a_dict_result_is_canonical_json_that_round_trips() -> None:
    """A dict becomes sorted, compact JSON, the same whatever the key order.

    Killed by: src/uclone_x/core/tool_results.py :: sort_keys=True,
    Becomes: sort_keys=False,
    """
    first = canonical_tool_text({"b": 1, "a": [1, 2, {"z": None, "y": True}]})
    second = canonical_tool_text({"a": [1, 2, {"y": True, "z": None}], "b": 1})
    assert first == second == '{"a":[1,2,{"y":true,"z":null}],"b":1}'
    assert json.loads(first) == {"a": [1, 2, {"y": True, "z": None}], "b": 1}


def test_non_ascii_is_kept_as_written() -> None:
    """Killed by: src/uclone_x/core/tool_results.py :: ensure_ascii=False,
    Becomes: ensure_ascii=True,
    """
    assert canonical_tool_text({"name": "한글 é"}) == '{"name":"한글 é"}'


def test_a_str_result_stays_a_str_and_none_is_null() -> None:
    assert canonical_tool_text("plain prose") == "plain prose"
    assert canonical_tool_text(None) == "null"
    assert canonical_tool_text(3) == "3"


def test_a_value_with_no_json_form_is_refused_not_repred() -> None:
    with pytest.raises(TypeError, match="no JSON form"):
        canonical_tool_text({"x": object()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_nan_or_infinite_number_is_refused_not_written_as_invalid_json(value: float) -> None:
    """Killed by: src/uclone_x/core/tool_results.py :: allow_nan=False,
    Becomes: allow_nan=True,
    """
    with pytest.raises(TypeError, match="NaN or infinite number"):
        canonical_tool_text({"score": value})


def test_bytes_are_refused_not_decoded_with_a_guess() -> None:
    with pytest.raises(TypeError, match="bytes, which has no JSON form"):
        canonical_tool_text({"blob": b"\xff\xfeabc"})


def test_sets_and_paths_render_the_same_on_every_run() -> None:
    assert canonical_tool_text({"s": {"b", "a", "c"}}) == '{"s":["a","b","c"]}'
    assert canonical_tool_text({"p": Path("a") / "b"}) == '{"p":"a/b"}'


# ======================================================================================
# Cap, stored body, reading it back
# ======================================================================================


def test_an_under_cap_result_is_kept_whole(tmp_path: Path) -> None:
    text = "short"
    assert ingest_tool_text(text, artifacts_dir=tmp_path, session_id="s1") == text
    assert not any(tmp_path.iterdir())


def test_an_over_cap_result_is_stored_and_read_back_whole(tmp_path: Path) -> None:
    """The history holds an excerpt under the cap; the pages rebuild the exact body."""
    body = _big_text()
    excerpt = ingest_tool_text(body, artifacts_dir=tmp_path, session_id="s1")

    assert len(excerpt.encode("utf-8")) <= TOOL_RESULT_CAP_BYTES
    handle = handle_in(excerpt)
    assert handle is not None and handle == result_handle(body)
    assert excerpt.startswith(f"{STORED_RESULT_PREFIX}{handle}:")
    head_line = body.splitlines()[0]
    tail_line = body.splitlines()[-1]
    assert head_line in excerpt and tail_line in excerpt

    rebuilt = ""
    offset = 0
    while True:
        page = read_tool_result_page(tmp_path, "s1", handle, offset)
        assert len(page.encode("utf-8")) <= TOOL_RESULT_CAP_BYTES
        header, _, text = page.partition("\n")
        rebuilt += text
        offset += len(text)
        if "This is the end of the result." in header:
            break
        assert f"offset={offset})" in header
    assert rebuilt == body


def test_the_excerpt_names_the_offset_where_the_hidden_part_starts(tmp_path: Path) -> None:
    body = _big_text()
    excerpt = ingest_tool_text(body, artifacts_dir=tmp_path, session_id="s1")
    handle = handle_in(excerpt)
    assert handle is not None
    head = excerpt.split("\n", 1)[1]
    shown = len(head.split("\n[... characters ", 1)[0])
    assert f"offset={shown})" in excerpt.splitlines()[0]
    page = read_tool_result_page(tmp_path, "s1", handle, shown)
    assert page.split("\n", 1)[1] == body[shown : shown + len(page.split("\n", 1)[1])]


def test_a_result_is_the_same_excerpt_on_every_run(tmp_path: Path) -> None:
    body = _big_text()
    one = ingest_tool_text(body, artifacts_dir=tmp_path / "a", session_id="s1")
    two = ingest_tool_text(body, artifacts_dir=tmp_path / "b", session_id="s1")
    assert one == two


def test_with_nowhere_to_store_the_excerpt_says_the_rest_is_gone() -> None:
    excerpt = ingest_tool_text(_big_text(), artifacts_dir=None, session_id="s1")
    assert handle_in(excerpt) is None
    assert "cannot be read back" in excerpt
    assert "tool_result_read" not in excerpt


def test_without_the_reader_the_excerpt_does_not_offer_it(tmp_path: Path) -> None:
    excerpt = ingest_tool_text(_big_text(), artifacts_dir=tmp_path, session_id="s1", readable=False)
    assert handle_in(excerpt) is not None
    assert "no tool to read it" in excerpt
    assert "tool_result_read(" not in excerpt


def test_credentials_are_redacted_before_the_body_is_stored(tmp_path: Path) -> None:
    secret = "sk-ant-api03-" + "A" * 90
    body = _big_text() + f"\nkey={secret}\n"
    excerpt = ingest_tool_text(body, artifacts_dir=tmp_path, session_id="s1")
    stored = next((tmp_path / "s1").iterdir()).read_text(encoding="utf-8")
    assert secret not in stored and secret not in excerpt


@pytest.mark.parametrize(
    ("handle", "offset", "length", "expected"),
    [
        ("not-a-handle", 0, None, "not a stored tool result name"),
        ("tr_0000000000000000", 0, None, "No stored tool result named"),
        (None, -1, None, "must be 0 or more"),
        (None, 0, 0, "must be at least 1"),
        (None, 10**9, None, "past the end of this result"),
    ],
)
def test_bad_reads_are_refused_in_plain_words(
    tmp_path: Path, handle: str | None, offset: int, length: int | None, expected: str
) -> None:
    stored = handle_in(ingest_tool_text(_big_text(), artifacts_dir=tmp_path, session_id="s1"))
    assert stored is not None
    with pytest.raises((StoredResultNotFoundError, ValueError)) as info:
        read_tool_result_page(tmp_path, "s1", handle or stored, offset, length)
    message = str(info.value)
    assert expected in message
    assert "Traceback" not in message and str(tmp_path) not in message


def test_one_conversation_cannot_read_anothers_results(tmp_path: Path) -> None:
    handle = handle_in(ingest_tool_text(_big_text(), artifacts_dir=tmp_path, session_id="s1"))
    assert handle is not None
    with pytest.raises(StoredResultNotFoundError):
        read_tool_result_page(tmp_path, "s2", handle)
    with pytest.raises(ValueError):
        read_tool_result_page(tmp_path, "../s1", handle)


@pytest.mark.asyncio
async def test_the_reader_tool_reads_within_its_own_session(tmp_path: Path) -> None:
    body = _big_text()
    handle = handle_in(
        ingest_tool_text(body, artifacts_dir=artifacts_dir_for(tmp_path), session_id="s1")
    )
    context = ToolContext(agent_id="a", session_id="s1", trace_id="t", workspace_root=tmp_path)
    result = await ToolResultReadTool().execute({"handle": handle, "offset": 5}, context)
    assert result.success is True
    assert isinstance(result.output, str)
    assert result.output.split("\n", 1)[1].startswith(body[5:40])


# ======================================================================================
# Through the agent
# ======================================================================================


@pytest.mark.asyncio
async def test_the_agent_keeps_a_dict_result_as_canonical_json(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: content = canonical_tool_text(res.output) if res.success
    Becomes: content = str(res.output) if res.success
    """
    llm = _ScriptedLLM([[ToolCallRequest(id="c1", name="facts", arguments={})]])
    agent = _agent(tmp_path, llm, [_returning("facts", {"b": 2, "a": "é"})])
    await agent.execute_turn("go")

    (tool_msg,) = _tool_messages(agent.history)
    assert tool_msg.content == '{"a":"é","b":2}'
    assert json.loads(tool_msg.content) == {"a": "é", "b": 2}


@pytest.mark.asyncio
async def test_an_over_cap_result_reaches_history_as_an_excerpt_and_reads_back(
    tmp_path: Path,
) -> None:
    """The model reads the stored body with the reader; nothing earlier is rewritten.

    Killed by: src/uclone_x/agent/base.py :: self._ingest_tool_message(m, readable=reader_offered)
    Becomes: m
    """
    body = _big_text()
    handle = result_handle(body)
    llm = _ScriptedLLM(
        [
            [ToolCallRequest(id="c1", name="dump", arguments={})],
            [ToolCallRequest(id="c2", name="tool_result_read", arguments={"handle": handle})],
        ]
    )
    agent = _agent(tmp_path, llm, [_returning("dump", body), ToolResultReadTool()])
    await agent.execute_turn("go")

    first, second = _tool_messages(agent.history)
    assert first.content is not None and second.content is not None
    assert len(first.content.encode("utf-8")) <= TOOL_RESULT_CAP_BYTES
    assert handle_in(first.content) == handle
    assert 'tool_result_read(handle="' in first.content
    assert (artifacts_dir_for(tmp_path) / agent.session_id / f"{handle}.txt").is_file()
    # The page is appended at the tail, and the excerpt before it is what the model saw.
    assert second.content.split("\n", 1)[1] == body[: len(second.content.split("\n", 1)[1])]
    later_request_tools = _tool_messages(llm.requests[-1].messages)
    assert later_request_tools[0].content == first.content


@pytest.mark.asyncio
async def test_an_agent_without_the_reader_is_not_told_to_call_it(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: reader_offered = any(d.name == TOOL_RESULT_READ_TOOL for d in tool_defs)
    Becomes: reader_offered = True
    """
    llm = _ScriptedLLM([[ToolCallRequest(id="c1", name="dump", arguments={})]])
    agent = _agent(tmp_path, llm, [_returning("dump", _big_text())])
    await agent.execute_turn("go")

    assert "tool_result_read" not in {t.name for t in llm.requests[0].tools}
    (tool_msg,) = _tool_messages(agent.history)
    assert tool_msg.content is not None
    assert "no tool to read it" in tool_msg.content
    assert "tool_result_read(" not in tool_msg.content


@pytest.mark.asyncio
async def test_a_sub_agent_result_goes_through_the_same_cap(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequest(
                    id="c1",
                    name="delegate_subagent",
                    arguments={"role": "r", "goal": "g", "prompt": "p"},
                )
            ]
        ]
    )
    agent = _agent(tmp_path, llm, [SubagentDelegationTool()], enable_subagent_tools=True)
    child = MagicMock()
    child.agent_id = "child"
    child.turn_counter = 0
    child._config = AgentConfig(agent_id="child", name="child")
    agent.spawn_subagent = AsyncMock(return_value=child)  # type: ignore[method-assign]
    child_result = MagicMock()
    child_result.error = None
    child_result.content = _big_text()
    child_result.provenance = Provenance.primary(provider="test", model="test")
    agent.delegate_task = AsyncMock(return_value=child_result)  # type: ignore[method-assign]

    await agent.execute_turn("go")

    (tool_msg,) = _tool_messages(agent.history)
    assert tool_msg.content is not None
    assert tool_msg.content.startswith(STORED_RESULT_PREFIX)
    assert len(tool_msg.content.encode("utf-8")) <= TOOL_RESULT_CAP_BYTES


# ======================================================================================
# Compaction: the trigger, between steps, and where it cuts
# ======================================================================================


def test_the_trigger_counts_tool_schemas_and_system_sections() -> None:
    """A request whose messages alone are small trips on its tool schemas.

    Killed by: src/uclone_x/llm/compactor.py :: return estimate_request_tokens(request) >= threshold_tokens
    Becomes: return estimate_request_tokens(LLMRequest(messages=request.messages)) >= threshold_tokens
    """
    compactor = ContextCompactor()
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="s " * 400),
        ChatMessage(role=MessageRole.USER, content="hi"),
    )
    big_schema = ToolDefinition(
        name="wide",
        description="d " * 2000,
        parameters={"type": "object", "properties": {}},
    )
    bare = LLMRequest(messages=messages[1:])
    with_system = LLMRequest(messages=messages)
    with_tools = LLMRequest(messages=messages[1:], tools=(big_schema,))
    assert not compactor.should_compact_request_at(bare, 150)
    assert compactor.should_compact_request_at(with_system, 150)
    assert compactor.should_compact_request_at(with_tools, 150)


@pytest.mark.asyncio
async def test_a_mid_turn_overflow_compacts_before_the_next_step(tmp_path: Path) -> None:
    """A step that crosses the threshold is compacted before the next request is built,
    and that request carries the step's own result exactly as ingested.

    The pass prunes what the model has already read -- the earlier step's result -- and
    leaves the step that just ran alone: the model has not seen it yet (#1422 review B1).

    Killed by: src/uclone_x/agent/base.py :: if await self._auto_compact_if_needed(
    Becomes: if False and await self._auto_compact_if_needed(
    """
    older = "o" * 2_000
    fresh = "y" * 6_000  # under the ingest cap, far over this agent's threshold
    llm = _ScriptedLLM(
        [
            [ToolCallRequest(id="c1", name="small", arguments={})],
            [ToolCallRequest(id="c2", name="dump", arguments={})],
        ]
    )
    agent = _agent(
        tmp_path,
        llm,
        [_returning("small", older), _returning("dump", fresh)],
        llm_config=AgentLLMConfig(model_name="mock-model", compaction_threshold_tokens=1_200),
    )
    reasons: list[tuple[str, int]] = []
    original = agent._compact_session  # pyright: ignore[reportPrivateUsage]

    async def recording(sid: str, reason: str, **kwargs: Any) -> Any:
        reasons.append((reason, len(llm.requests)))
        return await original(sid, reason, **kwargs)

    agent._compact_session = recording  # type: ignore[method-assign]
    await agent.execute_turn("go")

    assert ("auto_threshold_mid_turn", 2) in reasons
    assert len(llm.requests) == 3
    third = llm.requests[2]
    by_id = {m.tool_call_id: m for m in _tool_messages(third.messages)}
    # The step the model has not read reaches it verbatim ...
    assert by_id["c2"].content == fresh
    # ... while the one it has read was pruned by the pass.
    assert by_id["c1"].content is not None and len(by_id["c1"].content) < len(older)
    # Both groups survive: each call and its result reach the next request.
    calls = [m for m in third.messages if m.role is MessageRole.ASSISTANT and m.tool_calls]
    assert [c.id for m in calls for c in m.tool_calls] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_a_freshly_read_file_survives_mid_turn_compaction_on_an_8k_window(
    tmp_path: Path,
) -> None:
    """The review's reproduction: the full default tool set, an 8K window, one small file.

    The tool schemas alone are most of the trigger, so the first `file_read` crosses it.
    Every later request must carry the read it follows exactly as ingested; before the
    fix each carried a 410-character offload line and the model re-read forever.

    Killed by: src/uclone_x/agent/base.py :: hold_unseen_step=True,
    Becomes: hold_unseen_step=False,
    """
    from uclone_x.tools.registry import create_default_registry

    text = "\n".join(f"line {i:05d} " + "x" * 60 for i in range(3_000 // 72))
    (tmp_path / "a.py").write_text(text)
    llm = _ScriptedLLM(
        [
            [ToolCallRequest(id=f"c{i}", name="file_read", arguments={"path": "a.py"})]
            for i in range(4)
        ]
    )
    registry = create_default_registry(workspace_root=tmp_path, enable_mcp=False)
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="ingest",
            name="Ingest",
            workspace_dir=tmp_path,
            llm_config=AgentLLMConfig(model_name="mock-model", context_limit=8_192),
        ),
        llm=llm,
        tools=registry,
    )
    reasons: list[str] = []
    original = agent._compact_session  # pyright: ignore[reportPrivateUsage]

    async def recording(sid: str, reason: str, **kwargs: Any) -> Any:
        reasons.append(reason)
        return await original(sid, reason, **kwargs)

    agent._compact_session = recording  # type: ignore[method-assign]
    await agent.execute_turn("read a.py")

    assert "auto_threshold_mid_turn" in reasons
    assert len(llm.requests) == 5
    ingested = {m.tool_call_id: m.content for m in _tool_messages(agent.history)}
    for n, request in enumerate(llm.requests[1:]):
        latest = _tool_messages(request.messages)[-1]
        assert latest.tool_call_id == f"c{n}"
        assert latest.content is not None and text.splitlines()[-1] in latest.content
        if n == 3:
            # The last read is never compacted afterwards, so history holds it as ingested.
            assert latest.content == ingested[f"c{n}"]


@pytest.mark.asyncio
async def test_a_mid_turn_compaction_is_rebuilt_from_the_request_record(tmp_path: Path) -> None:
    """The request after a pass between steps is recorded as a divergence, not an extension.

    #1421 records each request's conversation as a delta on the previous one, found by
    comparing prefixes. A pass between steps rewrites an earlier result, so the delta must
    start there, and the rebuilt request must equal the one sent.

    Killed by: src/uclone_x/agent/base.py :: if before != now:
    Becomes: if False:
    """
    from uclone_x.agent.models import AgentContext, AgentState
    from uclone_x.agent.request_record import rebuild_requests
    from uclone_x.log.reader import read_session_log

    store = SessionStore(tmp_path / "store")
    sid = "sess_1422"
    older = "o" * 2_000
    fresh = "y" * 6_000
    llm = _ScriptedLLM(
        [
            [ToolCallRequest(id="c1", name="small", arguments={})],
            [ToolCallRequest(id="c2", name="dump", arguments={})],
        ]
    )
    registry = ToolRegistry()
    registry.register(_returning("small", older))
    registry.register(_returning("dump", fresh))
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="ingest",
            name="Ingest",
            workspace_dir=tmp_path,
            llm_config=AgentLLMConfig(model_name="mock-model", compaction_threshold_tokens=1_200),
        ),
        llm=llm,
        tools=registry,
        store=store,
        context=AgentContext(session_id=sid, agent_id="ingest", current_state=AgentState.IDLE),
    )
    reasons: list[str] = []
    original = agent._compact_session  # pyright: ignore[reportPrivateUsage]

    async def recording(session_id: str, reason: str, **kwargs: Any) -> Any:
        reasons.append(reason)
        return await original(session_id, reason, **kwargs)

    agent._compact_session = recording  # type: ignore[method-assign]
    await agent.start()
    assert (await agent.execute_turn("go")).is_completed
    agent.persist_session()
    assert "auto_threshold_mid_turn" in reasons

    state = store.load(sid)
    log = store.event_log_path(sid)
    assert state is not None and log is not None
    events = [dict(e) for e in read_session_log(log)]
    contexts = [e for e in events if e.get("type") == "REQUEST_CONTEXT"]
    assert len(contexts) == len(llm.requests) == 3
    # The third request diverges where the pass pruned the first result.
    second_len = contexts[1]["kept_message_count"] + len(contexts[1]["appended_messages"])
    assert contexts[2]["kept_message_count"] < second_len

    rebuilt = rebuild_requests(store, state, events)
    for number, (request, sent) in enumerate(zip(rebuilt, llm.requests, strict=True)):
        assert request.verified, number
        assert [m.model_dump() for m in request.request.messages] == [
            m.model_dump() for m in sent.messages
        ], number


@pytest.mark.asyncio
async def test_the_turn_start_check_counts_the_tool_schemas(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/base.py :: await self._auto_compact_if_needed(tool_defs, turn_extra_sections)
    Becomes: await self._auto_compact_if_needed((), turn_extra_sections)
    """
    wide = LocalTool(
        "wide",
        "d " * 3_000,
        handler=None,
        writes_files=False,
    )
    llm = _ScriptedLLM([])
    agent = _agent(
        tmp_path,
        llm,
        [wide],
        llm_config=AgentLLMConfig(model_name="mock-model", compaction_threshold_tokens=1_000),
    )
    reasons: list[str] = []
    original = agent._compact_session  # pyright: ignore[reportPrivateUsage]

    async def recording(sid: str, reason: str, **kwargs: Any) -> Any:
        reasons.append(reason)
        return await original(sid, reason, **kwargs)

    agent._compact_session = recording  # type: ignore[method-assign]
    await agent.execute_turn("hi")
    assert "auto_threshold" in reasons


def _random_dialog(rng: random.Random) -> list[ChatMessage]:
    messages = [ChatMessage(role=MessageRole.SYSTEM, content="anchor")]
    call_no = 0
    for turn in range(rng.randint(0, 7)):
        messages.append(ChatMessage(role=MessageRole.USER, content=f"q{turn}"))
        for _ in range(rng.randint(0, 3)):
            ids: list[str] = []
            for _ in range(rng.randint(1, 3)):
                call_no += 1
                ids.append(f"call{call_no}")
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=tuple(ToolCallRequest(id=i, name="t") for i in ids),
                )
            )
            for i in ids:
                size = rng.choice((5, 50, 900))
                messages.append(
                    ChatMessage(role=MessageRole.TOOL, content="r" * size, tool_call_id=i, name="t")
                )
        if rng.random() < 0.8:
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=f"a{turn}"))
    return messages


@pytest.mark.asyncio
async def test_no_compaction_cut_orphans_a_tool_call_or_its_results() -> None:
    """Seeded property check over 400 random histories and window sizes.

    Killed by: src/uclone_x/llm/compactor.py :: boundaries = [i for i in range(1, len(dialog)) if dialog[i].role == MessageRole.USER]
    Becomes: boundaries = list(range(1, len(dialog)))
    """
    rng = random.Random(1422)
    for case in range(400):
        messages = _random_dialog(rng)
        keep = rng.randint(0, 9)
        outcome = await ContextCompactor(keep_recent_turns=keep).compact(messages)
        out = list(outcome.messages)
        dialog = [m for m in out if m.role is not MessageRole.SYSTEM]
        where = f"case {case}, keep={keep}"

        asked: dict[str, int] = {}
        for index, msg in enumerate(dialog):
            if msg.role is MessageRole.ASSISTANT:
                for call in msg.tool_calls:
                    asked[call.id] = index
            if msg.role is MessageRole.TOOL:
                assert msg.tool_call_id in asked, f"orphan result {msg.tool_call_id}; {where}"
        answered = {m.tool_call_id for m in dialog if m.role is MessageRole.TOOL}
        assert set(asked) <= answered, f"call kept without its results; {where}"
        if dialog and any(m.compaction_ledger for m in out[len(out) - len(dialog) - 1 :]):
            assert dialog[0].role is MessageRole.USER, f"kept window starts mid-turn; {where}"
        # The last user turn is never summarised, and at least `keep` messages are kept
        # whenever a boundary allows it.
        original_dialog = [m for m in messages if m.role is not MessageRole.SYSTEM]
        users = [i for i, m in enumerate(original_dialog) if m.role is MessageRole.USER]
        if len(original_dialog) <= keep:
            assert len(dialog) == len(original_dialog), where
        if users:
            assert len(dialog) >= len(original_dialog) - users[-1], where
        if any(0 < i <= len(original_dialog) - keep for i in users):
            assert len(dialog) >= keep, where


@pytest.mark.asyncio
async def test_a_single_long_turn_is_pruned_not_summarised_away() -> None:
    messages = [
        ChatMessage(role=MessageRole.USER, content="the only question"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=tuple(ToolCallRequest(id=f"c{i}", name="t") for i in range(6)),
        ),
        *(
            ChatMessage(role=MessageRole.TOOL, content="r" * 900, tool_call_id=f"c{i}", name="t")
            for i in range(6)
        ),
    ]
    outcome = await ContextCompactor(keep_recent_turns=2).compact(messages)
    assert outcome.messages[0].content == "the only question"
    assert all(len(m.content or "") < 900 for m in outcome.messages[2:])


@pytest.mark.asyncio
async def test_compaction_shrinks_a_stored_excerpt_to_a_stub(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/llm/compactor.py :: if stub is not None:
    Becomes: if False:
    """
    artifacts = artifacts_dir_for(tmp_path)
    excerpt = ingest_tool_text(_big_text(), artifacts_dir=artifacts, session_id="s1")
    handle = handle_in(excerpt)
    messages = [
        ChatMessage(role=MessageRole.USER, content="go"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(ToolCallRequest(id="c1", name="t"),)),
        ChatMessage(role=MessageRole.TOOL, content=excerpt, tool_call_id="c1", name="t"),
    ]
    compactor = ContextCompactor(workspace_root=tmp_path, session_id="s1")
    outcome = await compactor.compact(messages)
    tool_msg = _tool_messages(outcome.messages)[0]
    assert tool_msg.content is not None
    assert tool_msg.content.startswith(f"{STORED_RESULT_PREFIX}{handle}:")
    assert "since the conversation was compacted" in tool_msg.content
    assert len(tool_msg.content) < len(excerpt)


@pytest.mark.asyncio
async def test_without_the_reader_compaction_does_not_stub(tmp_path: Path) -> None:
    """The short form names `tool_result_read`, so it is not used when that is not offered.

    Killed by: src/uclone_x/llm/compactor.py :: self.tool_result_reader = tool_result_reader
    Becomes: self.tool_result_reader = True
    """
    artifacts = artifacts_dir_for(tmp_path)
    excerpt = ingest_tool_text(
        _big_text(), artifacts_dir=artifacts, session_id="s1", readable=False
    )
    messages = [
        ChatMessage(role=MessageRole.USER, content="go"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(ToolCallRequest(id="c1", name="t"),)),
        ChatMessage(role=MessageRole.TOOL, content=excerpt, tool_call_id="c1", name="t"),
    ]
    compactor = ContextCompactor(workspace_root=tmp_path, session_id="s1", tool_result_reader=False)
    outcome = await compactor.compact(messages)
    tool_msg = _tool_messages(outcome.messages)[0]
    assert tool_msg.content is not None
    assert "tool_result_read" not in tool_msg.content
    assert len(tool_msg.content) < len(excerpt)


@pytest.mark.asyncio
async def test_an_agent_without_the_reader_is_not_told_to_call_it_after_compaction(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/base.py :: self._tools is not None and self._tools.get(TOOL_RESULT_READ_TOOL) is not None
    Becomes: True
    """
    llm = _ScriptedLLM([[ToolCallRequest(id="c1", name="dump", arguments={})]])
    agent = _agent(tmp_path, llm, [_returning("dump", _big_text())])
    await agent.execute_turn("go")
    await agent.compact_session()

    for message in agent.history:
        assert "tool_result_read" not in (message.content or "")


def test_text_that_only_looks_like_a_stored_result_is_not_stubbed(tmp_path: Path) -> None:
    forged = f"{STORED_RESULT_PREFIX}tr_0123456789abcdef: …]\n" + "x" * 5_000
    assert stub_tool_result(forged, tmp_path, "s1", keep_chars=10) is None


# ======================================================================================
# Lifetime of stored bodies
# ======================================================================================


@pytest.mark.asyncio
async def test_resetting_or_deleting_a_session_removes_its_stored_results(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/base.py :: cleanup_session_artifacts(artifacts_dir_for(ws_root), sid)
    Becomes: None
    """
    for action in ("reset", "delete"):
        workspace = tmp_path / action
        workspace.mkdir()
        llm = _ScriptedLLM([[ToolCallRequest(id="c1", name="dump", arguments={})]])
        agent = _agent(workspace, llm, [_returning("dump", _big_text())])
        await agent.execute_turn("go")
        session_dir = artifacts_dir_for(workspace) / agent.session_id
        assert any(session_dir.iterdir())
        if action == "reset":
            agent.reset_session()
        else:
            agent.delete_session()
        assert not session_dir.exists(), action


def test_the_reaper_spares_a_live_session_and_reaps_an_orphan(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/session.py :: if is_live is not None and is_live(entry.name):
    Becomes: if False:
    """
    artifacts = tmp_path / "artifacts"
    for sid in ("live", "gone"):
        ingest_tool_text(_big_text(), artifacts_dir=artifacts, session_id=sid)
    store = SessionStore(storage_dir=tmp_path / "sessions", artifacts_dir=artifacts)
    store.save(SessionState(session_id="live", agent_id="a"))
    store.reap_orphaned_temp_files(max_age_seconds=0)
    assert (artifacts / "live").is_dir()
    assert not (artifacts / "gone").exists()


def test_the_reaper_without_a_liveness_check_keeps_its_old_behaviour(tmp_path: Path) -> None:
    ingest_tool_text(_big_text(), artifacts_dir=tmp_path, session_id="s1")
    assert reap_orphaned_tool_artifacts(tmp_path, max_age_seconds=0) == 1


# ======================================================================================
# One step's results, together, within the window (#1480)
# ======================================================================================


def _default_registry_agent(workspace: Path, llm: MockLLMConnector) -> BaseAgent:
    from uclone_x.tools.registry import create_default_registry

    return BaseAgent(
        config=AgentConfig(
            agent_id="ingest",
            name="Ingest",
            workspace_dir=workspace,
            llm_config=AgentLLMConfig(model_name="mock-model", context_limit=8_192),
        ),
        llm=llm,
        tools=create_default_registry(workspace_root=workspace, enable_mcp=False),
    )


def _near_cap_file(workspace: Path, name: str) -> str:
    """A file whose read comes back just under the result cap, so ingest keeps it whole."""
    lines = [f"{name} line {i:04d} " + "x" * 50 for i in range(TOOL_RESULT_CAP_BYTES // 90)]
    text = "\n".join(lines)
    (workspace / name).write_text(text)
    return text


def test_step_shares_keep_small_results_whole_and_split_the_rest_evenly() -> None:
    """Smallest first: a result under an equal share keeps its size; the rest split what
    is left, and together they fit the budget.

    Killed by: src/uclone_x/core/tool_results.py :: if sizes[index] <= share:
    Becomes: if sizes[index] < 0:
    """
    caps = step_result_caps([100, 8_000, 8_000, 50], 6_150)
    assert caps is not None
    assert caps[0] == 100 and caps[3] == 50
    assert caps[1] == caps[2] == 3_000
    assert sum(caps) <= 6_150


def test_step_shares_below_the_excerpt_floor_are_refused() -> None:
    """A share too small for an excerpt with its handle is no share at all.

    Killed by: src/uclone_x/core/tool_results.py :: elif share < floor:
    Becomes: elif share < 0:
    """
    assert step_result_caps([8_000] * 3, 3 * STEP_EXCERPT_MIN_BYTES) is not None
    assert step_result_caps([8_000] * 3, 3 * STEP_EXCERPT_MIN_BYTES - 3) is None


@pytest.mark.asyncio
async def test_a_step_over_the_window_reaches_the_next_request_as_readable_excerpts(
    tmp_path: Path,
) -> None:
    """The issue's reproduction: an 8K window, the default registry, three parallel reads
    each just under the result cap. Together they exceed what the schemas leave.

    The next request stays within the window, and every result is in it as an excerpt
    whose handle reads back the whole file.

    Killed by: src/uclone_x/agent/base.py :: step_refusal = self._fit_step_to_window(
    Becomes: step_refusal = None and self._fit_step_to_window(
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    names = ["a.txt", "b.txt", "c.txt"]
    texts = {name: _near_cap_file(tmp_path, name) for name in names}
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequest(id=f"c{i}", name="file_read", arguments={"path": name})
                for i, name in enumerate(names)
            ]
        ]
    )
    agent = _default_registry_agent(tmp_path, llm)
    result = await agent.execute_turn("read all three")

    assert result.is_completed, result.error
    assert len(llm.requests) == 2
    second = llm.requests[1]
    assert estimate_request_tokens(second) <= 8_192
    by_id = {m.tool_call_id: m.content for m in _tool_messages(second.messages)}
    assert sorted(str(k) for k in by_id) == ["c0", "c1", "c2"]
    for i, name in enumerate(names):
        content = by_id[f"c{i}"]
        assert content is not None
        handle = handle_in(content)
        assert handle is not None, content[:200]
        assert "tool_result_read" in content
        # The handle reads back the whole file, from its first line to its last.
        body = load_tool_result(artifacts_dir_for(tmp_path), agent.session_id, handle)
        assert texts[name].splitlines()[0] in body
        assert texts[name].splitlines()[-1] in body


@pytest.mark.asyncio
async def test_a_step_that_cannot_fit_even_as_excerpts_is_refused_in_plain_words(
    tmp_path: Path,
) -> None:
    """Forty parallel reads on an 8K window: no share can hold an excerpt and its handle.

    The step is refused; the over-window request is never sent, and the refusal says so
    without a path, a handle, a class name or any other internal.

    Killed by: src/uclone_x/agent/base.py :: error=step_refusal,
    Becomes: error=repr(step_results[0]),
    """
    names = [f"f{i:02d}.txt" for i in range(40)]
    for name in names:
        _near_cap_file(tmp_path, name)
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequest(id=f"c{i}", name="file_read", arguments={"path": name})
                for i, name in enumerate(names)
            ]
        ]
    )
    agent = _default_registry_agent(tmp_path, llm)
    result = await agent.execute_turn("read them all")

    assert len(llm.requests) == 1
    assert not result.is_completed
    assert result.stop_reason == "step_results_over_window"
    error = result.error or ""
    assert error == STEP_OVER_WINDOW_MESSAGE
    for internal in ("/", "\\", "tr_", "Error", "Traceback", "token", str(tmp_path), "{"):
        assert internal not in error


@pytest.mark.asyncio
async def test_a_refused_step_sends_nothing_over_the_window(tmp_path: Path) -> None:
    """With the refusal gone, the same step goes out over the window.

    Killed by: src/uclone_x/agent/base.py :: if step_refusal is not None:
    Becomes: if False:
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    names = [f"f{i:02d}.txt" for i in range(40)]
    for name in names:
        _near_cap_file(tmp_path, name)
    llm = _ScriptedLLM(
        [
            [
                ToolCallRequest(id=f"c{i}", name="file_read", arguments={"path": name})
                for i, name in enumerate(names)
            ]
        ]
    )
    agent = _default_registry_agent(tmp_path, llm)
    await agent.execute_turn("read them all")

    assert all(estimate_request_tokens(request) <= 8_192 for request in llm.requests)


# ======================================================================================
# The step budget, continued (#1509)
# ======================================================================================


def _parallel_reads(workspace: Path, count: int) -> list[ToolCallRequest]:
    names = [f"f{i:03d}.txt" for i in range(count)]
    for name in names:
        _near_cap_file(workspace, name)
    return [
        ToolCallRequest(id=f"c{i}", name="file_read", arguments={"path": name})
        for i, name in enumerate(names)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [160, 300])
async def test_the_turns_after_a_refused_step_stay_within_the_window(
    tmp_path: Path, count: int
) -> None:
    """The review's probe: after a refused step of many parallel near-cap reads, the next
    turns were sent over an 8K window (10,506 tokens at 160 reads, 15,371 at 300).

    The refused step never reached the model, so it leaves the conversation; the next
    turn is told, in its turn context, which calls ran. Every request sent, on every
    turn, stays within the window, and the turns after the refusal answer.

    Killed by: src/uclone_x/agent/base.py :: del history[start:]
    Becomes: del history[len(history):]
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    llm = _ScriptedLLM([_parallel_reads(tmp_path, count)])
    agent = _default_registry_agent(tmp_path, llm)

    refused = await agent.execute_turn("read them all")
    second = await agent.execute_turn("then just tell me what you can")
    third = await agent.execute_turn("and once more")

    assert refused.stop_reason == "step_results_over_window"
    # The step's own calls fill the window here, not the conversation before it.
    assert refused.error == STEP_OVER_WINDOW_MESSAGE
    assert second.is_completed, second.error
    assert third.is_completed, third.error
    sizes = [estimate_request_tokens(request) for request in llm.requests]
    assert len(sizes) == 3, sizes
    assert all(size <= 8_192 for size in sizes), sizes
    # The next turn is told what the withheld step called; the one after is not.
    assert "[Undone Attempt]" in (llm.requests[1].messages[-1].content or "")
    assert "file_read" in (llm.requests[1].messages[-1].content or "")
    assert "[Undone Attempt]" not in (llm.requests[2].messages[-1].content or "")


def _three_near_cap_reads(workspace: Path) -> list[ToolCallRequest]:
    names = ["a.txt", "b.txt", "c.txt"]
    for name in names:
        _near_cap_file(workspace, name)
    return [
        ToolCallRequest(id=f"c{i}", name="file_read", arguments={"path": name})
        for i, name in enumerate(names)
    ]


def _budget_agent(
    workspace: Path, llm: MockLLMConnector, *, max_tokens: int | None = None
) -> BaseAgent:
    from uclone_x.tools.registry import create_default_registry

    return BaseAgent(
        config=AgentConfig(
            agent_id="ingest",
            name="Ingest",
            workspace_dir=workspace,
            llm_config=AgentLLMConfig(
                model_name="mock-model", context_limit=8_192, max_tokens=max_tokens
            ),
        ),
        llm=llm,
        tools=create_default_registry(workspace_root=workspace, enable_mcp=False),
    )


@pytest.mark.asyncio
async def test_a_fitted_step_leaves_room_for_the_reply_the_agent_asks_for(
    tmp_path: Path,
) -> None:
    """The window counts the reply too: a request cut to the window's last token leaves
    the model no room to answer. With `max_tokens` set, that much is kept free.

    Killed by: src/uclone_x/agent/base.py :: return max_tokens if max_tokens is not None else STEP_REPLY_RESERVE_TOKENS
    Becomes: return 0
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    llm = _ScriptedLLM([_three_near_cap_reads(tmp_path)])
    agent = _budget_agent(tmp_path, llm, max_tokens=1_500)
    result = await agent.execute_turn("read all three")

    assert result.is_completed, result.error
    assert len(llm.requests) == 2
    assert estimate_request_tokens(llm.requests[1]) + 1_500 <= 8_192


@pytest.mark.asyncio
async def test_a_fitted_step_leaves_room_for_a_reply_when_no_length_is_set(
    tmp_path: Path,
) -> None:
    """With no `max_tokens` the server picks the reply's length; room is kept anyway.

    Killed by: src/uclone_x/agent/base.py :: else STEP_REPLY_RESERVE_TOKENS
    Becomes: else 0
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    llm = _ScriptedLLM([_three_near_cap_reads(tmp_path)])
    agent = _budget_agent(tmp_path, llm)
    result = await agent.execute_turn("read all three")

    assert result.is_completed, result.error
    assert len(llm.requests) == 2
    assert estimate_request_tokens(llm.requests[1]) + STEP_REPLY_RESERVE_TOKENS <= 8_192


@pytest.mark.asyncio
async def test_a_step_of_many_one_byte_results_is_fitted_not_refused(tmp_path: Path) -> None:
    """Each message's estimate rounds up: a one-byte result costs a whole token, not a
    quarter. Two hundred of them beside one near-cap result are four times dearer than
    their bytes say. The budget keeps a token back per result, so the step is cut to
    fit and sent, rather than cut too little and then refused.

    The window is set from the step's own size, measured on an agent with no window, so
    the step is over it by a fixed amount whatever the fixed part weighs.

    Killed by: src/uclone_x/agent/base.py :: budget = window - reserve - rest - len(tail) - 2
    Becomes: budget = window - reserve - rest
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    near_cap = "\n".join(
        f"big line {i:04d} " + "y" * 60 for i in range(TOOL_RESULT_CAP_BYTES // 80)
    )
    tools = [_returning("big", near_cap), _returning("tiny", "x")]

    def step() -> list[ToolCallRequest]:
        return [
            ToolCallRequest(id="big0", name="big", arguments={}),
            *(ToolCallRequest(id=f"t{i}", name="tiny", arguments={}) for i in range(200)),
        ]

    probe_llm = _ScriptedLLM([step()])
    await _agent(tmp_path / "probe", probe_llm, tools).execute_turn("go")
    whole = estimate_request_tokens(probe_llm.requests[1])

    reserve = 64
    window = whole + reserve - 1_000
    llm = _ScriptedLLM([step()])
    agent = _agent(
        tmp_path / "real",
        llm,
        tools,
        llm_config=AgentLLMConfig(
            model_name="mock-model", context_limit=window, max_tokens=reserve
        ),
    )
    result = await agent.execute_turn("go")

    assert result.is_completed, result.error
    assert len(llm.requests) == 2
    assert estimate_request_tokens(llm.requests[1]) + reserve <= window
    by_id = {m.tool_call_id: m.content for m in _tool_messages(llm.requests[1].messages)}
    assert handle_in(by_id["big0"]) is not None
    assert all(by_id[f"t{i}"] == "x" for i in range(200))


@pytest.mark.asyncio
async def test_a_step_still_over_the_window_after_cutting_is_refused_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last check stands behind the share arithmetic: whatever the shares say, a
    request still over the window after cutting is refused, not sent.

    The shares are replaced with ones that cut nothing, the fault the check exists for.

    Killed by: src/uclone_x/agent/base.py :: if fitted + reserve > window:
    Becomes: if False:
    """
    import uclone_x.agent.base as base_module
    from uclone_x.llm.compactor import estimate_request_tokens

    def cut_nothing(sizes: Sequence[int], budget_bytes: int, **_: Any) -> list[int]:
        return list(sizes)

    monkeypatch.setattr(base_module, "step_result_caps", cut_nothing)
    llm = _ScriptedLLM([_three_near_cap_reads(tmp_path)])
    agent = _budget_agent(tmp_path, llm)
    result = await agent.execute_turn("read all three")

    assert len(llm.requests) == 1
    assert all(estimate_request_tokens(r) <= 8_192 for r in llm.requests)
    assert result.stop_reason == "step_results_over_window"
    assert result.error == STEP_OVER_WINDOW_MESSAGE


@pytest.mark.asyncio
async def test_a_step_refused_for_a_full_conversation_does_not_blame_the_tools(
    tmp_path: Path,
) -> None:
    """When the conversation before the step already takes the window, a step of one
    one-byte result is refused -- and the refusal says the conversation is the cause, in
    plain words, rather than that the tools returned too much.

    The window is set just above the first request, measured on an agent with no window.

    Killed by: src/uclone_x/agent/base.py :: if outside + reserve >= window:
    Becomes: if False:
    """
    from uclone_x.llm.compactor import estimate_request_tokens

    prompt = "Please keep all of this in mind. " * 480
    tools = [_returning("tiny", "x")]

    def step() -> list[list[ToolCallRequest]]:
        return [[ToolCallRequest(id="t0", name="tiny", arguments={})]]

    probe_llm = _ScriptedLLM(step())
    await _agent(tmp_path / "probe", probe_llm, tools).execute_turn(prompt)
    first = estimate_request_tokens(probe_llm.requests[0])

    window = first + 32
    llm = _ScriptedLLM(step())
    agent = _agent(
        tmp_path / "real",
        llm,
        tools,
        llm_config=AgentLLMConfig(
            model_name="mock-model", context_limit=window, max_tokens=64, auto_compact=False
        ),
    )
    result = await agent.execute_turn(prompt)

    assert len(llm.requests) == 1
    assert result.stop_reason == "step_results_over_window"
    assert result.error == STEP_NO_ROOM_MESSAGE
    for internal in ("/", "\\", "tr_", "Error", "Traceback", "token", str(tmp_path), "{"):
        assert internal not in STEP_NO_ROOM_MESSAGE
