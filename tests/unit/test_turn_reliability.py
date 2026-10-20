"""A turn never ends in an empty reply (#2264), and a runaway thought is cut (#2202).

Both are core: they hold for every clone and every model, and nothing here reads a
story or a language beyond the person's message choosing the sentence's language.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast, get_args

import httpx
import pytest
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.work_done import WORK_DONE_TEXT, WorkDoneKey, work_done_text
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.core.tool_results import STEP_REFUSAL_TEXT, StepRefusalCode
from uclone_x.errors import PlainRefusalError
from uclone_x.i18n.language import Language
from uclone_x.i18n.refusals import refusal_text
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
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
from uclone_x.room.one_seat import HeadTurn
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.a2a import A2ACallTool
from uclone_x.tools.builtin.shell import BashRunTool
from uclone_x.tools.client import MCPTool
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model="m"),
    served_by=ServiceRef(provider="scripted", model="m"),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model="m", input_tokens=0, output_tokens=0)


def _reply(content: str | None, *calls: ToolCallRequest) -> ModelResponse:
    return ModelResponse(
        content=content,
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=calls,
        usage=_USAGE,
        provenance=_PROV,
    )


class _Scripted(BaseLLMConnector):
    """Answers each request with the next scripted reply, streamed as one chunk."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self._responses.pop(0)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        resp = await self.generate(request)
        yield StreamChunk(
            delta_content=resp.content,
            finish_reason=resp.finish_reason,
            tool_calls=resp.tool_calls,
            usage=_USAGE,
        )


class _NoParams(BaseModel):
    pass


class _RefusingTool(BaseTool[_NoParams]):
    """Refuses every call, as the story tools refused the live Writer's writes."""

    name = "refusing"
    description = "Refuses every call."

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        raise PlainRefusalError("Nothing was written: the person has not answered yet.")


def _agent(tmp_path: Path, llm: BaseLLMConnector, *more: BaseTool[Any], **config: Any) -> BaseAgent:
    tools = ToolRegistry()
    tools.register(_RefusingTool())
    for tool in more:
        tools.register(tool)
    return BaseAgent(
        config=AgentConfig(
            agent_id="clone",
            name="Clone",
            llm_config=AgentLLMConfig(model_name="m"),
            **config,
        ),
        llm=llm,
        tools=tools,
        context=AgentContext(agent_id="clone", session_id="s", workspace_root=tmp_path),
    )


async def _quiet(kind: str, data: dict[str, Any]) -> None:
    del kind, data


def _events(agent: BaseAgent, kind: str) -> list[Mapping[str, Any]]:
    return [e for e in agent.pending_durable_events if e.get("type") == kind]


# --- #2264: an empty reply ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_answer_still_empty_after_the_re_ask_says_so_plainly(tmp_path: Path) -> None:
    """The re-ask got an empty answer too: the turn ends with the plain sentence, not "".

    Killed by: src/uclone_x/agent/turn_executor.py :: if not shown.strip():
    Becomes: if False:
    """
    llm = _Scripted([_reply(""), _reply("")])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("Write the next scene.")

    assert len(llm.requests) == 2
    assert _events(agent, "EMPTY_REPLY_NUDGE")
    assert result.content == ""
    assert result.error == STEP_REFUSAL_TEXT["step.empty_reply"]
    assert result.error_code == "step.empty_reply"
    assert result.stop_reason == "empty_reply"
    assert result.is_completed is False


@pytest.mark.asyncio
async def test_the_live_shape_refused_calls_then_empty_after_the_re_ask(tmp_path: Path) -> None:
    """The recorded mind_changer T3/T5 shape (#2264): the turn's calls were all refused, the
    re-ask was spent on an empty step, the model called again, and the turn ended empty.
    Before, that was a completed row with content "" and nothing saying why.

    Killed by: src/uclone_x/agent/turn_executor.py :: raise EmptyReplyError(self._turn_language)
    Becomes: pass
    """
    call = ToolCallRequest(id="c1", name="refusing", arguments={})
    again = ToolCallRequest(id="c2", name="refusing", arguments={"x": 1})
    llm = _Scripted([_reply(None, call), _reply(""), _reply(None, again), _reply("")])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("네, 1번으로 시작해 주세요.", stream_callback=_quiet)

    assert len(llm.requests) == 4
    assert len(_events(agent, "EMPTY_REPLY_NUDGE")) == 1
    assert result.error_code == "step.empty_reply"
    # In the person's language, as a refused step's row is (#2300).
    assert result.error == refusal_text("step.empty_reply", "ko")
    # The room records it on the row, as it records a refused step.
    row = HeadTurn.from_result("네", result)
    assert row.step_refusal == "step.empty_reply"
    assert row.error == refusal_text("step.empty_reply", "ko")


@pytest.mark.asyncio
async def test_an_answer_the_re_ask_got_is_kept(tmp_path: Path) -> None:
    llm = _Scripted([_reply(""), _reply("Here it is.")])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("Write the next scene.")

    assert result.content == "Here it is."
    assert result.error is None
    assert result.error_code is None


# --- #2202: a runaway thought --------------------------------------------------------------


class _Thinker(BaseLLMConnector):
    """Streams `thinking[i]` thinking pieces for request i, then an answer.

    A request whose plan is `None` thinks without end: only the caller closing the stream
    stops it, and `closed` counts the streams that were closed before they finished.
    """

    def __init__(self, plans: Sequence[int | None]) -> None:
        super().__init__()
        self._plans = list(plans)
        self.requests: list[LLMRequest] = []
        self.closed = 0
        self.closed_before_next_request: list[bool] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        raise AssertionError("a streamed turn never calls generate")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.closed_before_next_request.append(self.closed == len(self.requests))
        self.requests.append(request)
        plan = self._plans.pop(0)
        finished = False
        try:
            sent = 0
            while plan is None or sent < plan:
                yield StreamChunk(delta_thinking="hmm ")
                sent += 1
                if sent > 100_000:
                    raise AssertionError("the stream was never cut")
            yield StreamChunk(
                delta_content="The answer.", finish_reason=FinishReason.STOP, usage=_USAGE
            )
            finished = True
        finally:
            if not finished:
                self.closed += 1


@pytest.mark.asyncio
async def test_a_thought_at_the_budget_is_left_alone(tmp_path: Path) -> None:
    """Normal requests think far less (p99 about 1.8k on qwen3:8b): one at exactly the
    budget is not cut, retried or re-marked.

    Killed by: src/uclone_x/agent/turn_executor.py :: if budget is not None and len(thinking_chunks) > budget:
    Becomes: if budget is not None and len(thinking_chunks) >= budget:
    """
    llm = _Thinker([8_192])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("Hello.", stream_callback=_quiet)

    assert result.content == "The answer."
    assert len(llm.requests) == 1
    assert llm.requests[0].thinking is None
    assert llm.closed == 0
    assert result.provenance is not None
    assert result.provenance.path is ExecutionPath.PRIMARY
    assert not _events(agent, "THINKING_BUDGET_STOP")


@pytest.mark.asyncio
async def test_a_runaway_thought_is_cut_closed_and_asked_once_without_thinking(
    tmp_path: Path,
) -> None:
    """Cut at 8,192 thinking pieces; the stream is closed before the retry is sent; the
    retry is the same messages with thinking off, and the answer says it was a retry.

    Killed by: src/uclone_x/agent/turn_executor.py :: req = req if cut is None else req.model_copy(update={"thinking": False})
    Becomes: req = req
    Killed by: src/uclone_x/agent/turn_executor.py :: await aclose()
    Becomes: pass
    """
    llm = _Thinker([None, 0])
    agent = _agent(tmp_path, llm)
    statuses: list[str] = []

    async def on_event(kind: str, data: dict[str, Any]) -> None:
        if kind == "status" and data.get("status") == "retrying":
            statuses.append(str(data.get("detail")))

    result = await agent.execute_turn("Hello.", stream_callback=on_event)

    assert result.content == "The answer."
    assert result.error is None
    assert len(llm.requests) == 2
    assert llm.closed == 1
    assert llm.closed_before_next_request == [True, True]
    first, retry = llm.requests
    assert first.thinking is None
    assert retry.thinking is False
    assert retry.messages == first.messages
    stops = _events(agent, "THINKING_BUDGET_STOP")
    assert len(stops) == 1
    assert stops[0]["reason"] == "thinking_budget"
    assert stops[0]["thinking_tokens"] == 8_193
    assert stops[0]["budget"] == 8_192
    assert stops[0]["elapsed_s"] >= 0
    # P6: the retry is attributable in band, with the span that records it.
    assert result.provenance is not None
    assert result.provenance.path is ExecutionPath.RETRY
    (attempt,) = result.provenance.attempts
    assert attempt.error_class == "ThinkingBudgetExceededError"
    assert attempt.span_id
    # The person is told in plain words while the retry runs.
    assert statuses and "thinking" in statuses[0].lower()
    assert "8192" not in statuses[0] and "_" not in statuses[0]


@pytest.mark.asyncio
async def test_a_thought_that_runs_away_again_stops_with_the_plain_sentence(
    tmp_path: Path,
) -> None:
    """One retry, not two: a model that thinks past the budget again ends the turn.

    Killed by: src/uclone_x/agent/turn_executor.py :: if cut is not None:
    Becomes: if False:
    """
    llm = _Thinker([None, None, 0])
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("다음 장면을 써줘", stream_callback=_quiet)

    assert len(llm.requests) == 2
    assert llm.closed == 2
    assert result.content == ""
    assert result.error_code == "step.thinking_budget"
    assert result.stop_reason == "thinking_budget"
    assert result.error == refusal_text("step.thinking_budget", "ko")


@pytest.mark.asyncio
async def test_no_budget_set_means_no_cut(tmp_path: Path) -> None:
    llm = _Thinker([9_000])
    agent = _agent(tmp_path, llm, thinking_budget_tokens=None)

    result = await agent.execute_turn("Hello.", stream_callback=_quiet)

    assert result.content == "The answer."
    assert len(llm.requests) == 1


# --- the connector: closing the stream ends the request ----------------------------------


@pytest.mark.asyncio
async def test_closing_an_ollama_stream_closes_its_http_response() -> None:
    """Closing the connector's generator closes the HTTP response it reads, which is what
    stops generation on the Ollama server (#2202).

    Killed by: src/uclone_x/llm/connectors/ollama.py :: async with client.stream("POST", url, json=payload, timeout=self.timeout) as resp:
    Becomes: for resp in [await client.send(client.build_request("POST", url, json=payload, timeout=self.timeout), stream=True)]:
    """
    closed: list[bool] = []

    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                yield b'{"message": {"thinking": "hmm"}}\n'

        async def aclose(self) -> None:
            closed.append(True)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=_Body())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = OllamaConnector(base_url="http://ollama.test", http_client=client)
    stream = cast(
        AsyncGenerator[StreamChunk, None],
        connector.stream(
            LLMRequest(model="m", messages=(ChatMessage(role=MessageRole.USER, content="hi"),))
        ),
    )
    seen = 0
    async for chunk in stream:
        assert chunk.delta_thinking == "hmm"
        seen += 1
        if seen == 3:
            break
    assert closed == []
    await stream.aclose()

    assert closed == [True]
    await client.aclose()


# --- the app says every code the Core sends ----------------------------------------------


_FRONTEND_LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "i18n" / "locales"


@pytest.mark.parametrize("language", ["en", "ko"])
def test_the_app_has_the_cores_sentence_for_every_step_code(language: Language) -> None:
    """A row's `step_refusal` the app had no sentence for would show the generic failure
    line instead of saying why (#2264): every code the Core can send is in each catalog,
    as the Core says it.

    `Killed by:` hand-run: deleting the `"step.empty_reply"` entry from
    `frontend/src/i18n/locales/en/conversation.json` fails the `en` case.
    """
    catalog = json.loads((_FRONTEND_LOCALES / language / "conversation.json").read_text("utf-8"))
    causes = catalog["outcome"]["stepCause"]

    for code in get_args(StepRefusalCode):
        assert causes.get(code) == refusal_text(code, language), code


# --- #2322 item 3: a model that can only think refuses the thinking-off retry ------------


@pytest.mark.asyncio
async def test_a_retry_the_ollama_server_refuses_ends_with_the_stop_sentence(
    tmp_path: Path,
) -> None:
    """A model that can only think may be refused `think: false` by Ollama with a 400. The
    retry then never ran, and the turn ends as a second cut does -- the plain
    `step.thinking_budget` sentence -- not as a provider error with the server's text.

    Killed by: src/uclone_x/llm/connectors/ollama.py :: if status_code == 400:
    Becomes: if False:
    """
    sent: list[dict[str, Any]] = []

    class _Thinking(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                yield b'{"message": {"thinking": "hmm"}}\n'

        async def aclose(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if body.get("think") is False:
            return httpx.Response(400, json={"error": "this model cannot turn thinking off"})
        return httpx.Response(200, stream=_Thinking())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = OllamaConnector(base_url="http://ollama.test", http_client=client)
    agent = _agent(tmp_path, llm)

    result = await agent.execute_turn("다음 장면을 써줘", stream_callback=_quiet)
    await client.aclose()

    assert [body.get("think") for body in sent] == [None, False]
    assert result.error_code == "step.thinking_budget"
    assert result.stop_reason == "thinking_budget"
    assert result.error == refusal_text("step.thinking_budget", "ko")
    assert "400" not in (result.error or "")


@pytest.mark.asyncio
async def test_a_retry_that_fails_for_another_reason_stays_that_failure(tmp_path: Path) -> None:
    """Only a refusal of the request is read as "cannot think less": an outage on the retry
    is still an outage, reported as one.

    Killed by: src/uclone_x/agent/turn_executor.py :: if _in_cause_chain(retry_exc, LLMRequestRefusedError) is None:
    Becomes: if False:
    """

    class _Thinking(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                yield b'{"message": {"thinking": "hmm"}}\n'

        async def aclose(self) -> None:
            return None

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("think") is False:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, stream=_Thinking())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    agent = _agent(tmp_path, OllamaConnector(base_url="http://ollama.test", http_client=client))

    result = await agent.execute_turn("Hello.", stream_callback=_quiet)
    await client.aclose()

    assert result.error is not None
    assert result.error_code is None
    assert result.stop_reason != "thinking_budget"


# --- #2322 item 4: an empty reply after real work says what was done ----------------------


class _PictureTool(BaseTool[_NoParams]):
    name = "picture_maker"
    description = "Makes pictures."
    writes_files = False

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        return {"relative_url": "/api/artifacts/content?path=images/cat_01.png"}

    def produced_artifacts(self, output: Any) -> tuple[str, ...]:
        del output
        return ("images/cat_01.png", "images/cat_02.png")


class _FileTool(BaseTool[_NoParams]):
    name = "file_maker"
    description = "Writes a file."
    writes_files = True

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params
        assert context.workspace_root is not None
        written = context.workspace_root / "notes" / "plan_v2.md"
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("plan", encoding="utf-8")
        return {"path": "notes/plan_v2.md"}


class _ReadingTool(BaseTool[_NoParams]):
    name = "page_reader"
    description = "Reads."
    writes_files = False

    def run(self, params: _NoParams, context: ToolContext) -> dict[str, Any]:
        del params, context
        return {"text": "some text"}


def _silent_after(*names: str) -> _Scripted:
    """A model that calls `names` in one step, then writes nothing, re-ask included."""
    calls = [ToolCallRequest(id=f"c{i}", name=name, arguments={}) for i, name in enumerate(names)]
    return _Scripted([_reply(None, *calls), _reply(""), _reply("")])


_INTERNALS = ("/", "\\", ".png", ".md", "_", "{", "}", "picture_maker", "file_maker")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "language"), [("Draw a cat.", "en"), ("고양이를 그려 줘", "ko")]
)
async def test_pictures_made_then_silence_says_they_were_made(
    tmp_path: Path, message: str, language: Language
) -> None:
    """After a picture was made, a silent model must not tell the person "nothing to show,
    send again": they would ask for the same pictures twice (#2322).

    Killed by: src/uclone_x/agent/turn_executor.py :: done = work_done_lines(work_done(records), self._turn_language)
    Becomes: done = ()
    """
    agent = _agent(tmp_path, _silent_after("picture_maker"), _PictureTool())

    result = await agent.execute_turn(message, stream_callback=_quiet)

    assert result.error is None
    assert result.error_code is None
    assert result.content == work_done_text_of("work.pictures", language, count=2)
    for internal in _INTERNALS:
        assert internal not in result.content, internal
    final = [m for m in agent.history if m.role is MessageRole.ASSISTANT][-1]
    assert final.content == result.content


@pytest.mark.asyncio
async def test_one_picture_reads_as_one(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/work_done.py :: key: WorkDoneKey = "work.picture" if done.pictures == 1 else "work.pictures"
    Becomes: key: WorkDoneKey = "work.pictures"
    """

    class _One(_PictureTool):
        def produced_artifacts(self, output: Any) -> tuple[str, ...]:
            del output
            return ("images/cat_01.png",)

    agent = _agent(tmp_path, _silent_after("picture_maker"), _One())

    result = await agent.execute_turn("Draw a cat.", stream_callback=_quiet)

    assert result.content == work_done_text_of("work.picture", "en")


@pytest.mark.asyncio
async def test_a_file_written_then_silence_says_it_was_written(tmp_path: Path) -> None:
    """A file is not a picture: told apart by the produced file's suffix, not by words.

    Killed by: src/uclone_x/agent/work_done.py :: if PurePosixPath(path).suffix.lower() in PICTURE_SUFFIXES
    Becomes: if True
    """
    agent = _agent(tmp_path, _silent_after("file_maker"), _FileTool())

    result = await agent.execute_turn("파일로 정리해 줘", stream_callback=_quiet)

    assert result.error is None
    assert result.content == work_done_text_of("work.file", "ko")
    for internal in _INTERNALS:
        assert internal not in result.content, internal


@pytest.mark.asyncio
async def test_only_reading_then_silence_is_still_the_empty_reply_stop(tmp_path: Path) -> None:
    """Reading changed nothing, and a refused write wrote nothing, so asking again costs
    nothing: the empty-reply stop stands.

    Killed by: src/uclone_x/agent/work_done.py :: produced.extend(p for p in record.produced_paths if p not in produced)
    Becomes: produced.extend(["changes"] if record.writes_files else [])
    """
    agent = _agent(tmp_path, _silent_after("page_reader", "refusing"), _ReadingTool())

    result = await agent.execute_turn("Read it.", stream_callback=_quiet)

    assert result.error_code == "step.empty_reply"


def work_done_text_of(key: WorkDoneKey, language: Language, **values: int) -> str:
    return work_done_text(key, language).format(**values)


_WORK_KEYS: tuple[WorkDoneKey, ...] = get_args(WorkDoneKey)


@pytest.mark.parametrize("key", _WORK_KEYS)
def test_every_work_line_is_plain_in_korean_with_the_same_values(key: WorkDoneKey) -> None:
    """Plain copy: Korean only, 합니다체, no code or path, and the same `{count}` as English.

    Killed by: src/uclone_x/agent/work_done.py :: return _korean_catalog()[key]
    Becomes: return WORK_DONE_TEXT[key]
    """
    import re

    english = WORK_DONE_TEXT[key]
    korean = work_done_text(key, "ko")

    assert re.search(r"[가-힣]", korean)
    assert not re.search(r"[A-Za-z_/\\]", korean.replace("{count}", "")), korean
    assert korean.rstrip().endswith("니다.")
    assert set(re.findall(r"\{(\w+)\}", korean)) == set(re.findall(r"\{(\w+)\}", english))
    assert not re.search(r"[_/\\]", english.replace("{count}", ""))


# --- #2325 review: a declared capability to write is not a write ---------------------------


class _ReadOnlyMcpClient:
    """An MCP server whose tool answered with text and named no file."""

    async def call_tool(self, name: str, params: dict[str, Any], context: ToolContext) -> Any:
        del name, params, context
        return ToolResult(
            success=True, output={"content": "3 rows"}, provenance=Provenance.primary("mcp")
        )


class _PeerAnsweredA2A(A2ACallTool):
    """The real `a2a_call` (its declarations), with a peer that answered in words only."""

    async def execute(self, *args: Any, **kwargs: Any) -> ToolResult:
        del args, kwargs
        return ToolResult(
            success=True,
            output={"reply": "Done, see you tomorrow."},
            provenance=Provenance.primary("a2a"),
        )


def _has_a_peer(self: BaseAgent) -> bool:
    del self
    return True


def _read_only_tool(kind: str) -> tuple[str, dict[str, Any], Any]:
    if kind == "bash_run":
        return "bash_run", {"action": "run", "command": "ls"}, BashRunTool()
    if kind == "mcp":
        tool = MCPTool("lookup_rows", "Looks rows up.", {}, cast(Any, _ReadOnlyMcpClient()))
        return "lookup_rows", {}, tool
    return "a2a_call", {"agent": "peer", "task": "say hi"}, _PeerAnsweredA2A()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["bash_run", "mcp", "a2a_call"])
async def test_a_tool_that_may_write_but_produced_nothing_keeps_the_empty_reply_stop(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`writes_files` is a declared capability, true for all three, not a record that the
    call wrote: after an `ls`, a read-only MCP call or a peer that only answered, a silent
    model ends with `step.empty_reply`, never "I saved the changes." (#2325 review).

    Killed by: src/uclone_x/agent/work_done.py :: produced.extend(p for p in record.produced_paths if p not in produced)
    Becomes: produced.extend(["changes"] if record.writes_files else [])
    """
    # `a2a_call` is offered only to an agent with a peer to call; this one has one.
    monkeypatch.setattr(BaseAgent, "_may_call_peers", _has_a_peer)
    name, arguments, tool = _read_only_tool(kind)
    call = ToolCallRequest(id="c1", name=name, arguments=arguments)
    llm = _Scripted([_reply(None, call), _reply(""), _reply("")])
    agent = _agent(tmp_path, llm, tool)

    result = await agent.execute_turn("Look.", stream_callback=_quiet)

    (record,) = result.tool_executions
    assert record.writes_files is True
    assert record.produced_paths == ()
    assert result.content == ""
    assert result.error_code == "step.empty_reply"


# --- #2331: a path a tool names is produced only if the call made or changed it ------------


class _PathNamingMcpClient:
    """An MCP server whose tool reads a file and names it, as a lookup does."""

    async def call_tool(self, name: str, params: dict[str, Any], context: ToolContext) -> Any:
        del name, params, context
        return ToolResult(
            success=True,
            output={"path": "data/rows.csv", "content": "3 rows"},
            provenance=Provenance.primary("mcp"),
        )


class _PathWritingMcpClient:
    """An MCP server whose tool writes the file it names, during the call."""

    async def call_tool(self, name: str, params: dict[str, Any], context: ToolContext) -> Any:
        del name, params
        assert context.workspace_root is not None
        (context.workspace_root / "data" / "rows.csv").write_text("a,b\n", encoding="utf-8")
        return ToolResult(
            success=True,
            output={"paths": ["data/rows.csv"]},
            provenance=Provenance.primary("mcp"),
        )


def _old_file(tmp_path: Path) -> Path:
    existing = tmp_path / "data" / "rows.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text("a,b\n", encoding="utf-8")
    os.utime(existing, (time.time() - 3600, time.time() - 3600))
    return existing


@pytest.mark.asyncio
async def test_a_read_only_mcp_tool_that_names_an_existing_file_produced_nothing(
    tmp_path: Path,
) -> None:
    """The file was there before the call and the call did not touch it: naming it is not
    making it, and a silent model still ends with `step.empty_reply` (#2331).

    Killed by: src/uclone_x/agent/tool_execution.py :: if stat.st_mtime >= since:
    Becomes: if True:
    """
    existing = _old_file(tmp_path)
    tool: Any = MCPTool("lookup_rows", "Looks rows up.", {}, cast(Any, _PathNamingMcpClient()))
    llm = _Scripted(
        [
            _reply(None, ToolCallRequest(id="c1", name="lookup_rows", arguments={})),
            _reply(""),
            _reply(""),
        ]
    )
    agent = _agent(tmp_path, llm, tool)

    result = await agent.execute_turn("Look.", stream_callback=_quiet)

    (record,) = result.tool_executions
    assert record.writes_files is True
    assert record.produced_paths == ()
    assert result.error_code == "step.empty_reply"
    assert existing.read_text(encoding="utf-8") == "a,b\n"


@pytest.mark.asyncio
async def test_an_mcp_tool_that_rewrote_the_file_it_names_produced_it(tmp_path: Path) -> None:
    """The same file, changed during the call, is produced: the check reads the file, not
    whether it existed before (#2331).

    Killed by: src/uclone_x/agent/tool_execution.py :: written = changed_output_paths(res.output, tool_ctx.workspace_root, call_started)
    Becomes: written = ()
    """
    _old_file(tmp_path)
    tool: Any = MCPTool("write_rows", "Writes rows.", {}, cast(Any, _PathWritingMcpClient()))
    llm = _Scripted(
        [
            _reply(None, ToolCallRequest(id="c1", name="write_rows", arguments={})),
            _reply(""),
            _reply(""),
        ]
    )
    agent = _agent(tmp_path, llm, tool)

    result = await agent.execute_turn("Write it.", stream_callback=_quiet)

    (record,) = result.tool_executions
    assert record.produced_paths == ("data/rows.csv",)
    assert result.error is None
    assert result.content == work_done_text_of("work.file", "en")
