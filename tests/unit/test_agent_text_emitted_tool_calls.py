"""A tool call written into message content is counted, and never executed (#694).

`qwen2.5-coder:14b` on Ollama answers "read pyproject.toml" with the *call* as its content
and an empty `tool_calls`. `execute_turn` sees no structured call, stops, and hands the
blob back as the answer. A 100-problem `frontier_live` run graded 100 of those and
published 1.1% correct for a model that was trying to use tools throughout.

Two things are pinned here and they pull in opposite directions on purpose:

* the discarded call is **counted**, so a run can tell "declined to use tools" from "every
  call fell on the floor" — two readings of the same zero with opposite meanings;
* the discarded call is **not executed**, because recovering it would make model prose an
  entry point into the tool path. See the commit message for the argument.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentLLMConfig
from uclone_x.agent.text_tool_calls import detect_text_emitted_tool_calls
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.models import FinishReason, ModelResponse, TokenUsage, ToolCallRequest
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

#: Verbatim from a live `qwen2.5-coder:14b` turn, 2026-09-11, asked to read
#: `pyproject.toml` through `build_agent_answerer(provider="ollama", ...)`. Kept exactly as
#: the model emitted it — a hand-tidied approximation would stop being the reproduction.
QWEN_CONTENT = """{
  "name": "file_read",
  "arguments": {
    "path": "pyproject.toml",
    "start_line": 1,
    "end_line": null,
    "max_bytes": 45000,
    "max_lines": 800
  }
}"""


class _NoParams(BaseModel):
    path: str = "pyproject.toml"


class _CountingTool(BaseTool[_NoParams]):
    """Records every execution, so "it was never run" is an assertion and not a hope."""

    name = "file_read"
    description = "Read a file"

    def __init__(self) -> None:
        super().__init__()
        self.runs = 0

    def run(self, params: _NoParams, context: ToolContext) -> str:
        self.runs += 1
        return '[project]\nname = "uclone-x"'


def _provenance() -> Provenance:
    return Provenance(
        path=ExecutionPath.PRIMARY,
        requested=ServiceRef(provider="test", model="mock"),
        served_by=ServiceRef(provider="test", model="mock"),
    )


def _agent(tmp_path: Path, response: ModelResponse) -> tuple[BaseAgent, _CountingTool, MagicMock]:
    tool = _CountingTool()
    registry = ToolRegistry()
    registry.register(tool)
    llm = MagicMock(spec=LLMProviderProtocol)
    llm.generate = AsyncMock(return_value=response)
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="text_calls",
            name="TextCalls",
            workspace_dir=str(tmp_path),
            llm_config=AgentLLMConfig(model_name="mock"),
        ),
        llm=llm,
        tools=registry,
    )
    return agent, tool, llm


def _answer(content: str, tool_calls: tuple[ToolCallRequest, ...] = ()) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=content,
        tool_calls=tool_calls,
        usage=TokenUsage(
            provider="test", model="mock", input_tokens=1, output_tokens=1, total_tokens=2
        ),
        provenance=_provenance(),
    )


# ---------------------------------------------------------------------------
# Part 1 — it is counted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tool_call_written_into_content_is_counted(tmp_path: Path) -> None:
    """The observed `qwen2.5-coder:14b` turn must leave a count behind, not just a blob.

    Without this the turn is indistinguishable from a model that answered in prose: zero
    tool calls, a completed turn, and a string that grades. That indistinguishability is
    what let a hundred discarded invocations reach a published capability figure.

    Killed by: src/uclone_x/agent/base.py :: detect_text_emitted_tool_calls(resp_content, shown_tool_names)
    """
    agent, _, _llm = _agent(tmp_path, _answer(QWEN_CONTENT))

    turn = await agent.execute_turn("Read pyproject.toml and tell me the project name.")

    assert turn.tool_calls == ()
    assert turn.text_emitted_tool_calls == ("file_read",)

    # The control. A count is only worth reading if a well-behaved turn scores zero; this
    # half has no mutation of its own to be killed by, which is why it lives inside the
    # positive test rather than claiming a `Killed by` line it cannot support.
    plain, _, _plain_llm = _agent(tmp_path, _answer("The project name is uclone-x."))
    assert (await plain.execute_turn("What is the project name?")).text_emitted_tool_calls == ()


@pytest.mark.asyncio
async def test_content_alongside_a_real_call_is_not_counted(tmp_path: Path) -> None:
    """A step that *did* call a tool discarded nothing, whatever else it printed.

    The defect is an invocation falling on the floor. A model that both invoked and
    narrated used the structured channel successfully, and counting it would make the
    metric fire on working runs -- a field that asserts work was discarded, on a turn where
    the tools ran.

    Killed by: src/uclone_x/agent/base.py :: if not tool_calls:
    """
    agent, tool, llm = _agent(
        tmp_path,
        _answer(
            QWEN_CONTENT,
            (ToolCallRequest(id="call_1", name="file_read", arguments={"path": "x"}),),
        ),
    )
    # The stub asks for the same tool forever; let the second step answer plainly.
    from tests.conftest import finish_after_tools

    finish_after_tools(llm)

    turn = await agent.execute_turn("Read pyproject.toml.")

    assert tool.runs == 1
    assert turn.text_emitted_tool_calls == ()


def test_an_unregistered_name_is_not_counted() -> None:
    """JSON naming some other system's API is not this defect.

    Membership in the registry is what separates "an invocation was discarded" from "the
    model emitted JSON". Drop it and the count measures output formatting.

    Killed by: src/uclone_x/agent/text_tool_calls.py :: if name is not None and name in registered:
    """
    blob = '{"name": "stripe_charge", "arguments": {"amount": 100}}'

    assert detect_text_emitted_tool_calls(blob, {"file_read", "bash_run"}) == ()
    assert detect_text_emitted_tool_calls(blob, {"stripe_charge"}) == ("stripe_charge",)


def test_a_record_that_merely_has_a_name_field_is_not_counted() -> None:
    """`{"name": "file_read", "size": 10}` is a row about a tool, not a call to one.

    A false positive here cannot refuse a run -- `_toolless_run_refusal` decides from the
    zero-tool-call share alone -- but it misdiagnoses one: the turn asserts work was
    discarded when none was, and a toolless run's stated cause flips to the #694
    attribution and sends a triager to the wrong fix. So the shape check is a whitelist
    rather than "has a name". It is a whitelist with a known hole, pinned below.

    Killed by: src/uclone_x/agent/text_tool_calls.py :: if not keys or not keys <= _CALL_KEYS:
    """
    assert detect_text_emitted_tool_calls('{"name": "file_read", "size": 10}', {"file_read"}) == ()


def test_a_tool_definition_is_currently_counted_and_that_is_the_known_hole() -> None:
    """Characterisation, not approval: `parameters` is in `_CALL_KEYS`, so a schema counts.

    A correct answer to "what arguments does file_read take?" is call-shaped, and so is a
    bare `{"name": ...}`. What saves the OpenAI-style definition echo is only that
    `description` falls outside the key set -- accidental, and the module docstring says
    plainly that it is not meant to be load-bearing. The designed narrowing is to require
    an argument-carrying key (`arguments`/`args`), which a definition does not have; that
    is a behaviour change with its own tests and its own issue.

    This test carries no `Killed by:` line because it pins no fix -- it exists so the
    narrowing, when it comes, arrives as a failing test someone has to update deliberately
    rather than as a silent change to what the count means. There is no mutation to watch:
    the same key-set check already has a marker on the test above, and a second claim on
    it would not discriminate.
    """
    schema = '{"name": "file_read", "parameters": {"path": "string", "max_lines": "integer"}}'
    assert detect_text_emitted_tool_calls(schema, {"file_read"}) == ("file_read",)

    echo = '{"type": "function", "function": {"name": "file_read", "description": "Read"}}'
    assert detect_text_emitted_tool_calls(echo, {"file_read"}) == ()


def test_a_fenced_call_is_counted() -> None:
    """Models that emit a call as prose usually dress it as a code block.

    Killed by: src/uclone_x/agent/text_tool_calls.py :: fenced = _FENCE_RE.match(content)
    """
    fenced = '```json\n{"name": "file_read", "arguments": {"path": "a"}}\n```'

    assert detect_text_emitted_tool_calls(fenced, {"file_read"}) == ("file_read",)


def test_tagged_calls_are_counted_once_each() -> None:
    """`<tool_call>` reaching `content` means the provider did not lift it out.

    Both qwen Ollama templates ask for this wrapper. Seeing it in the answer is the same
    defect with a different surface, and a message may carry several.

    Killed by: src/uclone_x/agent/text_tool_calls.py :: tagged = [m.group("body") for m in _TOOL_CALL_TAG_RE.finditer(content)]
    """
    tagged = (
        'Sure.<tool_call>{"name": "file_read", "arguments": {"path": "a"}}</tool_call>'
        '<tool_call>{"name": "bash_run", "arguments": {"command": "pwd"}}</tool_call>'
    )

    assert detect_text_emitted_tool_calls(tagged, {"file_read", "bash_run"}) == (
        "file_read",
        "bash_run",
    )


# ---------------------------------------------------------------------------
# Part 2 — it is not executed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_discarded_call_is_not_executed(tmp_path: Path) -> None:
    """Detection must not become invocation.

    The decision argued in this commit is that model-authored text is not an invocation:
    the channel is the only evidence of whether the model was addressing the runtime or the
    user, and a recovery path would make any tool-shaped JSON that reaches `content` — from
    a file the agent read, a page it fetched, a prompt injected into it — a candidate entry
    point. The hooks bound what an executed call may do; they cannot restore the evidence
    of who authored it.

    The non-execution half of this test has no mutation to be killed by -- absence is not
    a line -- so it is a guard against a future change rather than a pinned behaviour, and
    it is written here plainly rather than dressed up as one. The marker below pins the
    half that *is* a line and that no other test asserts: the discarded blob is handed back
    as the answer, unaltered. "Surface it" must not quietly become "rewrite it", because a
    runtime that edits the answer has started making the model's text into something else
    -- which is one short step from making it into a call. (The detection call was this
    test's earlier marker; it is already pinned by
    `test_a_tool_call_written_into_content_is_counted`, and a marker that cannot fail alone
    is not evidence about this test.)

    Killed by: src/uclone_x/agent/base.py :: resp_content = resp.content or ""
    Becomes: resp_content = (resp.content or "") + " "
    """
    agent, tool, _ = _agent(tmp_path, _answer(QWEN_CONTENT))

    turn = await agent.execute_turn("Read pyproject.toml.")

    assert turn.text_emitted_tool_calls == ("file_read",)
    assert tool.runs == 0
    assert turn.tool_executions == ()
    # The blob is returned as the answer, unaltered. Pinned because "surface it" must not
    # quietly become "rewrite the answer": the caller sees what the model actually said.
    assert turn.content == QWEN_CONTENT


@pytest.mark.asyncio
async def test_the_runtime_warns_and_names_the_remedy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The person holding the broken model must learn this from the runtime.

    `text_emitted_tool_calls` travels on `TurnResult` and into the eval report, and neither
    is in front of a `qwen2.5-coder` user: they get a JSON fragment where an answer should
    be and no indication anything went wrong. The remedy -- a model whose calls arrive
    structured, or an `ollama create` variant -- lived only in a PR body. A log line is
    where it actually reaches them.

    The marker names the remedy text rather than the condition, so that it fails if the
    advice is deleted from the message -- a warning that says only "this happened" is the
    version this test exists to prevent. The condition itself is pinned by the test below.

    Killed by: src/uclone_x/agent/base.py :: (qwen3:8b, qwen2.5:7b-instruct are verified)
    """
    agent, _, _llm = _agent(tmp_path, _answer(QWEN_CONTENT))

    with caplog.at_level("WARNING", logger="uclone_x.agent.base"):
        await agent.execute_turn("Read pyproject.toml.")

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("file_read" in m and "#694" in m for m in warnings), warnings
    assert any("qwen3:8b" in m for m in warnings), warnings


@pytest.mark.asyncio
async def test_a_turn_whose_tools_ran_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn whose tools ran discarded nothing, whatever a later step printed.

    The hard case, and the reason the condition is a conjunction: step one calls the tool
    through the structured channel, step two prints the call as text. `text_emitted` is
    non-empty and the work happened anyway. Warning here would fire on healthy turns, which
    is how the warning above stops reaching the people it was written for.

    Killed by: src/uclone_x/agent/base.py :: if text_emitted and not tool_executions:
    """
    agent, tool, llm = _agent(
        tmp_path,
        _answer(
            "",
            (ToolCallRequest(id="call_1", name="file_read", arguments={"path": "x"}),),
        ),
    )
    # Step 1 invokes properly; step 2 answers with the blob and no structured call.
    llm.generate.side_effect = [llm.generate.return_value, _answer(QWEN_CONTENT)]

    with caplog.at_level("WARNING", logger="uclone_x.agent.base"):
        turn = await agent.execute_turn("Read pyproject.toml.")

    assert tool.runs == 1
    assert turn.text_emitted_tool_calls == ("file_read",)
    assert not [r for r in caplog.records if "#694" in r.getMessage()]


def test_the_detector_returns_names_and_not_invocations() -> None:
    """There is nothing here a caller could `await`.

    The argument against recovery is only as strong as the shape of the thing this returns.
    Hand back a parsed call object and the next caller is one line from executing it, which
    is how a deliberate refusal decays into an accident.

    Killed by: src/uclone_x/agent/text_tool_calls.py :: found.append(name)
    """
    found = detect_text_emitted_tool_calls(QWEN_CONTENT, {"file_read"})

    assert found == ("file_read",)
    assert all(isinstance(entry, str) for entry in found)
