"""A call that found nothing must not look like a call that answered the question.

`ToolResultStatus` is binary, so a search matching nothing is a success, and
`{"total_matches": 0, "matches": []}` reaches the model as an unremarkable result.
Measured (#698): `qwen3:8b` missed with a regex and answered *"The line does not appear in
the file"*. It was there. That was one call of the median one per problem against a median
declared horizon of six (#697) — the turn did not run out of budget, it ran out of
willingness, and the empty result is what ended it.
"""

from __future__ import annotations

from typing import Any

import pytest

from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.tools.models import ToolResult
from uclone_x.tools.outcome import ToolOutcome, classify_tool_outcome

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="tool", model="t"),
    served_by=ServiceRef(provider="tool", model="t"),
    attempts=(),
)


def _result(output: Any, *, success: bool = True, error: str | None = None) -> ToolResult:
    return ToolResult(success=success, output=output, error=error, provenance=_PROV)


def test_a_search_that_matched_nothing_is_empty_not_productive() -> None:
    """The exact payload from the measured failure.

    Three keys and no findings: structurally non-empty, semantically nothing. A generic
    "is the dict empty" rule reports this as productive, which is how the misread survived.

    Killed by: src/uclone_x/tools/outcome.py :: for key in _COLLECTION_KEYS:
    """
    miss = _result({"query": "ZZZ", "is_regex": False, "total_matches": 0, "matches": []})

    assert classify_tool_outcome(miss) is ToolOutcome.EMPTY


def test_a_zero_count_with_no_collection_is_still_empty() -> None:
    """The count rule has to carry its own weight.

    The real `file_search` payload reports both a zero count *and* an empty `matches`
    list, so the collection rule alone classifies it — which means the count rule was
    dead code as far as the tests were concerned, and a first version of this file
    declared it as the killer of a test it does not kill. A tool that reports only a
    count is what makes it load-bearing.

    Killed by: src/uclone_x/tools/outcome.py :: for key in _COUNT_KEYS:
    """
    counted = _result({"query": "ZZZ", "total_matches": 0})

    assert classify_tool_outcome(counted) is ToolOutcome.EMPTY
    assert classify_tool_outcome(_result({"query": "x", "total_matches": 3})) is (
        ToolOutcome.PRODUCTIVE
    )


def test_a_search_that_matched_something_is_productive() -> None:
    """The other half of the same payload shape, so the rule is not simply "always empty".

    Killed by: src/uclone_x/tools/outcome.py :: value == 0
    Becomes: value >= 0
    """
    hit = _result({"query": "max_steps", "total_matches": 16, "matches": [{"line_number": 146}]})

    assert classify_tool_outcome(hit) is ToolOutcome.PRODUCTIVE


def test_an_empty_collection_is_empty_even_with_no_count() -> None:
    """Not every tool reports a count; some only return the findings.

    Killed by: src/uclone_x/tools/outcome.py :: for key in _COLLECTION_KEYS:
    """
    assert classify_tool_outcome(_result({"results": []})) is ToolOutcome.EMPTY
    assert classify_tool_outcome(_result({"results": [1]})) is ToolOutcome.PRODUCTIVE


def test_a_failed_call_is_errored_not_empty() -> None:
    """The two need different responses: a bad call can be retried, a bad query cannot.

    Collapsing them would make the note tell a model with a rejected path that its *query*
    found nothing, which is false and points it away from the fix.

    Killed by: src/uclone_x/tools/outcome.py :: if not result.success:
    """
    failed = _result(None, success=False, error="Path traversal rejected")

    assert classify_tool_outcome(failed) is ToolOutcome.ERRORED


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (None, ToolOutcome.EMPTY),
        ("", ToolOutcome.EMPTY),
        ("   ", ToolOutcome.EMPTY),
        ("text", ToolOutcome.PRODUCTIVE),
        ([], ToolOutcome.EMPTY),
        ([1], ToolOutcome.PRODUCTIVE),
        ({}, ToolOutcome.EMPTY),
        (0, ToolOutcome.PRODUCTIVE),
        (False, ToolOutcome.PRODUCTIVE),
    ],
)
def test_the_generic_shapes(output: Any, expected: ToolOutcome) -> None:
    """`0` and `False` are answers, not absences.

    A tool that returns a count of zero as its whole output has answered the question —
    "how many?" — and calling that empty would tell the model its own correct answer found
    nothing.

    Killed by: src/uclone_x/tools/outcome.py :: if isinstance(output, str):
    """
    assert classify_tool_outcome(_result(output)) is expected


def test_a_zero_count_that_is_a_bool_is_not_a_count() -> None:
    """`False` is an `int` in Python, and `{"count": False}` is not a count of zero.

    Without the `bool` exclusion a tool reporting a flag under one of the count keys would
    have its result declared empty.

    Killed by: src/uclone_x/tools/outcome.py :: and not isinstance(value, bool)
    """
    assert classify_tool_outcome(_result({"count": False})) is ToolOutcome.PRODUCTIVE


@pytest.mark.asyncio
async def test_the_empty_note_reaches_the_model_in_the_tool_message() -> None:
    """The classifier is worth nothing if the model never sees its conclusion.

    The turn message is the only channel through which the runtime can correct the reading
    that ended the measured turn. A classification recorded in a field the model does not
    read would be the #584 failure in another costume.

    Killed by: src/uclone_x/agent/base.py :: if classify_tool_outcome(res) is ToolOutcome.EMPTY:
    """
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from uclone_x.agent.base import BaseAgent
    from uclone_x.agent.models import AgentConfig, AgentLLMConfig
    from uclone_x.llm.connectors.base import BaseLLMConnector
    from uclone_x.llm.models import (
        FinishReason,
        LLMRequest,
        ModelResponse,
        StreamChunk,
        TokenUsage,
        ToolCallRequest,
    )
    from uclone_x.tools.base import BaseTool
    from uclone_x.tools.models import ToolContext
    from uclone_x.tools.outcome import EMPTY_RESULT_NOTE
    from uclone_x.tools.registry import ToolRegistry

    usage = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)

    class _Params(BaseModel):
        pass

    class _EmptySearch(BaseTool[_Params]):
        name = "empty_search"
        description = "always matches nothing"

        def run(self, params: _Params, context: ToolContext) -> dict[str, Any]:
            return {"query": "anything", "total_matches": 0, "matches": []}

    class _Scripted(BaseLLMConnector):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[LLMRequest] = []

        @property
        def provider_name(self) -> str:
            return "dummy"

        async def generate(self, request: LLMRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    finish_reason=FinishReason.TOOL_CALLS,
                    content=None,
                    tool_calls=(ToolCallRequest(id="tc_1", name="empty_search", arguments={}),),
                    usage=usage,
                    provenance=_PROV,
                )
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                content="nothing found",
                tool_calls=(),
                usage=usage,
                provenance=_PROV,
            )

        async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
            yield StreamChunk(delta_content="")

    registry = ToolRegistry()
    registry.register(_EmptySearch())
    llm = _Scripted()
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="a_empty",
            name="Agent",
            llm_config=AgentLLMConfig(model_name="dummy"),
        ),
        llm=llm,
        tools=registry,
    )
    await agent.start()

    await agent.execute_turn("find the thing")

    tool_messages = [
        m.content
        for req in llm.requests
        for m in req.messages
        if m.content and "total_matches" in str(m.content)
    ]
    assert tool_messages, "the tool result never reached the model"
    assert any(EMPTY_RESULT_NOTE in str(c) for c in tool_messages), (
        "the empty result reached the model with nothing to distinguish it from a hit"
    )
