"""The log must be able to answer why a step run ended.

#697 was diagnosed by computing median steps against median declared horizon **from the
report**, because the log could not say it: `TURN_END` carried `completed` or `error`, so a
turn that answered after one step and a turn that gave up after one step were the same
line. #698 has the same shape one level down — `TOOL_RESULT` recorded `success` for a
search that matched nothing, so a healthy-looking call sat in front of an unmotivated
stop.

And `EVIDENCE_NUDGE`, added in #702, was never registered with the reader, which fails
closed on unknown event types. A session in which the nudge fired could not be read back
at all. No test covered it because nothing read a log containing one.
"""

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from uclone_x.agent.base import BaseAgent, _tool_outcome_of  # pyright: ignore[reportPrivateUsage]
from uclone_x.agent.hooks import BaseHook, HookAction, HookContext, HookDecision
from uclone_x.agent.models import AgentConfig, AgentLLMConfig, ToolExecutionRecord
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.errors import BudgetExceededError, LLMStreamInterruptedError, LLMTimeoutError
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.log.reader import (
    CURRENT_LOG_FORMAT_VERSION,
    CURRENT_LOG_SCHEMA,
    KNOWN_LOG_EVENT_TYPES,
    UnknownLogEventError,
    read_session_log,
)
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResult, ToolResultStatus
from uclone_x.tools.outcome import ToolOutcome, classify_payload_shape, classify_tool_outcome
from uclone_x.tools.registry import ToolRegistry

_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="agent", model="dummy"),
    served_by=ServiceRef(provider="agent", model="dummy"),
    attempts=(),
)
_USAGE = TokenUsage(provider="dummy", model="dummy", input_tokens=0, output_tokens=0)


class _Params(BaseModel):
    pass


class _EmptySearch(BaseTool[_Params]):
    name = "empty_search"
    description = "always matches nothing"

    def run(self, params: _Params, context: ToolContext) -> dict[str, Any]:
        return {"query": "anything", "total_matches": 0, "matches": []}


class _Scripted(BaseLLMConnector):
    def __init__(self, responses: list[ModelResponse]) -> None:
        super().__init__()
        self.responses = responses
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "dummy"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        content=text,
        tool_calls=(),
        usage=_USAGE,
        provenance=_PROV,
    )


def _call() -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id="tc_1", name="empty_search", arguments={}),),
        usage=_USAGE,
        provenance=_PROV,
    )


def _agent(llm: _Scripted, *, require_evidence: bool = False, max_steps: int = 50) -> BaseAgent:
    registry = ToolRegistry()
    registry.register(_EmptySearch())
    return BaseAgent(
        config=AgentConfig(
            agent_id="a_log",
            name="Agent",
            llm_config=AgentLLMConfig(model_name="dummy"),
            require_evidence_before_answer=require_evidence,
            max_steps=max_steps,
        ),
        llm=llm,
        tools=registry,
    )


def _turn_end(agent: BaseAgent) -> dict[str, Any]:
    events = [dict(e) for e in agent.pending_durable_events]
    ends = [e for e in events if e.get("type") == "TURN_END"]
    assert ends, f"no TURN_END among {[e.get('type') for e in events]}"
    return ends[-1]


@pytest.mark.asyncio
async def test_the_log_says_the_model_stopped_on_its_own() -> None:
    """The ordinary ending, named rather than left as "completed".

    A turn that answered and a turn that gave up both reached `outcome: completed`, so a
    step-run post-mortem could not start from the log at all.

    Killed by: src/uclone_x/agent/base.py :: "stop_reason": stop_reason,
    """
    agent = _agent(_Scripted([_answer("done")]))
    await agent.start()

    await agent.execute_turn("hello")
    end = _turn_end(agent)

    assert end["stop_reason"] == "model_stopped"
    assert end["steps"] == 1
    assert end["tool_executions"] == 0


@pytest.mark.asyncio
async def test_the_log_distinguishes_a_nudged_stop_from_an_ordinary_one() -> None:
    """The #697 question, asked of the log.

    Both turns end with the model declining to call a tool. One of them was asked again
    first, and that is the difference between "answered" and "gave up" — which is exactly
    what a reader needs and could not get.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "model_stopped_after_nudge"
    """
    agent = _agent(_Scripted([_answer("no tools")]), require_evidence=True)
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")
    end = _turn_end(agent)

    assert end["stop_reason"] == "model_stopped_after_nudge"
    assert end["steps"] == 2


@pytest.mark.asyncio
async def test_the_log_distinguishes_a_grounding_stop_from_an_unevidenced_one() -> None:
    """Two nudges, two reasons, and the log has to say which one was spent.

    This turn *did* call a tool — so it is not the `model_stopped_after_nudge` case — and
    answered with a figure the tool never returned. Collapsing the two into one name loses
    the distinction #700 needs: an answer refused for resting on nothing and an answer
    refused for outrunning what it read are different failures with different remedies.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "model_stopped_after_grounding_nudge"
    """
    agent = _agent(
        _Scripted([_call(), _answer("The configured maximum is 100.")]),
        require_evidence=True,
    )
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")
    end = _turn_end(agent)

    assert end["stop_reason"] == "model_stopped_after_grounding_nudge"
    assert end["steps"] == 3
    nudges = [dict(e) for e in agent.pending_durable_events if e.get("type") == "GROUNDING_NUDGE"]
    assert len(nudges) == 1, nudges
    assert nudges[0]["unsupported"] == ["100"], nudges[0]


@pytest.mark.asyncio
async def test_the_log_keeps_both_reasons_when_a_turn_spent_both_nudges() -> None:
    """A turn nudged for *both* reasons must not be summarised as only the second.

    The script is the reachable double: answer with nothing read (evidence nudge), then
    call a tool, then answer with a figure that tool never returned (grounding nudge). Both
    latches end set, and this is the most interesting turn in the distribution — the model
    was asked to look, looked, and still answered past what it found.

    `stop_reason` used to be assigned for evidence and then unconditionally overwritten for
    grounding, so this turn was indistinguishable in the summary from one that only outran
    its reading. The `GROUNDING_NUDGE` and `EVIDENCE_NUDGE` events carried the pair the
    whole time, so nothing was lost from the log — but a reader aggregating `stop_reason`,
    which is the cheap way to read a run, was undercounting the evidence nudge.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "model_stopped_after_both_nudges"
    """
    agent = _agent(
        _Scripted([_answer("no tools"), _call(), _answer("The configured maximum is 100.")]),
        require_evidence=True,
    )
    await agent.start()

    await agent.execute_turn("what is the configured maximum?")
    end = _turn_end(agent)
    kinds = [e.get("type") for e in agent.pending_durable_events]

    assert end["stop_reason"] == "model_stopped_after_both_nudges"
    assert kinds.count("EVIDENCE_NUDGE") == 1, kinds
    assert kinds.count("GROUNDING_NUDGE") == 1, kinds
    # Two extra steps and no more: one answer, one nudge, one tool call, one nudge, one
    # answer. That is the per-turn bound both latches exist to hold.
    assert end["steps"] == 4


@pytest.mark.asyncio
async def test_the_grounding_nudge_event_is_registered_with_the_reader() -> None:
    """The #702 defect, which #716 fixed once and nothing stopped recurring.

    The reader fails closed on an unregistered type and `read_session_log` is a generator,
    so an unregistered `GROUNDING_NUDGE` would make every event after it unreachable in a
    log that was written correctly.

    Killed by: src/uclone_x/log/reader.py :: "GROUNDING_NUDGE",
    """
    assert "GROUNDING_NUDGE" in KNOWN_LOG_EVENT_TYPES


@pytest.mark.asyncio
async def test_the_log_says_the_budget_ran_out() -> None:
    """A refusal and a completion must not read alike.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "step_budget_exceeded"
    """
    agent = _agent(_Scripted([_call()]), max_steps=1)
    await agent.start()

    result = await agent.execute_turn("search forever")
    end = _turn_end(agent)

    assert result.is_completed is False
    assert end["stop_reason"] == "step_budget_exceeded"


@pytest.mark.asyncio
async def test_the_log_says_a_step_did_not_fit_the_window() -> None:
    """A step refused because its results cannot fit the window is named as such (#1480).

    A 64-token window cannot hold even the request's fixed part, so no share is left for
    the step's result and the step is refused before a second request is sent.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "step_results_over_window"
    Becomes: stop_reason = "step_budget_exceeded"
    """
    registry = ToolRegistry()
    registry.register(_EmptySearch())
    llm = _Scripted([_call()])
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="a_log",
            name="Agent",
            llm_config=AgentLLMConfig(model_name="dummy", context_limit=64),
        ),
        llm=llm,
        tools=registry,
    )
    await agent.start()

    result = await agent.execute_turn("search")
    end = _turn_end(agent)

    assert llm.calls == 1
    assert result.is_completed is False
    assert end["stop_reason"] == "step_results_over_window"


@pytest.mark.asyncio
async def test_a_tool_that_found_nothing_is_recorded_as_empty() -> None:
    """`status: success` in front of a stop makes the stop look unmotivated.

    The search succeeded and matched nothing. Without the classification the log shows a
    healthy call and then a turn that ended, and the reader has to open the payload and
    judge it — which is the judgement #698 moved into code.

    Killed by: src/uclone_x/agent/base.py :: "outcome": _tool_outcome_of(tr),
    """
    agent = _agent(_Scripted([_call(), _answer("nothing found")]))
    await agent.start()

    await agent.execute_turn("find the thing")
    results = [dict(e) for e in agent.pending_durable_events if e.get("type") == "TOOL_RESULT"]

    assert results, "no TOOL_RESULT recorded"
    assert results[0]["status"] == "success"
    assert results[0]["outcome"] == ToolOutcome.EMPTY.value


def test_the_record_mapping_agrees_with_the_classifier() -> None:
    """The log's mapping and `classify_tool_outcome` must not drift.

    `_tool_outcome_of` reads a `ToolExecutionRecord` and the classifier reads a
    `ToolResult`, so they are two code paths over the same question. Pinned here rather
    than trusted, because a divergence would show up as a log that disagrees with the
    report built from the same run.

    Killed by: src/uclone_x/tools/outcome.py :: for key in _COUNT_KEYS:
    """
    cases: list[tuple[Any, ToolOutcome]] = [
        ({"total_matches": 0, "matches": []}, ToolOutcome.EMPTY),
        # Count-only, deliberately: the payload above carries an empty `matches` as well,
        # so the collection rule classifies it and the count rule is never exercised.
        # Without this case the declaration below names a line no mutation of it can kill.
        ({"query": "x", "total_matches": 0}, ToolOutcome.EMPTY),
        ({"total_matches": 3, "matches": [1]}, ToolOutcome.PRODUCTIVE),
        (None, ToolOutcome.EMPTY),
        ("text", ToolOutcome.PRODUCTIVE),
    ]
    for payload, expected in cases:
        assert classify_payload_shape(payload) is expected, payload


def test_the_nudge_event_is_one_the_reader_accepts(tmp_path: Path) -> None:
    """A session the nudge touched must still be readable.

    The reader fails closed on an unknown event type, and `EVIDENCE_NUDGE` was added in
    #702 without being registered — so a log from any turn in which the nudge fired raised
    `UnknownLogEventError` on read. Nothing caught it because nothing read such a log.

    Killed by: src/uclone_x/log/reader.py :: "EVIDENCE_NUDGE",
    """
    assert "EVIDENCE_NUDGE" in KNOWN_LOG_EVENT_TYPES

    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps({"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION})
        + "\n"
        + json.dumps({"type": "TURN_START", "turn_index": 0})
        + "\n"
        + json.dumps({"type": "EVIDENCE_NUDGE", "step": 1, "turn_index": 0})
        + "\n"
        + json.dumps({"type": "TURN_END", "turn_index": 0, "outcome": "completed"})
        + "\n",
        encoding="utf-8",
    )

    events = list(read_session_log(log))

    assert any(str(e.get("type")) == "EVIDENCE_NUDGE" for e in events)


def test_an_event_the_reader_does_not_know_still_fails_closed(tmp_path: Path) -> None:
    """Registering one event must not have loosened the rule.

    Killed by: src/uclone_x/log/reader.py :: known_event_types=known_event_types,
    Becomes: known_event_types=known_event_types | {str(record.get("type"))},
    """
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps({"schema": CURRENT_LOG_SCHEMA, "version": CURRENT_LOG_FORMAT_VERSION})
        + "\n"
        + json.dumps({"type": "SOMETHING_NOBODY_DECLARED", "turn_index": 0})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(UnknownLogEventError):
        list(read_session_log(log))


def _toolless_agent(llm: _Scripted) -> BaseAgent:
    """An agent with no tool registry at all.

    `BaseAgent` only materialises a default `ToolRegistry` when skills are configured, so
    passing neither leaves `_tools` as `None` — the configuration `no_tools_registered`
    exists to name.
    """
    return BaseAgent(
        config=AgentConfig(
            agent_id="a_log_toolless",
            name="Agent",
            llm_config=AgentLLMConfig(model_name="dummy"),
        ),
        llm=llm,
        tools=None,
    )


@pytest.mark.asyncio
async def test_the_log_says_the_agent_had_no_tools_to_call() -> None:
    """A model that answered because it *could* not act reads like one that chose not to.

    Without this the log records `model_stopped` for an agent wired with no registry, and
    a post-mortem attributes a toolless run to the model's judgement. #694 established
    that a toolless run is not gradeable; this is the same fact recorded at turn level.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "no_tools_registered"
    """
    agent = _toolless_agent(_Scripted([_answer("I cannot check that here")]))
    await agent.start()

    await agent.execute_turn("what is in the file?")
    end = _turn_end(agent)

    assert end["stop_reason"] == "no_tools_registered"
    assert end["tool_executions"] == 0


@pytest.mark.asyncio
async def test_the_log_says_the_run_never_reached_a_stop_decision() -> None:
    """A turn that raised mid-step must not borrow a stop reason it never took.

    `stop_reason` is seeded `not_started` above the `try`, so a turn that dies before any
    of the three stop branches assigns one carries the seed to `TURN_END`. That is the
    intended reading — `outcome: error` says it failed, `not_started` says the step run
    never got far enough to decide why it was over — and it is the value a reader will
    meet most often on a broken run.

    Killed by: src/uclone_x/agent/base.py :: stop_reason: TurnStopReason = "not_started"
    Becomes: stop_reason: TurnStopReason = "model_stopped"
    """

    class _Exploding(_Scripted):
        async def generate(self, request: LLMRequest) -> ModelResponse:
            raise RuntimeError("provider down")

    agent = _agent(_Exploding([_answer("unreachable")]))
    await agent.start()

    await agent.execute_turn("hello")
    end = _turn_end(agent)

    assert end["outcome"] == "error"
    assert end["stop_reason"] == "not_started"


@pytest.mark.asyncio
async def test_the_log_says_a_provider_ran_past_the_ceiling_the_caller_set() -> None:
    """A turn cut off at its deadline is not a turn that failed for unknown reasons.

    Both arrive as `outcome: error` carrying a string in `error`, and #1277 is what the
    conflation costs one layer up: an eval probe that was truncated at 60s and a probe
    whose host was never there both reached the report as `UNREACHABLE`, tellable apart
    only by reading prose. `TurnResult.error` is `str | None`, so the exception type does
    not survive the agent boundary; the stop reason is the channel that does.

    `not_started` stays right for every other provider failure -- the step run really did
    not reach a stop decision -- but a deadline the caller chose is a decision the caller
    made, and naming it is what lets a consumer keep it out of a model's grade.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "provider_timeout"
    Becomes: stop_reason = "not_started"
    """
    agent = _agent(_Raising(LLMTimeoutError("did not answer within 600s", seconds=600.0)))
    await agent.start()

    result = await agent.execute_turn("hello")
    end = _turn_end(agent)

    assert end["outcome"] == "error"
    assert end["stop_reason"] == "provider_timeout"
    assert result.stop_reason == "provider_timeout"


@pytest.mark.asyncio
async def test_a_stream_cut_off_by_a_ceiling_is_named_through_its_chained_cause() -> None:
    """The streaming path wraps the provider's error, so the fact is one link down.

    `_invoke_model` raises `LLMStreamInterruptedError` -- deliberately not an
    `LLMProviderError` -- with the provider's own error chained as `__cause__`. A check
    that looked only at the raised exception would name a streamed cut-off `not_started`
    and a non-streamed one `provider_timeout`, which is the same conflation #1277 is
    about, reintroduced for half the runs.

    Killed by: src/uclone_x/agent/base.py :: cause = seen.__cause__
    Becomes: cause = None
    """
    interrupted = LLMStreamInterruptedError(
        "stream stopped after 2 chunks",
        provider="ollama",
        model="dummy",
        chunks_received=2,
        discarded_tool_calls=0,
    )
    interrupted.__cause__ = LLMTimeoutError("did not answer within 600s", seconds=600.0)

    agent = _agent(_Raising(interrupted))
    await agent.start()

    result = await agent.execute_turn("hello")

    assert result.stop_reason == "provider_timeout"
    assert _turn_end(agent)["stop_reason"] == "provider_timeout"


@pytest.mark.asyncio
async def test_a_cause_chain_too_deep_to_walk_is_not_called_a_ceiling_expiring() -> None:
    """The walk is bounded, and past the bound the answer is "no", not a guess.

    A `__cause__` chain can be arbitrarily long and can cycle -- an `except` that re-raises
    what it caught is enough -- and this runs inside the handler that exists to report a
    failure, which is the one place that must not itself fail or spin. So the walk stops,
    and stopping means the fact was not established. Reporting `provider_timeout` on a
    chain it could not finish reading would be the fallback P6 forbids: a claim made where
    there is no knowledge, and here it would pull a real model failure out of the grade.

    Killed by: src/uclone_x/agent/base.py :: _MAX_CAUSE_DEPTH = 10
    Becomes: _MAX_CAUSE_DEPTH = 100
    """
    buried: BaseException = LLMTimeoutError("did not answer within 600s", seconds=600.0)
    for depth in range(12):
        wrapper = RuntimeError(f"wrapped {depth}")
        wrapper.__cause__ = buried
        buried = wrapper

    agent = _agent(_Raising(cast(Exception, buried)))
    await agent.start()

    result = await agent.execute_turn("hello")

    assert result.stop_reason == "not_started"


@pytest.mark.asyncio
async def test_the_log_says_the_run_was_cancelled() -> None:
    """A turn cancelled by task cancellation or stop signal records cancelled stop reason.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "cancelled"
    Becomes: stop_reason = "not_started"
    """

    class _Hanging(_Scripted):
        def __init__(self) -> None:
            super().__init__([])
            self.started = asyncio.Event()

        async def generate(self, request: LLMRequest) -> ModelResponse:
            self.started.set()
            await asyncio.sleep(10)
            return _answer("unreachable")

    llm = _Hanging()
    agent = _agent(llm)
    await agent.start()

    task = asyncio.create_task(agent.execute_turn("hello"))
    await llm.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    end = _turn_end(agent)
    assert end["outcome"] == "cancelled"
    assert end["stop_reason"] == "cancelled"


#: The guard below reads this module's own assertions. Its own `assert` statements name
#: stop reasons too, and counting those would let the guard satisfy itself.
_STOP_REASON_GUARD = "test_every_stop_reason_the_turn_loop_can_write_is_asserted_in_this_module"


def _string_constants_under(node: ast.AST) -> set[str]:
    return {
        n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def test_every_stop_reason_the_turn_loop_can_write_is_asserted_in_this_module() -> None:
    """The enumeration is derived from the assignments rather than listed.

    Since #970 `stop_reason` is typed `TurnStopReason`, so Pyright refuses a value outside
    that `Literal` -- but not a value added to it with no test. Nothing stops a new value
    being added with no test
    and no reader ever learning of it — which is how `no_tools_registered` and
    `not_started` shipped uncovered. This reads the assignments out of `execute_turn` by
    AST and requires each one to appear inside an `assert` in this module, so adding a
    value without a test is a red gate rather than a silent gap.

    Matching `ast.Assert` rather than the raw module text is the whole difference between
    the name and the thing. Against raw text, the repository's own `Killed by:` anchor
    convention satisfies the guard: deleting the only real assertion on
    `no_tools_registered` left this green, because the docstring line naming the mutation
    still contained the literal. A future author could add a stop reason, write an anchor,
    and pass. Two exclusions keep the rule honest — this function's own assertions, which
    name reasons in order to check them, and docstrings, which are not assertions.
    """
    source = Path(__file__).resolve().parents[2] / "src" / "uclone_x" / "agent" / "base.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    written: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr]
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            # The seed is annotated (`stop_reason: TurnStopReason = "not_started"`, #970).
            # Reading only `ast.Assign` would drop it from the sweep without a sound.
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "stop_reason" for t in targets):
            continue
        assert node.value is not None
        written |= _string_constants_under(node.value)

    # Positive control: the sweep must find the values this module already pins -- one
    # plain assignment and the annotated seed -- or a green result below means "found
    # nothing" rather than "everything is covered".
    assert {"model_stopped", "not_started"} <= written, (
        f"stop_reason sweep found only {sorted(written)}"
    )

    this_module = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    guard = next(
        n
        for n in ast.walk(this_module)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == _STOP_REASON_GUARD
    )
    own = {id(n) for n in ast.walk(guard)}

    asserted: set[str] = set()
    for node in ast.walk(this_module):
        if isinstance(node, ast.Assert) and id(node) not in own:
            asserted |= _string_constants_under(node)

    # Second positive control: the assertion sweep must see the module's real assertions.
    # `_STOP_REASON_GUARD` is excluded above, so this literal cannot satisfy itself.
    assert "model_stopped" in asserted, f"assertion sweep found only {sorted(asserted)}"

    untested = sorted(r for r in written if r not in asserted)

    assert not untested, (
        f"stop_reason values assigned in {source.name} with no assertion in this module: "
        f"{untested}. A `Killed by:` anchor naming the value is not one."
    )


def test_the_record_status_the_agent_writes_is_the_one_the_outcome_mapping_reads() -> None:
    """`ToolExecutionRecord.status` is an un-enumerated `str` on both sides of a seam.

    `_tool_outcome_of` branches on `record.status != "success"`; the producers in
    `_execute_single_tool` write the bare literals `"success"` and `"error"`.
    `ToolResultStatus` exists and spells the same two values, but the field is not typed
    to it, so producer and consumer agree by convention. Retyping the field is a wider
    change than this issue (a `strict=True` frozen model consumed across the suite) and is
    its own card, #731; what is cheap here is to stop the convention being unwritten.

    This pins the **consumer** half: if `_tool_outcome_of` drifts to compare against `"ok"`
    or `ToolResultStatus.SUCCESS.name`, this fails. It does not pin the producer — mutating
    `_execute_single_tool` to write `"ok"` leaves this green, because the records here are
    built in the test. Producer drift is not uncovered, it is covered elsewhere: that
    mutation kills 25 nodes across `test_agent_core`, `test_agent_hooks`,
    `test_evaluation_answerer` and `test_ui_server`. What was missing, and is what this
    adds, is the assertion that the consumer keeps reading the spelling those produce.

    Killed by: src/uclone_x/agent/base.py :: if record.status != ToolResultStatus.SUCCESS:
    Becomes: if record.status != ToolResultStatus.ERROR:
    """
    assert ToolResultStatus.SUCCESS.value == "success"
    assert ToolResultStatus.ERROR.value == "error"

    productive = ToolExecutionRecord(
        tool_name="t", output={"matches": [1]}, status=ToolResultStatus.SUCCESS
    )
    empty = ToolExecutionRecord(
        tool_name="t", output={"matches": []}, status=ToolResultStatus.SUCCESS
    )
    failed = ToolExecutionRecord(
        tool_name="t", output={"matches": [1]}, status=ToolResultStatus.ERROR
    )

    assert _tool_outcome_of(productive) == ToolOutcome.PRODUCTIVE.value
    assert _tool_outcome_of(empty) == ToolOutcome.EMPTY.value
    # The payload is productive; only the status makes this errored. That is the axis the
    # existing agreement test does not cover.
    assert _tool_outcome_of(failed) == ToolOutcome.ERRORED.value

    assert (
        classify_tool_outcome(ToolResult(success=False, output={"matches": [1]}, provenance=_PROV))
        is ToolOutcome.ERRORED
    )


def test_tool_execution_record_status_is_typed_and_rejects_loose_strings() -> None:
    """`ToolExecutionRecord.status` is strictly typed to `ToolResultStatus` (#731).

    Arbitrary string literals and un-enumerated strings are rejected by
    Pydantic's strict model validation, preventing convention-only drift.

    Killed by: src/uclone_x/agent/models.py :: status: ToolResultStatus = ToolResultStatus.SUCCESS
    Becomes: status: str = "success"
    """
    rec_success = ToolExecutionRecord(tool_name="t", status=ToolResultStatus.SUCCESS)
    assert rec_success.status is ToolResultStatus.SUCCESS

    rec_error = ToolExecutionRecord(tool_name="t", status=ToolResultStatus.ERROR)
    assert rec_error.status is ToolResultStatus.ERROR

    # Default value is ToolResultStatus.SUCCESS enum member
    rec_default = ToolExecutionRecord(tool_name="t")
    assert rec_default.status is ToolResultStatus.SUCCESS

    # Loose strings are rejected under strict=True
    with pytest.raises(ValidationError):
        ToolExecutionRecord(tool_name="t", status="success")  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValidationError):
        ToolExecutionRecord(tool_name="t", status="arbitrary_value")  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_an_agent_that_does_not_require_evidence_still_records_why_it_stopped() -> None:
    """The diagnosis must survive the setting being off.

    With `require_evidence_before_answer` off nothing is nudged, so the log is the only
    place the difference between "answered" and "gave up having found nothing" can live.
    That is the configuration a benchmark does *not* run under and a user does, which
    makes it the one where a post-mortem has least else to go on.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "model_stopped_after_unproductive_tools"
    Becomes: stop_reason = "model_stopped"
    """
    llm = _Scripted([_call(), _answer("The line does not appear in the file.")])
    agent = _agent(llm)
    await agent.start()

    await agent.execute_turn("find the value")
    end = _turn_end(agent)

    assert end["stop_reason"] == "model_stopped_after_unproductive_tools"


# ======================================================================================
# The result carries the stop reason too, so a consumer need not parse `content` (#970)
# ======================================================================================


class _BlockEveryTurn(BaseHook):
    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        return HookDecision(action=HookAction.BLOCK, reason="refused 970")


class _Raising(_Scripted):
    def __init__(self, error: Exception) -> None:
        super().__init__([_answer("unreachable")])
        self.error = error

    async def generate(self, request: LLMRequest) -> ModelResponse:
        raise self.error


def _agent_ending_by(scenario: str) -> BaseAgent:
    if scenario == "answered":
        return _agent(_Scripted([_answer("done")]))
    if scenario == "step_budget":
        return _agent(_Scripted([_call()]), max_steps=1)
    if scenario == "provider_error":
        return _agent(_Raising(RuntimeError("provider down")))
    assert scenario == "budget_ceiling", scenario
    return _agent(_Raising(BudgetExceededError("ceiling 970")))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("answered", "model_stopped"),
        ("step_budget", "step_budget_exceeded"),
        ("provider_error", "not_started"),
        ("budget_ceiling", "budget_exceeded"),
    ],
)
async def test_the_turn_result_carries_the_stop_reason_its_turn_end_records(
    scenario: str, expected: str
) -> None:
    """Every return from `execute_turn` says why the step run ended, as `TURN_END` does.

    #970: the result carried no reason, so a consumer had to read one out of `content`.
    The `stop_reason=stop_reason` keyword sits in each return and is not unique as a line,
    so each return was mutation-checked with a multi-line `./swx mutate` (recorded in the
    PR) rather than declared here. The hook block has its own test below.
    """
    agent = _agent_ending_by(scenario)
    await agent.start()

    result = await agent.execute_turn("hello")

    assert result.stop_reason == expected
    assert _turn_end(agent)["stop_reason"] == expected


@pytest.mark.asyncio
async def test_the_log_and_the_result_say_a_budget_ceiling_refused_the_turn() -> None:
    """A token or cost ceiling is named as one, in the result and in `TURN_END` (#969).

    It was recorded as `not_started`, the value a provider outage also carries, so a head
    could not tell a refusal every retry meets again from a failure a retry can get past.
    """
    agent = _agent_ending_by("budget_ceiling")
    await agent.start()

    result = await agent.execute_turn("hello")

    assert result.stop_reason == "budget_exceeded"
    assert _turn_end(agent)["stop_reason"] == "budget_exceeded"


@pytest.mark.asyncio
async def test_the_log_and_the_result_say_a_hook_refused_the_turn() -> None:
    """A `PRE_TURN` refusal is named as one, in the result and in `TURN_END`.

    Before #970 the CLI recognised a hook block only by the wording of `content`, and the
    log recorded it as `not_started`, which reads as a turn that never got going rather
    than one that was refused.

    Killed by: src/uclone_x/agent/base.py :: stop_reason = "blocked_by_hook"
    Becomes: stop_reason = "not_started"
    """
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="a_log", name="Agent", llm_config=AgentLLMConfig(model_name="dummy")
        ),
        llm=_Scripted([_answer("unreachable")]),
        hooks=[_BlockEveryTurn()],
    )
    await agent.start()

    result = await agent.execute_turn("hello")
    end = _turn_end(agent)

    assert result.error == "refused 970"
    assert result.stop_reason == "blocked_by_hook"
    assert end["stop_reason"] == "blocked_by_hook"
