"""IT-06, Invariant 5: real token counts, the ledger, and the owner's rule on limits (#2431).

Owner rule: the limit is checked when a turn starts; a started turn finishes even past it; the
runaway ceiling is one 10-minute limit per turn, stopping at a step boundary; No limit means no
ceiling. The gate logic is the desktop's own (`llm/usage/`), run here over the cluster's store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply
from uclone_x.llm.models import ToolCallRequest
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.shells.cluster.worker_runner import TurnDisposition

_STEP_TOKENS = 100  # input + output of every scripted step below


def _step(call_id: str) -> Reply:
    """A model step that asks for a tool, so the turn goes on to another step."""
    return Reply(
        text="",
        tool_calls=(ToolCallRequest(id=call_id, name="directory_list", arguments={"path": "."}),),
        input_tokens=60,
        output_tokens=40,
    )


def _answer(text: str = "All done.") -> Reply:
    return Reply(text=text, input_tokens=60, output_tokens=40)


def _rig(tmp_path: Path, limits: UsageLimits) -> ClusterRig:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ClusterRig(tmp_path, limits=limits, workspace=workspace)


@pytest.mark.asyncio
async def test_token_counts_come_from_the_response_and_accumulate_across_workers(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    rig.script.replies.extend(
        [
            Reply(text="one", input_tokens=211, output_tokens=17),
            Reply(text="two", input_tokens=389, output_tokens=23),
        ]
    )
    first = rig.command("sess-counts", "first")
    second = rig.command("sess-counts", "second")
    await rig.worker("w1").execute_single_turn(first)
    await rig.worker("w2").execute_single_turn(second)

    assert rig.usage.tokens_of(first.turn_id) == [(211, 17)]
    assert rig.usage.tokens_of(second.turn_id) == [(389, 23)]
    assert all(row["scope"] == "user-1" for row in rig.usage.rows)
    budget = rig.sessions.load_budget_snapshot("sess-counts")
    assert budget is not None
    assert budget.consumed_prompt_tokens == 600
    assert budget.consumed_completion_tokens == 40
    assert budget.per_provider == {"scripted": 640}
    assert budget.turn_count == 2


@pytest.mark.asyncio
async def test_a_turn_over_the_limit_is_refused_at_its_start_with_no_model_call(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/llm/usage/admission.py :: admit()  # the one check a turn gets
    Becomes: pass
    """
    rig = _rig(tmp_path, UsageLimits(per_5_hours=50))
    rig.script.replies.append(_answer())
    await rig.worker().execute_single_turn(rig.command("sess-limit", "uses 100 tokens"))
    assert rig.script.calls == 1

    refused = rig.command("sess-limit", "over the limit now")
    outcome = await rig.worker().execute_single_turn(refused)

    assert rig.script.calls == 1  # no call was made for it
    assert outcome.disposition is TurnDisposition.FAILED
    assert outcome.result is not None and outcome.result.stop_reason == "usage_limit"
    assert rig.terminal_events("sess-limit", refused.turn_id) == [TurnDisposition.FAILED.value]
    # Its prompt is not left in the conversation, and a redelivery will not run it later.
    assert ("user", "over the limit now") not in rig.history("sess-limit")
    assert await rig.sessions.settled_terminal("sess-limit", refused.turn_id)


@pytest.mark.asyncio
async def test_a_started_turn_finishes_past_the_limit_and_every_call_is_booked(
    tmp_path: Path,
) -> None:
    """The 5-hour limit is crossed by step 1; steps 2 and 3 still run."""
    rig = _rig(tmp_path, UsageLimits(per_5_hours=50))
    rig.script.replies.extend([_step("c1"), _step("c2"), _answer("Finished past the limit.")])
    cmd = rig.command("sess-past", "a long job")
    outcome = await rig.worker().execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.COMPLETED
    assert outcome.result is not None and outcome.result.content == "Finished past the limit."
    assert rig.script.calls == 3
    assert rig.usage.tokens_of(cmd.turn_id) == [(60, 40)] * 3

    # The next turn is the one refused.
    next_turn = await rig.worker().execute_single_turn(rig.command("sess-past", "next"))
    assert next_turn.result is not None and next_turn.result.stop_reason == "usage_limit"


@pytest.mark.asyncio
async def test_a_runaway_turn_stops_at_a_step_boundary_and_continues_in_place(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path, UsageLimits(per_10_minutes=_STEP_TOKENS + 1))
    rig.script.replies.extend([_step("c1"), _step("c2"), _step("c3"), _answer("Resumed.")])
    cmd = rig.command("sess-ceiling", "runs away")
    outcome = await rig.worker().execute_single_turn(cmd)

    # 100 then 200 tokens: past the ceiling after the second call, stopped before the third.
    assert rig.script.calls == 2
    assert outcome.disposition is TurnDisposition.STOPPED
    assert outcome.result is not None and outcome.result.stop_reason == "usage_ceiling"
    assert rig.terminal_events("sess-ceiling", cmd.turn_id) == [TurnDisposition.STOPPED.value]
    kept = rig.sessions.load("sess-ceiling")
    assert kept is not None and kept.revision == 1  # kept, so it can go on

    # [Continue]: no second copy of the prompt, no check at the start, one more ceiling's worth.
    resumed = await rig.worker("w2").execute_single_turn(
        rig.command("sess-ceiling", "", continues_stopped_turn=True)
    )
    assert resumed.disposition is TurnDisposition.COMPLETED
    assert resumed.result is not None and resumed.result.content == "Resumed."
    prompts = [text for role, text in rig.history("sess-ceiling") if role == "user"]
    assert prompts == ["runs away"]


@pytest.mark.asyncio
async def test_with_no_limit_there_is_no_ceiling(tmp_path: Path) -> None:
    rig = _rig(tmp_path, UsageLimits())
    rig.script.replies.extend([_step(f"c{n}") for n in range(5)] + [_answer("Never stopped.")])
    outcome = await rig.worker().execute_single_turn(rig.command("sess-free", "go on"))
    assert outcome.disposition is TurnDisposition.COMPLETED
    assert rig.script.calls == 6


@pytest.mark.asyncio
async def test_a_model_that_costs_nothing_gets_a_ledger_row_and_never_a_window(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path, limits=UsageLimits(per_5_hours=1), paid=False)
    rig.script.replies.extend([Reply(text="one"), Reply(text="two")])
    first = rig.command("sess-local", "first")
    await rig.worker().execute_single_turn(first)
    second = await rig.worker().execute_single_turn(rig.command("sess-local", "second"))

    assert second.disposition is TurnDisposition.COMPLETED  # a local model is never limited
    assert len(rig.usage.rows) == 2
    assert not any(row["counts_against_limits"] for row in rig.usage.rows)
