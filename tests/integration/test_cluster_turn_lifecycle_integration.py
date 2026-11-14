"""IT-01: one turn, from ingress through a real agent to a committed session and SSE."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.cluster_harness import (
    SCRIPTED_INPUT_TOKENS,
    SCRIPTED_OUTPUT_TOKENS,
    ClusterRig,
    Reply,
    eventually,
)
from uclone_x.core.cluster_protocols import ClusterSessionStoreProtocol
from uclone_x.llm.models import ToolCallRequest
from uclone_x.shells.cluster.worker_runner import TurnDisposition


@pytest.mark.asyncio
async def test_a_submitted_turn_runs_a_real_agent_and_commits(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    assert isinstance(rig.sessions, ClusterSessionStoreProtocol)
    rig.script.replies.append(Reply(text="Hello from the model."))
    worker = rig.start_worker()
    try:
        cmd = rig.command("sess-life", "Say hello")
        accepted = await rig.ingress.submit_turn(cmd)
        assert accepted.status_code == 202
        assert accepted.sse_channel == "ucx:session:sess-life"
        await eventually(lambda: rig.committed_offset("sess-life") == 1)
    finally:
        await worker.shut_down()
        await rig.close()

    # The agent's own request reached the model, carrying the prompt.
    assert rig.script.calls == 1
    assert "Say hello" in rig.script.prompts_sent(0)

    # The session committed at revision 1, in the agent's own words.
    state = rig.sessions.load("sess-life")
    assert state is not None and state.revision == 1
    assert rig.history("sess-life") == [
        ("user", "Say hello"),
        ("assistant", "Hello from the model."),
    ]
    assert await rig.sessions.settled_terminal("sess-life", cmd.turn_id)

    # Rehydrated before the turn, evicted after it, lease given back.
    assert rig.sessions.prefetched == ["sess-life"]
    assert rig.sessions.evicted == ["sess-life"]
    assert rig.leases.holder("sess-life") is None

    # SSE: monotonic from 1, started first, the model's text, the terminal event last.
    stream = rig.stream("sess-life")
    assert [e["seq"] for e in stream] == list(range(1, len(stream) + 1))
    named = [e["payload"].get("event") for e in stream]
    assert named[0] == "TURN_STARTED"
    assert named[-1] == TurnDisposition.COMPLETED.value
    assert stream[-1]["payload"]["content"] == "Hello from the model."
    assert stream[-1]["payload"]["revision"] == 1
    assert any(
        e["payload"].get("event") == "TOKEN" and e["payload"].get("turn_id") == cmd.turn_id
        for e in stream
    )

    # The counts are the provider's, and the ledger row names the turn.
    assert rig.usage.tokens_of(cmd.turn_id) == [(SCRIPTED_INPUT_TOKENS, SCRIPTED_OUTPUT_TOKENS)]
    budget = rig.sessions.load_budget_snapshot("sess-life")
    assert budget is not None
    assert budget.consumed_prompt_tokens == SCRIPTED_INPUT_TOKENS
    assert budget.consumed_completion_tokens == SCRIPTED_OUTPUT_TOKENS
    assert budget.turn_count == 1


@pytest.mark.asyncio
async def test_a_second_turn_on_another_worker_sees_the_first(tmp_path: Path) -> None:
    """Nothing is held between turns: the second agent knows only what was committed."""
    rig = ClusterRig(tmp_path)
    rig.script.replies.extend([Reply(text="First answer."), Reply(text="Second answer.")])
    await rig.worker("w1").execute_single_turn(rig.command("sess-two", "First question"))
    await rig.worker("w2").execute_single_turn(rig.command("sess-two", "Second question"))

    sent = rig.script.prompts_sent(1)
    assert "First question" in sent and "First answer." in sent and "Second question" in sent
    assert rig.history("sess-two")[-2:] == [
        ("user", "Second question"),
        ("assistant", "Second answer."),
    ]
    state = rig.sessions.load("sess-two")
    assert state is not None and state.revision == 2


@pytest.mark.asyncio
async def test_a_turn_the_provider_fails_leaves_nothing_partial_and_is_not_run_again(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/turn_executor.py :: if result.error is not None:
    Becomes: if False:
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rig = ClusterRig(tmp_path, workspace=workspace)
    rig.script.replies.extend(
        [
            Reply(text="Earlier answer."),
            Reply(
                tool_calls=(
                    ToolCallRequest(id="c1", name="directory_list", arguments={"path": "."}),
                )
            ),
            Reply(raises=RuntimeError("provider exploded with secret detail")),
        ]
    )
    await rig.worker().execute_single_turn(rig.command("sess-fail", "Earlier question"))

    cmd = rig.command("sess-fail", "Doomed question")
    outcome = await rig.worker().execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.FAILED
    assert rig.history("sess-fail") == [
        ("user", "Earlier question"),
        ("assistant", "Earlier answer."),
    ]
    assert rig.terminal_events("sess-fail", cmd.turn_id) == [TurnDisposition.FAILED.value]
    assert await rig.sessions.settled_terminal("sess-fail", cmd.turn_id)

    calls_so_far = rig.script.calls
    again = await rig.worker("w2").execute_single_turn(cmd)
    assert again.disposition is TurnDisposition.DUPLICATE
    assert rig.script.calls == calls_so_far
