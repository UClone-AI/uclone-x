"""IT-05, Invariant 4: an interrupt stops a queued turn and a running turn; neither commits."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply, eventually
from uclone_x.core.cluster_protocols import InterruptControlEnvelope
from uclone_x.shells.cluster.turn_executor import TurnExecution
from uclone_x.shells.cluster.worker_runner import TurnDisposition


def _interrupt(session_id: str, turn_id: str | None) -> InterruptControlEnvelope:
    return InterruptControlEnvelope(
        conversation_id=session_id, turn_id=turn_id, requested_by="user-1", issued_at_ms=2000
    )


@pytest.mark.asyncio
async def test_a_queued_turn_is_refused_before_any_model_call(tmp_path: Path) -> None:
    """The rig's lease refuses the tombstoned turn; the acquire script itself is in cluster_live."""
    rig = ClusterRig(tmp_path)
    entered, release = rig.script.hold("running turn")
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_turn(rig.command("sess-q", "running turn"))
        await entered.wait()
        queued = rig.command("sess-q", "queued turn")
        await rig.ingress.submit_turn(queued)
        await rig.ingress.interrupt_turn(_interrupt("sess-q", queued.turn_id))
        release.set()
        await eventually(lambda: rig.committed_offset("sess-q") == 2)
    finally:
        await worker.shut_down()
        await rig.close()

    assert rig.script.calls == 1  # the queued turn reached no model
    assert rig.terminal_events("sess-q", queued.turn_id) == [TurnDisposition.INTERRUPTED.value]
    assert ("user", "queued turn") not in rig.history("sess-q")


@pytest.mark.asyncio
async def test_a_running_turn_stops_commits_nothing_and_its_offset_advances(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if running is None or (turn_id is not None and running.turn_id != turn_id):
    Becomes: if True:
    """
    rig = ClusterRig(tmp_path)
    entered, _never = rig.script.hold("long question")
    worker = rig.start_worker()
    try:
        cmd = rig.command("sess-run", "long question")
        await rig.ingress.submit_turn(cmd)
        await entered.wait()
        await rig.ingress.interrupt_turn(_interrupt("sess-run", cmd.turn_id))
        await eventually(lambda: rig.committed_offset("sess-run") == 1)
    finally:
        await worker.shut_down()
        await rig.close()

    assert rig.sessions.commits == []
    assert rig.sessions.load("sess-run") is None
    assert rig.leases.holder("sess-run") is None
    assert rig.terminal_events("sess-run", cmd.turn_id) == [TurnDisposition.INTERRUPTED.value]


@pytest.mark.asyncio
async def test_an_interrupt_naming_no_turn_stops_what_is_running_only(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    entered, _never = rig.script.hold("stop me")
    rig.script.replies.append(Reply(text="Next answer."))
    worker = rig.start_worker()
    try:
        running = rig.command("sess-any", "stop me")
        following = rig.command("sess-any", "next question")
        await rig.ingress.submit_turn(running)
        await rig.ingress.submit_turn(following)
        await entered.wait()
        await rig.ingress.interrupt_turn(_interrupt("sess-any", None))
        await eventually(lambda: rig.committed_offset("sess-any") == 2)
    finally:
        await worker.shut_down()
        await rig.close()

    assert rig.terminal_events("sess-any", running.turn_id) == [TurnDisposition.INTERRUPTED.value]
    assert rig.history("sess-any") == [
        ("user", "next question"),
        ("assistant", "Next answer."),
    ]


@pytest.mark.asyncio
async def test_a_broadcast_for_another_turn_leaves_the_running_turn_alone(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    entered, release = rig.script.hold("keep going")
    rig.script.replies.append(Reply(text="Finished."))
    turn = asyncio.create_task(
        (worker := rig.worker()).execute_single_turn(rig.command("sess-keep", "keep going"))
    )
    await entered.wait()
    assert worker.interrupt("sess-keep", "some-other-turn") is False
    release.set()
    outcome = await turn
    assert outcome.disposition is TurnDisposition.COMPLETED


@pytest.mark.asyncio
async def test_an_interrupt_after_the_model_answered_still_wins_over_the_commit(
    tmp_path: Path,
) -> None:
    """The tombstone is read once more immediately before the write.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if interrupted_late:
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path)
    cmd = rig.command("sess-late", "question")
    real_executor = rig.executor()

    async def executor_then_interrupt(turn: object, bus: object) -> TurnExecution:
        execution = await real_executor(turn, bus)  # pyright: ignore[reportArgumentType]
        await rig.ingress.interrupt_turn(_interrupt("sess-late", cmd.turn_id))
        return execution

    worker = rig.worker()
    worker._turn_executor = executor_then_interrupt  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    outcome = await worker.execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.INTERRUPTED
    assert rig.sessions.commits == []
    assert rig.sessions.load("sess-late") is None


@pytest.mark.asyncio
async def test_a_tombstone_the_broadcast_missed_is_caught_by_the_lease_keeper(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    rig.leases = type(rig.leases)(ttl_seconds=1)
    entered, _never = rig.script.hold("unheard")
    worker = rig.worker()
    worker._lease_manager = rig.leases  # pyright: ignore[reportPrivateUsage]
    cmd = rig.command("sess-miss", "unheard")
    turn = asyncio.create_task(worker.execute_single_turn(cmd))
    await entered.wait()
    await rig.leases.write_cancel_tombstone("sess-miss", cmd.turn_id)  # no broadcast
    outcome = await asyncio.wait_for(turn, timeout=5.0)
    assert outcome.disposition is TurnDisposition.INTERRUPTED
