"""IT-09: a worker dies; its turn is redelivered and commits exactly once."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply, eventually
from uclone_x.shells.cluster.worker_runner import TurnDisposition


@pytest.mark.asyncio
async def test_a_turn_killed_in_the_model_call_is_redelivered_and_commits_once(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    entered, _never = rig.script.hold("question")
    w1 = rig.start_worker("w1")
    cmd = rig.command("sess-kill", "question")
    await rig.ingress.submit_turn(cmd)
    await entered.wait()

    await w1.kill()
    assert rig.committed_offset("sess-kill") is None
    assert rig.sessions.commits == []

    rig.script.holds.clear()
    rig.script.replies.append(Reply(text="Answered after the crash."))
    rig.leases.expire("sess-kill")  # a dead worker's deadline passes
    w2 = rig.start_worker("w2")
    try:
        await eventually(lambda: rig.committed_offset("sess-kill") == 1)
    finally:
        await w2.shut_down()
        await rig.close()

    assert rig.sessions.commits == [("sess-kill", cmd.turn_id)]
    assert rig.history("sess-kill") == [
        ("user", "question"),
        ("assistant", "Answered after the crash."),
    ]


@pytest.mark.asyncio
async def test_a_new_owner_waits_for_the_lease_of_a_worker_that_may_still_be_running(
    tmp_path: Path,
) -> None:
    """Giving up on a held lease would advance the offset of a turn nobody ran.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await asyncio.sleep(delay)
    Becomes: raise
    """
    # With no time left: waiting for a lease is not a failure, so the deadline does not apply.
    rig = ClusterRig(tmp_path, turn_deadline_seconds=0.0)
    await rig.leases.acquire("sess-held", "other-turn", "other-worker")
    cmd = rig.command("sess-held", "question")
    turn = asyncio.create_task(rig.worker("w2").execute_single_turn(cmd))
    await asyncio.sleep(0.05)
    assert not turn.done()
    assert rig.script.calls == 0

    rig.leases.expire("sess-held")
    outcome = await asyncio.wait_for(turn, timeout=5.0)
    assert outcome.disposition is TurnDisposition.COMPLETED


@pytest.mark.asyncio
async def test_a_crash_after_the_commit_runs_no_model_on_redelivery(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if told is not None:
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path)
    rig.script.replies.append(Reply(text="Only once."))
    cmd = rig.command("sess-dup", "question")
    await rig.worker("w1").execute_single_turn(cmd)
    assert rig.script.calls == 1

    outcome = await rig.worker("w2").execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.DUPLICATE
    assert rig.script.calls == 1
    state = rig.sessions.load("sess-dup")
    assert state is not None and state.revision == 1
    assert rig.terminal_events("sess-dup", cmd.turn_id) == [
        TurnDisposition.COMPLETED.value,
        TurnDisposition.COMPLETED.value,
    ]


@pytest.mark.asyncio
async def test_an_earlier_turn_redelivered_behind_a_later_one_does_not_run_again(
    tmp_path: Path,
) -> None:
    """Why the guard is a set of receipts: A and B committed behind another session's offset."""
    rig = ClusterRig(tmp_path)
    rig.script.replies.extend([Reply(text="A"), Reply(text="B")])
    turn_a = rig.command("sess-ab", "turn A")
    turn_b = rig.command("sess-ab", "turn B")
    await rig.worker().execute_single_turn(turn_a)
    await rig.worker().execute_single_turn(turn_b)

    # The crash redelivers both, in order. B is the last turn; A is not.
    first = await rig.worker("w2").execute_single_turn(turn_a)
    second = await rig.worker("w2").execute_single_turn(turn_b)

    assert first.disposition is TurnDisposition.DUPLICATE
    assert second.disposition is TurnDisposition.DUPLICATE
    assert rig.script.calls == 2


@pytest.mark.asyncio
async def test_a_worker_that_loses_its_lease_mid_turn_stops_and_tries_again(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    rig.leases = type(rig.leases)(ttl_seconds=1)
    entered, _never = rig.script.hold("question")
    worker = rig.worker("w1")
    worker._lease_manager = rig.leases  # pyright: ignore[reportPrivateUsage]
    cmd = rig.command("sess-lost", "question")
    turn = asyncio.create_task(worker.execute_single_turn(cmd))
    await entered.wait()

    rig.leases.expire("sess-lost")  # the renew that follows is refused
    rig.script.holds.clear()
    rig.script.replies.append(Reply(text="Second attempt."))
    outcome = await asyncio.wait_for(turn, timeout=5.0)

    assert outcome.disposition is TurnDisposition.COMPLETED
    assert rig.sessions.commits == [("sess-lost", cmd.turn_id)]


@pytest.mark.asyncio
async def test_a_revoked_partition_redelivers_and_the_old_tracker_commits_nothing(
    tmp_path: Path,
) -> None:
    """Eager rebalance: every partition is revoked, even one that comes straight back.

    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: if commit_offset is not None and still_owned:
    Becomes: if commit_offset is not None:
    """
    rig = ClusterRig(tmp_path)
    entered, release = rig.script.hold("question")
    rig.script.replies.append(Reply(text="Once."))
    w1 = rig.start_worker("w1")
    cmd = rig.command("sess-rev", "question")
    await rig.ingress.submit_turn(cmd)
    await entered.wait()

    partition = rig.broker.partition_of("sess-rev")
    w1.process._on_revoked([("ucx.cmd.turn", partition)])  # pyright: ignore[reportPrivateUsage]
    release.set()
    await eventually(lambda: bool(rig.sessions.commits))
    await asyncio.sleep(0.02)
    assert rig.committed_offset("sess-rev") is None  # not this consumer's to commit any more

    w2 = rig.start_worker("w2")  # the new owner reads from the last commit
    try:
        await eventually(lambda: rig.committed_offset("sess-rev") == 1)
    finally:
        await w1.shut_down()
        await w2.shut_down()
        await rig.close()
    assert rig.script.calls == 1
    assert rig.terminal_events("sess-rev", cmd.turn_id)[-1] == TurnDisposition.COMPLETED.value
