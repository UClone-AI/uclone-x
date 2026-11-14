"""IT-04, Invariant 3: no head-of-line blocking, and only contiguous offsets are committed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply, eventually
from uclone_x.adapters.cluster.kafka_broker import TOPIC_CMD_TURN
from uclone_x.core.cluster_protocols import TurnCommandEnvelope


def _same_partition_sessions(rig: ClusterRig) -> tuple[str, str]:
    """Two session ids the rig's partitioner puts on one partition."""
    first = "sess-aa"
    second = next(
        f"sess-{n}"
        for n in range(1000)
        if f"sess-{n}" != first
        and rig.broker.partition_of(f"sess-{n}") == rig.broker.partition_of(first)
    )
    return first, second


@pytest.mark.asyncio
async def test_a_slow_turn_does_not_block_a_fast_one_and_holds_the_watermark(
    tmp_path: Path,
) -> None:
    """Offset 1 finishes while offset 0 runs: committed nothing until 0 ends, then 2.

    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: commit_offset = tracker.mark_completed(offset)
    Becomes: commit_offset = (tracker.mark_completed(offset), offset + 1)[1]
    """
    rig = ClusterRig(tmp_path)
    slow, fast = _same_partition_sessions(rig)
    entered, release = rig.script.hold("slow question")
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_turn(rig.command(slow, "slow question"))
        await entered.wait()
        fast_cmd = rig.command(fast, "fast question")
        await rig.ingress.submit_turn(fast_cmd)

        await eventually(lambda: bool(rig.sessions.commits))
        assert rig.sessions.commits == [(fast, fast_cmd.turn_id)]
        await asyncio.sleep(0.02)
        assert rig.committed_offset(slow) is None  # the crash window the watermark exists for

        release.set()
        await eventually(lambda: rig.committed_offset(slow) == 2)
    finally:
        await worker.shut_down()
        await rig.close()


@pytest.mark.asyncio
async def test_turns_of_one_session_run_strictly_in_order(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    entered, release = rig.script.hold("first")
    rig.script.replies.extend([Reply(text="one"), Reply(text="two")])
    worker = rig.start_worker()
    try:
        await rig.ingress.submit_turn(rig.command("sess-fifo", "first"))
        await rig.ingress.submit_turn(rig.command("sess-fifo", "second"))
        await entered.wait()
        await asyncio.sleep(0.02)
        assert rig.script.calls == 1  # the second waits behind the first on this worker

        release.set()
        await eventually(lambda: rig.committed_offset("sess-fifo") == 2)
    finally:
        await worker.shut_down()
        await rig.close()
    assert rig.history("sess-fifo") == [
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
        ("assistant", "two"),
    ]


@pytest.mark.asyncio
async def test_a_turn_that_keeps_raising_keeps_its_offset_and_the_worker_stays(
    tmp_path: Path,
) -> None:
    """A failure around the turn must not look like a finished offset (it would be lost).

    Killed by: src/uclone_x/shells/cluster/dispatcher.py :: if finished:
    Becomes: if True:
    """
    rig = ClusterRig(tmp_path)
    called = asyncio.Event()

    async def boom(session_id: str, turn_id: str) -> None:
        called.set()
        raise ConnectionError("postgres went away")

    rig.sessions.settled_terminal = boom  # type: ignore[method-assign]
    worker = rig.start_worker()
    await rig.ingress.submit_turn(rig.command("sess-boom", "anything"))
    await asyncio.wait_for(called.wait(), timeout=5.0)
    await asyncio.sleep(0.05)
    assert not worker.task.done()
    await rig.close()
    assert rig.committed_offset("sess-boom") is None
    assert rig.script.calls == 0


@pytest.mark.asyncio
async def test_a_malformed_record_is_stepped_over_and_the_partition_moves_on(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    partition = rig.broker.partition_of("sess-bad")
    rig.broker._append(TOPIC_CMD_TURN, "sess-bad", {"not": "a turn"}, partition)  # pyright: ignore[reportPrivateUsage]
    worker = rig.start_worker()
    try:
        cmd = rig.command("sess-bad", "a real turn")
        assert isinstance(cmd, TurnCommandEnvelope)
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: rig.committed_offset("sess-bad") == 2)
    finally:
        await worker.shut_down()
        await rig.close()
    assert rig.sessions.commits == [("sess-bad", cmd.turn_id)]


@pytest.mark.asyncio
async def test_a_full_worker_pauses_fetching_and_resumes_when_a_slot_frees(tmp_path: Path) -> None:
    rig = ClusterRig(tmp_path)
    entered, release = rig.script.hold("hold this")
    worker = rig.start_worker(concurrency=1)
    try:
        await rig.ingress.submit_turn(rig.command("sess-full", "hold this"))
        await entered.wait()
        await eventually(lambda: worker.consumer.paused)
        release.set()
        await eventually(lambda: not worker.consumer.paused)
    finally:
        await worker.shut_down()
        await rig.close()
    assert worker.consumer.pause_count >= 1
