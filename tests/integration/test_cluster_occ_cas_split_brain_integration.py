"""IT-02, Invariant 1: two workers on one session cannot both commit.

The SQL revision check itself is covered against PostgreSQL in `cluster_live/`; here the
worker's handling of a refused commit runs against the file store's revision check.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.integration.cluster_harness import ClusterRig, Reply
from uclone_x.shells.cluster.worker_runner import TurnDisposition


@pytest.mark.asyncio
async def test_zombie_worker_of_one_turn_cannot_commit_a_second_time(tmp_path: Path) -> None:
    """W1 stalls in the model, its lease lapses, W2 runs the turn; W1's commit changes nothing."""
    rig = ClusterRig(tmp_path)
    cmd = rig.command("sess-zombie", "Stalled question", turn_id="turn-z")
    entered, release = rig.script.hold("Stalled question")
    rig.script.replies.extend([Reply(text="The new owner's answer."), Reply(text="Zombie's.")])
    zombie = asyncio.create_task(rig.worker("w1").execute_single_turn(cmd))
    await entered.wait()

    rig.leases.expire("sess-zombie")
    rig.script.holds.clear()
    owner = await rig.worker("w2").execute_single_turn(cmd)
    assert owner.disposition is TurnDisposition.COMPLETED

    release.set()
    outcome = await zombie

    # Refused by the revision check, retried, and found the turn's receipt.
    assert outcome.disposition is TurnDisposition.DUPLICATE
    assert rig.sessions.invalidated == ["sess-zombie"]
    assert rig.sessions.commits == [("sess-zombie", "turn-z")]
    state = rig.sessions.load("sess-zombie")
    assert state is not None and state.revision == 1
    assert rig.history("sess-zombie") == [
        ("user", "Stalled question"),
        ("assistant", "The new owner's answer."),
    ]


@pytest.mark.asyncio
async def test_zombie_of_an_older_turn_runs_again_on_the_newer_state(tmp_path: Path) -> None:
    """Another turn landed meanwhile: the stale write is refused, nothing of it is merged in."""
    rig = ClusterRig(tmp_path)
    entered, release = rig.script.hold("Older question")
    rig.script.replies.extend(
        [Reply(text="Newer answer."), Reply(text="Stale answer."), Reply(text="Older, redone.")]
    )
    stalled = asyncio.create_task(
        rig.worker("w1").execute_single_turn(rig.command("sess-order", "Older question"))
    )
    await entered.wait()

    rig.leases.expire("sess-order")
    rig.script.holds.clear()
    await rig.worker("w2").execute_single_turn(rig.command("sess-order", "Newer question"))
    release.set()
    outcome = await stalled

    assert outcome.disposition is TurnDisposition.COMPLETED
    state = rig.sessions.load("sess-order")
    assert state is not None and state.revision == 2
    assert rig.history("sess-order") == [
        ("user", "Newer question"),
        ("assistant", "Newer answer."),
        ("user", "Older question"),
        ("assistant", "Older, redone."),
    ]
