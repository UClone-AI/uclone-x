"""A turn that fails is that session's trouble alone, and it ends within its bounds.

Whatever the failure is: nothing here looks at its type. The session's queue backs off in process,
its neighbours go on, and when the attempts or the deadline run out the person is told, a receipt
is written and a description of the record goes to the dead-letter topic.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.integration.cluster_harness import SHUTDOWN_SECONDS, ClusterRig, Reply, eventually
from uclone_x.adapters.cluster.kafka_broker import TOPIC_CMD_TURN, KafkaCommandBroker
from uclone_x.adapters.cluster.sse_bridge import MAX_EVENT_BYTES
from uclone_x.core.cluster_protocols import (
    DeadLetterEnvelope,
    InterruptControlEnvelope,
    SSEStreamEnvelope,
    TurnCommandEnvelope,
)
from uclone_x.core.session_state import SessionState
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.models import ModelResponse, StreamChunk, TokenUsage, ToolCallRequest
from uclone_x.shells.cluster.turn_executor import TurnExecution
from uclone_x.shells.cluster.worker_runner import (
    DEFAULT_MAX_TURN_ATTEMPTS,
    DEFAULT_MAX_TURN_STARTS,
    DEFAULT_TURN_DEADLINE_SECONDS,
    TurnDisposition,
)

_NOT_ANSWERED = "This message could not be answered. Please try again later."
#: A dead letter is far under Kafka's 1 MB, whatever it describes.
_FITS = 16 * 1024


def _terminal_payload(rig: ClusterRig, session_id: str, turn_id: str) -> dict[str, Any]:
    payloads = [
        e["payload"]
        for e in rig.stream(session_id)
        if e["payload"].get("turn_id") == turn_id
        and str(e["payload"].get("event")).startswith("TURN_")
        and e["payload"].get("event") != "TURN_STARTED"
    ]
    assert payloads, "no terminal event"
    return payloads[-1]


def _state_validation_error() -> ValidationError:
    try:
        SessionState.model_validate_json("{not json at all")
    except ValidationError as exc:
        return exc
    raise AssertionError("that was supposed to be invalid")


def _corrupt_session(rig: ClusterRig, session_id: str) -> None:
    """As a corrupt `state_json` behaves: reading this session raises, on every worker."""
    real = rig.sessions.prefetch_session

    async def raising(asked: str) -> SessionState | None:
        if asked == session_id:
            raise _state_validation_error()
        return await real(asked)

    rig.sessions.prefetch_session = raising  # type: ignore[method-assign]


def _same_partition_as(rig: ClusterRig, session_id: str) -> str:
    return next(
        f"neighbour-{n}"
        for n in range(1000)
        if rig.broker.partition_of(f"neighbour-{n}") == rig.broker.partition_of(session_id)
    )


def _failing_for(worker: Any, session_id: str, error: BaseException) -> list[str]:
    """Make the executor raise `error` for `session_id`; returns who it was run for."""
    real = worker.turn_executor
    runs: list[str] = []

    async def executor(turn: TurnCommandEnvelope, bus: Any) -> TurnExecution:
        if turn.session_id != session_id:
            return await real(turn, bus)
        runs.append(turn.turn_id)
        raise error

    worker._turn_executor = executor
    return runs


def test_the_default_bounds() -> None:
    assert DEFAULT_MAX_TURN_ATTEMPTS == 5
    assert DEFAULT_TURN_DEADLINE_SECONDS == 900.0
    assert DEFAULT_MAX_TURN_STARTS == 10


@pytest.mark.asyncio
async def test_a_turn_that_keeps_failing_uses_its_attempts_and_the_person_is_told(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if delivery.attempts >= self._max_turn_attempts:
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=3)
    worker = rig.start_worker()
    runs = _failing_for(worker.worker, "sess-bad", AttributeError("/srv/secret/path"))
    good = rig.command(_same_partition_as(rig, "sess-bad"), "a healthy turn")
    bad = rig.command("sess-bad", "anything")
    try:
        await rig.ingress.submit_turn(bad)
        await rig.ingress.submit_turn(good)
        await eventually(lambda: rig.committed_offset("sess-bad") == 2)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()

    assert len(runs) == 3
    assert [(d.reason, d.attempts) for d in rig.broker.dead_letters] == [("attempts_used", 3)]
    letter = rig.broker.dead_letters[0]
    assert letter.key == "sess-bad"
    assert letter.source_topic == TOPIC_CMD_TURN
    assert letter.source_offset == 0
    assert letter.error_kind == "AttributeError"
    assert json.loads(base64.b64decode(letter.value_head_base64))["turn_id"] == bad.turn_id

    terminal = _terminal_payload(rig, "sess-bad", bad.turn_id)
    assert terminal["event"] == TurnDisposition.FAILED.value
    assert terminal["dead_lettered"] is True
    assert terminal["error"] == _NOT_ANSWERED
    assert "secret" not in str(terminal)
    assert await rig.sessions.settled_terminal("sess-bad", bad.turn_id)
    assert rig.sessions.receipt(good.session_id, good.turn_id) == "committed"


@pytest.mark.asyncio
async def test_a_bad_turn_stops_nothing_beside_it(tmp_path: Path) -> None:
    """The neighbour's model call spans every failure, and is the only one it makes."""
    rig = ClusterRig(tmp_path, max_turn_attempts=3)
    entered, release = rig.script.hold("slow")
    worker = rig.start_worker()
    runs = _failing_for(worker.worker, "sess-bad", AttributeError("a bug"))
    good = rig.command(_same_partition_as(rig, "sess-bad"), "a slow one")
    bad = rig.command("sess-bad", "anything")
    try:
        await rig.ingress.submit_turn(good)
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        await rig.ingress.submit_turn(bad)
        await eventually(lambda: bool(rig.broker.dead_letters))

        assert len(runs) == 3
        assert not worker.task.done()
        assert rig.sessions.attempts[(good.session_id, good.turn_id)] == {"starts": 1}
        assert rig.terminal_events(good.session_id, good.turn_id) == []

        release.set()
        await eventually(lambda: rig.committed_offset("sess-bad") == 2)
        await worker.shut_down()
    finally:
        release.set()
        await rig.close()

    assert rig.script.calls == 1
    assert rig.terminal_events(good.session_id, good.turn_id) == [TurnDisposition.COMPLETED.value]
    assert [d.key for d in rig.broker.dead_letters] == ["sess-bad"]


@pytest.mark.asyncio
async def test_a_failure_before_the_executor_uses_no_attempt_and_ends_at_the_deadline(
    tmp_path: Path,
) -> None:
    """It billed nothing, so an outage is bounded by the clock alone.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: return "deadline_passed" if age_seconds >= self._turn_deadline_seconds else None
    Becomes: return None
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=1, turn_deadline_seconds=0.2)
    tries: list[str] = []

    async def down(session_id: str) -> SessionState | None:
        tries.append(session_id)
        raise ConnectionError("refused")

    rig.sessions.prefetch_session = down  # type: ignore[method-assign]
    worker = rig.start_worker()
    cmd = rig.command("sess-out", "during an outage")
    try:
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: bool(rig.broker.dead_letters))
        await eventually(lambda: rig.committed_offset("sess-out") == 1)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()

    assert len(tries) > 1  # far more tries than the one attempt: none of them was one
    assert rig.script.calls == 0
    assert [(d.reason, d.attempts) for d in rig.broker.dead_letters] == [("deadline_passed", 0)]
    assert _terminal_payload(rig, "sess-out", cmd.turn_id)["error"] == _NOT_ANSWERED


@pytest.mark.asyncio
async def test_an_outage_that_ends_within_the_deadline_costs_the_turn_nothing(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    real = rig.sessions.prefetch_session
    failures = [ConnectionError("refused")] * 3

    async def recovering(session_id: str) -> SessionState | None:
        if failures:
            raise failures.pop()
        return await real(session_id)

    rig.sessions.prefetch_session = recovering  # type: ignore[method-assign]
    rig.script.replies.append(Reply(text="Answered after it."))
    worker = rig.start_worker()
    cmd = rig.command("sess-back", "during an outage")
    try:
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: rig.committed_offset("sess-back") == 1)
        await worker.shut_down()
    finally:
        await rig.close()

    assert rig.broker.dead_letters == []
    assert rig.script.calls == 1
    assert _terminal_payload(rig, "sess-back", cmd.turn_id)["content"] == "Answered after it."


@pytest.mark.asyncio
async def test_the_deadline_counts_from_the_record_and_never_refuses_a_first_try(
    tmp_path: Path,
) -> None:
    """A backlog is not a failure: an old record runs. If it fails, that was its one try.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if source is not None and source.timestamp_ms is not None
    Becomes: if False
    """
    rig = ClusterRig(tmp_path, turn_deadline_seconds=60.0)
    rig.broker.now_ms = lambda: 1_000  # long ago
    worker = rig.start_worker()
    runs = _failing_for(worker.worker, "sess-old-bad", AttributeError("a bug"))
    rig.script.replies.append(Reply(text="Late, but answered."))
    good = rig.command("sess-old-good", "an old question")
    bad = rig.command("sess-old-bad", "another old question")
    try:
        await rig.ingress.submit_turn(good)
        await rig.ingress.submit_turn(bad)
        await eventually(lambda: bool(rig.terminal_events(good.session_id, good.turn_id)))
        await eventually(lambda: bool(rig.broker.dead_letters))
        await worker.shut_down()
    finally:
        await rig.close()

    assert _terminal_payload(rig, good.session_id, good.turn_id)["content"] == "Late, but answered."
    assert len(runs) == 1
    assert [d.reason for d in rig.broker.dead_letters] == ["deadline_passed"]


@pytest.mark.asyncio
async def test_a_turn_backing_off_gives_up_its_slot(tmp_path: Path) -> None:
    """One slot, one bad session asleep for a long backoff: the neighbour still runs at once.

    Killed by: src/uclone_x/shells/cluster/worker_process.py :: cmd, source=record, wait=self._dispatcher.wait_outside_slot
    Becomes: cmd, source=record
    """
    rig = ClusterRig(tmp_path, retry_delay_seconds=30.0)
    worker = rig.start_worker(concurrency=1)
    runs = _failing_for(worker.worker, "sess-bad", AttributeError("a bug"))
    bad = rig.command("sess-bad", "anything")
    good = rig.command("sess-good", "a healthy turn")
    try:
        await rig.ingress.submit_turn(bad)
        await eventually(lambda: bool(runs))
        await rig.ingress.submit_turn(good)
        await eventually(
            lambda: bool(rig.terminal_events(good.session_id, good.turn_id)), timeout=3.0
        )
    finally:
        await rig.close()
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_work_that_raises_says_so_because_its_offset_is_stuck(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_process.py :: task.add_done_callback(self._note_if_broken)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path)
    worker = rig.start_worker()

    async def broken(cmd: object, **_: object) -> None:
        raise OverflowError("a bug in the retry itself")

    worker.worker.execute_single_turn = broken  # type: ignore[method-assign,assignment]
    await rig.ingress.submit_turn(rig.command("sess-bug", "question"))
    try:
        await eventually(lambda: "A turn's work raised" in caplog.text)
        assert rig.committed_offset("sess-bug") is None
    finally:
        await rig.close()


@pytest.mark.asyncio
async def test_a_turn_cancelled_from_inside_is_a_failure_like_any_other(tmp_path: Path) -> None:
    """An executor awaiting a cancelled child: no flag set, and nobody asked.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: raise _TurnCancelledInsideError(cmd.turn_id) from None
    Becomes: raise
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=2)
    worker = rig.start_worker()
    _failing_for(worker.worker, "sess-wedge", asyncio.CancelledError())
    cmd = rig.command("sess-wedge", "question")
    try:
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: rig.committed_offset("sess-wedge") == 1)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()
    assert [d.error_kind for d in rig.broker.dead_letters] == ["_TurnCancelledInsideError"]
    assert rig.terminal_events("sess-wedge", cmd.turn_id) == [TurnDisposition.FAILED.value]


@pytest.mark.asyncio
async def test_a_workspace_that_stops_being_writable_fails_that_turn_within_its_bounds(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    rig = ClusterRig(tmp_path, workspace=workspace, max_turn_attempts=2)
    workspace.chmod(0o500)
    try:
        worker = rig.start_worker()
        cmd = rig.command("sess-mount", "question")
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: rig.committed_offset("sess-mount") == 1)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        workspace.chmod(0o700)
        await rig.close()
    assert [d.error_kind for d in rig.broker.dead_letters] == ["PermissionError"]
    assert rig.terminal_events("sess-mount", cmd.turn_id) == [TurnDisposition.FAILED.value]


@pytest.mark.asyncio
async def test_a_record_that_kills_its_worker_each_time_is_bounded_by_its_starts(
    tmp_path: Path,
) -> None:
    """Nothing runs after an out-of-memory kill, so the starts are counted before the turn.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if starts > self._max_turn_starts:
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path)
    cmd = rig.command("sess-oom", "this one kills the process")
    rig.sessions.attempts[("sess-oom", cmd.turn_id)] = {"starts": DEFAULT_MAX_TURN_STARTS}

    outcome = await rig.worker().execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.FAILED
    assert rig.script.calls == 0
    assert [d.reason for d in rig.broker.dead_letters] == ["started_repeatedly"]
    assert await rig.sessions.settled_terminal("sess-oom", cmd.turn_id)


@pytest.mark.asyncio
async def test_an_attempt_that_comes_to_an_end_gives_its_start_back(tmp_path: Path) -> None:
    """Retries do not move a turn toward the crash bound.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await self._refund_start(cmd)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=DEFAULT_MAX_TURN_STARTS + 5)
    worker = rig.start_worker()
    runs = _failing_for(worker.worker, "sess-many", AttributeError("a bug"))
    cmd = rig.command("sess-many", "question")
    try:
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: bool(rig.broker.dead_letters))
        await worker.shut_down()
    finally:
        await rig.close()
    assert len(runs) == DEFAULT_MAX_TURN_STARTS + 5
    assert [d.reason for d in rig.broker.dead_letters] == ["attempts_used"]


@pytest.mark.asyncio
async def test_a_shutdown_leaves_a_running_turn_pending_with_no_start_against_it(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    entered, _never = rig.script.hold("question")
    worker = rig.start_worker()
    worker.process._drain_seconds = 0.0  # pyright: ignore[reportPrivateUsage]
    cmd = rig.command("sess-bye", "question")
    await rig.ingress.submit_turn(cmd)
    await asyncio.wait_for(entered.wait(), timeout=SHUTDOWN_SECONDS)
    assert rig.sessions.attempts[("sess-bye", cmd.turn_id)] == {"starts": 1}

    worker.stop.set()
    await asyncio.wait({worker.task}, timeout=SHUTDOWN_SECONDS)
    assert worker.task.done()
    assert worker.task.exception() is None  # a shutdown is not a failure

    assert rig.committed_offset("sess-bye") is None
    assert await rig.sessions.settled_terminal("sess-bye", cmd.turn_id) is None
    assert rig.broker.dead_letters == []
    assert rig.sessions.attempts[("sess-bye", cmd.turn_id)] == {"starts": 0}


@pytest.mark.asyncio
async def test_giving_up_is_retried_until_it_is_written_and_the_letter_goes_once(
    tmp_path: Path,
) -> None:
    """Nothing is lost silently: the offset stays until the person can be told.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: delivery.letter_published = True
    Becomes: pass
    """
    rig = ClusterRig(tmp_path, max_turn_attempts=1)
    worker = rig.start_worker()
    _failing_for(worker.worker, "sess-late", AttributeError("a bug"))
    real = rig.sessions.settle_turn
    refusals = [ConnectionError("refused")] * 3

    async def settling(session_id: str, turn_id: str, terminal: Mapping[str, Any]) -> None:
        if refusals:
            raise refusals.pop()
        await real(session_id, turn_id, terminal)

    rig.sessions.settle_turn = settling  # type: ignore[method-assign]
    cmd = rig.command("sess-late", "question")
    try:
        await rig.ingress.submit_turn(cmd)
        await eventually(lambda: rig.committed_offset("sess-late") == 1)
        await worker.shut_down()
    finally:
        await rig.close()
    assert refusals == []
    assert len(rig.broker.dead_letters) == 1
    assert rig.terminal_events("sess-late", cmd.turn_id) == [TurnDisposition.FAILED.value]


@pytest.mark.asyncio
async def test_a_letter_that_cannot_be_published_is_retried_and_the_offset_waits(
    tmp_path: Path,
) -> None:
    """For a turn and for a record that does not parse, alike."""
    rig = ClusterRig(tmp_path, max_turn_attempts=1)
    worker = rig.start_worker()
    _failing_for(worker.worker, "sess-a", AttributeError("a bug"))
    real = rig.broker.publish_dead_letter
    refusals = [ConnectionError("refused")] * 4

    async def publishing(envelope: DeadLetterEnvelope) -> None:
        if refusals:
            raise refusals.pop()
        await real(envelope)

    rig.broker.publish_dead_letter = publishing  # type: ignore[method-assign]
    try:
        await rig.ingress.submit_turn(rig.command("sess-a", "question"))
        rig.broker._append(  # pyright: ignore[reportPrivateUsage]
            TOPIC_CMD_TURN, "sess-b", {"not": "a turn"}, rig.broker.partition_of("sess-b")
        )
        await eventually(lambda: len(rig.broker.dead_letters) == 2)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()
    assert sorted(d.reason for d in rig.broker.dead_letters) == ["attempts_used", "malformed"]


@pytest.mark.parametrize(
    "fields",
    [
        {"session_id": "with\x00nul"},
        {"turn_id": "with\x00nul"},
        {"user_id": "with\x00nul"},
        {"agent_id": "with\ud800surrogate"},
        {"session_id": "with\nnewline"},
        {"session_id": "x" * 201},
        {"session_id": "   "},
        {"prompt": "with\x00nul"},
        {"prompt": "with\ud800surrogate"},
    ],
    ids=lambda fields: repr(fields)[:30],
)
def test_an_envelope_that_could_never_be_stored_is_rejected_at_parse_time(
    fields: dict[str, str],
) -> None:
    """Killed by: src/uclone_x/core/cluster_protocols.py :: raise ValueError(f"An identifier on TurnCommandEnvelope {problem}")
    Becomes: return value
    """
    valid = {
        "turn_id": "t",
        "session_id": "s",
        "agent_id": "a",
        "user_id": "u",
        "prompt": "p",
        "enqueued_at_ms": 1,
    }
    with pytest.raises(ValidationError):
        TurnCommandEnvelope.model_validate({**valid, **fields})


def test_an_interrupt_with_an_unsafe_id_is_rejected_at_parse_time() -> None:
    with pytest.raises(ValidationError):
        InterruptControlEnvelope(conversation_id="with\x00nul", requested_by="u", issued_at_ms=1)


@pytest.mark.asyncio
async def test_a_malformed_record_goes_to_the_dead_letter_topic_with_no_retry(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    partition = rig.broker.partition_of("sess-nul")
    bad = {"turn_id": "t-1", "session_id": "bad\x00id", "agent_id": "a", "user_id": "u"}
    rig.broker._append(TOPIC_CMD_TURN, "sess-nul", bad, partition)  # pyright: ignore[reportPrivateUsage]
    worker = rig.start_worker()
    try:
        await eventually(lambda: rig.committed_offset("sess-nul") == 1)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()

    assert [d.reason for d in rig.broker.dead_letters] == ["malformed"]
    assert rig.broker.dead_letters[0].error_kind == "ValidationError"
    assert json.loads(base64.b64decode(rig.broker.dead_letters[0].value_head_base64)) == bad
    assert rig.broker.sse == []  # it names no session that can be written to
    assert rig.script.calls == 0


@pytest.mark.asyncio
async def test_a_malformed_record_naming_a_good_session_tells_it_and_leaves_a_receipt(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    partition = rig.broker.partition_of("sess-ok")
    bad = {"turn_id": "t-2", "session_id": "sess-ok", "prompt": "bad\ud800text"}
    rig.broker._append(TOPIC_CMD_TURN, "sess-ok", bad, partition)  # pyright: ignore[reportPrivateUsage]
    worker = rig.start_worker()
    try:
        await eventually(lambda: rig.committed_offset("sess-ok") == 1)
        await worker.shut_down()
    finally:
        await rig.close()

    terminal = _terminal_payload(rig, "sess-ok", "t-2")
    assert terminal["event"] == TurnDisposition.FAILED.value
    assert "surrogate" not in str(terminal)
    assert await rig.sessions.settled_terminal("sess-ok", "t-2")


def _letter_describes(letter_json: str, value: bytes) -> None:
    letter = json.loads(letter_json)
    assert len(letter_json) < _FITS
    assert letter["value_size"] == len(value)
    assert letter["value_sha256"] == hashlib.sha256(value).hexdigest()
    assert base64.b64decode(letter["value_head_base64"]) == value[:4096]
    assert "raw_value_base64" not in letter


@pytest.mark.asyncio
async def test_a_dead_letter_of_a_huge_turn_fits_and_describes_the_value(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: value_head_base64=base64.b64encode(value[:_DEAD_LETTER_HEAD_BYTES]).decode("ascii"),
    Becomes: value_head_base64=base64.b64encode(value).decode("ascii"),
    """
    rig = ClusterRig(tmp_path, turn_deadline_seconds=0.0)
    _corrupt_session(rig, "sess-huge")
    cmd = rig.command("sess-huge", "x" * 900_000)
    await rig.ingress.submit_turn(cmd)
    worker = rig.start_worker()
    try:
        await eventually(lambda: rig.committed_offset("sess-huge") == 1)
        await worker.shut_down()
    finally:
        await rig.close()
    assert [d.reason for d in rig.broker.dead_letters] == ["deadline_passed"]
    assert rig.broker.dead_letters[0].error_kind == "ValidationError"
    value = json.dumps(cmd.model_dump(mode="json")).encode()
    _letter_describes(rig.broker.dead_letters[0].model_dump_json(), value)


@pytest.mark.asyncio
async def test_a_dead_letter_of_a_huge_malformed_record_fits_key_and_all(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: key=key[:_DEAD_LETTER_KEY_CHARS],
    Becomes: key=key,
    """
    rig = ClusterRig(tmp_path)
    key = "k" * 900_000
    bad = {"turn_id": "t", "session_id": "s", "prompt": "with\x00nul" + "x" * 900_000}
    rig.broker._append(TOPIC_CMD_TURN, key, bad, rig.broker.partition_of(key))  # pyright: ignore[reportPrivateUsage]
    worker = rig.start_worker()
    try:
        await eventually(lambda: bool(rig.broker.dead_letters))
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()
    _letter_describes(rig.broker.dead_letters[0].model_dump_json(), json.dumps(bad).encode())


class _Message:
    topic = TOPIC_CMD_TURN
    partition = 0
    key = b"k"
    timestamp = 1_700_000_000_000

    def __init__(self, offset: int, value: bytes | None) -> None:
        self.offset = offset
        self.value = value


class _OneBatchConsumer:
    def __init__(self, *values: bytes | None) -> None:
        self._values = values

    async def getmany(self, timeout_ms: int, max_records: int) -> Mapping[str, list[_Message]]:
        return {"tp": [_Message(n, value) for n, value in enumerate(self._values)]}


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (b"[" * 400_000, "RecursionError"),
        (b"\xff\xfe not text", "UnicodeDecodeError"),
        (b"{not json", "JSONDecodeError"),
        (None, "JSONDecodeError"),
    ],
    ids=["deep", "not-utf8", "not-json", "no-value"],
)
@pytest.mark.asyncio
async def test_no_value_stops_the_poll_and_each_counts_as_malformed(
    tmp_path: Path, value: bytes | None, error: str
) -> None:
    """Killed by: src/uclone_x/adapters/cluster/kafka_broker.py :: except Exception as exc:  # `RecursionError`, too, on a deeply nested value
    Becomes: except ValueError as exc:
    """
    broker = KafkaCommandBroker(bootstrap_servers=["unused:1"], group_id="g")
    broker._consumer = _OneBatchConsumer(value, b'{"fine": true}')  # pyright: ignore[reportPrivateUsage]

    try:
        records = await broker.poll()
    except Exception as exc:
        pytest.fail(f"a value stopped the poll: {type(exc).__name__}")

    assert [r.offset for r in records] == [0, 1]
    assert records[0].payload == {}
    assert records[0].decode_error == error
    assert records[0].raw_value == (value or b"")
    assert records[0].timestamp_ms == _Message.timestamp
    assert records[1].payload == {"fine": True} and records[1].decode_error is None

    rig = ClusterRig(tmp_path)
    rig.broker.log[(TOPIC_CMD_TURN, 0)] = [records[0]]
    worker = rig.start_worker()
    try:
        await eventually(lambda: bool(rig.broker.dead_letters))
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()
    assert rig.broker.dead_letters[0].reason == "malformed"
    assert rig.broker.dead_letters[0].error_kind == error


@pytest.mark.asyncio
async def test_whatever_parsing_an_envelope_raises_counts_as_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only parsing is in that `try`, on the poll loop, where no value may stop the process.

    Killed by: src/uclone_x/shells/cluster/worker_process.py :: except Exception as exc:  # only parsing is in here, and no value may stop this loop
    Becomes: except ValidationError as exc:
    """

    def breaks(payload: object) -> TurnCommandEnvelope:
        raise RecursionError("too deep to validate")

    monkeypatch.setattr(TurnCommandEnvelope, "model_validate", breaks)
    rig = ClusterRig(tmp_path)
    rig.broker._append(TOPIC_CMD_TURN, "k", {"prompt": "deep"}, rig.broker.partition_of("k"))  # pyright: ignore[reportPrivateUsage]
    worker = rig.start_worker()
    try:
        await eventually(lambda: rig.committed_offset("k") == 1)
        assert not worker.task.done()
        await worker.shut_down()
    finally:
        await rig.close()
    assert [d.error_kind for d in rig.broker.dead_letters] == ["RecursionError"]


@pytest.mark.parametrize(
    "ending",
    ["interrupted_queued", "interrupted_running", "stale_expected_revision", "plain_refusal"],
)
@pytest.mark.asyncio
async def test_every_terminal_event_has_a_receipt_so_a_redelivery_never_reruns_it(
    tmp_path: Path, ending: str
) -> None:
    """The rule: anything a person was told ended has a receipt written first.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await self._session_store.settle_turn(cmd.session_id, cmd.turn_id, terminal)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path)
    worker = rig.worker()
    if ending == "interrupted_queued":
        cmd = rig.command("sess-r", "question")
        await rig.leases.write_cancel_tombstone("sess-r", cmd.turn_id)
        await worker.execute_single_turn(cmd)
        # A tombstone expires after five minutes; the receipt does not.
        rig.leases._tombstones.clear()  # pyright: ignore[reportPrivateUsage]
    elif ending == "interrupted_running":
        cmd = rig.command("sess-r", "question")
        entered, _never = rig.script.hold("question")
        running = asyncio.create_task(worker.execute_single_turn(cmd))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        assert worker.interrupt("sess-r", cmd.turn_id)
        await asyncio.wait_for(running, timeout=5.0)
        rig.script.holds.clear()
    elif ending == "stale_expected_revision":
        cmd = rig.command("sess-r", "question", expected_revision=7)
        await worker.execute_single_turn(cmd)
    else:
        cmd = rig.command("sess-r", "question")

        async def refuses(turn: object, bus: object) -> TurnExecution:
            raise PlainRefusalError("This clone cannot answer.", reason_code="test")

        worker._turn_executor = refuses  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        await worker.execute_single_turn(cmd)

    calls_before = rig.script.calls
    told = rig.terminal_events("sess-r", cmd.turn_id)
    assert len(told) == 1

    outcome = await rig.worker("w2").execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.DUPLICATE
    assert rig.script.calls == calls_before
    assert await rig.sessions.settled_terminal("sess-r", cmd.turn_id)


def _fail_the_terminal_publish_once(rig: ClusterRig) -> list[str | None]:
    """The next terminal event cannot be sent; returns the receipts seen at each such moment."""
    seen: list[str | None] = []
    real = rig.broker.publish_sse

    async def publish(conversation_id: str, envelope: SSEStreamEnvelope) -> None:
        payload = envelope.data["payload"]
        event = str(payload.get("event"))
        if event.startswith("TURN_") and event != "TURN_STARTED" and not seen:
            seen.append(rig.sessions.receipt(conversation_id, str(payload["turn_id"])))
            raise ConnectionError("the publish failed")
        await real(conversation_id, envelope)

    rig.broker.publish_sse = publish  # type: ignore[method-assign]
    return seen


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        ("committed", {"event": "TURN_COMPLETED", "content": "Answered.", "revision": 1}),
        (
            "stale_expected_revision",
            {
                "event": "TURN_FAILED",
                "error": "This conversation has changed. Please try again.",
            },
        ),
        ("plain_refusal", {"event": "TURN_FAILED", "error": "This clone cannot answer."}),
        ("interrupted_running", {"event": "TURN_INTERRUPTED"}),
        ("given_up", {"event": "TURN_FAILED", "error": _NOT_ANSWERED, "dead_lettered": True}),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
@pytest.mark.asyncio
async def test_a_receipt_with_a_failed_publish_is_told_again_and_nothing_reruns(
    tmp_path: Path, ending: str, expected: dict[str, Any]
) -> None:
    """The receipt comes first and holds the event, so a person is never left without one.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: await self._publish_terminal(cmd, told)
    Becomes: pass
    """
    rig = ClusterRig(tmp_path, turn_deadline_seconds=0.0 if ending == "given_up" else 900.0)
    worker = rig.worker()
    fields: dict[str, Any] = {}
    if ending == "stale_expected_revision":
        fields["expected_revision"] = 7
    cmd = rig.command("sess-told", "question", **fields)
    if ending == "plain_refusal":

        async def refuses(turn: object, bus: object) -> TurnExecution:
            raise PlainRefusalError("This clone cannot answer.", reason_code="test")

        worker._turn_executor = refuses  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    elif ending == "given_up":
        _corrupt_session(rig, "sess-told")
    rig.script.replies.append(Reply(text="Answered."))
    receipts_at_publish = _fail_the_terminal_publish_once(rig)

    if ending == "interrupted_running":
        entered, _never = rig.script.hold("question")
        running = asyncio.create_task(worker.execute_single_turn(cmd))
        await asyncio.wait_for(entered.wait(), timeout=5.0)
        rig.script.holds.clear()
        assert worker.interrupt("sess-told", cmd.turn_id)
        await asyncio.wait_for(running, timeout=5.0)
    else:
        await asyncio.wait_for(worker.execute_single_turn(cmd), timeout=5.0)

    assert receipts_at_publish[0] is not None  # written before the event was tried
    told = _terminal_payload(rig, "sess-told", cmd.turn_id)
    assert {k: told.get(k) for k in expected} == expected
    assert rig.script.calls == (1 if ending in ("committed", "interrupted_running") else 0)
    assert len(rig.broker.dead_letters) == (1 if ending == "given_up" else 0)
    assert "TURN_ALREADY_SETTLED" not in rig.terminal_events("sess-told", cmd.turn_id)
    # The retry took the next number: the channel's sequence keeps rising.
    seqs = [int(e["seq"]) for e in rig.stream("sess-told")]
    assert seqs == sorted(set(seqs))


@pytest.mark.asyncio
async def test_a_turn_already_settled_gets_no_letter_when_its_deadline_passes(
    tmp_path: Path,
) -> None:
    """It ended, and telling the person is what failed: there is nothing to give up on.

    Killed by: src/uclone_x/shells/cluster/worker_runner.py :: if settled is not None:  # the failure was telling them: nothing to give up on
    Becomes: if False:
    """
    rig = ClusterRig(tmp_path, turn_deadline_seconds=0.0)
    _fail_the_terminal_publish_once(rig)
    cmd = rig.command("sess-settled", "question", expected_revision=7)

    await asyncio.wait_for(rig.worker().execute_single_turn(cmd), timeout=5.0)

    assert rig.broker.dead_letters == []
    told = _terminal_payload(rig, "sess-settled", cmd.turn_id)
    assert told["error"] == "This conversation has changed. Please try again."
    assert "dead_lettered" not in told


@pytest.mark.asyncio
async def test_a_second_redelivery_tells_again_and_a_client_takes_one_per_turn(
    tmp_path: Path,
) -> None:
    """At-least-once: the event is the same one, so one per `turn_id` is enough for a client."""
    rig = ClusterRig(tmp_path)
    rig.script.replies.append(Reply(text="Once."))
    cmd = rig.command("sess-twice", "question")
    await rig.worker().execute_single_turn(cmd)
    await rig.worker("w2").execute_single_turn(cmd)
    await rig.worker("w3").execute_single_turn(cmd)

    terminals = [
        e["payload"]
        for e in rig.stream("sess-twice")
        if e["payload"].get("event") == TurnDisposition.COMPLETED.value
    ]
    assert len(terminals) == 3
    assert all(t == terminals[0] for t in terminals)
    assert rig.script.calls == 1


@pytest.mark.asyncio
async def test_a_reply_too_big_for_kafka_is_told_in_a_cut_event_and_kept_whole(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/adapters/cluster/sse_bridge.py :: if _encoded_size(data) <= MAX_EVENT_BYTES:
    Becomes: if True:
    """
    rig = ClusterRig(tmp_path)
    reply = "x" * 1_200_000
    rig.script.replies.append(Reply(text=reply))
    cmd = rig.command("sess-big", "y" * 950_000)
    outcome = await rig.worker().execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.COMPLETED
    sizes = [len(json.dumps(e.data)) for e in rig.broker.sse]
    assert max(sizes) <= MAX_EVENT_BYTES
    terminal = _terminal_payload(rig, "sess-big", cmd.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["truncated"] is True
    assert terminal["content"] == reply[: len(terminal["content"])]
    state = rig.sessions.load("sess-big")
    assert state is not None and terminal["revision"] == state.revision
    started = next(
        e["payload"] for e in rig.stream("sess-big") if e["payload"]["event"] == "TURN_STARTED"
    )
    assert started["truncated"] is True and started["turn_id"] == cmd.turn_id
    assert ("assistant", reply) in rig.history("sess-big")
    seqs = [int(e["seq"]) for e in rig.stream("sess-big")]
    assert seqs == list(range(1, len(seqs) + 1))


@pytest.mark.asyncio
async def test_an_event_under_the_limit_is_not_touched_and_any_bulk_can_be_cut(
    tmp_path: Path,
) -> None:
    rig = ClusterRig(tmp_path)
    cmd = rig.command("sess-small", "z" * 800_000)
    await rig.worker().execute_single_turn(cmd)
    started = next(
        e["payload"] for e in rig.stream("sess-small") if e["payload"]["event"] == "TURN_STARTED"
    )
    assert started["prompt"] == cmd.prompt
    assert "truncated" not in started

    from uclone_x.engine.event_bus import AgentEvent, EventType  # the bus is what publishes

    await rig.sse.publish_event(
        "sess-small",
        AgentEvent(
            type=EventType.AGENT_REPLY,
            topic="session.sess-small",
            session_id="sess-small",
            sender_id="a",
            payload={
                "event": "TOOL_CALL",
                "turn_id": "t",
                "args": {"path": "p" * 1_000_000},
                "text": "q" * 200,
            },
        ),
    )
    cut = rig.stream("sess-small")[-1]["payload"]
    assert cut["event"] == "TOOL_CALL" and cut["turn_id"] == "t"
    assert cut["args"] is None
    assert cut["text"] == "q" * 200
    assert cut["truncated"] is True


@pytest.mark.parametrize(
    ("said", "kept"),
    [
        ("bad\ud800text", "bad\ufffdtext"),
        ("\U0001f600", "\U0001f600"),
        ("bad\x00nul", "bad\x00nul"),
    ],
    ids=["lone", "emoji", "nul"],
)
@pytest.mark.asyncio
async def test_a_reply_a_store_could_not_hold_commits_on_its_first_call(
    tmp_path: Path, said: str, kept: str
) -> None:
    """Killed by: src/uclone_x/llm/models.py :: return _SURROGATES.sub(_REPLACEMENT, value)
    Becomes: return value
    """
    rig = ClusterRig(tmp_path)
    rig.script.replies.append(Reply(text=said))
    cmd = rig.command("sess-odd", "question")

    outcome = await rig.worker().execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.COMPLETED
    assert rig.script.calls == 1
    assert rig.broker.dead_letters == []
    assert ("assistant", kept) in rig.history("sess-odd")
    assert _terminal_payload(rig, "sess-odd", cmd.turn_id)["content"] == kept


def test_everything_a_model_says_is_made_storable_where_it_enters() -> None:
    usage = TokenUsage(provider="p", model="m", input_tokens=1, output_tokens=1, total_tokens=2)
    response = ModelResponse(content="a\ud800", thinking="b\ud800", usage=usage, provenance=None)
    assert (response.content, response.thinking) == ("a\ufffd", "b\ufffd")
    chunk = StreamChunk(delta_content="c\udfff", delta_thinking="d\udfff")
    assert (chunk.delta_content, chunk.delta_thinking) == ("c\ufffd", "d\ufffd")
    call = ToolCallRequest(
        id="i\ud800", name="n\ud800", arguments={"k": ["v\ud800", {"z": "\ud800"}]}
    )
    assert (call.id, call.name) == ("i\ufffd", "n\ufffd")
    assert call.model_dump(mode="json")["arguments"] == {"k": ["v\ufffd", {"z": "\ufffd"}]}


@pytest.mark.asyncio
async def test_what_a_failed_turn_tells_a_person_has_no_internals_in_it(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/shells/cluster/worker_runner.py :: error=_plain_failure(result) if disposition is TurnDisposition.FAILED else None,
    Becomes: error=result.error,
    """
    rig = ClusterRig(tmp_path)
    secret = "pydantic_core ValidationError at /srv/secret/path line 42"
    rig.script.replies.append(
        Reply(raises=UnicodeEncodeError("utf-8", secret, 0, 1, "surrogates not allowed"))
    )
    cmd = rig.command("sess-copy", "question")
    outcome = await rig.worker().execute_single_turn(cmd)

    assert outcome.disposition is TurnDisposition.FAILED
    terminal = _terminal_payload(rig, "sess-copy", cmd.turn_id)
    assert terminal["error"] == "The turn could not be finished. Please try again."
    for internal in ("utf-8", "codec", "surrogate", "secret", "Error", "Traceback"):
        assert internal not in str(terminal["error"])
