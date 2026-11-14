"""Real `ucx worker` processes against the docker stack, with a scripted model over HTTP.

Everything between the two ends is real: Kafka, the worker process, `BaseAgent`, PostgreSQL,
Valkey, Centrifugo's Kafka consumer and its history. The worker is killed with SIGKILL, so nothing
in it gets to clean up.

Mutation evidence for these tests is recorded in the PR, not as `Killed by:` declarations: the
gate's lethality ratchet would run them offline, where they are skipped.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiokafka.partitioner import murmur2  # pyright: ignore[reportUnknownVariableType]

from tests.integration.cluster_live.conftest import (
    REPO_ROOT,
    container,
    docker,
    eventually,
    raw_valkey,
    stack_env_file,
)
from tests.integration.cluster_live.scripted_gemini import ScriptedGemini
from uclone_x.adapters.cluster.config import ClusterRuntimeConfig
from uclone_x.adapters.cluster.kafka_broker import (
    DEFAULT_TOPIC_PARTITIONS,
    TOPIC_CMD_TURN,
    TOPIC_DEAD_LETTER,
    KafkaCommandBroker,
)
from uclone_x.adapters.cluster.sse_bridge import KafkaSSEBridge
from uclone_x.adapters.cluster.turn_lease import ValkeyTurnLeaseManager
from uclone_x.core.cluster_protocols import (
    DeadLetterEnvelope,
    InterruptControlEnvelope,
    TurnCommandEnvelope,
)
from uclone_x.core.session_state import SessionState
from uclone_x.errors import StaleSessionWriteError
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.usage.gate import UsageGate, gate_if_paid
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack
from uclone_x.shells.cluster.turn_executor import AgentTurnExecutor
from uclone_x.shells.cluster.worker_runner import StatelessAgentWorker, TurnDisposition

pytestmark = [pytest.mark.cluster_live, pytest.mark.allow_network]

_LEASE_TTL_SECONDS = 3
_KAFKA_SESSION_TIMEOUT_MS = 6000
_WAIT_SECONDS = 60.0


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def provider() -> Iterator[ScriptedGemini]:
    server = ScriptedGemini()
    yield server
    server.close()


class WorkerProcesses:
    """Starts `./ucx worker` processes, by recorded PID, with a scratch HOME each."""

    def __init__(self, root: Path, provider: ScriptedGemini) -> None:
        self._root = root
        self._provider = provider
        self._running: list[subprocess.Popen[bytes]] = []
        self._logs: dict[int, Path] = {}
        self._ids: dict[int, str] = {}

    def start(self, *args: str, **env: str) -> subprocess.Popen[bytes]:
        name = _uid("worker")
        home = self._root / name
        home.mkdir(parents=True)
        log = (home / "worker.log").open("wb")
        process = subprocess.Popen(
            [
                str(REPO_ROOT / "ucx"),
                "worker",
                "--worker-id",
                name,
                "--lease-ttl-seconds",
                str(_LEASE_TTL_SECONDS),
                "--kafka-session-timeout-ms",
                str(_KAFKA_SESSION_TIMEOUT_MS),
                *args,
            ],
            cwd=home,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                "PATH": os.environ["PATH"],
                "HOME": str(home),
                "UCX_CLUSTER_ENV_FILE": str(stack_env_file()),
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "scripted",
                "GEMINI_BASE_URL": self._provider.base_url,
                "GEMINI_MODEL": "scripted-model",
                **env,
            },
        )
        self._running.append(process)
        self._logs[process.pid] = home / "worker.log"
        self._ids[process.pid] = name
        return process

    def id_of(self, process: subprocess.Popen[bytes]) -> str:
        return self._ids[process.pid]

    def log_of(self, process: subprocess.Popen[bytes]) -> str:
        return self._logs[process.pid].read_text(errors="replace")

    @staticmethod
    def kill(process: subprocess.Popen[bytes]) -> None:
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=10)

    def close(self) -> None:
        for process in self._running:
            if process.poll() is None:
                process.send_signal(signal.SIGKILL)
                process.wait(timeout=10)
        for log in sorted(self._root.glob("worker-*/worker.log")):
            print(f"--- {log.parent.name}\n{log.read_text(errors='replace')[-4000:]}")

    def assert_stopped_cleanly(self, process: subprocess.Popen[bytes]) -> None:
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0


@pytest.fixture
def workers(tmp_path: Path, provider: ScriptedGemini) -> Iterator[WorkerProcesses]:
    running = WorkerProcesses(tmp_path, provider)
    yield running
    running.close()


def _command(
    session_id: str, prompt: str, *, user_id: str | None = None, **fields: Any
) -> TurnCommandEnvelope:
    return TurnCommandEnvelope(
        turn_id=_uid("turn"),
        session_id=session_id,
        agent_id="default",
        user_id=user_id or _uid("user"),
        prompt=prompt,
        enqueued_at_ms=int(time.time() * 1000),
        **fields,
    )


async def _terminal(stack: ClusterRuntimeStack, session_id: str, turn_id: str) -> dict[str, Any]:
    """The turn's terminal event, once it has crossed Kafka into Centrifugo's history."""

    async def probe() -> dict[str, Any] | None:
        for envelope in await stack.channel_history(f"ucx:session:{session_id}"):
            payload: dict[str, Any] = envelope.data["payload"]
            if payload.get("turn_id") == turn_id and payload.get("event") in {
                d.value for d in TurnDisposition
            }:
                return payload
        return None

    return await eventually(probe, timeout=_WAIT_SECONDS)


async def _committed_row(stack: ClusterRuntimeStack, session_id: str) -> Any:
    async with stack.sql_pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT revision, state_json, budget_json FROM session_snapshots WHERE session_id = $1",
            session_id,
        )


async def _receipts(stack: ClusterRuntimeStack, session_id: str) -> list[str]:
    async with stack.sql_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT turn_id FROM session_turn_receipts "
            "WHERE session_id = $1 AND outcome = 'committed' ORDER BY revision",
            session_id,
        )
    return [str(r["turn_id"]) for r in rows]


def _history(stack: ClusterRuntimeStack, session_id: str) -> list[tuple[str, str]]:
    state = stack.session_store.load(session_id)
    stack.session_store.evict_local_turn_cache(session_id)
    assert state is not None
    return [(m.role.value, m.content or "") for m in state.messages if m.role.value != "system"]


async def test_a_submitted_turn_runs_on_a_worker_commits_and_streams(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    """Kafka -> `ucx worker` -> real agent -> PostgreSQL + Valkey -> Centrifugo history."""
    sid = _uid("live")
    cmd = _command(sid, "Please say hello")
    provider.script(
        "say hello", "Hello from the scripted model.", input_tokens=211, output_tokens=17
    )
    worker = workers.start()

    accepted = await stack.ingress.submit_turn(cmd)
    assert accepted.sse_channel == f"ucx:session:{sid}"
    terminal = await _terminal(stack, sid, cmd.turn_id)

    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["content"] == "Hello from the scripted model."
    assert terminal["revision"] == 1

    # PostgreSQL holds the commit, the receipt and the budget, with the provider's counts.
    row = await _committed_row(stack, sid)
    assert int(row["revision"]) == 1
    assert await _receipts(stack, sid) == [cmd.turn_id]
    budget = await asyncio.to_thread(stack.session_store.load_budget_snapshot, sid)
    assert budget is not None
    assert (budget.consumed_prompt_tokens, budget.consumed_completion_tokens) == (211, 17)
    assert _history(stack, sid) == [
        ("user", "Please say hello"),
        ("assistant", "Hello from the scripted model."),
    ]

    # The Valkey head moved with it, and carries no newer state than PostgreSQL.
    client = await raw_valkey(stack)
    head = await client.get(f"sess:head:{{{sid}}}".encode())
    assert head is not None and head.startswith(b"1\n")

    # The ledger and the window, both from the response's numbers.
    ledger = await stack.usage_store.list_ledger_entries(sid)
    assert [(int(r["prompt_tokens"]), int(r["completion_tokens"])) for r in ledger] == [(211, 17)]
    assert ledger[0]["turn_id"] == cmd.turn_id and ledger[0]["scope"] == cmd.user_id
    window = await client.hgetall(stack.usage_store.window_key(cmd.user_id))
    assert sum(int(v) for k, v in window.items() if not k.startswith(b"c")) == 228

    # The stream crossed Kafka and Centrifugo in one rising sequence, with the model's text in it.
    history = await stack.channel_history(f"ucx:session:{sid}")
    seqs = sorted(int(e.data["seq"]) for e in history)
    assert seqs == list(range(1, len(seqs) + 1))
    assert any(e.data["payload"].get("event") == "TOKEN" for e in history)

    workers.assert_stopped_cleanly(worker)


async def test_a_killed_workers_turn_is_redelivered_and_commits_once(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    sid = _uid("kill")
    cmd = _command(sid, "A question to survive a crash")
    held = threading.Event()
    stalled = provider.script("survive a crash", "The dead worker's answer.", hold=held)
    first = workers.start()
    await stack.ingress.submit_turn(cmd)
    await eventually(lambda: asyncio.to_thread(stalled.entered.wait, 0.1), timeout=_WAIT_SECONDS)

    workers.kill(first)
    assert await _committed_row(stack, sid) is None
    assert int((await _attempts(stack, sid))["starts"]) == 1  # only a process that died keeps one

    provider.script("survive a crash", "The second worker's answer.")
    second = workers.start()
    terminal = await _terminal(stack, sid, cmd.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value

    row = await _committed_row(stack, sid)
    assert int(row["revision"]) == 1
    assert await _receipts(stack, sid) == [cmd.turn_id]
    assert _history(stack, sid) == [
        ("user", "A question to survive a crash"),
        ("assistant", "The second worker's answer."),
    ]
    # The dead worker's call never completed, so it has no ledger row.
    assert len(await stack.usage_store.list_ledger_entries(sid)) == 1
    workers.assert_stopped_cleanly(second)


async def test_a_queued_turn_and_a_running_turn_can_both_be_interrupted(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    sid = _uid("stop")
    running = _command(sid, "A running question")
    queued = _command(sid, "A queued question")
    held = threading.Event()
    in_call = provider.script("running question", "Never heard.", hold=held)
    never = provider.script("queued question", "Never asked.")
    worker = workers.start()

    await stack.ingress.submit_turn(running)
    await eventually(lambda: asyncio.to_thread(in_call.entered.wait, 0.1), timeout=_WAIT_SECONDS)
    await stack.ingress.submit_turn(queued)

    for turn in (queued, running):
        await stack.ingress.interrupt_turn(
            InterruptControlEnvelope(
                conversation_id=sid,
                turn_id=turn.turn_id,
                requested_by="test",
                issued_at_ms=int(time.time() * 1000),
            )
        )

    assert (await _terminal(stack, sid, running.turn_id))["event"] == (
        TurnDisposition.INTERRUPTED.value
    )
    assert (await _terminal(stack, sid, queued.turn_id))["event"] == (
        TurnDisposition.INTERRUPTED.value
    )
    assert never.calls == 0
    assert await _committed_row(stack, sid) is None
    assert await _receipts(stack, sid) == []
    held.set()

    # The worker goes on: the next turn of the session runs.
    next_turn = _command(sid, "Still there?")
    provider.script("Still there", "Yes.")
    await stack.ingress.submit_turn(next_turn)
    assert (await _terminal(stack, sid, next_turn.turn_id))["content"] == "Yes."
    workers.assert_stopped_cleanly(worker)


def _on_one_partition(*prefixes: str) -> tuple[str, ...]:
    """One new session id per prefix, all on one partition of `ucx.cmd.turn`."""
    partitions = DEFAULT_TOPIC_PARTITIONS[TOPIC_CMD_TURN]

    def partition_of(session_id: str) -> int:
        return (murmur2(session_id.encode()) & 0x7FFFFFFF) % partitions

    first = _uid(prefixes[0])
    chosen = [first]
    for prefix in prefixes[1:]:
        candidate = _uid(prefix)
        while partition_of(candidate) != partition_of(first):
            candidate = _uid(prefix)
        chosen.append(candidate)
    return tuple(chosen)


async def _dead_letters(config: ClusterRuntimeConfig, key: str) -> Mapping[str, Any]:
    """The dead letter keyed `key`, read from the topic with a group of its own."""
    reader = KafkaCommandBroker(
        bootstrap_servers=config.kafka_bootstrap_servers, group_id=_uid("dlq-reader")
    )
    await reader.subscribe([TOPIC_DEAD_LETTER])
    try:

        async def probe() -> Mapping[str, Any] | None:
            for record in await reader.poll(timeout_ms=500):
                if record.key == key:
                    return record.payload
            return None

        return await eventually(probe, timeout=_WAIT_SECONDS)
    finally:
        await reader.stop()


def _in_process_worker(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workspace: Path | None = None
) -> StatelessAgentWorker:
    def llm(gate: UsageGate) -> GeminiConnector:
        connector = GeminiConnector(
            api_key="scripted", base_url=provider.base_url, model="scripted-model"
        )
        gate_if_paid(connector, gate)
        return connector

    return StatelessAgentWorker(
        worker_id=_uid("inproc"),
        session_store=stack.session_store,
        lease_manager=stack.lease_manager,
        sse_publisher=stack.sse_publisher,
        turn_executor=AgentTurnExecutor(
            sessions=stack.session_store,
            usage=stack.usage_store,
            sse=stack.sse_publisher,
            llm_factory=llm,
            limits=UsageLimits,
            workspace_root=workspace,
        ),
    )


async def test_two_workers_on_one_session_cannot_both_commit(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    """A process stalls in the model, its lease is lost, another worker finishes the turn."""
    sid = _uid("two")
    cmd = _command(sid, "One answer only")
    held = threading.Event()
    stalled = provider.script("One answer only", "The stalled answer.", hold=held)
    process = workers.start()
    await stack.ingress.submit_turn(cmd)
    await eventually(lambda: asyncio.to_thread(stalled.entered.wait, 0.1), timeout=_WAIT_SECONDS)

    # The lease is taken from under it, as a failover can lose it.
    client = await raw_valkey(stack)
    await client.delete(ValkeyTurnLeaseManager.lease_key(sid))
    provider.script("One answer only", "The other worker's answer.")
    other = await _in_process_worker(stack, provider).execute_single_turn(cmd)
    assert other.disposition is TurnDisposition.COMPLETED

    held.set()  # the stalled worker's call returns; its commit must not land
    await asyncio.sleep(_LEASE_TTL_SECONDS)

    row = await _committed_row(stack, sid)
    assert int(row["revision"]) == 1
    assert await _receipts(stack, sid) == [cmd.turn_id]
    assert _history(stack, sid)[-1] == ("assistant", "The other worker's answer.")
    workers.assert_stopped_cleanly(process)


async def test_the_commit_statement_refuses_a_stale_revision_and_writes_no_receipt(
    stack: ClusterRuntimeStack,
) -> None:
    sid = _uid("occ")
    told = {"event": TurnDisposition.COMPLETED.value, "content": "Said."}
    first = await asyncio.to_thread(
        stack.session_store.commit_turn,
        SessionState(session_id=sid, agent_id="default"),
        turn_id="t1",
        terminal=told,
    )
    second = await asyncio.to_thread(
        stack.session_store.commit_turn, first, turn_id="t2", terminal=told
    )
    with pytest.raises(StaleSessionWriteError):
        await asyncio.to_thread(stack.session_store.commit_turn, first, turn_id="t3", terminal=told)
    assert await _receipts(stack, sid) == ["t1", "t2"]
    assert await stack.session_store.settled_terminal(sid, "t2") == {
        **told,
        "revision": second.revision,
    }
    assert await stack.session_store.settled_terminal(sid, "t3") is None


async def test_a_limit_refuses_at_turn_start_and_a_started_turn_finishes(
    stack: ClusterRuntimeStack,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    tmp_path: Path,
) -> None:
    """Over two model calls a turn crosses the 5-hour limit, finishes, and the next is refused."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sid = _uid("limit")
    user = _uid("user")
    crossing = _command(sid, "Write the file please", user_id=user)
    provider.script(
        "Write the file please",
        {"name": "file_write", "args": {"path": "a.txt", "content": "contents"}},
        "I read it.",
        input_tokens=60,
        output_tokens=40,
    )
    worker = workers.start("--workspace", str(workspace), UCLONE_USAGE_LIMIT_5_HOURS="150")

    await stack.ingress.submit_turn(crossing)
    terminal = await _terminal(stack, sid, crossing.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["content"] == "I read it."
    rows = await stack.usage_store.list_ledger_entries(sid)
    assert [(int(r["prompt_tokens"]), int(r["completion_tokens"])) for r in rows] == [(60, 40)] * 2

    refused = _command(sid, "One more please", user_id=user)
    never = provider.script("One more please", "Never asked.")
    await stack.ingress.submit_turn(refused)
    terminal = await _terminal(stack, sid, refused.turn_id)
    assert terminal["event"] == TurnDisposition.FAILED.value
    assert terminal["stop_reason"] == "usage_limit"
    assert never.calls == 0
    assert ("user", "One more please") not in _history(stack, sid)

    # Limits belong to the user: someone else on the same worker is not refused.
    other = _command(_uid("limit-other"), "Hello from another user")
    provider.script("another user", "Hi.")
    await stack.ingress.submit_turn(other)
    assert (await _terminal(stack, other.session_id, other.turn_id))["content"] == "Hi."
    workers.assert_stopped_cleanly(worker)


async def test_a_turn_prefetches_everything_in_one_batch_and_fetches_nothing_mid_turn(
    stack: ClusterRuntimeStack,
    provider: ScriptedGemini,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Invariant 2 (IT-03), against Valkey: one `MGET`, then no synchronous body fetch."""
    sid = _uid("prefetch")
    worker = _in_process_worker(stack, provider, tmp_path)
    provider.script(
        "first question",
        {"name": "file_write", "args": {"path": "notes.txt", "content": "the marker is 73124"}},
        {"name": "file_read", "args": {"path": "notes.txt"}},
        "First answer.",
    )
    await worker.execute_single_turn(_command(sid, "first question"))
    await worker.execute_single_turn(_command(sid, "second question"))

    fetched: list[str] = []
    real = stack.session_store._load_context_body_async  # pyright: ignore[reportPrivateUsage]

    async def counting(session_id: str, digest: str) -> str | None:
        fetched.append(digest)
        return await real(session_id, digest)

    monkeypatch.setattr(stack.session_store, "_load_context_body_async", counting)
    before = stack.session_store.multi_get_calls
    outcome = await worker.execute_single_turn(_command(sid, "third question"))

    assert outcome.disposition is TurnDisposition.COMPLETED
    assert stack.session_store.multi_get_calls == before + 1
    assert fetched == []
    # And the tool result of the first turn was in what the third turn sent.
    assert "73124" in str(provider.requests_for("third question")[-1])


async def test_a_turn_after_the_whole_cache_is_lost_reads_through_postgres(
    stack: ClusterRuntimeStack, provider: ScriptedGemini
) -> None:
    """IT-10: head, bodies and budget gone from Valkey; the next turn loses nothing."""
    sid = _uid("lost")
    worker = _in_process_worker(stack, provider)
    provider.script("before the loss", "Remembered answer.", input_tokens=11, output_tokens=7)
    await worker.execute_single_turn(_command(sid, "before the loss"))

    client = await raw_valkey(stack)
    async for key in client.scan_iter(match=f"*{{{sid}}}*".encode(), target_nodes="all"):
        if key.startswith((b"sess:", b"blob:")):
            await client.delete(key)
    assert await client.get(f"sess:head:{{{sid}}}".encode()) is None

    provider.script("after the loss", "Still here.", input_tokens=13, output_tokens=5)
    outcome = await worker.execute_single_turn(_command(sid, "after the loss"))

    assert outcome.disposition is TurnDisposition.COMPLETED
    assert outcome.state is not None and outcome.state.revision == 2
    sent = str(provider.requests_for("after the loss")[-1])
    assert "before the loss" in sent and "Remembered answer." in sent
    budget = await asyncio.to_thread(stack.session_store.load_budget_snapshot, sid)
    assert budget is not None
    assert (budget.consumed_prompt_tokens, budget.turn_count) == (24, 2)
    head = await client.get(f"sess:head:{{{sid}}}".encode())
    assert head is not None and head.startswith(b"2\n")


async def test_cluster_verify_passes_on_a_scripted_model(stack: ClusterRuntimeStack) -> None:
    """`ucx cluster verify` needs no key, and runs the real path on a model it names as scripted."""
    checks = await stack.run_live_verification_suite()
    assert [c.name for c in checks if not c.healthy] == []
    assert "scripted model" in checks[-1].detail


async def test_a_poison_record_reaches_the_dead_letter_topic_and_the_partition_goes_on(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """A corrupt session and a record with a NUL, beside a healthy session on their partition."""
    bad, healthy, nul_key = _on_one_partition("poison", "healthy", "nul")
    async with stack.sql_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO session_snapshots (session_id, agent_id, revision, state_json, updated_at)"
            " VALUES ($1, 'default', 1, '{not json at all', 'now')",
            bad,
        )
    poisoned = _command(bad, "Read my corrupt session")
    provider.script("corrupt session", "Never asked.")
    provider.script("healthy question", "Healthy answer.")
    provider.script("after the poison", "Still serving.")
    worker = workers.start("--turn-deadline-seconds", "2")

    await stack.ingress.submit_turn(poisoned)
    await stack.broker._send(  # pyright: ignore[reportPrivateUsage]
        TOPIC_CMD_TURN,
        nul_key,
        {"turn_id": "t", "session_id": "bad\x00id", "agent_id": "a", "user_id": "u", "prompt": "p"},
    )
    first_healthy = _command(healthy, "A healthy question")
    await stack.ingress.submit_turn(first_healthy)

    assert (await _terminal(stack, healthy, first_healthy.turn_id))["content"] == "Healthy answer."
    terminal = await _terminal(stack, bad, poisoned.turn_id)
    assert terminal["event"] == TurnDisposition.FAILED.value
    assert terminal["dead_lettered"] is True
    for internal in ("JSON", "pydantic", "ValidationError", "state_json", "Traceback"):
        assert internal not in str(terminal)

    letter = await _dead_letters(cluster_config, bad)
    assert letter["reason"] == "deadline_passed"
    assert letter["source_topic"] == TOPIC_CMD_TURN
    assert letter["error_kind"] == "ValidationError"
    malformed = await _dead_letters(cluster_config, nul_key)
    assert malformed["reason"] == "malformed"

    # The worker is the same process, and goes on serving that partition.
    assert worker.poll() is None
    after = _command(healthy, "Serving after the poison?")
    await stack.ingress.submit_turn(after)
    assert (await _terminal(stack, healthy, after.turn_id))["content"] == "Still serving."
    async with stack.sql_pool.acquire() as conn:
        receipt = await conn.fetchrow(
            "SELECT outcome FROM session_turn_receipts WHERE session_id = $1", bad
        )
        counted = await conn.fetchrow(
            "SELECT 1 FROM turn_delivery_attempts WHERE session_id = $1", bad
        )
    assert receipt is not None and receipt["outcome"] == TurnDisposition.FAILED.value
    assert counted is None
    workers.assert_stopped_cleanly(worker)


async def test_an_outage_is_waited_out_by_the_same_worker(
    stack: ClusterRuntimeStack,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """PostgreSQL stops while a turn is in its model call: the worker stays, and finishes it."""
    sid = _uid("outage")
    cmd = _command(sid, "A question during an outage")
    held = threading.Event()
    in_call = provider.script("during an outage", "Answered after it.", hold=held)
    worker = workers.start()
    await stack.ingress.submit_turn(cmd)
    await eventually(lambda: asyncio.to_thread(in_call.entered.wait, 0.1), timeout=_WAIT_SECONDS)

    docker("stop", container("postgres"))
    try:
        held.set()
        await asyncio.sleep(5)
        assert worker.poll() is None
    finally:
        docker("start", container("postgres"))

    terminal = await _terminal(stack, sid, cmd.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["content"] == "Answered after it."
    assert await _receipts(stack, sid) == [cmd.turn_id]
    assert worker.poll() is None
    workers.assert_stopped_cleanly(worker)


async def _send_raw(stack: ClusterRuntimeStack, key: str, value: bytes) -> None:
    """A record as anyone might produce it: no envelope, no encoder."""
    producer = await stack.broker._producers.get()  # pyright: ignore[reportPrivateUsage]
    await producer.send_and_wait(TOPIC_CMD_TURN, value, key=key.encode())


@pytest.mark.parametrize("record", ["nul_900kb", "corrupt_session_850kb", "deep_400kb"])
async def test_a_huge_or_pathological_record_reaches_the_dead_letter_topic_and_the_worker_stays(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    record: str,
) -> None:
    """A dead letter describes the record, so it fits; each of these once killed every worker."""
    bad, healthy = _on_one_partition("huge", "healthy")
    never = provider.script("a huge question", "Never asked.")
    provider.script("healthy question", "Healthy answer.")
    provider.script("after the huge one", "Still serving.")
    turn_id: str | None = None
    if record == "deep_400kb":
        value = b"[" * 400_000
        reason, error_kind = "malformed", "RecursionError"
    else:
        padding = "x" * (900_000 if record == "nul_900kb" else 850_000)
        prompt = f"a huge question {padding}"
        if record == "nul_900kb":
            prompt = f"a huge question \x00 {padding}"
            reason, error_kind = "malformed", "ValidationError"
        else:
            reason, error_kind = "deadline_passed", "ValidationError"
            async with stack.sql_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO session_snapshots "
                    "(session_id, agent_id, revision, state_json, updated_at)"
                    " VALUES ($1, 'default', 1, '{not json at all', 'now')",
                    bad,
                )
        turn_id = _uid("turn")
        value = json.dumps(
            {
                "turn_id": turn_id,
                "session_id": bad,
                "agent_id": "default",
                "user_id": _uid("user"),
                "prompt": prompt,
                "enqueued_at_ms": int(time.time() * 1000),
            }
        ).encode()
    worker = workers.start("--turn-deadline-seconds", "2")

    await _send_raw(stack, bad, value)
    good = _command(healthy, "A healthy question")
    await stack.ingress.submit_turn(good)

    assert (await _terminal(stack, healthy, good.turn_id))["content"] == "Healthy answer."
    letter = await _dead_letters(cluster_config, bad)
    assert len(json.dumps(letter)) < 16 * 1024
    assert letter["reason"] == reason
    assert letter["error_kind"] == error_kind
    assert letter["source_topic"] == TOPIC_CMD_TURN
    assert letter["source_offset"] is not None
    assert letter["value_size"] == len(value)
    assert letter["value_sha256"] == hashlib.sha256(value).hexdigest()
    assert base64.b64decode(letter["value_head_base64"]) == value[:4096]
    if turn_id is not None:
        terminal = await _terminal(stack, bad, turn_id)
        assert terminal["event"] == TurnDisposition.FAILED.value
        assert terminal["dead_lettered"] is True
    assert never.calls == 0

    # The same process, and it goes on serving that partition.
    assert worker.poll() is None
    after = _command(healthy, "Serving after the huge one?")
    await stack.ingress.submit_turn(after)
    assert (await _terminal(stack, healthy, after.turn_id))["content"] == "Still serving."
    workers.assert_stopped_cleanly(worker)


async def _letters_for(
    stack: ClusterRuntimeStack, config: ClusterRuntimeConfig, key: str
) -> list[str]:
    """Every dead letter keyed `key`, by reason: a marker on its partition says the read is whole."""
    marker = "test-marker"
    await stack.broker.publish_dead_letter(
        DeadLetterEnvelope(
            source_topic="test",
            key=key,
            value_size=0,
            value_sha256="",
            value_head_base64="",
            reason=marker,
            worker_id="test",
            dead_lettered_at_ms=1,
        )
    )
    reader = KafkaCommandBroker(
        bootstrap_servers=config.kafka_bootstrap_servers, group_id=_uid("dlq-reader")
    )
    await reader.subscribe([TOPIC_DEAD_LETTER])
    reasons: list[str] = []
    try:

        async def probe() -> bool:
            for record in await reader.poll(timeout_ms=500):
                if record.key == key:
                    reasons.append(str(record.payload["reason"]))
            return marker in reasons

        await eventually(probe, timeout=_WAIT_SECONDS)
    finally:
        await reader.stop()
    return [reason for reason in reasons if reason != marker]


async def _sequence_of(stack: ClusterRuntimeStack, session_id: str) -> list[int]:
    history = await stack.channel_history(f"ucx:session:{session_id}")
    return sorted(int(e.data["seq"]) for e in history)


async def _attempts(stack: ClusterRuntimeStack, session_id: str) -> Any:
    async with stack.sql_pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT starts FROM turn_delivery_attempts WHERE session_id = $1", session_id
        )


@asynccontextmanager
async def _full_valkey(stack: ClusterRuntimeStack) -> AsyncGenerator[None]:
    """Every write refused while PING still answers, and then not."""
    client = await raw_valkey(stack)
    nodes = client.get_primaries()
    limits = [await client.config_get("maxmemory", target_nodes=node) for node in nodes]
    for node in nodes:
        await client.config_set("maxmemory", 1, target_nodes=node)
    try:
        yield
    finally:
        for node, limit in zip(nodes, limits, strict=True):
            await client.config_set("maxmemory", limit["maxmemory"], target_nodes=node)


async def test_a_full_valkey_is_waited_out_and_the_turn_runs_after_it(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """Refused writes say nothing about the turn: no letter, no model call, the same worker."""
    sid = _uid("full")
    cmd = _command(sid, "A question for a full Valkey")
    asked = provider.script("full Valkey", "Answered after it.")
    async with _full_valkey(stack):
        await stack.ingress.submit_turn(cmd)
        # One attempt to spare: if a refused lease used one, this would be dead-lettered.
        worker = workers.start("--max-turn-attempts", "2")
        await asyncio.sleep(10)  # long enough to have failed on the lease, several times
        assert worker.poll() is None

    assert asked.calls == 0
    assert await _letters_for(stack, cluster_config, sid) == []
    assert await stack.session_store.settled_terminal(sid, cmd.turn_id) is None

    terminal = await _terminal(stack, sid, cmd.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["content"] == "Answered after it."
    assert asked.calls == 1
    assert await _receipts(stack, sid) == [cmd.turn_id]
    seqs = await _sequence_of(stack, sid)
    assert seqs == list(range(1, len(seqs) + 1))
    assert await _letters_for(stack, cluster_config, sid) == []
    workers.assert_stopped_cleanly(worker)


async def test_a_full_valkey_past_the_deadline_tells_the_person_as_soon_as_it_can(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """The deadline ends the turn. The event itself has to wait: its number comes from Valkey."""
    sid = _uid("late")
    cmd = _command(sid, "A question that runs out of time")
    asked = provider.script("runs out of time", "Never asked.")
    async with _full_valkey(stack):
        await stack.ingress.submit_turn(cmd)
        worker = workers.start("--turn-deadline-seconds", "2")

        async def given_up() -> bool:
            return await stack.session_store.settled_terminal(sid, cmd.turn_id) is not None

        await eventually(given_up, timeout=_WAIT_SECONDS)
        assert worker.poll() is None

    terminal = await _terminal(stack, sid, cmd.turn_id)
    assert terminal["event"] == TurnDisposition.FAILED.value
    assert terminal["dead_lettered"] is True
    assert terminal["error"] == "This message could not be answered. Please try again later."
    assert asked.calls == 0
    assert await _letters_for(stack, cluster_config, sid) == ["deadline_passed"]
    workers.assert_stopped_cleanly(worker)


async def test_a_bad_turn_stops_nothing_beside_it_on_real_workers(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """Continuing a turn on a new session always raises. The neighbour's call spans all of it."""
    bad_id, good_id = _on_one_partition("bad", "slow")
    held = threading.Event()
    slow = provider.script("A slow question", "Slow, and fine.", hold=held)
    good = _command(good_id, "A slow question")
    bad = _command(bad_id, "Continue what never began", continues_stopped_turn=True)
    started_at = time.monotonic()
    worker = workers.start("--max-turn-attempts", "3")
    await stack.ingress.submit_turn(good)
    await eventually(lambda: asyncio.to_thread(slow.entered.wait, 0.1), timeout=_WAIT_SECONDS)
    await stack.ingress.submit_turn(bad)

    told = await _terminal(stack, bad_id, bad.turn_id)
    assert told["event"] == TurnDisposition.FAILED.value
    assert told["dead_lettered"] is True
    assert await _letters_for(stack, cluster_config, bad_id) == ["attempts_used"]
    assert time.monotonic() - started_at < _WAIT_SECONDS
    # All of that happened during one model call, in one process, on one start.
    assert worker.poll() is None
    assert slow.calls == 1
    assert int((await _attempts(stack, good_id))["starts"]) == 1

    await asyncio.sleep(max(0.0, 5.0 - (time.monotonic() - started_at)))
    held.set()
    terminal = await _terminal(stack, good_id, good.turn_id)
    assert terminal["content"] == "Slow, and fine."
    assert slow.calls == 1
    assert await _letters_for(stack, cluster_config, good_id) == []
    workers.assert_stopped_cleanly(worker)


async def test_a_stopped_worker_leaves_no_start_against_the_turn_it_was_running(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    sid = _uid("stopped")
    cmd = _command(sid, "A question held through a stop")
    held = threading.Event()
    in_call = provider.script("through a stop", "Answered by the next one.", hold=held)
    worker = workers.start()
    await stack.ingress.submit_turn(cmd)
    await eventually(lambda: asyncio.to_thread(in_call.entered.wait, 0.1), timeout=_WAIT_SECONDS)
    assert int((await _attempts(stack, sid))["starts"]) == 1

    workers.assert_stopped_cleanly(worker)

    assert int((await _attempts(stack, sid))["starts"]) == 0
    assert await stack.session_store.settled_terminal(sid, cmd.turn_id) is None
    held.set()
    second = workers.start()
    assert (await _terminal(stack, sid, cmd.turn_id))["event"] == TurnDisposition.COMPLETED.value
    workers.assert_stopped_cleanly(second)


@pytest.mark.parametrize("entry", ["head", "body_not_text", "body_other_text", "budget"])
async def test_a_corrupt_cache_entry_is_a_miss_and_the_turn_runs_from_postgres(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    entry: str,
) -> None:
    """PostgreSQL is the record: a bad Valkey entry costs one read, and the session goes on."""
    sid = _uid("cache")
    first = _command(sid, "Remember the first question")
    provider.script("first question", "The first answer.", input_tokens=11, output_tokens=7)
    worker = workers.start()
    await stack.ingress.submit_turn(first)
    assert (await _terminal(stack, sid, first.turn_id))["content"] == "The first answer."

    client = await raw_valkey(stack)
    if entry == "head":
        corrupted = [f"sess:head:{{{sid}}}".encode()]
    elif entry == "budget":
        corrupted = [f"sess:budget:{{{sid}}}".encode()]
    else:
        corrupted = [
            key
            async for key in client.scan_iter(
                match=f"blob:{{{sid}}}:*".encode(), target_nodes="all"
            )
        ]
    assert corrupted
    garbage = b"\xff\xfe not text" if entry != "body_other_text" else b'{"role": "assistant"}'
    for key in corrupted:
        await client.set(key, garbage)

    second = _command(sid, "And the second question")
    asked = provider.script(
        "second question", "The second answer.", input_tokens=13, output_tokens=5
    )
    await stack.ingress.submit_turn(second)

    terminal = await _terminal(stack, sid, second.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["content"] == "The second answer."
    assert asked.calls == 1
    assert "The first answer." in str(provider.requests_for("second question")[-1])
    assert await _letters_for(stack, cluster_config, sid) == []
    assert await _receipts(stack, sid) == [first.turn_id, second.turn_id]
    budget = await asyncio.to_thread(stack.session_store.load_budget_snapshot, sid)
    assert budget is not None and budget.consumed_prompt_tokens == 24
    # What was bad has been replaced by what PostgreSQL holds.
    for key in corrupted:
        assert await client.get(key) != garbage
    workers.assert_stopped_cleanly(worker)


async def test_a_reply_too_big_for_kafka_is_told_and_the_offset_advances(
    stack: ClusterRuntimeStack,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
) -> None:
    """The terminal event is cut to fit, and says so. The whole reply is in the session."""
    big, other = _on_one_partition("big", "other")
    reply = "x" * 1_200_000
    cmd = _command(big, "A question with a very long answer")
    provider.script("very long answer", reply)
    provider.script("After the long answer", "Still serving.")
    worker = workers.start()
    await stack.ingress.submit_turn(cmd)

    terminal = await _terminal(stack, big, cmd.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["truncated"] is True
    assert terminal["content"] == reply[: len(terminal["content"])]
    assert ("assistant", reply) in _history(stack, big)
    assert await _receipts(stack, big) == [cmd.turn_id]
    seqs = await _sequence_of(stack, big)
    assert seqs == list(range(1, len(seqs) + 1))

    # The offset moved on: a later turn commits past it, and a new worker does not tell it again.
    after = _command(other, "After the long answer?")
    await stack.ingress.submit_turn(after)
    assert (await _terminal(stack, other, after.turn_id))["content"] == "Still serving."
    workers.assert_stopped_cleanly(worker)
    told_before = await _sequence_of(stack, big)
    probe = _command(other, "After the long answer, again?")
    second = workers.start()
    await stack.ingress.submit_turn(probe)
    await _terminal(stack, other, probe.turn_id)
    assert await _sequence_of(stack, big) == told_before
    workers.assert_stopped_cleanly(second)


@pytest.mark.parametrize(
    ("said", "kept"),
    [("bad\ud800text", "bad\ufffdtext"), ("bad\x00nul", "bad\x00nul")],
    ids=["surrogate", "nul"],
)
async def test_a_reply_a_store_could_not_hold_commits_on_real_postgres(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    said: str,
    kept: str,
) -> None:
    sid = _uid("odd")
    cmd = _command(sid, "An odd question")
    asked = provider.script("odd question", said)
    worker = workers.start()
    await stack.ingress.submit_turn(cmd)

    terminal = await _terminal(stack, sid, cmd.turn_id)
    assert terminal["event"] == TurnDisposition.COMPLETED.value
    assert terminal["content"] == kept
    assert asked.calls == 1
    assert await _receipts(stack, sid) == [cmd.turn_id]
    assert await _letters_for(stack, cluster_config, sid) == []

    # And the session reads back: the next turn sends it to the model.
    after = _command(sid, "The next question")
    provider.script("next question", "Fine.")
    await stack.ingress.submit_turn(after)
    assert (await _terminal(stack, sid, after.turn_id))["content"] == "Fine."
    assert ("assistant", kept) in _history(stack, sid)
    workers.assert_stopped_cleanly(worker)


async def test_a_worker_refuses_a_workspace_it_cannot_write_to_and_leaves_the_turn_alone(
    stack: ClusterRuntimeStack,
    cluster_config: ClusterRuntimeConfig,
    provider: ScriptedGemini,
    workers: WorkerProcesses,
    tmp_path: Path,
) -> None:
    sid = _uid("mount")
    cmd = _command(sid, "A question for a good mount")
    provider.script("a good mount", "Answered on a good mount.")
    await stack.ingress.submit_turn(cmd)
    mount = tmp_path / "mount"
    mount.mkdir()
    mount.chmod(0o500)
    try:
        refused = workers.start("--workspace", str(mount))
        assert await asyncio.to_thread(refused.wait, _WAIT_SECONDS) == 1
    finally:
        mount.chmod(0o700)
    assert "cannot write to --workspace" in workers.log_of(refused)
    assert await _attempts(stack, sid) is None
    assert await _letters_for(stack, cluster_config, sid) == []

    worker = workers.start("--workspace", str(mount))
    terminal = await _terminal(stack, sid, cmd.turn_id)
    assert terminal["content"] == "Answered on a good mount."
    workers.assert_stopped_cleanly(worker)


async def test_a_receipt_written_and_a_failed_publish_is_told_on_redelivery(
    stack: ClusterRuntimeStack, provider: ScriptedGemini, workers: WorkerProcesses
) -> None:
    """The sequence key holds what `INCR` refuses, so the event fails after its receipt."""
    sid = _uid("told")
    earlier = _command(sid, "An earlier question")
    provider.script("earlier question", "An earlier answer.")
    first = workers.start()
    await stack.ingress.submit_turn(earlier)
    await _terminal(stack, sid, earlier.turn_id)

    client = await raw_valkey(stack)
    key = KafkaSSEBridge.sequence_key(sid)
    reached = await client.get(key)
    await client.set(key, b"not a number")
    stale = _command(sid, "An answer to a moved conversation", expected_revision=99)
    await stack.ingress.submit_turn(stale)

    async def receipt() -> Any:
        return await stack.session_store.settled_terminal(sid, stale.turn_id)

    told = await eventually(receipt, timeout=_WAIT_SECONDS)
    assert told["event"] == TurnDisposition.FAILED.value
    assert stale.turn_id not in {
        e.data["payload"].get("turn_id") for e in await stack.channel_history(f"ucx:session:{sid}")
    }
    # The worker dies before it can tell anyone: the next one reads the receipt.
    assert first.poll() is None
    workers.kill(first)
    await client.set(key, reached)
    second = workers.start()
    terminal = await _terminal(stack, sid, stale.turn_id)
    assert terminal["event"] == TurnDisposition.FAILED.value
    assert terminal["error"] == "This conversation has changed. Please try again."

    # The redelivered event took the next number.
    seqs = await _sequence_of(stack, sid)
    assert seqs == list(range(1, len(seqs) + 1))
    assert seqs[-1] == int(reached) + 1
    workers.assert_stopped_cleanly(second)
