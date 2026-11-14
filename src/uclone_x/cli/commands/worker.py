"""`ucx worker`: one stateless agent worker process, consuming the command and learn topics."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import tempfile
import uuid
from pathlib import Path

import typer

from uclone_x.adapters.cluster.config import ClusterRuntimeConfig, stack_env_path
from uclone_x.adapters.cluster.kafka_broker import TOPIC_CTRL_INTERRUPT, KafkaCommandBroker
from uclone_x.llm.usage.gate import UsageGate
from uclone_x.llm.usage.limits import UsageLimits
from uclone_x.llm.usage.store import MemoryUsageStore
from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack
from uclone_x.shells.cluster.turn_executor import environment_llm
from uclone_x.shells.cluster.worker_process import ClusterWorkerProcess
from uclone_x.shells.cluster.worker_runner import (
    DEFAULT_MAX_TURN_ATTEMPTS,
    DEFAULT_TURN_DEADLINE_SECONDS,
    StatelessAgentWorker,
)
from uclone_x.skills.auditor import load_runtime_skill_registry

__all__ = ["run_worker"]

#: The one consumer group every worker joins, so each partition has one owner.
WORKER_GROUP = "ucx-worker"
#: Its own group: a learn job's rebalance moves no turn, and the other way round.
LEARN_GROUP = "ucx-learn-worker"


async def _serve(
    *,
    config: ClusterRuntimeConfig,
    worker_id: str,
    concurrency: int,
    workspace: Path | None,
    lease_ttl_seconds: int | None,
    session_timeout_ms: int,
    max_turn_attempts: int,
    turn_deadline_seconds: float,
) -> None:
    # Refuse to start with no model, rather than fail every turn it is handed (P6).
    await environment_llm(UsageGate(store=MemoryUsageStore(), limits=UsageLimits)).aclose()
    if lease_ttl_seconds is not None:
        config = config.model_copy(
            update={
                "cache_policy": config.cache_policy.model_copy(
                    update={"lease_ttl_seconds": lease_ttl_seconds}
                )
            }
        )
    stack = ClusterRuntimeStack.create_docker_stack(
        config=config,
        num_workers=1,
        workspace_root=workspace,
        skills=await load_runtime_skill_registry(),
    )
    # The schema and topics are `ucx cluster up`'s: N workers starting at once must not race to
    # create them, and a worker on a stack with neither fails on its first statement.
    template = stack.workers[0]
    worker = StatelessAgentWorker(
        worker_id=worker_id,
        session_store=stack.session_store,
        lease_manager=stack.lease_manager,
        sse_publisher=stack.sse_publisher,
        turn_executor=template.turn_executor,
        room_steps=stack.room_steps,
        broker=stack.broker,
        max_turn_attempts=max_turn_attempts,
        turn_deadline_seconds=turn_deadline_seconds,
    )
    process = ClusterWorkerProcess(
        worker=worker,
        consumer=KafkaCommandBroker(
            bootstrap_servers=config.kafka_bootstrap_servers,
            group_id=WORKER_GROUP,
            session_timeout_ms=session_timeout_ms,
        ),
        control=KafkaCommandBroker(bootstrap_servers=config.kafka_bootstrap_servers),
        control_topic=TOPIC_CTRL_INTERRUPT,
        concurrency=concurrency,
        learner=stack.learner,
        learn_consumer=KafkaCommandBroker(
            bootstrap_servers=config.kafka_bootstrap_servers,
            group_id=LEARN_GROUP,
            session_timeout_ms=session_timeout_ms,
        ),
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    try:
        await process.run(stop)
    finally:
        await stack.close()


def _refuse_an_unwritable_workspace(workspace: Path) -> None:
    """Stop here if the worker cannot write where each user's folder goes (P6).

    Past this point a bad mount shows only as every turn with files failing, one at a time.
    """
    users = workspace / "users"
    try:
        users.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=users):
            pass
    except OSError as exc:
        typer.echo(f"The worker cannot write to --workspace {workspace}: {exc.strerror}", err=True)
        raise typer.Exit(code=1) from exc


def run_worker(
    worker_id: str = typer.Option(
        "", "--worker-id", help="This worker's id (default: host, process and a random suffix)."
    ),
    concurrency: int = typer.Option(
        16, "--concurrency", min=1, help="Turns this worker runs at once."
    ),
    workspace: Path | None = typer.Option(  # noqa: B008
        None,
        "--workspace",
        help="A folder for users' files. Each user gets a subfolder of their own, and file tools "
        "cannot leave it. Without one the worker has no file tools.",
    ),
    lease_ttl_seconds: int | None = typer.Option(
        None, "--lease-ttl-seconds", min=1, help="Turn lease lifetime (default: the stack's)."
    ),
    session_timeout_ms: int = typer.Option(
        10000,
        "--kafka-session-timeout-ms",
        min=1000,
        help="How long Kafka waits before moving a silent worker's partitions.",
    ),
    max_turn_attempts: int = typer.Option(
        DEFAULT_MAX_TURN_ATTEMPTS,
        "--max-turn-attempts",
        min=1,
        help="Times one turn may run and fail before the person is told. Each can bill a model.",
    ),
    turn_deadline_seconds: float = typer.Option(
        DEFAULT_TURN_DEADLINE_SECONDS,
        "--turn-deadline-seconds",
        min=0.0,
        help="Seconds after a turn was submitted that a failure ends it instead of a retry.",
    ),
) -> None:
    """Run one stateless agent worker: consume turns and room commands until told to stop."""
    if workspace is not None:
        _refuse_an_unwritable_workspace(workspace.resolve())
    asyncio.run(
        _serve(
            config=ClusterRuntimeConfig.from_stack_env(stack_env_path()),
            worker_id=worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}",
            concurrency=concurrency,
            workspace=workspace.resolve() if workspace is not None else None,
            lease_ttl_seconds=lease_ttl_seconds,
            session_timeout_ms=session_timeout_ms,
            max_turn_attempts=max_turn_attempts,
            turn_deadline_seconds=turn_deadline_seconds,
        )
    )
