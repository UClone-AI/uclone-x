"""Cluster runtime stack orchestrator and FastAPI gateway for stateless distributed turns."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.adapters.cluster.catalog_cache import ValkeyCatalogCache
from uclone_x.adapters.cluster.centrifugo_jwt import CentrifugoTokenSigner
from uclone_x.adapters.cluster.config import ClusterRuntimeConfig
from uclone_x.adapters.cluster.kafka_broker import KafkaCommandBroker
from uclone_x.adapters.cluster.knowledge_store import PostgresKnowledgeStore
from uclone_x.adapters.cluster.postgres_pool import PostgresPool, bootstrap_schema
from uclone_x.adapters.cluster.room_store import ValkeyPostgresRoomStore
from uclone_x.adapters.cluster.session_store import ValkeyPostgresSessionStore
from uclone_x.adapters.cluster.sse_bridge import KafkaSSEBridge
from uclone_x.adapters.cluster.turn_lease import ValkeyTurnLeaseManager
from uclone_x.adapters.cluster.usage_store import DistributedUsageStore
from uclone_x.adapters.cluster.valkey_cache import ValkeyClusterCache
from uclone_x.core.cluster_protocols import (
    CacheClientProtocol,
    InterruptControlEnvelope,
    RoomCommandEnvelope,
    SqlPoolProtocol,
    SSEStreamEnvelope,
    TokenBudgetSnapshot,
    TurnCommandEnvelope,
    TurnLeaseConflictError,
    is_worker_id,
)
from uclone_x.core.session_state import SessionState
from uclone_x.engine.event_bus import AgentEvent, EventType
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.usage.gate import UsageGate
from uclone_x.llm.usage.limits import UsageLimits, load_limits
from uclone_x.shells.cluster.ingress_routes import ClusterIngressService
from uclone_x.shells.cluster.learn_worker import RoomLearnWorker
from uclone_x.shells.cluster.room_steps import RoomStepExecutor
from uclone_x.shells.cluster.turn_executor import (
    AgentTurnExecutor,
    LlmFactory,
    environment_llm,
)
from uclone_x.shells.cluster.worker_runner import (
    StatelessAgentWorker,
    TurnDisposition,
    TurnExecutorFn,
)
from uclone_x.skills.protocols import SkillRegistryProtocol

__all__ = [
    "ClusterHealthReport",
    "ClusterRuntimeStack",
    "ClusterTurnRunResult",
    "InfraComponentStatus",
    "create_cluster_http_app",
    "register_cluster_routes",
]

_DETAIL_WORKER_ID = "this id has the form of one the workers make; choose another"


class InfraComponentStatus(BaseModel):
    """Health or verification status for a single distributed infrastructure component."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str
    endpoint: str
    healthy: bool
    latency_ms: float
    detail: str = ""


class ClusterHealthReport(BaseModel):
    """Aggregate health report across Valkey, PostgreSQL, Kafka, and Centrifugo."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    backend: Literal["docker"] = "docker"
    all_healthy: bool
    components: tuple[InfraComponentStatus, ...]


class ClusterTurnRunResult(BaseModel):
    """Result of one turn run by `execute_turn_end_to_end`.

    `sse_events` are read back from Centrifugo's channel history, so they prove the events
    travelled Kafka -> Centrifugo's consumer -> the Valkey history store.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    session_id: str
    turn_id: str
    worker_id: str
    disposition: str
    committed_revision: int
    assistant_text: str
    sse_events: tuple[SSEStreamEnvelope, ...]
    budget_snapshot: TokenBudgetSnapshot | None = None


class _ExecuteTurnRequest(BaseModel):
    """JSON request body for `POST /api/cluster/execute`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    prompt: str = Field(min_length=1)
    session_id: str = "sess-cli"
    agent_id: str = "default"
    turn_id: str | None = None


class ClusterRuntimeStack:
    """Wires the library adapters to one running cluster stack, plus ingress and workers."""

    def __init__(
        self,
        *,
        config: ClusterRuntimeConfig,
        cache: CacheClientProtocol,
        sql_pool: SqlPoolProtocol,
        broker: KafkaCommandBroker,
        sse_publisher: KafkaSSEBridge,
        session_store: ValkeyPostgresSessionStore,
        room_store: ValkeyPostgresRoomStore,
        lease_manager: ValkeyTurnLeaseManager,
        usage_store: DistributedUsageStore,
        knowledge_store: PostgresKnowledgeStore,
        catalog_cache: ValkeyCatalogCache,
        token_signer: CentrifugoTokenSigner,
        ingress: ClusterIngressService,
        workers: tuple[StatelessAgentWorker, ...],
        room_sse: KafkaSSEBridge,
        room_steps: RoomStepExecutor,
        learner: RoomLearnWorker | None,
    ) -> None:
        self.backend: Literal["docker"] = "docker"
        self.config = config
        self.cache = cache
        self.sql_pool = sql_pool
        self.broker = broker
        self.sse_publisher = sse_publisher
        self.session_store = session_store
        self.room_store = room_store
        self.lease_manager = lease_manager
        self.usage_store = usage_store
        self.knowledge_store = knowledge_store
        self.catalog_cache = catalog_cache
        self.token_signer = token_signer
        self.ingress = ingress
        self.workers = workers
        self.room_sse = room_sse
        self.room_steps = room_steps
        self.learner = learner

    @classmethod
    def create_docker_stack(
        cls,
        *,
        config: ClusterRuntimeConfig,
        num_workers: int = 2,
        turn_executor: TurnExecutorFn | None = None,
        llm_factory: LlmFactory = environment_llm,
        workspace_root: Path | None = None,
        limits: Callable[[], UsageLimits] = load_limits,
        skills: SkillRegistryProtocol | None = None,
    ) -> ClusterRuntimeStack:
        """Build a stack against the compose topology described by `config`.

        Connections open lazily on first use, one per event loop; nothing here touches the
        network. Its workers run real agents on `llm_factory`'s connector; with no provider
        named anywhere, the first turn ends as `TURN_FAILED`, and `ucx worker` refuses to
        start. There is no stand-in model.
        """
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        policy = config.cache_policy
        cache = ValkeyClusterCache(
            nodes=config.valkey_nodes,
            password=config.valkey_password,
            address_map=config.valkey_address_map,
        )
        sql_pool = PostgresPool(config.postgres_dsn)
        broker = KafkaCommandBroker(bootstrap_servers=config.kafka_bootstrap_servers)
        sse_publisher = KafkaSSEBridge(cache, broker, channel_kind="session")
        session_store = ValkeyPostgresSessionStore(sql_pool, cache, policy=policy)
        room_store = ValkeyPostgresRoomStore(sql_pool, cache, policy=policy)
        lease_manager = ValkeyTurnLeaseManager(cache, default_ttl_seconds=policy.lease_ttl_seconds)
        ingress = ClusterIngressService(broker, lease_manager)
        usage_store = DistributedUsageStore(cache, sql_pool)
        room_sse = KafkaSSEBridge(cache, broker, channel_kind="room")
        clones = AgentTurnExecutor(
            sessions=session_store,
            usage=usage_store,
            sse=sse_publisher,
            llm_factory=llm_factory,
            workspace_root=workspace_root,
            limits=limits,
            skills=skills,
            rooms=room_store,
            room_sse=room_sse,
        )
        effective_executor = turn_executor or clones
        room_steps = RoomStepExecutor(
            rooms=room_store,
            sessions=session_store,
            broker=broker,
            sse=room_sse,
            leases=lease_manager,
            decide=clones.decide_for_room,
            clone_learns=clones.clone_learns,
        )
        workers = tuple(
            StatelessAgentWorker(
                worker_id=f"worker-{idx}",
                session_store=session_store,
                lease_manager=lease_manager,
                sse_publisher=sse_publisher,
                turn_executor=effective_executor,
                room_steps=room_steps,
                broker=broker,
            )
            for idx in range(num_workers)
        )
        return cls(
            config=config,
            cache=cache,
            sql_pool=sql_pool,
            broker=broker,
            sse_publisher=sse_publisher,
            session_store=session_store,
            room_store=room_store,
            lease_manager=lease_manager,
            usage_store=usage_store,
            knowledge_store=PostgresKnowledgeStore(sql_pool),
            catalog_cache=ValkeyCatalogCache(cache, default_ttl_seconds=policy.catalog_ttl_seconds),
            token_signer=CentrifugoTokenSigner(config.centrifugo_hmac_secret),
            ingress=ingress,
            workers=workers,
            room_sse=room_sse,
            room_steps=room_steps,
            learner=RoomLearnWorker(rooms=room_store, clones=clones),
        )

    async def bootstrap_schema(self) -> None:
        """Create the PostgreSQL tables and the Kafka topics that do not exist yet."""
        await bootstrap_schema(self.sql_pool)
        await self.broker.ensure_topics()

    async def close(self) -> None:
        """Close every client this stack opened, on every event loop."""
        await self.broker.stop()
        await self.cache.close()
        await self.sql_pool.close()

    async def _probe(self, name: str, endpoint: str, probe: Any) -> InfraComponentStatus:
        t0 = time.perf_counter()
        try:
            detail = str(await probe())
            healthy = True
        except Exception as exc:  # a probe reports, it never raises
            healthy = False
            detail = f"{type(exc).__name__}: {exc}"
        return InfraComponentStatus(
            name=name,
            endpoint=endpoint,
            healthy=healthy,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            detail=detail,
        )

    async def _probe_valkey(self) -> str:
        if not await self.cache.ping():
            raise RuntimeError("PING did not answer PONG")
        if not isinstance(self.cache, ValkeyClusterCache):
            return "PING OK"
        # Healthy means failover-ready: every primary has an online replica. Right after
        # `docker compose up` the slots are served before the replicas have synced.
        primaries, ready = await self.cache.replication_status()
        if primaries == 0 or ready < primaries:
            raise RuntimeError(f"{ready}/{primaries} primaries have an online replica")
        return f"cluster PING OK; {ready}/{primaries} primaries have an online replica"

    async def _probe_postgres(self) -> str:
        async with self.sql_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT 1 AS ok")
        if row is None or int(row["ok"]) != 1:
            raise RuntimeError("SELECT 1 returned no row")
        return "SELECT 1 OK"

    async def _probe_kafka(self) -> str:
        await self.broker.ping()
        return "metadata OK"

    async def _probe_centrifugo(self) -> str:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.config.centrifugo_url}/health")
        if resp.status_code != 200:
            raise RuntimeError(f"/health answered HTTP {resp.status_code}")
        return "/health OK"

    async def check_health(self) -> ClusterHealthReport:
        """Probe Valkey, PostgreSQL, Kafka, and Centrifugo and return a health report."""
        components = (
            await self._probe("valkey", ",".join(self.config.valkey_nodes), self._probe_valkey),
            await self._probe("postgres", "postgresql://***", self._probe_postgres),
            await self._probe(
                "kafka", ",".join(self.config.kafka_bootstrap_servers), self._probe_kafka
            ),
            await self._probe("centrifugo", self.config.centrifugo_url, self._probe_centrifugo),
        )
        return ClusterHealthReport(
            all_healthy=all(c.healthy for c in components), components=components
        )

    async def channel_history(
        self, channel: str, limit: int = 100
    ) -> tuple[SSEStreamEnvelope, ...]:
        """Read a channel's history from Centrifugo's server API (the Valkey history store)."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self.config.centrifugo_url}/api/history",
                headers={"X-API-Key": self.config.centrifugo_api_key},
                json={"channel": channel, "limit": limit},
            )
        resp.raise_for_status()
        body: Any = resp.json()
        if "error" in body:
            raise RuntimeError(f"Centrifugo history error: {body['error']}")
        publications: list[Mapping[str, Any]] = body.get("result", {}).get("publications", [])
        return tuple(
            SSEStreamEnvelope(
                channel=channel,
                idempotency_key=f"{channel}:{pub.get('offset', 0)}",
                data=dict(cast(Mapping[str, Any], pub.get("data", {}))),
            )
            for pub in publications
        )

    async def wait_for_history(
        self,
        channel: str,
        predicate: Callable[[tuple[SSEStreamEnvelope, ...]], bool],
        timeout_seconds: float = 15.0,
    ) -> tuple[SSEStreamEnvelope, ...]:
        """Poll `channel_history` until `predicate(envelopes)` holds or the timeout passes."""
        deadline = time.monotonic() + timeout_seconds
        envelopes = await self.channel_history(channel)
        while not predicate(envelopes) and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            envelopes = await self.channel_history(channel)
        return envelopes

    async def execute_turn_end_to_end(
        self,
        *,
        prompt: str,
        session_id: str = "sess-cli",
        agent_id: str = "default",
        turn_id: str | None = None,
        worker: StatelessAgentWorker | None = None,
    ) -> ClusterTurnRunResult:
        """Run one turn on an in-process worker; its SSE still crosses Kafka and Centrifugo.

        The command itself does not cross Kafka: this is the diagnostic path of `ucx cluster
        run` and `verify`, which work with no `ucx worker` running. A turn published to
        `ucx.cmd.turn` is what a `ucx worker` consumes.
        """
        effective_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        cmd = TurnCommandEnvelope(
            turn_id=effective_turn_id,
            session_id=session_id,
            agent_id=agent_id,
            user_id="cli-user",
            prompt=prompt,
            enqueued_at_ms=int(time.time() * 1000),
        )
        runner = worker if worker is not None else self.workers[0]
        outcome = await runner.execute_single_turn(cmd)
        budget = await asyncio.to_thread(self.session_store.load_budget_snapshot, session_id)

        def _terminal_arrived(envs: tuple[SSEStreamEnvelope, ...]) -> bool:
            return any(
                _payload_field(e, "event") == outcome.disposition.value
                and _payload_turn_id(e) == effective_turn_id
                for e in envs
            )

        history = await self.wait_for_history(f"ucx:session:{session_id}", _terminal_arrived)
        committed = outcome.state.revision if outcome.state is not None else 0
        return ClusterTurnRunResult(
            session_id=session_id,
            turn_id=effective_turn_id,
            worker_id=runner.worker_id,
            disposition=outcome.disposition.value,
            committed_revision=committed,
            assistant_text=outcome.result.content if outcome.result is not None else "",
            sse_events=tuple(e for e in history if _payload_turn_id(e) == effective_turn_id),
            budget_snapshot=budget,
        )

    async def run_live_verification_suite(self) -> tuple[InfraComponentStatus, ...]:
        """Five end-to-end checks against the running stack."""
        await self.bootstrap_schema()
        return (
            await self._probe(
                "1. Valkey lease, fencing & tombstone script",
                ",".join(self.config.valkey_nodes),
                self._verify_lease,
            ),
            await self._probe(
                "2. PostgreSQL revision OCC commit", "postgresql://***", self._verify_occ
            ),
            await self._probe(
                "3. Kafka command publish",
                ",".join(self.config.kafka_bootstrap_servers),
                self._verify_publish,
            ),
            await self._probe(
                "4. SSE via Kafka -> Centrifugo history",
                self.config.centrifugo_url,
                self._verify_sse,
            ),
            await self._probe(
                "5. Stateless worker turn lifecycle",
                f"workers={len(self.workers)}",
                self._verify_turn,
            ),
        )

    async def _verify_lease(self) -> str:
        conv = f"verify-lease-{uuid.uuid4().hex[:8]}"
        first = await self.lease_manager.acquire(conv, "t1", "verifier-a")
        try:
            await self.lease_manager.acquire(conv, "t1", "verifier-b")
        except TurnLeaseConflictError:
            pass
        else:
            raise RuntimeError("a second worker acquired a held lease")
        await self.lease_manager.release(first)
        second = await self.lease_manager.acquire(conv, "t2", "verifier-b")
        await self.lease_manager.release(second)
        if second.fencing_token <= first.fencing_token:
            raise RuntimeError(
                f"fencing regressed: {first.fencing_token} -> {second.fencing_token}"
            )
        return f"exclusive; fencing {first.fencing_token} -> {second.fencing_token}"

    async def _verify_occ(self) -> str:
        sid = f"verify-occ-{uuid.uuid4().hex[:8]}"
        saved = await asyncio.to_thread(
            self.session_store.save, SessionState(session_id=sid, agent_id="default")
        )
        loaded = await self.session_store.prefetch_session(sid)
        self.session_store.evict_local_turn_cache(sid)
        if loaded is None or saved.revision != 1 or loaded.revision != 1:
            raise RuntimeError(f"expected revision 1, got {saved.revision}/{loaded}")
        return f"committed {sid} at revision 1"

    async def _verify_publish(self) -> str:
        # An interrupt of a conversation nobody has, not a turn: a running `ucx worker` would
        # run a published turn, on a real model.
        conversation = f"verify-sess-{uuid.uuid4().hex[:6]}"
        await self.ingress.interrupt_turn(
            InterruptControlEnvelope(
                conversation_id=conversation,
                turn_id="verify-turn",
                requested_by="verifier",
                issued_at_ms=int(time.time() * 1000),
            )
        )
        return f"published an interrupt of {conversation}"

    async def _verify_sse(self) -> str:
        sid = f"verify-sse-{uuid.uuid4().hex[:6]}"
        evt = AgentEvent(
            type=EventType.AGENT_REPLY,
            topic=f"session.{sid}",
            session_id=sid,
            sender_id="default",
            payload={"verify": True},
        )
        seq = await self.sse_publisher.publish_event(sid, evt)

        def _any(envs: tuple[SSEStreamEnvelope, ...]) -> bool:
            return len(envs) >= 1

        history = await self.wait_for_history(f"ucx:session:{sid}", _any)
        if not history:
            raise RuntimeError(
                "Centrifugo history stayed empty: the Kafka consumer did not deliver"
            )
        return f"seq={seq} delivered to Centrifugo history"

    async def _verify_turn(self) -> str:
        """A real agent, commit and SSE, on a scripted model named as such in the report.

        `verify` checks the stack and must not spend a person's tokens or need their key.
        """

        def scripted(gate: UsageGate) -> BaseLLMConnector:
            _ = gate  # the mock is not paid; the executor books it for the ledger only
            return MockLLMConnector(default_response="Verified.", default_model="verify-mock")

        verifier = StatelessAgentWorker(
            worker_id="verifier",
            session_store=self.session_store,
            lease_manager=self.lease_manager,
            sse_publisher=self.sse_publisher,
            turn_executor=AgentTurnExecutor(
                sessions=self.session_store,
                usage=self.usage_store,
                sse=self.sse_publisher,
                llm_factory=scripted,
                limits=UsageLimits,
            ),
            broker=self.broker,
        )
        result = await self.execute_turn_end_to_end(
            prompt="Verify stateless worker turn lifecycle",
            session_id=f"verify-e2e-{uuid.uuid4().hex[:6]}",
            worker=verifier,
        )
        if result.disposition != TurnDisposition.COMPLETED.value or result.committed_revision < 1:
            raise RuntimeError(f"{result.disposition} at revision {result.committed_revision}")
        return (
            f"scripted model: rev={result.committed_revision} sse_events={len(result.sse_events)}"
        )


def _payload_field(envelope: SSEStreamEnvelope, name: str) -> str | None:
    payload: Any = envelope.data.get("payload")
    if isinstance(payload, Mapping):
        value: Any = cast(Mapping[str, Any], payload).get(name)
        return str(value) if value is not None else None
    return None


def _payload_turn_id(envelope: SSEStreamEnvelope) -> str | None:
    return _payload_field(envelope, "turn_id")


def register_cluster_routes(app: Any, stack: ClusterRuntimeStack) -> None:
    """Mount stateless cluster ingress, and turn execution endpoints on `app`."""
    try:
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        from uclone_x.errors import MissingDependencyError

        raise MissingDependencyError(
            extra="http",
            package="fastapi",
            feature="Cluster HTTP gateway (ucx cluster serve)",
        ) from exc

    globals()["JSONResponse"] = JSONResponse

    def _mint_subscription_jwt(sub: str, channel: str) -> str:
        return stack.token_signer.mint_channel_token(user_id=sub, channel=channel)

    @app.get("/api/cluster/health")
    async def cluster_health() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        report = await stack.check_health()
        return report.model_dump(mode="json")

    @app.post("/api/cluster/turns")
    async def submit_turn(cmd: TurnCommandEnvelope) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        if cmd.seat is not None:  # a seat's turn is the room's to give, never a caller's
            return JSONResponse(status_code=422, content={"detail": "not a turn a client submits"})
        if is_worker_id(cmd.turn_id):
            return JSONResponse(status_code=422, content={"detail": _DETAIL_WORKER_ID})
        accepted = await stack.ingress.submit_turn(cmd)
        token = _mint_subscription_jwt(sub=cmd.user_id, channel=accepted.sse_channel)
        turn_body = {**accepted.model_dump(mode="json"), "subscription_token": token}
        return JSONResponse(status_code=202, content=turn_body)

    @app.post("/api/cluster/rooms")
    async def submit_room_message(cmd: RoomCommandEnvelope) -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
        if cmd.kind != "ROOM_POST":  # the rest of a cascade is the workers' to send
            return JSONResponse(status_code=422, content={"detail": "only a post is submitted"})
        if is_worker_id(cmd.turn_id):
            return JSONResponse(status_code=422, content={"detail": _DETAIL_WORKER_ID})
        accepted = await stack.ingress.submit_room_message(cmd)
        subscription = _mint_subscription_jwt(sub=cmd.user_id, channel=accepted.sse_channel)
        room_body = {**accepted.model_dump(mode="json"), "subscription_token": subscription}
        return JSONResponse(status_code=202, content=room_body)

    @app.post("/api/cluster/interrupt")
    async def interrupt_turn(ctrl: InterruptControlEnvelope) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        await stack.ingress.interrupt_turn(ctrl)
        return {
            "status": "interrupted",
            "conversation_id": ctrl.conversation_id,
            "turn_id": ctrl.turn_id,
        }

    @app.post("/api/cluster/execute")
    async def execute_turn(body: _ExecuteTurnRequest) -> Any:  # pyright: ignore[reportUnusedFunction]
        if body.turn_id is not None and is_worker_id(body.turn_id):
            return JSONResponse(status_code=422, content={"detail": _DETAIL_WORKER_ID})
        result = await stack.execute_turn_end_to_end(
            prompt=body.prompt,
            session_id=body.session_id,
            agent_id=body.agent_id,
            turn_id=body.turn_id,
        )
        return result.model_dump(mode="json")


def create_cluster_http_app(stack: ClusterRuntimeStack) -> Any:
    """Create the FastAPI application serving stateless cluster ingress and turn execution endpoints."""
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        from uclone_x.errors import MissingDependencyError

        raise MissingDependencyError(
            extra="http",
            package="fastapi",
            feature="Cluster HTTP gateway (ucx cluster serve)",
        ) from exc

    app = FastAPI(title="UClone-X Stateless Cluster Gateway")
    register_cluster_routes(app, stack)
    return app
