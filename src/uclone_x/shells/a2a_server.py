"""A2A HTTP/SSE Server implementation exposing RFC 8615 discovery and REST/SSE endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Final, cast

try:
    import uvicorn
    from fastapi import FastAPI, Header, HTTPException, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:
    pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    from uclone_x.errors import MissingDependencyError

    raise MissingDependencyError(
        extra="http",
        package=pkg,
        feature="A2A HTTP/SSE server",
    ) from exc
from pydantic import BaseModel, ConfigDict, Field

from uclone_x.a2a.models import AgentCard, TaskMessage, TaskResult, TaskStatus
from uclone_x.a2a.wire import task_result_to_wire, task_result_to_wire_json
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import TurnResult
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.core.set_aside import SESSION_SET_ASIDE_NOTICE
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import (
    A2AError,
    MissingProvenanceError,
    TaskNotFoundError,
)

logger = logging.getLogger(__name__)

TaskHandler = Callable[[TaskMessage], Awaitable[TaskResult]]
StreamHandler = Callable[[TaskMessage], AsyncIterator[str]]

#: Builds the agent that answers one A2A conversation (`contextId`), with its saved
#: history restored (#1836). The CLI builds it on the clone's seat session in the
#: conversation's one-seat room (§5.9); the server keeps one agent per context.
ContextAgentFactory = Callable[[str], BaseAgent]


@dataclass(frozen=True, slots=True)
class A2ATurnFailure:
    """A turn that returned no result: it raised (`cause` is the raw text), or was cancelled."""

    cause: str
    completed: bool = True


#: Records a turn a context ran, beyond the agent's own save (#1837): the CLI records it
#: in the context's one-seat room transcript. Called with the context id, the prompt, the
#: turn's result or an `A2ATurnFailure`, and whether the save after the turn kept an
#: unreadable earlier record aside (#1921). A recorder that raises is logged; the task's
#: outcome stands.
A2ATurnRecorder = Callable[[str, str, "TurnResult | A2ATurnFailure", bool], None]

#: The names the person on the other end of an A2A conversation goes by, read before each
#: turn with the context id (#1893 item 1). The CLI reads them from the context's one-seat
#: room, as the room orchestrator gives a seat's turn its room's. Unset, a turn is given none.
A2APersonNames = Callable[[str], tuple[str, ...]]

#: How long a cancel waits for the task to finish its own cancellation before answering.
CANCEL_SETTLE_SECONDS: Final[float] = 5.0

#: How many per-context agents a server holds at once. See `A2AServer`.
DEFAULT_MAX_CONTEXT_AGENTS: Final[int] = 8

#: A task's error when every held agent is busy or unsaved. Read by the calling agent's
#: owner, so it says what to do and nothing about the server.
CONTEXTS_BUSY_MESSAGE: Final[str] = (
    "Too many conversations are in progress with this agent. Please try again shortly."
)

#: A task's error when the conversation's agent could not be built or its history read.
CONTEXT_UNOPENABLE_MESSAGE: Final[str] = "This conversation could not be opened. Please try again."

#: A task's error when the agent's turn failed (#1885 item 8). The calling agent's owner
#: reads it, so it is plain: the cause -- an exception's text, a provider's answer, a
#: path -- goes to this server's log, never over the wire.
TURN_FAILED_MESSAGE: Final[str] = (
    "The agent could not finish this task. The reason is in the log of the agent's server."
)

#: A task's error when the turn came back with nobody named as its producer (P6). The
#: answer is refused rather than attributed by whoever holds it; plain for the same reader.
TURN_UNATTRIBUTED_MESSAGE: Final[str] = (
    "The agent's answer could not be checked, so it was not sent. "
    "The reason is in the log of the agent's server."
)


class _ContextRefusedError(Exception):
    """A context's turn was not run; `message` is what the task says, in plain words."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TaskCreateRequest(BaseModel):
    """Payload model for POST /a2a/v1/tasks."""

    model_config = ConfigDict(extra="allow", strict=False, populate_by_name=True)

    task_id: str | None = None
    session_id: str | None = None
    context_id: str | None = Field(
        default=None,
        alias="contextId",
        description="The A2A conversation this task belongs to. A server serving one-seat "
        "rooms continues the room of a known context and starts one for a new context "
        "(#1836). Absent, `session_id` stands in for it, and a new id when neither is sent.",
    )
    input_data: dict[str, Any] = Field(default_factory=dict)
    sender_agent_id: str | None = None
    target_agent_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    prompt: str | None = None
    message: str | None = None


class ManagedTaskRecord:
    """In-memory tracking record for asynchronous A2A tasks."""

    task_id: str
    session_id: str
    context_id: str
    input_data: dict[str, Any]
    sender_agent_id: str
    target_agent_id: str
    metadata: dict[str, str]
    status: TaskStatus
    output_data: dict[str, Any]
    artifacts: list[dict[str, Any]]
    events: list[dict[str, Any]]
    error: str | None
    provenance: Provenance | None
    created_at: float
    updated_at: float
    async_task: asyncio.Task[None] | None
    listeners: list[asyncio.Queue[dict[str, Any] | None]]
    _lock: asyncio.Lock

    def __init__(
        self,
        task_id: str,
        session_id: str,
        input_data: dict[str, Any],
        sender_agent_id: str = "client",
        target_agent_id: str = "default",
        metadata: dict[str, str] | None = None,
        context_id: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.context_id = context_id if context_id is not None else session_id
        self.input_data = input_data
        self.sender_agent_id = sender_agent_id
        self.target_agent_id = target_agent_id
        self.metadata = metadata or {}
        self.status = TaskStatus.WORKING
        self.output_data: dict[str, Any] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None
        # Words for the person beside the task's outcome, as A2A's `TaskStatus.message`
        # carries them (#1921): set when the save after the turn kept a record aside.
        self.status_message: str | None = None
        self.provenance: Provenance | None = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.async_task: asyncio.Task[None] | None = None
        self.listeners: list[asyncio.Queue[dict[str, Any] | None]] = []
        self._lock = asyncio.Lock()

    @property
    def is_terminal(self) -> bool:
        """Check if task has reached a terminal execution status."""
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.REJECTED,
            TaskStatus.CANCELED,
        }

    async def emit_event(self, event: dict[str, Any]) -> None:
        """Record and broadcast an execution event to all active SSE subscribers."""
        async with self._lock:
            event_with_meta: dict[str, Any] = {
                **event,
                "task_id": self.task_id,
                "timestamp": time.time(),
            }
            self.events.append(event_with_meta)
            self.updated_at = time.time()
            for q in list(self.listeners):
                await q.put(event_with_meta)

    async def close_listeners(self) -> None:
        """Signal closure to all active SSE subscribers."""
        async with self._lock:
            for q in list(self.listeners):
                await q.put(None)
            self.listeners.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert task record to wire dictionary representation."""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "context_id": self.context_id,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "artifacts": self.artifacts,
            "error": self.error,
            "status_message": self.status_message,
            "provenance": self.provenance.model_dump(mode="json") if self.provenance else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class A2AServer:
    """HTTP/SSE wire server for A2A v1.0.1 protocol discovery and task dispatch.

    **One agent per conversation, when given `context_agent_factory` (#1836).** Each
    `contextId` is answered by an agent of its own, built by the factory on first use --
    the CLI builds it on the clone's seat session in the context's one-seat room -- so a
    known context continues its conversation and a new one starts fresh. One turn runs at
    a time per context. After each turn the agent's session is saved and the turn handed
    to `turn_recorder`. At most `max_context_agents` agents are held; past that the least
    recently used one that is idle and saved is released (its session is on disk, and its
    next task rebuilds it). An unsaved agent is never released, and if none can be the
    task fails with `CONTEXTS_BUSY_MESSAGE`. Without a factory, `agent` answers every task
    in its one session, as before.
    """

    def __init__(
        self,
        agent_card: AgentCard,
        handler: TaskHandler | None = None,
        stream_handler: StreamHandler | None = None,
        agent: BaseAgent | None = None,
        bus: EventBus | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        context_agent_factory: ContextAgentFactory | None = None,
        turn_recorder: A2ATurnRecorder | None = None,
        max_context_agents: int = DEFAULT_MAX_CONTEXT_AGENTS,
        person_names: A2APersonNames | None = None,
    ) -> None:
        if max_context_agents < 1:
            raise ValueError(f"max_context_agents must be at least 1, got {max_context_agents}")
        self._agent_card = agent_card
        self._handler = handler
        self._stream_handler = stream_handler
        self._agent = agent
        self._context_agent_factory = context_agent_factory
        self._turn_recorder = turn_recorder
        self._person_names = person_names
        self._max_context_agents = max_context_agents
        # Least recently used first.
        self._context_agents: OrderedDict[str, BaseAgent] = OrderedDict()
        # A context's lock lives while a turn holds or awaits it, or while its agent is
        # held; `_context_users` counts the turns. Kept for every context ever seen, the
        # map grew by one lock per caller conversation for the server's life (#1885).
        self._context_locks: dict[str, asyncio.Lock] = {}
        self._context_users: dict[str, int] = {}
        # Contexts whose last save failed: their agent holds what the store does not.
        self._unsaved_contexts: set[str] = set()
        self._bus = bus
        self._host = host
        self._requested_port = port
        self._bound_port: int | None = port if port != 0 else None
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._startup_event = asyncio.Event()
        self._tasks: dict[str, ManagedTaskRecord] = {}
        self.processing_errors: list[dict[str, Any]] = []
        self._app = self._build_app()

    @property
    def agent_card(self) -> AgentCard:
        """Declared local agent card."""
        return self._agent_card

    @property
    def host(self) -> str:
        """Bound host address."""
        return self._host

    @property
    def port(self) -> int:
        """Bound port number."""
        if self._bound_port is None:
            raise RuntimeError("Server is not running; port is not bound yet")
        return self._bound_port

    @property
    def url(self) -> str:
        """Base URL for the running server."""
        return f"http://{self.host}:{self.port}"

    @property
    def app(self) -> FastAPI:
        """Underlying FastAPI ASGI application."""
        return self._app

    def context_agent(self, context_id: str) -> BaseAgent | None:
        """The agent currently held for `context_id`, or `None`."""
        return self._context_agents.get(context_id)

    @property
    def tasks(self) -> dict[str, ManagedTaskRecord]:
        """Managed task registry dictionary."""
        return self._tasks

    def get_local_agent_card(self) -> AgentCard:
        """Export local agent capabilities."""
        return self._agent_card

    def _verify_version(self, a2a_version: str | None) -> None:
        if a2a_version is None or a2a_version != "1.0":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "VersionNotSupportedError",
                    "message": f"Unsupported A2A version: {a2a_version}. Expected '1.0'",
                },
            )

    async def get_agent_card_endpoint(
        self,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> JSONResponse:
        """Serve RFC 8615 /.well-known/agent-card.json discovery endpoint."""
        self._verify_version(a2a_version)
        card_dict = self._agent_card.model_dump(mode="json")
        return JSONResponse(
            content=card_dict,
            headers={
                "Cache-Control": "max-age=3600",
                "ETag": f'"{self._agent_card.version}"',
                "Content-Type": "application/json",
            },
        )

    async def send_task_endpoint(
        self,
        message: TaskMessage,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> JSONResponse:
        """Process remote task dispatch."""
        self._verify_version(a2a_version)
        if self._handler is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "TaskNotFoundError", "message": "No task handler registered"},
            )
        try:
            result = await self._handler(message)
            if result.provenance is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "MissingProvenanceError",
                        "message": "TaskResult missing provenance (P6)",
                    },
                )
            return JSONResponse(content=task_result_to_wire(result))
        except HTTPException:
            raise
        except TaskNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail={"error": "TaskNotFoundError", "message": str(e)},
            ) from e
        except MissingProvenanceError as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "MissingProvenanceError", "message": str(e)},
            ) from e
        except A2AError as e:
            raise HTTPException(
                status_code=e.http_status,
                detail={"error": type(e).__name__, "message": str(e)},
            ) from e
        except Exception as e:
            logger.exception("A2A task %s: the task handler failed", message.task_id)
            raise HTTPException(
                status_code=500,
                detail={"error": "InvalidAgentResponseError", "message": TURN_FAILED_MESSAGE},
            ) from e

    async def stream_task_endpoint(
        self,
        message: TaskMessage,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> Response:
        """Stream task results via SSE text/event-stream."""
        self._verify_version(a2a_version)
        if self._stream_handler is not None:
            stream_iter = self._stream_handler(message)
        elif self._handler is not None:

            async def _gen() -> AsyncIterator[str]:
                assert self._handler is not None
                res = await self._handler(message)
                if res.provenance is None:
                    raise MissingProvenanceError("TaskResult missing provenance (P6)")
                yield task_result_to_wire_json(res)

            stream_iter = _gen()
        else:
            raise HTTPException(
                status_code=404,
                detail={"error": "TaskNotFoundError", "message": "No stream handler registered"},
            )

        async def event_generator() -> AsyncIterator[str]:
            try:
                async for chunk in stream_iter:
                    yield f"data: {chunk}\n\n"
            except Exception:
                logger.exception("A2A task %s: the task stream failed", message.task_id)
                yield f"data: {json.dumps({'error': TURN_FAILED_MESSAGE})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def create_task_endpoint(
        self,
        payload: TaskCreateRequest,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> JSONResponse:
        """Create and dispatch an asynchronous A2A task (POST /a2a/v1/tasks)."""
        self._verify_version(a2a_version)
        task_id = payload.task_id or f"task_{uuid.uuid4().hex[:12]}"
        session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:8]}"
        # The conversation: the caller's `contextId`, else its session id, else the new
        # session id -- so a task sent with neither starts a conversation of its own.
        context_id = payload.context_id or session_id
        input_data = dict(payload.input_data)
        if payload.prompt is not None and "prompt" not in input_data:
            input_data["prompt"] = payload.prompt
        if payload.message is not None and "message" not in input_data:
            input_data["message"] = payload.message
        sender_agent_id = payload.sender_agent_id or "client"
        target_agent_id = payload.target_agent_id or self._agent_card.name
        metadata = dict(payload.metadata)

        record = ManagedTaskRecord(
            task_id=task_id,
            session_id=session_id,
            input_data=input_data,
            sender_agent_id=sender_agent_id,
            target_agent_id=target_agent_id,
            metadata=metadata,
            context_id=context_id,
        )
        self._tasks[task_id] = record

        record.async_task = asyncio.create_task(self._execute_managed_task(record))
        return JSONResponse(status_code=201, content=record.to_dict())

    async def get_task_endpoint(
        self,
        task_id: str,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> JSONResponse:
        """Retrieve task execution status and artifacts (GET /a2a/v1/tasks/{task_id})."""
        self._verify_version(a2a_version)
        record = self._tasks.get(task_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "TaskNotFoundError", "message": f"Task '{task_id}' not found"},
            )
        return JSONResponse(content=record.to_dict())

    async def stream_task_events_endpoint(
        self,
        task_id: str,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> Response:
        """SSE stream for streaming task tokens and state events (GET /a2a/v1/tasks/{task_id}/events)."""
        self._verify_version(a2a_version)
        record = self._tasks.get(task_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "TaskNotFoundError", "message": f"Task '{task_id}' not found"},
            )

        async def event_generator() -> AsyncIterator[str]:
            client_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            record.listeners.append(client_q)
            try:
                # Replay past events snapshot
                for past_evt in list(record.events):
                    yield f"data: {json.dumps(past_evt)}\n\n"

                if record.is_terminal:
                    return

                while True:
                    evt = await client_q.get()
                    if evt is None:
                        break
                    yield f"data: {json.dumps(evt)}\n\n"
                    if record.is_terminal:
                        break
            finally:
                if client_q in record.listeners:
                    record.listeners.remove(client_q)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def cancel_task_endpoint(
        self,
        task_id: str,
        a2a_version: str | None = Header(default="1.0", alias="A2A-Version"),
    ) -> JSONResponse:
        """Cancel a running task (POST /a2a/v1/tasks/{task_id}/cancel)."""
        self._verify_version(a2a_version)
        record = self._tasks.get(task_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "TaskNotFoundError", "message": f"Task '{task_id}' not found"},
            )
        if record.is_terminal:
            return JSONResponse(
                status_code=200,
                content={
                    "task_id": task_id,
                    "status": record.status.value,
                    "message": "Task already in terminal state",
                },
            )

        task = record.async_task
        if task is not None and not task.done():
            task.cancel()
            # The task finishes its own cancellation first: the turn's save runs there, and
            # a `notice` that it kept a record aside reaches the listeners before the task
            # closes them (#1921). Bounded, so a task slow to stop cannot hold the reply.
            await asyncio.wait({task}, timeout=CANCEL_SETTLE_SECONDS)
        if record.is_terminal:  # it ended on its own while the cancel waited
            return JSONResponse(
                status_code=200, content={"task_id": task_id, "status": record.status.value}
            )
        record.status = TaskStatus.CANCELED
        await record.emit_event({"event": "canceled"})
        await record.close_listeners()
        return JSONResponse(status_code=200, content={"task_id": task_id, "status": "canceled"})

    async def _execute_managed_task(self, record: ManagedTaskRecord) -> None:
        """Execute managed task in background and broadcast status events."""
        try:
            await record.emit_event({"event": "status_changed", "status": "WORKING"})
            if self._agent is not None or self._context_agent_factory is not None:
                prompt = str(
                    record.input_data.get("prompt")
                    or record.input_data.get("message")
                    or json.dumps(record.input_data)
                )
                await record.emit_event({"event": "state", "state": "REASONING"})
                try:
                    answering_id, turn_result = await self._run_turn(
                        record.context_id, prompt, on_set_aside=record
                    )
                except _ContextRefusedError as refused:
                    record.status = TaskStatus.FAILED
                    record.error = refused.message
                    await record.emit_event({"event": "failed", "error": record.error})
                    return
                await self._tell_set_aside(record)  # the turn's save
                await record.emit_event({"event": "token", "content": turn_result.content})
                if turn_result.provenance is None:
                    # P6, and the policy this file already applies twice: an unattributed
                    # result is refused, not attributed by whoever happens to be holding
                    # it. `send_task_endpoint` answers 500 MissingProvenanceError and the
                    # SSE generator raises it; this path used to substitute
                    # `Provenance.primary(self._agent.agent_id)` instead, naming *this
                    # gateway's agent* as the primary producer of a value that arrived
                    # with no producer at all — and then ship it to a remote peer that
                    # has no way to know the attribution was invented (#157).
                    record.status = TaskStatus.FAILED
                    record.provenance = None
                    logger.warning(
                        "A2A task %s: agent %r produced turn %s with no provenance; refused "
                        "(P6). The turn reported: %s",
                        record.task_id,
                        answering_id,
                        turn_result.turn_index,
                        turn_result.error or "no error",
                    )
                    record.error = TURN_UNATTRIBUTED_MESSAGE
                    await record.emit_event({"event": "failed", "error": record.error})
                elif turn_result.is_completed:
                    record.status = TaskStatus.COMPLETED
                    record.output_data = {
                        "result": turn_result.content,
                        "turn_index": turn_result.turn_index,
                    }
                    record.artifacts = [
                        {"name": "response", "content": turn_result.content, "type": "text"}
                    ]
                    record.provenance = turn_result.provenance
                    await record.emit_event(
                        {
                            "event": "completed",
                            "output": record.output_data,
                            "artifacts": record.artifacts,
                        }
                    )
                else:
                    record.status = TaskStatus.FAILED
                    logger.warning(
                        "A2A task %s: agent %r's turn failed (%s): %s",
                        record.task_id,
                        answering_id,
                        turn_result.stop_reason,
                        turn_result.error,
                    )
                    record.error = TURN_FAILED_MESSAGE
                    record.provenance = turn_result.provenance
                    await record.emit_event({"event": "failed", "error": record.error})
            elif self._handler is not None:
                msg = TaskMessage(
                    task_id=record.task_id,
                    session_id=record.session_id,
                    input_data=record.input_data,
                    sender_agent_id=record.sender_agent_id,
                    target_agent_id=record.target_agent_id,
                    metadata=record.metadata,
                )
                result = await self._handler(msg)
                if result.provenance is None:
                    # Same refusal as above, and the same one `send_task_endpoint`
                    # already applies to this very handler's output (#157). The
                    # substituted `Provenance.primary("handler")` named a provider that
                    # does not exist — there is no service called "handler".
                    record.status = TaskStatus.FAILED
                    logger.warning(
                        "A2A task %s: the task handler returned a result with no "
                        "provenance; refused (P6)",
                        record.task_id,
                    )
                    record.error, record.provenance = TURN_UNATTRIBUTED_MESSAGE, None
                    await record.emit_event({"event": "failed", "error": record.error})
                    return
                record.status = result.status
                record.output_data = cast(dict[str, Any], unwrap_immutable(result.output_data))
                record.error = result.error
                record.provenance = result.provenance
                if result.output_data:
                    record.artifacts = [
                        {
                            "name": "result",
                            "content": cast(dict[str, Any], unwrap_immutable(result.output_data)),
                        }
                    ]
                if result.status == TaskStatus.COMPLETED:
                    await record.emit_event(
                        {
                            "event": "completed",
                            "output": record.output_data,
                            "artifacts": record.artifacts,
                        }
                    )
                else:
                    await record.emit_event(
                        {"event": "failed", "error": record.error or "Task failed"}
                    )
            else:
                prompt = str(
                    record.input_data.get("prompt")
                    or record.input_data.get("message")
                    or "Task processed"
                )
                record.status = TaskStatus.COMPLETED
                record.output_data = {"result": f"Executed: {prompt}"}
                record.artifacts = [{"name": "result", "content": record.output_data}]
                record.provenance = Provenance.primary("a2a_server")
                await record.emit_event(
                    {
                        "event": "completed",
                        "output": record.output_data,
                        "artifacts": record.artifacts,
                    }
                )
        except asyncio.CancelledError:
            record.status = TaskStatus.CANCELED
            await self._tell_set_aside(record)  # the cancelled turn's save
            await record.emit_event({"event": "canceled"})
        except Exception as exc:
            logger.exception("Task %s failed with unexpected exception", record.task_id)
            self.processing_errors.append(
                {
                    "task_id": record.task_id,
                    "error_class": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": time.time(),
                }
            )
            record.status = TaskStatus.FAILED
            # The cause is logged above and kept in `processing_errors`; the caller is told
            # plainly (#1885 item 8).
            record.error, record.provenance = TURN_FAILED_MESSAGE, None
            await self._tell_set_aside(record)  # the failed turn's save
            await record.emit_event({"event": "error", "error": record.error})
        finally:
            await record.close_listeners()

    async def _run_turn(
        self, context_id: str, prompt: str, *, on_set_aside: ManagedTaskRecord | None = None
    ) -> tuple[str, TurnResult]:
        """Run `prompt` as `context_id`'s turn; the answering agent's id and the result.

        With a context factory: one turn at a time per context, then the agent's session
        is saved and the turn recorded -- a turn that raised or was cancelled is saved and
        recorded as failed before it propagates. When that save kept an unreadable earlier
        record aside, the room row is flagged and `on_set_aside` is given the notice,
        before the turn's result or exception reaches the caller (#1921).

        Raises:
            _ContextRefusedError: The context has no agent and none can be released, or
                its agent could not be built. Nothing was run.
        """
        factory = self._context_agent_factory
        if factory is None:
            if self._agent is None:
                raise A2AError("No agent is configured to answer this task.")
            return self._agent.agent_id, await self._agent.execute_turn(prompt)
        lock = self._context_locks.setdefault(context_id, asyncio.Lock())
        self._context_users[context_id] = self._context_users.get(context_id, 0) + 1
        try:
            async with lock:
                return await self._run_context_turn(context_id, prompt, factory, on_set_aside)
        finally:
            self._release_context_lock(context_id)

    def _release_context_lock(self, context_id: str) -> None:
        """One turn is done with `context_id`'s lock; drop it once nothing needs it."""
        users = self._context_users[context_id] - 1
        if users:
            self._context_users[context_id] = users
            return
        del self._context_users[context_id]
        if context_id not in self._context_agents:
            del self._context_locks[context_id]

    async def _run_context_turn(
        self,
        context_id: str,
        prompt: str,
        factory: ContextAgentFactory,
        on_set_aside: ManagedTaskRecord | None = None,
    ) -> tuple[str, TurnResult]:
        """`_run_turn`'s body, under `context_id`'s lock."""
        agent = self._agent_for_context(context_id, factory)
        outcome: TurnResult | A2ATurnFailure
        try:
            outcome = await agent.execute_turn(
                prompt, person_names=self._turn_person_names(context_id)
            )
        except asyncio.CancelledError:
            self._finish_context_turn(
                context_id,
                agent,
                prompt,
                A2ATurnFailure(cause="Turn was interrupted", completed=False),
                on_set_aside,
            )
            raise
        except Exception as exc:
            self._finish_context_turn(
                context_id,
                agent,
                prompt,
                A2ATurnFailure(cause=f"{type(exc).__name__}: {exc}"),
                on_set_aside,
            )
            raise
        self._finish_context_turn(context_id, agent, prompt, outcome, on_set_aside)
        return agent.agent_id, outcome

    def _finish_context_turn(
        self,
        context_id: str,
        agent: BaseAgent,
        prompt: str,
        outcome: TurnResult | A2ATurnFailure,
        on_set_aside: ManagedTaskRecord | None,
    ) -> None:
        """Save `context_id`'s session after a turn and record the turn; never raises."""
        self._save_context(context_id, agent)
        set_aside = self._took_set_aside(context_id, agent)  # this turn's save
        self._record_turn(context_id, prompt, outcome, set_aside)
        if set_aside and on_set_aside is not None:
            on_set_aside.status_message = SESSION_SET_ASIDE_NOTICE

    @staticmethod
    def _took_set_aside(context_id: str, agent: BaseAgent) -> bool:
        """Whether the agent's last save kept an unreadable earlier record aside; once each.

        Only a store that sets records aside can say so; any other says it did not. Never
        raises: the task's outcome stands whatever the answer (#1921).
        """
        try:
            take = getattr(agent.store, "take_set_aside", None)
            return take is not None and take(agent.session_id) is True
        except Exception:
            logger.exception("Could not ask whether A2A context %s was set aside", context_id)
            return False

    @staticmethod
    async def _tell_set_aside(record: ManagedTaskRecord) -> None:
        """Tell the task's caller, once, in the words every head uses, that a record was kept aside.

        A `notice` event carrying the task's `status_message`: plain text for the person,
        no path and no cause (#1860, #1921). The task's result stays in its artifacts.
        """
        if record.status_message is None or any(
            event.get("event") == "notice" for event in record.events
        ):
            return
        await record.emit_event({"event": "notice", "message": record.status_message})

    def _turn_person_names(self, context_id: str) -> tuple[str, ...]:
        """The person's names for `context_id`'s next turn (`A2APersonNames`); none if unset.

        A reader that raises is logged and the turn is given none: the names sharpen where
        a fact is filed, and are not worth refusing the task over.
        """
        if self._person_names is None:
            return ()
        try:
            return self._person_names(context_id)
        except Exception:
            logger.exception("Could not read the person's names for A2A context %s", context_id)
            return ()

    def _agent_for_context(self, context_id: str, factory: ContextAgentFactory) -> BaseAgent:
        """`context_id`'s held agent, or a new one from the factory once there is room."""
        agent = self._context_agents.get(context_id)
        if agent is not None:
            self._context_agents.move_to_end(context_id)
            return agent
        if len(self._context_agents) >= self._max_context_agents:
            victim = next(
                (
                    held
                    for held in self._context_agents
                    if held not in self._unsaved_contexts and not self._context_locks[held].locked()
                ),
                None,
            )
            if victim is None:
                raise _ContextRefusedError(CONTEXTS_BUSY_MESSAGE)
            del self._context_agents[victim]
            if victim not in self._context_users:
                del self._context_locks[victim]
        try:
            agent = factory(context_id)
        except Exception as exc:
            logger.exception("Could not open A2A context %s", context_id)
            raise _ContextRefusedError(CONTEXT_UNOPENABLE_MESSAGE) from exc
        self._context_agents[context_id] = agent
        return agent

    def _save_context(self, context_id: str, agent: BaseAgent) -> None:
        """Persist `context_id`'s session after a turn; an unsaved agent is kept held."""
        try:
            agent.persist_session()
        except Exception:
            logger.exception("Could not save A2A context %s after its turn", context_id)
            self._unsaved_contexts.add(context_id)
            return
        self._unsaved_contexts.discard(context_id)

    def _record_turn(
        self,
        context_id: str,
        prompt: str,
        outcome: TurnResult | A2ATurnFailure,
        session_set_aside: bool = False,
    ) -> None:
        """Hand the turn to `turn_recorder`, if there is one; never raises."""
        if self._turn_recorder is None:
            return
        try:
            self._turn_recorder(context_id, prompt, outcome, session_set_aside)
        except Exception:
            logger.exception("Could not record the turn of A2A context %s", context_id)

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
            self._startup_event.set()
            yield

        app = FastAPI(title="UClone-X A2A Server", lifespan=lifespan)

        # Wire discovery & task routes
        app.add_api_route(
            "/.well-known/agent-card.json",
            self.get_agent_card_endpoint,
            methods=["GET"],
        )
        app.add_api_route("/message:send", self.send_task_endpoint, methods=["POST"])
        app.add_api_route("/tasks/send", self.send_task_endpoint, methods=["POST"])
        app.add_api_route("/tasks", self.send_task_endpoint, methods=["POST"])
        app.add_api_route("/message:stream", self.stream_task_endpoint, methods=["POST"])
        app.add_api_route("/tasks/stream", self.stream_task_endpoint, methods=["POST"])

        # A2A v1 Task lifecycle & SSE endpoints
        app.add_api_route("/a2a/v1/tasks", self.create_task_endpoint, methods=["POST"])
        app.add_api_route("/a2a/v1/tasks/{task_id}", self.get_task_endpoint, methods=["GET"])
        app.add_api_route(
            "/a2a/v1/tasks/{task_id}/events", self.stream_task_events_endpoint, methods=["GET"]
        )
        app.add_api_route(
            "/a2a/v1/tasks/{task_id}/cancel", self.cancel_task_endpoint, methods=["POST"]
        )

        return app

    async def start(self) -> None:
        """Start the A2A wire server in the background and wait until reactive readiness."""
        if self._server_task is not None:
            return

        if self._agent is not None:
            await self._agent.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._requested_port))
        self._bound_port = int(sock.getsockname()[1])

        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._bound_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._startup_event.clear()

        self._server_task = asyncio.create_task(self._server.serve(sockets=[sock]))
        await self._startup_event.wait()

    async def stop(self) -> None:
        """Gracefully stop the server and cancel any running tasks."""
        for record in list(self._tasks.values()):
            if record.async_task is not None and not record.async_task.done():
                record.async_task.cancel()
                try:
                    await record.async_task
                except (asyncio.CancelledError, Exception):
                    pass
            await record.close_listeners()

        if self._agent is not None:
            await self._agent.stop()

        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await self._server_task
            self._server_task = None
        self._server = None
