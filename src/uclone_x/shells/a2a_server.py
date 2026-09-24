"""A2A HTTP/SSE Server implementation exposing RFC 8615 discovery and REST/SSE endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

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
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.errors import (
    A2AError,
    MissingProvenanceError,
    TaskNotFoundError,
)

logger = logging.getLogger(__name__)

TaskHandler = Callable[[TaskMessage], Awaitable[TaskResult]]
StreamHandler = Callable[[TaskMessage], AsyncIterator[str]]


class TaskCreateRequest(BaseModel):
    """Payload model for POST /a2a/v1/tasks."""

    model_config = ConfigDict(extra="allow", strict=False)

    task_id: str | None = None
    session_id: str | None = None
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
    ) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.input_data = input_data
        self.sender_agent_id = sender_agent_id
        self.target_agent_id = target_agent_id
        self.metadata = metadata or {}
        self.status = TaskStatus.WORKING
        self.output_data: dict[str, Any] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None
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
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "artifacts": self.artifacts,
            "error": self.error,
            "provenance": self.provenance.model_dump(mode="json") if self.provenance else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class A2AServer:
    """HTTP/SSE wire server for A2A v1.0.1 protocol discovery and task dispatch."""

    def __init__(
        self,
        agent_card: AgentCard,
        handler: TaskHandler | None = None,
        stream_handler: StreamHandler | None = None,
        agent: BaseAgent | None = None,
        bus: EventBus | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._agent_card = agent_card
        self._handler = handler
        self._stream_handler = stream_handler
        self._agent = agent
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
            raise HTTPException(
                status_code=500,
                detail={"error": "InvalidAgentResponseError", "message": str(e)},
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
            except Exception as exc:
                yield f'data: {{"error": "{str(exc)}"}}\n\n'

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

        if record.async_task is not None and not record.async_task.done():
            record.async_task.cancel()
        record.status = TaskStatus.CANCELED
        await record.emit_event({"event": "canceled"})
        await record.close_listeners()
        return JSONResponse(status_code=200, content={"task_id": task_id, "status": "canceled"})

    async def _execute_managed_task(self, record: ManagedTaskRecord) -> None:
        """Execute managed task in background and broadcast status events."""
        try:
            await record.emit_event({"event": "status_changed", "status": "WORKING"})
            if self._agent is not None:
                prompt = str(
                    record.input_data.get("prompt")
                    or record.input_data.get("message")
                    or json.dumps(record.input_data)
                )
                await record.emit_event({"event": "state", "state": "REASONING"})
                turn_result = await self._agent.execute_turn(prompt)
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
                    turn_failure = (
                        f" The turn also reported: {turn_result.error}" if turn_result.error else ""
                    )
                    record.error = (
                        f"Agent '{self._agent.agent_id}' produced turn "
                        f"{turn_result.turn_index} with no provenance; refusing to "
                        "attribute it and forward it over the wire (P6: absence is a "
                        f"violation, not a default).{turn_failure}"
                    )
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
                    record.error = turn_result.error or "Agent turn failed"
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
                    record.provenance = None
                    record.error = (
                        "Task handler returned a TaskResult with no provenance; refusing "
                        "to attribute it and forward it over the wire (P6: absence is a "
                        "violation, not a default)."
                    )
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
            record.error = f"InternalError: {type(exc).__name__}: {exc}"
            record.provenance = None
            await record.emit_event({"event": "error", "error": record.error})
        finally:
            await record.close_listeners()

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
