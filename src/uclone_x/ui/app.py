"""FastAPI backend application for UClone-X developer UI dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import subprocess
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Sequence,
)
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError
from rich.console import Console

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.datastructures import Headers
    from starlette.types import ASGIApp, Receive, Scope, Send
    from starlette.websockets import WebSocketClose
except ImportError as exc:
    from uclone_x.errors import MissingDependencyError

    raise MissingDependencyError(
        extra="http",
        package="fastapi",
        feature="developer UI FastAPI application",
    ) from exc

from uclone_x import __version__
from uclone_x.acp import AcpConformanceReport, conformance_summary
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    AgentState,
    PersonaDefinition,
    TurnResult,
)
from uclone_x.agent.persona_registry import PersonaRegistry, split_appended_default_prompt
from uclone_x.agent.session import (
    CORE_RECORD_SUBDIR,
    UI_TRANSCRIPT_SUBDIR,
    SessionStore,
    default_session_root,
    reap_orphaned_temp_files,
    resolve_session_path,
    verify_record_identity,
)
from uclone_x.core.agent_home import AgentHomeError
from uclone_x.core.diagnostic_report import (
    issue_url,
    render_report,
    report_title,
    search_url,
    summarise,
)
from uclone_x.core.failure_journal import (
    clear_journal,
    consent_path,
    consent_state,
    journal_path,
    read_consent,
    read_journal,
    record_failure,
    set_consent,
)
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.session_diagnostics import (
    DEFAULT_MAX_CONVERSATION_TURNS,
    count_active_turns,
)
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventBus,
    EventPriority,
    EventSource,
    EventType,
    SubscriptionClosedError,
)
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMProviderError,
    LLMProviderNotConfiguredError,
    LLMTimeoutError,
    PathTraversalError,
    SessionHistoryRehydrationError,
    SessionIdCollisionError,
    SessionMutationDuringTurnError,
    SessionStoreNotConfiguredError,
    StaleSessionWriteError,
)
from uclone_x.evaluation import (
    EvalBackendUnavailableError,
    create_eval_runner,
    default_reports_dir,
)
from uclone_x.llm import create_llm_connector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connectors.factory import saved_choice_in_effect
from uclone_x.llm.connectors.ollama import (
    delete_model,
    pull_model,
    resolve_ollama_base_url,
    resolve_ollama_model,
)
from uclone_x.llm.connectors.saved_choice import (
    SETTINGS_FILE_NAME,
    key_owner,
    same_provider,
    update_settings_file,
)
from uclone_x.llm.connectors.vllm import (
    VLLM_MODEL_ENV_VAR,
    has_configured_vllm_endpoint,
    resolve_vllm_base_url,
    resolve_vllm_model,
)
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.memory.store import CrossSessionMemory, default_cross_session_memory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.room.models import turn_refusal
from uclone_x.room.service import (
    SESSION_ID_PREFIX as _ROOM_SESSION_PREFIX,
)
from uclone_x.room.service import (
    SESSION_ID_SEPARATOR as _ROOM_SESSION_SEPARATOR,
)
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.shells.ui_process import UI_BIND_HOST_ENV_VAR
from uclone_x.skills.auditor import SkillRegistry
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.builtin.comfy_client import (
    DEFAULT_COMFYUI_BASE_URL,
    ComfyClient,
)
from uclone_x.tools.builtin.comfy_image_tool import ComfyImageGenTool
from uclone_x.tools.mcp_manager import (
    DuplicateServerError,
    MCPServerManager,
    MCPServerSpec,
    UnknownServerError,
)
from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.registry import create_default_registry
from uclone_x.ui.knowledge import knowledge_graph
from uclone_x.ui.single_flight import SingleFlight

OFFLINE_LLM_DIAGNOSTIC_MESSAGE: str = (
    "⚠️ No active LLM provider connected.\n\n"
    "Pick a model in Settings: a local one through Ollama ('ollama serve'), "
    "or an API key for OpenAI, Anthropic or Gemini.\n"
    "From a terminal, 'ucx llm status' reports what is reachable."
)


logger = logging.getLogger(__name__)
_console = Console()

SERVER_START_TIME: str = datetime.now(UTC).isoformat()

#: `POST /api/models/pull` has no ceiling of its own any more, and that is the change (#1243).
#:
#: `PULL_ROUTE_TIMEOUT_SECONDS = 900.0` used to live here. It existed because this
#: route holds a socket and `ucx llm pull` does not, so the route could not afford
#: the connector's 1800 s — but both numbers measured elapsed time, and elapsed time
#: cannot tell a slow pull from a wedged one. Splitting a ceiling that measures the
#: wrong quantity only gives two callers two different wrong answers.
#:
#: `pull_model` now consumes Ollama's NDJSON and times out on *silence between
#: progress lines* (`PULL_SILENCE_TIMEOUT_SECONDS`), with a generous backstop
#: (`PULL_TOTAL_BACKSTOP_SECONDS`) for a stream that talks forever without finishing.
#: Those answer the question this route actually had — a wedged pull holds this
#: model's single-flight entry, so it must die fast — and they answer it identically
#: for any caller, which is why the split has nothing left to do and the route now
#: takes the connector's defaults.


def get_git_commit() -> str:
    """Return short Git commit SHA or environment fallback for build verification (#871)."""
    env_commit = os.getenv("GIT_COMMIT") or os.getenv("VITE_GIT_COMMIT")
    if env_commit:
        return env_commit[:7]
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "dev"


def _translate_session_error(exc: Exception) -> HTTPException:
    """Map a Core session failure onto the HTTP status that describes it.

    Every route used to catch `PathTraversalError` for its 400 and let everything else
    reach Starlette as a **bare 500 with no body**, which is the wrong answer twice over
    for the one that matters: a mid-turn reset is a *conflict* — the caller can retry
    when the turn finishes — and a 500 says the server is broken and gives the frontend
    nothing to say. The refusal was implemented in the Core and then not translated at
    the boundary, so from a client's point of view it did not exist.
    """
    if isinstance(exc, (LLMProviderNotConfiguredError, LLMCredentialsNotConfiguredError)):
        # Not a session fault. Reaching this translator at all is an artefact of the
        # broad handlers that wrap session loading; without a branch it fell through
        # to a 500 labelled "Session operation failed", naming the wrong subsystem for
        # a condition the caller can act on (P6).
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AgentHomeError):
        # 400, and not the "Session operation failed" 500 it fell through to: the agent
        # id is supplied by the client, the rule it broke is stated in the message, and a
        # caller that can fix the request should not be told the server is broken. It is
        # also not a session fault — naming that subsystem is the mis-attribution P6
        # forbids.
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, PathTraversalError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, SessionIdCollisionError):
        # 409, not 400: the request is well-formed and the id is legal (#256 states the P3
        # guard is not implicated). What makes it unserviceable is the *current state of
        # the store* — another session already occupies the one filename this filesystem
        # gives both ids — which is exactly what 409 Conflict describes. A 400 would tell
        # the frontend the id was malformed, and it is not.
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, SessionMutationDuringTurnError):
        # 409: the request is well-formed and legal, just not right now.
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, SessionHistoryRehydrationError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, SessionStoreNotConfiguredError):
        return HTTPException(status_code=500, detail=str(exc))
    if isinstance(exc, OSError):
        return HTTPException(
            status_code=500,
            detail=f"Session storage is unwritable or unreadable: {exc}",
        )
    return HTTPException(status_code=500, detail=f"Session operation failed: {exc}")


def _is_offline_llm_error(error: Exception | str | None) -> bool:
    """Check if an error string or exception indicates an offline, unreachable, or misconfigured LLM provider."""
    if error is None:
        return False
    err_str = str(error).lower()
    # If the provider responded with 404 or model not found, the server is ONLINE and responding.
    # Disguising a missing model as an offline provider creates false diagnostics (P6, P8).
    if "not found" in err_str or "404" in err_str:
        return False
    if isinstance(
        error,
        (LLMProviderError, LLMProviderNotConfiguredError, LLMCredentialsNotConfiguredError),
    ):
        # An unconfigured provider is an LLM condition, not a transport one. It gates the
        # same diagnostic, which is the only text naming `ollama serve` and
        # `ucx llm status` — without this the refusal is less actionable than the
        # connection error it replaced.
        return True
    keywords = (
        "failed to connect",
        "connection refused",
        "connecterror",
        "connecttimeout",
        "request error",
        "unreachable",
        "no active llm provider",
        "all connection attempts failed",
        "unsupported llm provider",
        "ollama provider returned",
        "errno 61",
        "errno 111",
        "llmprovidererror",
        "model 'default' not found",
    )
    return any(kw in err_str for kw in keywords)


def vllm_request_headers(api_key: str | None = None) -> dict[str, str]:
    """`Authorization` only when a key exists, because `vllm serve --api-key` is optional.

    An empty bearer is not the same as no header: a server started without `--api-key`
    accepts the request either way, but one behind a proxy that reads the header rejects
    `Bearer ` with a 401 that describes a credential nobody configured (#385).
    """
    key = (api_key or os.getenv("VLLM_API_KEY") or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def vllm_model_ids(payload: object) -> list[str]:
    """The `id`s in an OpenAI-compatible `/v1/models` listing, or an empty list.

    vLLM answers `{"object": "list", "data": [{"id": "<the --model argument>", ...}]}`, and
    that one entry is the model the server was started with — which is why this listing is
    worth showing at all: for vLLM it is not a catalogue of what *could* be run but a
    statement of what *is* running. Parsed defensively for the reason the Ollama branch
    below is: this is another process's JSON, and a gateway in front of it may answer
    something else entirely.
    """
    if not isinstance(payload, dict):
        return []
    raw = cast(dict[str, Any], payload).get("data")
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in cast(list[object], raw):
        if isinstance(item, dict):
            value = cast(dict[str, Any], item).get("id")
            if value is not None:
                ids.append(str(value))
    return ids


async def fetch_available_models(
    provider: str,
    base_url: str | None = None,
    timeout: float = 3.0,
) -> list[str]:
    """Enumerate installed or supported models for the provider (P0/Recognition over Recall)."""
    clean_provider = provider.strip().lower()
    if clean_provider == "ollama":
        ollama_url = (base_url or "").strip() or resolve_ollama_base_url()
        try:
            async with httpx.AsyncClient(timeout=timeout) as http_c:
                resp = await http_c.get(f"{ollama_url.rstrip('/')}/api/tags")
                if resp.status_code == 200:
                    data_obj: object = resp.json()
                    models: list[str] = []
                    if isinstance(data_obj, dict):
                        raw_models = cast(dict[str, Any], data_obj).get("models")
                        if isinstance(raw_models, list):
                            for item in cast(list[object], raw_models):
                                if isinstance(item, dict):
                                    name_val = cast(dict[str, Any], item).get("name")
                                    if name_val is not None:
                                        models.append(str(name_val))
                    return models
        except Exception as exc:
            logger.debug("Failed to query Ollama tags from %s: %s", ollama_url, exc)
            return []
    elif clean_provider == "vllm":
        if not has_configured_vllm_endpoint(base_url):
            # An unconfigured endpoint is not an empty inventory. Probing vLLM's documented
            # default port to fill the dropdown would be this module guessing where the
            # operator's server is, and reporting a refused connection as "no models" (P6).
            return []
        vllm_url = resolve_vllm_base_url(base_url)
        try:
            async with httpx.AsyncClient(timeout=timeout) as http_c:
                resp = await http_c.get(
                    f"{vllm_url.rstrip('/')}/models", headers=vllm_request_headers()
                )
                if resp.status_code == 200:
                    return vllm_model_ids(resp.json())
        except Exception as exc:
            logger.debug("Failed to query vLLM models from %s: %s", vllm_url, exc)
            return []
    elif clean_provider == "openai":
        return ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"]
    elif clean_provider == "anthropic":
        return ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]
    elif clean_provider in ("gemini", "google"):
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
    elif clean_provider == "mock":
        return ["mock-gpt-4o", "mock-llm"]
    return []


# Global bus and tracer instances for dashboard event monitoring
_ui_bus: EventBus | None = None
_ui_tracer: TelemetryTracer | None = None
_ui_session_mgr: AgentSessionManager | None = None
_sequence_counter: int = 0
_ui_shutdown_event: asyncio.Event | None = None

# Interaction turns held in a session's *active* context before it counts as saturated.
# Read against the live message sequence, never against the lifetime `turn_counter`:
# compaction discards context but deliberately keeps the counter (P5), so a signal keyed
# to the counter can never clear and the remediation button it recommends looks broken.
SATURATION_TURNS_THRESHOLD = DEFAULT_MAX_CONVERSATION_TURNS


async def _cancel_and_drain(tasks: Iterable[asyncio.Task[Any]]) -> None:
    """Cancel tasks and await them, so nothing finishes with an unretrieved exception.

    The awaiting is what matters here, not the cancelling. Cancelling a task whose
    awaited future has already resolved does **not** preserve its pending exception:
    measured on CPython 3.12.13, `_must_cancel` raises `CancelledError` and replaces it,
    on a bare `asyncio.Queue` and on a real `EventSubscription` alike.

    What leaked is the waiter that was never cancelled at all. When a subscription closes
    in the same window the shutdown event fires, the close sentinel wakes
    `EventSubscription.get()` and it completes with `SubscriptionClosedError`, so
    `asyncio.wait` returns it in `done` rather than `pending` -- untouched by a
    `for t in pending: t.cancel()` and unread by the `break` that followed. asyncio
    reports that at finalization as "Task exception was never retrieved" (#872).

    So a `done` task is deliberately not cancelled below, only gathered: reading it is
    the entire remedy.
    """
    collected = list(tasks)
    for task in collected:
        if not task.done():
            task.cancel()
    if collected:
        # `return_exceptions=True` so the drain itself never raises, and so every task's
        # result is *read* -- which is the whole point.
        await asyncio.gather(*collected, return_exceptions=True)


def _read_waiter_result(task: asyncio.Task[Any]) -> None:
    """Retrieve a finished waiter's outcome so asyncio does not report it as unread."""
    if not task.cancelled():
        task.exception()


def _discard_waiter(task: asyncio.Task[Any]) -> None:
    """Read or cancel one waiter **without awaiting**, for an exit that cannot await.

    `_cancel_and_drain` above is the remedy wherever the stream loop still owns its
    coroutine. It is unusable on the one exit where the loop does not: a `CancelledError`
    thrown into the generator at the `await asyncio.wait` itself -- an ordinary client
    closing the SSE stream. The drain call sits *below* that `await`, so it never runs,
    both waiters are left pending, and the `finally`'s `sub.close()` then wakes
    `sub.get()` with `SubscriptionClosedError` that nobody holds (#1038).

    A still-pending waiter is cancelled *and* given a done-callback, because cancelling
    does not settle the race: `close()` puts the sentinel on the queue in the same window,
    and whichever lands first, the callback reads the result. A waiter already finished is
    read here and now.
    """
    if task.done():
        _read_waiter_result(task)
        return
    task.cancel()
    task.add_done_callback(_read_waiter_result)


def get_ui_shutdown_event() -> asyncio.Event:
    """Retrieve or lazily initialize the shared UI shutdown event (#613)."""
    global _ui_shutdown_event
    if _ui_shutdown_event is None:
        _ui_shutdown_event = asyncio.Event()
    return _ui_shutdown_event


def trigger_ui_shutdown() -> None:
    """Signal all active UI background streams and generators to terminate cleanly (#613)."""
    global _ui_shutdown_event
    if _ui_shutdown_event is not None:
        _ui_shutdown_event.set()


def reset_ui_shutdown_event() -> None:
    """Reset the shared UI shutdown event (#613)."""
    global _ui_shutdown_event
    _ui_shutdown_event = None


def get_ui_event_bus() -> EventBus:
    """Retrieve or lazily initialize the shared UI event bus."""
    global _ui_bus
    if _ui_bus is None:
        _ui_bus = EventBus()
    return _ui_bus


def get_ui_tracer() -> TelemetryTracer:
    """Retrieve or lazily initialize the shared UI telemetry tracer.

    **The UI deliberately wires no span exporter, and this is the record of that so a
    reader does not infer a bug from its absence (#187).**

    Spans raised under the UI -- `agent.run`, and P6's `failover.event` since #179 --
    have no exporter, no `stream_spans` subscriber and no endpoint that returns them.
    That was worth checking rather than assuming, and it is a deliberate shape rather
    than an omission: the UI is a local developer dashboard with no collector to ship to,
    and the information a span carries about a failover **already reaches this UI in
    band** -- `PROVIDER_FAILOVER` events go to the browser over the existing SSE stream
    with the `span_id` in their payload, which is P6's decision-plane requirement
    (check 4). The span exists for a human reading a trace in a collector later (check
    5); locally there is no such trace, so exporting would ship to nothing.

    What was *not* deliberate is that the buffer holding them grew forever. This tracer
    is a module-global living for the life of the server process, and nothing drained it,
    so a long-running UI accumulated every span it had ever completed. `TelemetryTracer`
    is now bounded and counts what it evicts (`buffer_evicted_span_count`), so "local" is
    sustainable (conditioned on there being no `stream_spans` subscriber) and the eviction
    is observable instead of silent.

    If a collector is ever wanted here, wire an exporter and drain with
    `discard_exported` / `drop_unexported`; do not reach for `clear()`, which is the call
    that lost the CLI's spans on a failed export.
    """
    global _ui_tracer
    if _ui_tracer is None:
        _ui_tracer = TelemetryTracer()
    return _ui_tracer


#: The bind address assumed when nobody says: the `ucx ui --host` default. Unknown means
#: loopback, so a launcher that forgets to say is guarded rather than exposed.
DEFAULT_UI_BIND_HOST = "127.0.0.1"

#: Hosts a page can only be served from by something already running on
#: this machine. A hostile site has a public origin; it cannot forge one of
#: these, and the dashboard's own dev server is one of them.
# `0.0.0.0` is deliberately absent: it is a bind address, not a loopback
# host, and admitting it widens the set for nothing.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _is_loopback_bind(bind_host: str) -> bool:
    """Whether a server bound to `bind_host` can be reached only from this machine.

    `0.0.0.0` and `::` are not: they listen on every interface, which is how a user
    deliberately exposes the dashboard.
    """
    name = bind_host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _host_header_hostname(host: str) -> str | None:
    """The name in a `Host` header, lower-cased and without its port; None if unparseable."""
    try:
        return urlparse("//" + host).hostname
    except ValueError:  # e.g. an unclosed `[::1`
        return None


class LoopbackHostGuard:
    """Refuse every request not addressed to this machine by a loopback name (#1413).

    Installed on the whole app while it is bound to a loopback address, so no route, static
    file or stream can be added without it. It is the DNS-rebinding defence: a page on
    `http://attacker.example:5180` whose name has been re-pointed at 127.0.0.1 reaches this
    server with `Origin` and `Host` both naming `attacker.example`, which the origin check
    admits. The browser will not let that page send `Host: localhost`, so the `Host` name is
    what tells it apart. Without this, such a page could read conversations and post a chat
    turn that runs tools in the user's workspace.

    A server bound to a non-loopback address was exposed on purpose (`ucx ui --host
    0.0.0.0`) and is reached by names this cannot know, so it is not installed there.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return
        host = Headers(scope=scope).get("host", "")
        if _host_header_hostname(host) in _LOOPBACK_HOSTS:
            await self._app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await WebSocketClose(code=1008)(scope, receive, send)
            return
        port = _host_header_port(host, scope)
        message = (
            "This dashboard only answers when it is opened from this computer, and the "
            "address used here is not one it recognises. Open it at "
            f"http://localhost:{port} or http://127.0.0.1:{port} instead."
        )
        response: Response
        if str(scope.get("path", "")).startswith("/api"):
            response = JSONResponse({"detail": message}, status_code=403)
        else:
            response = PlainTextResponse(message, status_code=403)
        await response(scope, receive, send)


def _host_header_port(host: str, scope: Scope) -> int:
    """The port to name in the remedy: the one the request used, else the one served."""
    try:
        port = urlparse("//" + host).port
    except ValueError:
        port = None
    if port is not None:
        return port
    server = scope.get("server")
    if isinstance(server, tuple | list) and len(server) >= 2 and isinstance(server[1], int):  # pyright: ignore[reportUnknownArgumentType]
        return server[1]
    return 80


#: `sess_room` -- imported rather than spelled, so the filter above cannot drift from
#: the derivation in `uclone_x.room.service.participant_session_id`.
ROOM_SESSION_PREFIX = f"{_ROOM_SESSION_PREFIX}{_ROOM_SESSION_SEPARATOR}"


_CLIENT_TURN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _client_turn_id(req: dict[str, Any]) -> str | None:
    """The `client_turn_id` a head sent a turn with, or None when it sent none.

    The id is kept on the saved prompt (#1000), so it is bounded before anything runs: 1 to
    128 ASCII letters, digits, `-` or `_`, which a UUID or the dashboard's own ids satisfy.
    A field that is present and outside that -- empty, not a string, null, too long, other
    characters -- is refused rather than dropped, so a head never believes it sent an id the
    server did not keep (#1007). Leaving the field out sends the turn without one.

    Raises:
        HTTPException: 422, naming the field, the rule and the remedy.
    """
    if "client_turn_id" not in req:
        return None
    turn_id = req["client_turn_id"]
    if not isinstance(turn_id, str) or not _CLIENT_TURN_ID.fullmatch(turn_id):
        raise HTTPException(
            status_code=422,
            detail=(
                "client_turn_id must be a string of 1 to 128 ASCII letters, digits, '-' or "
                "'_'. Send an id of that shape, or leave the field out to send the turn "
                "without one."
            ),
        )
    return turn_id


#: Folders outside the workspace that clones may read, separated by `os.pathsep`; read
#: before the Settings list.
READ_ROOTS_ENV_VAR = "UCLONE_READ_ROOTS"


def _holds_folder(folder: Path, inner: Path) -> bool:
    """Whether `folder` is `inner` or one of its ancestors, compared by inode.

    Not by path: on a case-insensitive volume `/users/me` and `/Users/me` are one folder
    that `resolve()` leaves spelled two ways, and so are a firmlink and its target.
    """
    try:
        folder_stat = folder.stat()
    except OSError:
        return False
    resolved = inner.resolve()
    for candidate in (resolved, *resolved.parents):
        try:
            if os.path.samestat(candidate.stat(), folder_stat):
                return True
        except OSError:
            continue
    return False


def _read_root_problem(entry: str, storage_dir: Path) -> str | None:
    """Why `entry` cannot be a read root, or None when it can.

    A folder holding the storage directory is refused: `settings.json` there carries the
    model API key, and a clone that could read it could repeat it into a chat.
    """
    folder = Path(entry).expanduser()
    if not folder.is_absolute():
        return f"'{entry}' is not a full path; start it with / or ~"
    if not folder.is_dir():
        return f"'{entry}' is not a folder on this computer"
    if _holds_folder(folder, storage_dir):
        return f"'{entry}' contains uclone's own settings folder ({storage_dir}), which holds the API key"
    return None


def _validate_read_roots(
    entries: object, storage_dir: Path, already_saved: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """The Settings read-folder list, refused whole if any entry is not a usable folder.

    Refused rather than filtered: a folder the user typed and we silently dropped would
    leave them believing a clone can read it (P6). One exception: an entry that was already
    saved and has since disappeared is kept, so a deleted folder does not block every
    later edit of the list; `get_settings` reports it as missing.
    """
    if not isinstance(entries, list):
        raise ValueError("read_roots must be a list of folder paths")
    cleaned: list[str] = []
    for raw in cast(list[object], entries):
        if not isinstance(raw, str):
            raise ValueError(f"read_roots entries must be text, got {raw!r}")
        entry = raw.strip()
        if not entry:
            continue
        problem = _read_root_problem(entry, storage_dir)
        if problem is not None and not (
            entry in already_saved and not Path(entry).expanduser().exists()
        ):
            raise ValueError(problem)
        if entry not in cleaned:
            cleaned.append(entry)
    return tuple(cleaned)


def _persona_payload(registry: PersonaRegistry, persona: PersonaDefinition) -> dict[str, Any]:
    """One persona as the dashboard reads it, and as its editor sends it back.

    `system_prompt` is the persona's own text with the appended default prompt split off
    and reported as `append_default_prompt`, so an editor that saves what it was shown
    writes the flag back rather than a frozen copy of the default prompt.
    """
    own_prompt, appended = split_appended_default_prompt(persona.system_prompt)
    builtin = registry.is_builtin(persona.name)
    return {
        "name": persona.name,
        "role": persona.role,
        "description": persona.description,
        "system_prompt": own_prompt,
        "append_default_prompt": appended,
        "allowed_tools": list(persona.allowed_tools),
        "model_name": persona.llm_config.model_name,
        "model_tier": str(persona.llm_config.model_tier),
        "temperature": persona.llm_config.temperature,
        "max_tokens": persona.llm_config.max_tokens,
        "enable_write_tools": persona.enable_write_tools,
        "enable_subagent_tools": persona.enable_subagent_tools,
        "builtin": builtin,
        "overrides_builtin": not builtin and registry.has_builtin(persona.name),
    }


#: The picture formats a clone's avatar may be in, looked for in this order. Images only:
#: nothing here draws a face from the clone's name, so a clone with no picture gets the one
#: default the head ships rather than a generated one that would differ between heads.
_AVATAR_FORMATS: tuple[tuple[str, str], ...] = (
    (".png", "image/png"),
    (".webp", "image/webp"),
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".gif", "image/gif"),
)


def _persona_avatar(registry: PersonaRegistry, name: str) -> tuple[Path, str] | None:
    """The picture file beside one persona's own definition, or `None` when it has none.

    The path is built from `source_of`, never from the name in the request: a name that no
    loaded persona carries returns `None` before any path is composed, so `../` in a URL
    names nothing rather than reaching for a file outside the personas directory.
    """
    source = registry.source_of(name)
    if source is None:
        return None
    for suffix, mime in _AVATAR_FORMATS:
        candidate = source.with_suffix(suffix)
        if candidate.is_file():
            return candidate, mime
    return None


def _required_agent_id(raw: object) -> str:
    """Read the agent a request names, refusing the request when it names none.

    These endpoints used to fall back to a built-in persona name when the request
    carried no `agent_id`. On an install whose agents are its own -- which is every
    install, now that a persona is a file and an agent is a directory -- that
    substituted a name the caller never asked for: the turn ran, answered 200, and was
    recorded against an agent the user had not addressed. A missing name is a missing
    name (P6), and the name of a built-in is not a fact about this install.

    Raises:
        HTTPException: 400, saying the field is required.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing required 'agent_id'. Name the agent this request is for; "
                "there is no default agent. `GET /api/agents` lists the ones this "
                "install has."
            ),
        )
    return raw.strip()


def _refuse_a_reserved_session_id(session_id: str | None) -> None:
    """Refuse a caller-supplied session id that belongs to a room's agent.

    `participant_session_id` derives `sess_room__{room}__{agent}` so that no two
    participants share a `SessionState`. A chat endpoint that accepted any id could park
    an ordinary conversation on exactly that name -- two writers on one record, which is
    the failure the derivation exists to prevent -- and `/api/sessions` now filters the
    prefix, so it would not even be visible. The prefix is reserved at the door.

    Raises:
        HTTPException: 400, naming the prefix.
    """
    if session_id and session_id.startswith(ROOM_SESSION_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Session ids beginning {ROOM_SESSION_PREFIX!r} are reserved for the "
                f"agents seated in a conversation, one session each. Pick another id, or "
                f"use the conversation's own endpoints."
            ),
        )


def _next_sequence_number() -> int:
    """Increment and return global SSE sequence counter."""
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


def _turn_token_figures(
    usages: Sequence[TokenUsage] | None,
) -> tuple[int | None, int | None, str | None]:
    """The tokens a chat turn booked, as `(total, input, count source)`, from the budget ledger.

    The chat response used to report `100 + len(reply) // 4` and `100 + len(message) // 4`:
    figures nothing measured, shown as if something had (#939). The ledger holds what the
    budget actually charged for each step of the turn, provider counts and labelled
    estimates alike, so the figures are read from there. `usages` is what
    `TokenBudgetManager.collect_turn_usage` collected for this turn; `None` means the turn
    failed before collection began. It used to be a slice of the session's ledger from an
    index taken before the agent's turn lock, which also held any overlapping turn's steps
    on that session id (#982).

    The label follows the design document's §6.7 rule: a figure the provider did not count
    is an estimate, and a turn with any estimated step is estimated as a whole. A turn that
    booked nothing has no figure, not a zero.
    """
    if not usages:
        return None, None, None
    total = sum(u.input_tokens + u.output_tokens for u in usages)
    prompt = sum(u.input_tokens for u in usages)
    estimated = any(u.count_source is TokenCountSource.ESTIMATE for u in usages)
    source = TokenCountSource.ESTIMATE if estimated else TokenCountSource.PROVIDER
    return total, prompt, source.value


def _serialized_tool_calls(tool_calls: Sequence[ToolCallRequest]) -> list[dict[str, Any]]:
    """Tool calls as plain JSON-serialisable dicts, for a transcript that reaches disk.

    One function rather than two literals, because the two literals are what went wrong.
    `ToolCallRequest.arguments` is `ImmutableJsonMapping`, frozen *recursively*, so every
    nested object inside it is a `MappingProxyType` too; `dict(...)` copies only the top
    level and leaves the nested proxies for `json.dumps` to choke on (#665). Keeping one
    spelling of the unwrap means the next transcript writer cannot get half of it right,
    which is precisely what happened — the correct `unwrap_immutable` sat four lines from
    the broken `dict()`.

    **Both call sites failed loudly**, and an earlier version of this docstring said
    otherwise; the #673 review reverted the line and measured it (#665, #673):

    * history synthesised from the Core store — the `TypeError` propagates straight out
      of `truncate_session_history`, which sits in no `try`;
    * the tool calls of a completed turn — the same list goes into the response body as
      well as into `save_session_record`, so FastAPI's response serialiser raises
      `PydanticSerializationError: Unable to serialize unknown type: <class
      'mappingproxy'>` before the endpoint's `except Exception: logger.warning(...)`
      around the save can swallow anything.

    The `except Exception` around `save_session_record` is real and would have hidden a
    save-only failure; it is not what happened here. The one genuinely silent site in
    this family is elsewhere — the `PRE_TOOL_USE` script-hook path in `agent/base.py`,
    where `HookRunner` converts the `TypeError` into a fail-closed `BLOCK` with only a
    `WARNING`.
    """
    return [
        {
            "id": tc.id,
            "name": tc.name,
            "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
        }
        for tc in tool_calls
    ]


TRANSCRIPT_FAILURE_ROLE = "failure"
"""The transcript role of a turn that failed (#969).

A failure is not something the agent said. The chat head saved one as `role: "assistant"`
with `Error: ...` content, so the saved conversation claimed a reply and the only mark on it
was that wording. A record with this role keeps the text the page showed in `content` and
the turn's error in `error`; it is shown to the user and never rebuilt into model context.
"""


class ChatTurnOutcome(StrEnum):
    """How a chat turn ended, stated by the backend for a head to name (#1007 item 4).

    A head chipped a turn from `provenance.degraded`, which on a chat row is true for a
    failure, an offline turn, a cancelled one and an answer that was not persisted as well
    as for a substituted model -- so every one of them read "degraded", and a head wanting
    to tell them apart had only the row's text. This is the structured status it reads
    instead, carried on the `/api/turn` result and on every saved agent row.
    """

    #: The turn reached an answer from the model that was asked.
    COMPLETED = "completed"
    #: The turn reached an answer, from a model other than the one requested -- the Core's
    #: `Provenance.degraded` (`served_by != requested`) and nothing broader.
    DEGRADED = "degraded"
    #: The turn ended in an error; `error` says why, and `refusal` whether a retry is refused.
    FAILED = "failed"
    #: The turn was stopped before it finished.
    INTERRUPTED = "interrupted"


def _answered_outcome(turn_result: TurnResult) -> ChatTurnOutcome:
    """The outcome of a turn that returned a result rather than raising."""
    # `is_completed` is read too: a result without `error` that did not complete has no
    # answer, and the status line already reports it as an error.
    if turn_result.error is not None or not turn_result.is_completed:
        return ChatTurnOutcome.FAILED
    if turn_result.provenance is not None and turn_result.provenance.degraded:
        return ChatTurnOutcome.DEGRADED
    return ChatTurnOutcome.COMPLETED


LEGACY_FAILURE_PREFIX = "Error: "
"""How a failed turn's text began in a transcript saved before `TRANSCRIPT_FAILURE_ROLE`."""

TRANSCRIPT_CANCELLED_ROLE = "cancelled"
"""The transcript role of a turn Stop cancelled before it finished (#1031).

A cancellation is not a failure and not a reply: nothing went wrong and nothing was said.
Before this role existed a cancelled turn wrote **no row at all**, while the Core kept the
prompt and whatever the turn had already done -- so the page showed no sign a tool had run,
and the orphaned prompt stalled the truncation mapping, which then cut a turn the page was
still showing (#1031, measured end to end in PR #1033's probe).

The row carries the calls the Core records for the turn in `tool_calls`, so a stop after a
tool step is visible as one. It never re-enters model context, exactly as a failure row does
not: `reconstruct_history` drops both, through `_records_a_turn_not_spoken`.
"""

CANCELLED_TURN_TEXT = "⏹️ [Generation Stopped by User]"
"""What the saved row says, in the words the page already shows when Stop is pressed.

`App.tsx` writes this sentence into the bubble the moment the stream reports `cancelled`, so
a conversation reopened later reads as the user left it rather than in a second vocabulary
for the same event.
"""


def _is_failure_entry(role: str, entry: dict[str, Any]) -> bool:
    """Whether a persisted transcript entry records a failed turn rather than a message.

    A record written since #969 says so in its role. One written before says so only by
    the shape the chat head gave it: an assistant row whose text starts with
    `LEGACY_FAILURE_PREFIX` under degraded provenance. That wording is read only here, for
    transcripts that carry nothing better, and only to keep the row out of model context;
    the page is still served the row as it was saved.
    """
    if role == TRANSCRIPT_FAILURE_ROLE:
        return True
    content = entry.get("content")
    provenance = entry.get("provenance")
    return (
        role == MessageRole.ASSISTANT.value
        and isinstance(content, str)
        and content.startswith(LEGACY_FAILURE_PREFIX)
        and isinstance(provenance, dict)
        and cast(dict[str, Any], provenance).get("degraded") is True
    )


def _cancelled_turn_rows(
    *,
    agent_id: str,
    message: str,
    client_turn_id: str | None,
    model: str | None,
    latency_ms: float,
    turn_index: int,
    turn_messages: Sequence[ChatMessage],
    persist_error: str | None,
    persist_error_type: str | None,
) -> list[dict[str, Any]]:
    """The two transcript rows a cancelled turn owes the page (#1031).

    The prompt row, then a `TRANSCRIPT_CANCELLED_ROLE` row for the turn itself -- the same
    two-rows-per-turn shape `/api/turn` writes for every other outcome, which is what makes
    the truncation mapping work across a cancelled turn without teaching the index walk
    anything about turns (#1024's three blockers all came from doing that).

    **The prompt row is written even when the Core did not keep the prompt.** A turn
    cancelled before `BaseAgent` appended it, and a retry whose prompt repeats the
    unanswered one already in the Core (#1022), both leave a row the Core has no message
    for -- which is exactly a failure row's shape, and the walk already handles it: the row
    matches nothing and advances nothing. The page showed the prompt, so the record does.

    `tool_calls` comes from the Core messages this turn appended, which is the only record
    of it that exists: a cancelled turn produces no `TurnResult`. **No `tool_executions`
    key is written**, because the Core states which calls were made and not how each one
    ended, and a status nothing measured is the substituted default P6 forbids. Whether the
    partial `TOOL` messages themselves should be *shown* is #1024's open question, not
    decided here: they are treated exactly as a failed turn's are, and this change moves
    nothing. (Precisely: they are in the Core, no row of the live transcript renders them,
    and no row of the live transcript renders them -- identical for a failed turn.)
    """
    now = datetime.now(UTC).isoformat()
    prompt_row: dict[str, Any] = {
        "id": f"user-{uuid.uuid4().hex[:12]}",
        "sender": "user",
        "role": "user",
        "content": message,
        "timestamp": now,
        "model": model,
    }
    if client_turn_id is not None:
        prompt_row["client_turn_id"] = client_turn_id
    cancelled_row: dict[str, Any] = {
        "id": f"agent-{uuid.uuid4().hex[:12]}",
        "sender": "agent",
        "role": TRANSCRIPT_CANCELLED_ROLE,
        "outcome": ChatTurnOutcome.INTERRUPTED,
        "content": CANCELLED_TURN_TEXT,
        "timestamp": now,
        "agent_id": agent_id,
        "model": model,
        "latency_ms": latency_ms,
        "turn_count": turn_index,
        "tool_calls": [
            call for msg in turn_messages for call in _serialized_tool_calls(msg.tool_calls or ())
        ],
        "provenance": {
            "component": "uclone_x.ui.app",
            "producer": agent_id,
            "content_hash": hashlib.sha256(
                f"{agent_id}:{message}:{CANCELLED_TURN_TEXT}".encode()
            ).hexdigest(),
            # The turn produced no answer, which is what the page already records for a stop
            # it watched (`App.tsx`), and `path` says why rather than leaving it to be
            # inferred. It is not a provider failure: no failover was attempted.
            "degraded": True,
            "path": "CANCELLED",
            "served_by": None,
        },
        "durability": {
            "persisted": persist_error is None,
            "error": persist_error,
            "error_type": persist_error_type,
            "stale_conflict": persist_error_type == "StaleSessionWriteError",
        },
    }
    return [prompt_row, cancelled_row]


def _records_a_turn_not_spoken(role: str, entry: dict[str, Any]) -> bool:
    """Whether this row records how a turn ended rather than something that was said (#1031).

    A failed turn and a cancelled one are the same kind of row to the two callers that ask:
    `reconstruct_history` must keep both out of rebuilt model context (neither is anything
    the agent said), and `_with_turn_outcome_rows_restored` must carry both back through a
    compaction the Core cannot re-derive them from. Asking one question in one place is what
    stops the two answers drifting -- the defect #872 and #1022 each found a copy of.

    It is deliberately **not** asked where the Core is cut. That count is derived from the
    Core's own messages (`_core_index_for_transcript`), because a predicate that reads a
    row's wording decides a deletion by guess (#1024 review).
    """
    return role == TRANSCRIPT_CANCELLED_ROLE or _is_failure_entry(role, entry)


def _is_presentable_role(role: str, compaction_ledger: bool) -> bool:
    """Whether a message with this role belongs in the transcript the user reads.

    The rule is stated over the two fields it actually depends on because it is needed on
    both sides of the transcript boundary, where the message has two different types:
    `_is_presentable` asks it of a typed `ChatMessage` on the way out, and
    `reconstruct_history` asks it of a persisted transcript dict on the way back in.
    Those were independent copies that happened to agree — the same shape of defect the
    single predicate was extracted to prevent, one layer down (#872).

    A system message is an anchor, not conversation, *unless* it is a compaction ledger.
    """
    return role != MessageRole.SYSTEM.value or compaction_ledger


def _is_presentable(msg: ChatMessage) -> bool:
    """Whether a Core message belongs in the transcript the user reads.

    One predicate, because three callers need the same answer and a second copy of it is
    how they come to disagree: `_core_messages_to_transcript` renders these messages,
    `_truncate_core_messages` cuts the Core at an index counted over it, and
    `reconstruct_history` reaches the same rule through `_is_presentable_role`.
    """
    return _is_presentable_role(msg.role.value, bool(msg.compaction_ledger))


def _truncate_core_messages(messages: Sequence[ChatMessage], index: int) -> list[ChatMessage]:
    """Cut the Core's message sequence at an index over its presentable messages (#872).

    The index arrives from the UI as an offset into the transcript the user reads, and
    `_is_presentable` is what defines that list. Partitioning the Core on
    `role != SYSTEM` instead counted a different one: from the first compaction onward
    the transcript leads with a ledger that partition drops, so the same number cut the
    two lists in different places and a turn the user had just deleted stayed in the
    Core for the next turn to answer against.

    The offset is mapped through `_core_index_for_transcript` before it arrives here: the
    transcript holds rows the Core has no message for, and a tool-using turn is the reverse,
    one row over several messages (#1023, FR-13.7).

    Messages that are not presentable are anchors (the system prompt), not conversation:
    they are retained whole, ahead of the surviving prefix, exactly as before.
    """
    anchored = [m for m in messages if not _is_presentable(m)]
    presentable = [m for m in messages if _is_presentable(m)]
    return [*anchored, *presentable[:index]]


def _core_messages_to_transcript(
    messages: Sequence[ChatMessage],
    agent_id: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    """The rendered transcript alone, for callers with no use for the source messages."""
    return [entry for _, entry in _core_messages_to_transcript_pairs(messages, agent_id, timestamp)]


def _core_messages_to_transcript_pairs(
    messages: Sequence[ChatMessage],
    agent_id: str,
    timestamp: str,
) -> list[tuple[ChatMessage, dict[str, Any]]]:
    """Render the Core's message sequence as the UI transcript derived from it.

    Each rendered entry is returned beside the Core message it was derived from, so a
    caller that needs both reads them off one pass instead of re-deriving the filtered
    subsequence and zipping the two together. That zip was correct — both sides filtered
    through `_is_presentable`, so `strict=True` could not desync — but it held only
    while two expressions stayed in lockstep, and the failure mode if one ever drifted
    was an uncaught `ValueError` out of the compaction path: a presentation bug
    answering **500**. Pairing them here makes the correspondence structural.

    The Core session is the single source of truth for what the conversation *is*; the
    transcript is a presentation view over it, and this is the one place that derivation
    is written. Anchored system prompts are not conversation and are omitted, but a
    **compaction ledger is** — it is the only surviving record of the turns compaction
    discarded, and dropping it left the user unable to see what had happened to their
    conversation (#872, P6). It is rendered as its own `system` sender rather than as an
    agent turn: attribution is never inferred (FR-13.4).
    """
    transcript: list[tuple[ChatMessage, dict[str, Any]]] = []
    for idx, msg in enumerate(messages):
        if not _is_presentable(msg):
            continue
        role_str = msg.role.value
        if role_str == "system":
            transcript.append(
                (
                    msg,
                    {
                        "id": f"ledger-{idx}",
                        "sender": "system",
                        "role": "system",
                        "content": msg.content,
                        "timestamp": timestamp,
                        "agent_id": agent_id,
                        "compaction_ledger": True,
                    },
                )
            )
            continue
        sender = "user" if role_str == "user" else "agent"
        msg_dict: dict[str, Any] = {
            "id": f"{sender}-{idx}",
            "sender": sender,
            "role": role_str,
            "content": msg.content,
            "timestamp": timestamp,
            "agent_id": agent_id,
        }
        if msg.name is not None:
            msg_dict["name"] = msg.name
        if msg.tool_call_id is not None:
            msg_dict["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            msg_dict["tool_calls"] = _serialized_tool_calls(msg.tool_calls)
        if sender == "agent":
            msg_dict["provenance"] = {
                "component": "uclone_x.llm.orchestrator",
                "producer": agent_id,
                "content_hash": hashlib.sha256(
                    f"{agent_id}:{msg.content or ''}".encode()
                ).hexdigest(),
                "degraded": False,
                "path": "primary",
                "served_by": None,
            }
        transcript.append((msg, msg_dict))
    return transcript


def _transcript_role(entry: dict[str, Any]) -> str:
    """The role a persisted transcript entry states, falling back to its sender.

    One spelling of the fallback, because three callers need it: the matching key below,
    and the two places #1023 asks whether an entry is a failed turn. An entry saved by an
    older head carries `sender` alone, and the answer must not depend on which caller asks.
    """
    role = str(entry.get("role") or "").strip().lower()
    if role:
        return role
    sender = str(entry.get("sender") or "").strip().lower()
    return "user" if sender == "user" else "assistant" if sender == "agent" else sender


def _transcript_key(entry: dict[str, Any]) -> tuple[str, str | None]:
    """Identity of a transcript entry for matching it against a Core message.

    Role and content, because those are the only two fields both sides agree on: ids and
    timestamps exist on the transcript alone, and the Core message is the authority on
    everything else.
    """
    raw_content = entry.get("content")
    return _transcript_role(entry), (None if raw_content is None else str(raw_content))


def _turns_taken(transcript: Sequence[dict[str, Any]]) -> int:
    """Turns this transcript records the agent as having taken, failed ones included (#1023).

    A turn that failed was still a turn taken -- the Core's own `turn_counter` reads 4 after
    four sends of which one failed -- so a failure row counts here, and `turn_counter` stays
    the lifetime figure it is (P5).

    This is deliberately **not** the question `_core_index_for_transcript` answers. That one
    asks how far into the Core's messages a prefix of rows reaches, where a failed turn
    reaches nothing. The two numbers sit side by side in `truncate_session_history` and are
    different on purpose.
    """
    return sum(
        1
        for entry in transcript
        if entry.get("role") in ("assistant", "agent") or entry.get("sender") == "agent"
    )


def _core_key(msg: ChatMessage) -> tuple[str, str | None]:
    """A Core message's identity for matching a transcript row against it.

    The mirror of `_transcript_key`, so the two sides of the correspondence are written
    once each and cannot drift into asking different questions.
    """
    return msg.role.value, msg.content


def _is_folded_into_its_turns_row(msg: ChatMessage) -> bool:
    """Whether the live transcript folds this Core message into its turn's single row.

    `/api/turn` writes exactly two transcript rows per turn -- the prompt and one agent
    row (`updated_history = [*current_history, user_msg_entry, agent_msg_entry]`) -- while
    a tool-using turn leaves four or more messages in the Core: the prompt, an `ASSISTANT`
    message carrying that step's tool calls, the `TOOL` result, and the reply. The middle
    ones reach the page inside the agent row's `tool_calls` and `tool_executions`, never as
    rows of their own, so an index counted over the transcript steps past them.

    **The property is carrying tool calls, not lacking content.** `base.py` appends
    `ChatMessage(ASSISTANT, content=resp_content or None, tool_calls=tool_calls)` under
    `if tool_calls or resp_content:`, so a step that printed a preamble *and* called a tool
    has both. Keying on absent content left that message unfolded and stalled the walk
    exactly as strict lockstep does: measured on a cut that deleted nothing, the Core kept
    1 message where 4 is right, losing the preamble, the tool result and the turn's **own
    final reply** (#1024 review B2).

    They are *not* skipped when the transcript does render them as rows: a conversation
    rebuilt from the Core alone (`_core_messages_to_transcript_pairs`) gives every
    presentable message its own row, and a row that matches is always consumed before this
    question is asked.
    """
    return msg.role == MessageRole.TOOL or (
        msg.role == MessageRole.ASSISTANT and (msg.content is None or bool(msg.tool_calls))
    )


def _core_index_for_transcript(
    transcript: Sequence[dict[str, Any]], messages: Sequence[ChatMessage]
) -> int:
    """How far into the Core's presentable messages a transcript prefix reaches (#1023).

    The page counts a truncation index over the transcript it renders, and #872 made the
    Core's own cut agree with that count. The two lists are not the same length: a failed
    turn is a row with no Core message (the Core holds the prompt that went unanswered and
    no reply to add), and a tool-using turn is the reverse, one row over several messages.

    **The count is derived from the Core, never from what a row's text looks like.** An
    earlier version of this asked `_is_failure_entry`, which calls an assistant row a
    failed turn when its content begins `Error: ` under degraded provenance. That
    predicate answers a different question -- "may the model be shown this row?", where a
    false positive costs one row of rebuilt context (#1022 accepted that window). Deciding
    a *cut* with it **deletes** a message instead: a real reply beginning `Error: ` whose
    provenance is degraded (a failed `persist_session`, or plain provider-side model-alias
    resolution) was cut out of the Core, and off the disk, by a truncation that deleted
    nothing, while the page went on showing it (#1024 review).

    So a row advances the index when the Core's next message matches it, and a row the
    Core has no message for advances nothing of its own -- recognised by the Core's silence
    rather than by its wording, which makes both spellings of a failed turn fall out for
    free. It does carry with it whatever its turn folded in, which is what keeps a failed
    tool turn whole.

    Rejected: **strict lockstep** with no skipping (it stalls on the first tool turn and
    under-counts by three -- the same loss, differently caused); **searching ahead for the
    next match of any kind** (a repeated prompt matches a *later* Core message and swallows
    the reply between them -- measured 3 where 1 is right, worse than the defect it would
    replace).
    """
    presentable = [m for m in messages if _is_presentable(m)]
    index = 0
    for entry in transcript:
        key = _transcript_key(entry)
        candidate = index
        while (
            candidate < len(presentable)
            and _core_key(presentable[candidate]) != key
            and _is_folded_into_its_turns_row(presentable[candidate])
        ):
            candidate += 1
        matched = candidate < len(presentable) and _core_key(presentable[candidate]) == key
        # What was stepped over is kept either way. A row that matches nothing is still a
        # row *about* a turn -- a failed one -- and the messages folded into it are the
        # Core's record of that turn, so they survive with it. Leaving them uncounted cut a
        # turn that failed after its tool step to 1 message where 3 is right (#1024 B2).
        index = candidate + 1 if matched else candidate
    return index


def _sse_provenance_block(event: AgentEvent) -> dict[str, Any]:
    """Describe an event's in-band provenance for an SSE frame (#117, #150).

    `path`, `degraded`, `served_by`, and `persona` are reported **only when the event actually states them**
    or when a turn failed / produced an error reply (P6).
    """
    block: dict[str, Any] = {
        "component": "uclone_x.engine.event_bus",
        "producer": event.sender_id,
    }
    is_failed_turn = bool(
        event.payload.get("error") or str(event.payload.get("is_completed", "")).lower() == "false"
    )
    provenance = event.provenance
    if provenance is not None:
        block["path"] = provenance.path.value
        block["degraded"] = provenance.degraded or is_failed_turn
        block["served_by"] = provenance.served_by.provider
    elif is_failed_turn:
        block["degraded"] = True

    persona = event.payload.get("persona")
    if persona is not None and str(persona).strip():
        block["persona"] = str(persona).strip()
    return block


def _extract_markdown_title(path: Path) -> str:
    """Extract first # Heading 1 from markdown file, or fallback to cleaned file stem."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            chunk = f.read(4096)
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                title = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", title)
                title = re.sub(r"[*_`]", "", title)
                if title:
                    return title
    except Exception:
        pass
    return path.stem.replace("-", " ").replace("_", " ").title()


class AgentSessionManager:
    """Manages active BaseAgent instances per agent_id and session for the UI layer."""

    def __init__(
        self,
        bus: EventBus | None = None,
        llm: LLMProviderProtocol | None = None,
        tools: ToolRegistryProtocol | None = None,
        tracer: TelemetryTracer | None = None,
        fallback_to_mock: bool = False,
        storage_dir: Path | None = None,
        ontology_engine: OntologyEngine | None = None,
        skill_registry: SkillRegistry | None = None,
        budget_tracker: TokenBudgetManager | None = None,
        eval_reports_dir: Path | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        self._bus = bus if bus is not None else get_ui_event_bus()
        self._llm = llm
        #: Told when Settings replaces the connector. The chat agents in `_agents` are
        #: reloaded here, but a conversation's seats are built and cached by the room stack,
        #: which this class cannot see; without this they kept the connector they were
        #: built with -- `None`, for a room first used before a model was chosen (#1446).
        self._llm_listeners: list[Callable[[LLMProviderProtocol | None], None]] = []
        self._tools = tools if tools is not None else create_default_registry()
        self._tracer = tracer if tracer is not None else get_ui_tracer()
        self._fallback_to_mock = fallback_to_mock
        resolved_workspace = (
            workspace_dir.resolve()
            if workspace_dir is not None
            else Path(os.getenv("UCLONE_WORKSPACE_DIR", os.getcwd())).resolve()
        )
        self._workspace_dir: Path = resolved_workspace
        # `default_session_root()`, not an inline `Path.home() / ...`: the inline form
        # ignored `UCLONE_SESSION_DIR`, so the variable added to keep a headless run out
        # of the invoking user's home was not consulted by the busiest writer to that
        # directory. One resolver, two layers.
        self._storage_dir: Path = (
            storage_dir.resolve() if storage_dir is not None else default_session_root().resolve()
        )
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        # P8: the conversation is Core state and lives in the Core store. This class
        # keeps only the UI's *presentation transcript* — per-message ids, timestamps,
        # latency, token counts and the display provenance block, which are view data
        # the Core has no reason to model.
        #
        # Both artifacts are namespaced under the root, and the root itself is never
        # written. The earlier arrangement kept the transcript on `<root>/<id>.json` —
        # which is the path the **CLI's own Core store** owns — and put only the UI's
        # Core record under `core/`. So `ucx run` and the UI wrote different schemas to
        # one file, each destroying the other, and a record that does not validate reads
        # back as absent, so the loss was silent by construction.
        #
        # It also had the sharing backwards. The CLI and the UI should share **one** Core
        # record per session, which is what P8's single store means; they now both
        # resolve to `<root>/core/<id>.json`. The transcript is genuinely a different
        # artifact and gets `<root>/ui/`.
        self._core_store = SessionStore(storage_dir=self._storage_dir / CORE_RECORD_SUBDIR)
        self._transcript_dir = self._storage_dir / UI_TRANSCRIPT_SUBDIR
        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        reap_orphaned_temp_files(self._transcript_dir)
        self._ontology_engine = ontology_engine if ontology_engine is not None else OntologyEngine()
        self._skill_registry = skill_registry if skill_registry is not None else SkillRegistry()
        self._budget_tracker = (
            budget_tracker if budget_tracker is not None else TokenBudgetManager()
        )
        self._eval_reports_dir = (
            eval_reports_dir.resolve() if eval_reports_dir is not None else default_reports_dir()
        )
        self._agents: dict[str, BaseAgent] = {}
        # One memory store instance per agent id, not per session. Two live sessions of the
        # same agent constructing their own stores over the same file would each hold the
        # whole fact set in memory and each `save()` the whole of it, so the second writer
        # drops whatever the first recorded after it loaded (#1097).
        self._agent_memories: dict[str, CrossSessionMemory] = {}
        self._session_messages: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._configured_provider: str | None = None
        self._configured_base_url: str | None = None
        self._configured_api_key: str | None = None
        #: The provider `_configured_api_key` was saved for; `_api_key_for` applies it only there.
        self._configured_api_key_provider: str | None = None
        self._configured_model: str | None = None
        self._configured_comfyui_url: str | None = os.getenv(
            "COMFYUI_BASE_URL", DEFAULT_COMFYUI_BASE_URL
        )
        #: Folders outside the workspace that clones may read, as the user entered them.
        self._configured_read_roots: tuple[str, ...] = ()
        # The same file `ucx run` and `ucx install` read and seed (`saved_choice.py`).
        self._settings_file: Path = self._storage_dir / SETTINGS_FILE_NAME
        self._load_persisted_settings()
        for entry in self._env_read_root_entries():
            if (problem := _read_root_problem(entry, self._storage_dir)) is not None:
                logger.warning("Ignoring %s entry: %s", READ_ROOTS_ENV_VAR, problem)

    @property
    def workspace_dir(self) -> Path:
        return self._workspace_dir

    @staticmethod
    def _env_read_root_entries() -> list[str]:
        return [e.strip() for e in os.getenv(READ_ROOTS_ENV_VAR, "").split(os.pathsep) if e.strip()]

    @property
    def read_roots(self) -> tuple[Path, ...]:
        """Folders outside the workspace that clones' read-only file tools may read.

        `UCLONE_READ_ROOTS` (separated by `os.pathsep`) first, then the Settings list. An
        entry that is unusable now -- a folder deleted since it was saved, a bad
        environment entry -- is left out rather than failing every agent build, and
        `get_settings` reports it.
        """
        roots: list[Path] = []
        for entry in [*self._env_read_root_entries(), *self._configured_read_roots]:
            if _read_root_problem(entry, self._storage_dir) is not None:
                continue
            resolved = Path(entry).expanduser().resolve()
            if resolved not in roots:
                roots.append(resolved)
        return tuple(roots)

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def tools(self) -> ToolRegistryProtocol:
        return self._tools

    def apply_persona(self, persona: PersonaDefinition) -> int:
        """Put an edited persona in force on every chat agent seated as it; return how many.

        Chat agents only: a room's agents are built by `room/resolver.py` and are not held
        here, so they take the edit when the room is next seated.

        Each agent resolves its prompt from the persona on every read, but its tool scope is
        resolved once and stored, and `define_persona` is the call that recomputes it
        (#1153). An agent is counted once however many keys it is cached under. What an
        agent took at construction -- its model settings and the write and delegation
        switches -- stays until that agent is created again.
        """
        seen: set[int] = set()
        for agent in self._agents.values():
            if agent.persona != persona.name or id(agent) in seen:
                continue
            seen.add(id(agent))
            agent.define_persona(persona)
        return len(seen)

    @property
    def llm(self) -> LLMProviderProtocol | None:
        return self._llm

    @property
    def tracer(self) -> TelemetryTracer:
        return self._tracer

    @property
    def ontology_engine(self) -> OntologyEngine:
        return self._ontology_engine

    @property
    def skill_registry(self) -> SkillRegistry:
        return self._skill_registry

    @property
    def budget_tracker(self) -> TokenBudgetManager:
        return self._budget_tracker

    @property
    def fallback_to_mock(self) -> bool:
        return self._fallback_to_mock

    @property
    def core_store(self) -> SessionStore:
        """The Core session store this UI is a client of (P8)."""
        return self._core_store

    @property
    def eval_reports_dir(self) -> Path:
        """Directory path containing evaluation reports and scorecards."""
        return self._eval_reports_dir

    @eval_reports_dir.setter
    def eval_reports_dir(self, path: Path) -> None:
        """Update the directory path containing evaluation reports."""
        self._eval_reports_dir = path.resolve()

    @property
    def configured_model(self) -> str | None:
        """Configured model override for the UI session manager."""
        return self._configured_model

    @property
    def configured_provider(self) -> str | None:
        """Configured provider override for the UI session manager."""
        return self._configured_provider

    @property
    def configured_api_key(self) -> str | None:
        """Configured API key for the configured provider, when it was saved for that one."""
        return self._api_key_for(self._configured_provider)

    def _api_key_for(self, provider: str | None) -> str | None:
        """The configured key, only when it belongs to ``provider`` (see ``key_belongs_to``).

        The settings file keeps one key while the provider changes, so a key saved for
        OpenAI is still there after a switch to Anthropic -- and must not be sent to it. A
        key with no known owner (entered before any provider was) goes wherever it is used,
        as every key did before keys were tagged.
        """
        key = self._configured_api_key
        if not key:
            return None
        owner = self._configured_api_key_provider
        if owner is None or same_provider(owner, provider):
            return key
        return None

    @property
    def configured_base_url(self) -> str | None:
        """Configured provider base URL override for the UI session manager."""
        return self._configured_base_url

    @property
    def default_llm(self) -> LLMProviderProtocol | None:
        """Default or configured LLM provider connector."""
        return self._llm

    def _load_persisted_settings(self) -> None:
        """Load persisted settings from storage directory if available."""
        if self._settings_file.is_file():
            try:
                data = json.loads(self._settings_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    cfg = cast(dict[str, Any], data)
                    if "llm_provider" in cfg and isinstance(cfg["llm_provider"], str):
                        self._configured_provider = cfg["llm_provider"].strip().lower()
                    if "llm_base_url" in cfg and isinstance(cfg["llm_base_url"], str):
                        self._configured_base_url = cfg["llm_base_url"].strip()
                    if "llm_model" in cfg and isinstance(cfg["llm_model"], str):
                        self._configured_model = cfg["llm_model"].strip()
                    if "llm_api_key" in cfg and isinstance(cfg["llm_api_key"], str):
                        self._configured_api_key = cfg["llm_api_key"].strip()
                        owner = key_owner(cfg)
                        self._configured_api_key_provider = owner.lower() if owner else None
                    if "comfyui_base_url" in cfg and isinstance(cfg["comfyui_base_url"], str):
                        self._configured_comfyui_url = cfg["comfyui_base_url"].strip()
                    raw_roots: object = cfg.get("read_roots")
                    if isinstance(raw_roots, list):
                        self._configured_read_roots = tuple(
                            entry.strip()
                            for entry in cast(list[object], raw_roots)
                            if isinstance(entry, str) and entry.strip()
                        )

                    # Propagate loaded settings into process environment if not already overridden
                    eff_provider = self._configured_provider
                    if eff_provider and "LLM_PROVIDER" not in os.environ:
                        os.environ["LLM_PROVIDER"] = eff_provider

                    if self._configured_base_url:
                        if eff_provider == "ollama" and "OLLAMA_BASE_URL" not in os.environ:
                            os.environ["OLLAMA_BASE_URL"] = self._configured_base_url
                        elif eff_provider == "vllm" and "VLLM_BASE_URL" not in os.environ:
                            os.environ["VLLM_BASE_URL"] = self._configured_base_url
                        elif eff_provider == "openai" and "OPENAI_BASE_URL" not in os.environ:
                            os.environ["OPENAI_BASE_URL"] = self._configured_base_url
                        elif eff_provider == "anthropic" and "ANTHROPIC_BASE_URL" not in os.environ:
                            os.environ["ANTHROPIC_BASE_URL"] = self._configured_base_url

                    if self._configured_model:
                        if eff_provider == "ollama" and "OLLAMA_MODEL" not in os.environ:
                            os.environ["OLLAMA_MODEL"] = self._configured_model
                        elif eff_provider == "vllm" and VLLM_MODEL_ENV_VAR not in os.environ:
                            # Load-bearing, not symmetry: `VLLMConnector` refuses a request
                            # that names no model unless this variable does, so a saved
                            # vLLM configuration that skipped this line would come back
                            # after a restart as "no model is configured".
                            os.environ[VLLM_MODEL_ENV_VAR] = self._configured_model
                        elif eff_provider == "openai" and "OPENAI_MODEL" not in os.environ:
                            os.environ["OPENAI_MODEL"] = self._configured_model
                        elif eff_provider == "anthropic" and "ANTHROPIC_MODEL" not in os.environ:
                            os.environ["ANTHROPIC_MODEL"] = self._configured_model
                        elif (
                            eff_provider in ("gemini", "google")
                            and "GEMINI_MODEL" not in os.environ
                        ):
                            os.environ["GEMINI_MODEL"] = self._configured_model

                    # Only a key saved for this provider is exported: exporting another
                    # provider's key under this one's variable sends it to the wrong service.
                    own_key = self._api_key_for(eff_provider)
                    if own_key:
                        if eff_provider == "openai" and "OPENAI_API_KEY" not in os.environ:
                            os.environ["OPENAI_API_KEY"] = own_key
                        elif eff_provider == "vllm" and "VLLM_API_KEY" not in os.environ:
                            os.environ["VLLM_API_KEY"] = own_key
                        elif eff_provider == "anthropic" and "ANTHROPIC_API_KEY" not in os.environ:
                            os.environ["ANTHROPIC_API_KEY"] = own_key
                        elif (
                            eff_provider in ("gemini", "google")
                            and "GEMINI_API_KEY" not in os.environ
                        ):
                            os.environ["GEMINI_API_KEY"] = own_key

                    if self._llm is None and (
                        self._configured_provider or self._configured_base_url
                    ):
                        try:
                            self._llm = create_llm_connector(
                                provider=self._configured_provider or None,
                                api_key=self._api_key_for(self._configured_provider),
                                base_url=self._configured_base_url or None,
                                fallback_to_mock=self._fallback_to_mock,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to initialize active LLM connector from persisted settings %s: %s",
                                self._settings_file,
                                exc,
                            )
            except Exception as exc:
                logger.warning("Failed to read settings file %s: %s", self._settings_file, exc)

    def _adopt_saved_choice(self) -> None:
        """Pick up a model choice saved after this dashboard started, while it has none.

        `ucx install` (or `ucx llm use`) can save a choice while a dashboard is already
        running. That dashboard read the file at start and found nothing; without this it
        would report no provider, and build nothing from the choice, until restarted. A
        provider this dashboard was given or saved itself is never replaced, and a choice
        the environment overrides (`LLM_PROVIDER`, a key, an endpoint) is not adopted --
        the same test the connector factory applies.
        """
        if self._configured_provider:
            return
        saved = saved_choice_in_effect(path=self._settings_file)
        if saved is None:
            return
        self._configured_provider = saved.provider
        self._configured_model = self._configured_model or saved.model
        self._configured_base_url = self._configured_base_url or saved.base_url
        if not self._configured_api_key and saved.api_key:
            # `saved.api_key` is only ever the key saved for `saved.provider`.
            self._configured_api_key = saved.api_key
            self._configured_api_key_provider = saved.provider
        if self._llm is None:
            try:
                adopted = create_llm_connector(
                    provider=saved.provider,
                    api_key=self._api_key_for(saved.provider),
                    base_url=self._configured_base_url or None,
                    fallback_to_mock=self._fallback_to_mock,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize the LLM connector saved in %s: %s",
                    self._settings_file,
                    exc,
                )
                return
            self._install_llm(adopted)

    def _install_llm(self, new_llm: LLMProviderProtocol) -> None:
        """Make ``new_llm`` the connector, and hand it to every open agent and room (#1446).

        A Settings save and a choice adopted after startup both go through here, so a room
        opened before either picks the new connector up the same way.
        """
        self._llm = new_llm
        for agent in self._agents.values():
            agent.hot_reload_llm(new_llm, model_name=self._configured_model)
        for listener in self._llm_listeners:
            listener(new_llm)

    def _save_persisted_settings(self, changes: dict[str, Any]) -> None:
        """Merge the settings this save changed into the settings file.

        Only `changes` are written: the file is shared with setup and `ucx llm use`, and
        rewriting it whole from memory put `"llm_provider": null` back over a model saved
        after this dashboard started. When the file cannot be read, it is replaced with
        everything this dashboard holds, as a Settings save always did.
        """
        everything: dict[str, Any] = {
            "llm_provider": self._configured_provider,
            "llm_base_url": self._configured_base_url,
            "llm_model": self._configured_model,
            "llm_api_key": self._configured_api_key,
            "llm_api_key_provider": self._configured_api_key_provider,
            "comfyui_base_url": self._configured_comfyui_url,
            "read_roots": list(self._configured_read_roots),
        }
        try:
            update_settings_file(
                changes, path=self._settings_file, replace_unreadable_with=everything
            )
        except Exception as exc:
            logger.warning("Failed to write settings file %s: %s", self._settings_file, exc)

    def get_settings(self) -> dict[str, Any]:
        """Return active endpoints, configurations, and masked credentials."""
        self._adopt_saved_choice()
        active_provider = (
            self._configured_provider
            or (getattr(self._llm, "provider_name", None) if self._llm else None)
            or os.getenv("LLM_PROVIDER")
            or "ollama"
        )

        active_base_url = self._configured_base_url
        if not active_base_url and self._llm and hasattr(self._llm, "base_url"):
            active_base_url = cast(str | None, getattr(self._llm, "base_url", None))
        if not active_base_url:
            if active_provider == "ollama":
                active_base_url = resolve_ollama_base_url()
            elif active_provider == "vllm":
                # `resolve_vllm_base_url` refuses rather than defaults, and the refusal
                # belongs on a turn, not on opening the panel where the endpoint is typed.
                active_base_url = resolve_vllm_base_url() if has_configured_vllm_endpoint() else ""
            elif active_provider == "openai":
                active_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            elif active_provider == "anthropic":
                active_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
            else:
                active_base_url = ""

        active_model = self._configured_model
        if not active_model:
            for ag in self._agents.values():
                if ag.config.llm_config.model_name:
                    active_model = ag.config.llm_config.model_name
                    break
        if not active_model:
            if active_provider == "ollama":
                active_model = os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_INDEPTH_MODEL") or ""
            elif active_provider == "vllm":
                active_model = resolve_vllm_model() or ""
            elif active_provider == "openai":
                active_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            elif active_provider == "anthropic":
                active_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            elif active_provider in ("gemini", "google"):
                active_model = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            else:
                active_model = ""

        raw_key = self._api_key_for(active_provider)
        if not raw_key and self._llm and hasattr(self._llm, "api_key"):
            raw_key = cast(str | None, getattr(self._llm, "api_key", None))
        if not raw_key:
            if active_provider == "openai":
                raw_key = os.getenv("OPENAI_API_KEY")
            elif active_provider == "vllm":
                raw_key = os.getenv("VLLM_API_KEY")
            elif active_provider == "anthropic":
                raw_key = os.getenv("ANTHROPIC_API_KEY")
            elif active_provider in ("gemini", "google"):
                raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        key_set = bool(raw_key and raw_key.strip())
        masked_key = ""
        if key_set and raw_key:
            trimmed = raw_key.strip()
            if len(trimmed) > 8:
                masked_key = f"{trimmed[:3]}...{trimmed[-4:]}"
            else:
                masked_key = "***"

        comfy_url = self._configured_comfyui_url or os.getenv(
            "COMFYUI_BASE_URL", DEFAULT_COMFYUI_BASE_URL
        )

        available_providers = ["ollama", "vllm", "openai", "anthropic", "gemini"]
        if active_provider == "mock":
            available_providers.append("mock")

        return {
            "llm_provider": active_provider,
            "llm_base_url": active_base_url or "",
            "llm_model": active_model or "",
            "llm_api_key_set": key_set,
            "llm_api_key_masked": masked_key,
            "comfyui_base_url": comfy_url,
            "providers_available": available_providers,
            "workspace_dir": str(self._workspace_dir),
            "read_roots": list(self._configured_read_roots),
            "read_roots_missing": [
                entry
                for entry in self._configured_read_roots
                if _read_root_problem(entry, self._storage_dir) is not None
            ],
            # Set outside the app, so the Settings list cannot edit them; shown so the list
            # is not mistaken for everything a clone can read.
            "read_roots_env": [
                entry
                for entry in self._env_read_root_entries()
                if _read_root_problem(entry, self._storage_dir) is None
            ],
            "read_roots_env_ignored": [
                f"{problem} (from {READ_ROOTS_ENV_VAR})"
                for entry in self._env_read_root_entries()
                if (problem := _read_root_problem(entry, self._storage_dir)) is not None
            ],
        }

    def on_llm_replaced(self, listener: Callable[[LLMProviderProtocol | None], None]) -> None:
        """Call `listener` with the new connector whenever Settings replaces it."""
        self._llm_listeners.append(listener)

    def update_settings(
        self,
        llm_provider: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        comfyui_base_url: str | None = None,
        read_roots: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update configurations, hot-reload LLM connectors and tools across active agents."""
        self._adopt_saved_choice()
        clean_roots = (
            _validate_read_roots(read_roots, self._storage_dir, self._configured_read_roots)
            if read_roots is not None
            else None
        )
        env_updates: dict[str, str] = {}
        state_updates: dict[str, str] = {}

        if llm_provider is not None and llm_provider.strip():
            prov_clean = llm_provider.strip().lower()
            allowed = {"ollama", "vllm", "openai", "anthropic", "gemini", "google", "mock"}
            if prov_clean not in allowed:
                raise ValueError(f"Unsupported LLM provider: {prov_clean}")
            state_updates["_configured_provider"] = prov_clean
            env_updates["LLM_PROVIDER"] = prov_clean

        eff_provider = state_updates.get("_configured_provider", self._configured_provider)

        if llm_base_url is not None:
            clean_base = llm_base_url.strip()
            state_updates["_configured_base_url"] = clean_base
            if eff_provider == "ollama":
                env_updates["OLLAMA_BASE_URL"] = clean_base
            elif eff_provider == "vllm":
                env_updates["VLLM_BASE_URL"] = clean_base
            elif eff_provider == "openai":
                env_updates["OPENAI_BASE_URL"] = clean_base
            elif eff_provider == "anthropic":
                env_updates["ANTHROPIC_BASE_URL"] = clean_base

        if llm_model is not None and llm_model.strip():
            clean_model = llm_model.strip()
            state_updates["_configured_model"] = clean_model
            if eff_provider == "ollama":
                env_updates["OLLAMA_MODEL"] = clean_model
            elif eff_provider == "vllm":
                env_updates[VLLM_MODEL_ENV_VAR] = clean_model
            elif eff_provider == "openai":
                env_updates["OPENAI_MODEL"] = clean_model
            elif eff_provider == "anthropic":
                env_updates["ANTHROPIC_MODEL"] = clean_model
            elif eff_provider in ("gemini", "google"):
                env_updates["GEMINI_MODEL"] = clean_model

        if llm_api_key is not None:
            clean_key = llm_api_key.strip()
            if clean_key and not clean_key.startswith("***") and "..." not in clean_key:
                state_updates["_configured_api_key"] = clean_key
                key_for = (
                    eff_provider
                    or (getattr(self._llm, "provider_name", None) if self._llm else None)
                    or os.getenv("LLM_PROVIDER")
                )
                if key_for:
                    # Recorded with the key, so a later switch to another provider keeps
                    # the key without sending it there.
                    state_updates["_configured_api_key_provider"] = str(key_for).strip().lower()
                if eff_provider == "openai":
                    env_updates["OPENAI_API_KEY"] = clean_key
                elif eff_provider == "vllm":
                    env_updates["VLLM_API_KEY"] = clean_key
                elif eff_provider == "anthropic":
                    env_updates["ANTHROPIC_API_KEY"] = clean_key
                elif eff_provider in ("gemini", "google"):
                    env_updates["GEMINI_API_KEY"] = clean_key

        eff_provider_for_llm = str(
            eff_provider
            or (getattr(self._llm, "provider_name", None) if self._llm else None)
            or os.getenv("LLM_PROVIDER")
            or ""
        )
        new_llm = create_llm_connector(
            # Empty means "resolve from configuration". The literal "ollama" here made the
            # settings path build a localhost connector for a user who had configured
            # nothing, which is the case #533's refusal exists to report.
            provider=eff_provider_for_llm or None,
            api_key=state_updates.get("_configured_api_key")
            or self._api_key_for(eff_provider_for_llm or None),
            base_url=state_updates.get("_configured_base_url", self._configured_base_url) or None,
            fallback_to_mock=self._fallback_to_mock,
        )

        for k, v in env_updates.items():
            os.environ[k] = v
        for k, v in state_updates.items():
            setattr(self, k, v)
        self._install_llm(new_llm)

        logger.info(
            "⚙️ [UI Settings] Model/Settings updated: provider=%s, model=%s, base_url=%s",
            eff_provider,
            self._configured_model,
            self._configured_base_url,
        )
        _console.print(
            f"[bold green]⚙️ [UI Settings] Active model updated:[/bold green] [bold yellow]{self._configured_model}[/bold yellow] "
            f"(provider: [cyan]{eff_provider}[/cyan])"
        )

        if comfyui_base_url is not None and comfyui_base_url.strip():
            clean_comfy = comfyui_base_url.strip()
            self._configured_comfyui_url = clean_comfy
            os.environ["COMFYUI_BASE_URL"] = clean_comfy
            img_tool = self._tools.get("generate_image")
            if isinstance(img_tool, ComfyImageGenTool):
                img_tool.update_base_url(clean_comfy)

        if clean_roots is not None:
            self._configured_read_roots = clean_roots
            effective_roots = self.read_roots
            for agent in self._agents.values():
                agent.set_read_roots(effective_roots)

        changes: dict[str, Any] = {
            key: getattr(self, attr)
            for key, attr in (
                ("llm_provider", "_configured_provider"),
                ("llm_base_url", "_configured_base_url"),
                ("llm_model", "_configured_model"),
                ("llm_api_key", "_configured_api_key"),
                ("llm_api_key_provider", "_configured_api_key_provider"),
            )
            if attr in state_updates
        }
        if comfyui_base_url is not None and comfyui_base_url.strip():
            changes["comfyui_base_url"] = self._configured_comfyui_url
        if clean_roots is not None:
            changes["read_roots"] = list(self._configured_read_roots)
        self._save_persisted_settings(changes)
        return self.get_settings()

    def get_session_path(self, session_id: str) -> Path:
        """Resolve the UI transcript path for `session_id`, refusing escapes (P3).

        Under `<root>/ui/`, not `<root>/` — the root path belongs to the Core store and
        sharing it destroyed conversations in both directions.

        Delegates to `uclone_x.agent.session.resolve_session_path`, which is the single
        implementation of this guard. It previously existed twice — here and in the Core
        store — and a duplicated security control is one that gets fixed in a single copy.

        Raises:
            PathTraversalError: See `resolve_session_path`.
        """
        return resolve_session_path(self._transcript_dir, session_id)

    def legacy_session_path(self, session_id: str) -> Path:
        """Resolve the pre-namespacing transcript path, `<root>/<session_id>.json`.

        Read-only. Installs that predate the namespacing have transcripts at the root,
        and some of those files are Core records or hybrids left by the collision. They
        are still offered to the transcript hydration path so an existing conversation
        does not vanish from the UI, and nothing ever writes there again.
        """
        return resolve_session_path(self._storage_dir, session_id)

    def load_session_record(self, session_id: str) -> dict[str, Any] | None:
        """Load the UI transcript if present, falling back to the legacy root path.

        The legacy fallback is read-only and exists so a transcript written before the
        namespacing does not vanish from the UI. It may also encounter a Core record or
        a collision hybrid sitting at that path; those simply do not match the shape the
        transcript reader expects, and it returns `None` rather than pretending.

        **A transcript belonging to a different session is refused (#256).** The Core store
        is not the only artifact keyed by a session id in a filename, so it is not the only
        one the case- and normalization-insensitive filesystem folds. Measured on
        `e8b3e2f`: after `save_session_record("SessA", ...)`,
        `load_session_record("SESSA")` returned `SessA`'s transcript — including its
        `session_id` — and `get_session_history` cached it under `"SESSA"`, so the
        dashboard displayed one session's conversation as another's. `#256` names only the
        Core store; this door was found by enumerating the id-bearing surface. The rule is
        `agent.session.verify_record_identity`, the same one the Core store uses, not a
        second copy.

        The check is conditional on the record actually carrying a `session_id` string,
        which is deliberate: it refuses on **positive evidence** of a different owner and
        stays silent about a legacy file too damaged to name its session, which is already
        handled by the shape mismatch below.

        Raises:
            PathTraversalError: See `agent.session.resolve_session_path`.
            SessionIdCollisionError: If the transcript found identifies another session.
        """
        path = self.get_session_path(session_id)
        if not path.is_file():
            legacy = self.legacy_session_path(session_id)
            if not legacy.is_file():
                return None
            path = legacy
        try:
            content = path.read_text(encoding="utf-8")
            raw_data: object = json.loads(content)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(raw_data, dict):
            return None
        record = cast(dict[str, Any], raw_data)
        # Outside the `try`: this refusal must not be swallowed by the unreadable-record
        # handler above it. A transcript that parses and names another session is not an
        # unreadable transcript.
        recorded_id: object = record.get("session_id")
        if isinstance(recorded_id, str):
            verify_record_identity(session_id, recorded_id, path)
        return record

    def save_session_record(
        self,
        session_id: str,
        agent_id: str,
        messages: list[dict[str, Any]],
        turns: int = 0,
    ) -> None:
        """Atomically persist session record to disk."""
        path = self.get_session_path(session_id)
        existing = self.load_session_record(session_id)
        now_iso = datetime.now(UTC).isoformat()
        created_at = existing.get("created_at", now_iso) if existing else now_iso
        record: dict[str, Any] = {
            "session_id": session_id,
            "agent_id": agent_id,
            "created_at": created_at,
            "updated_at": now_iso,
            "turns": turns,
            "messages": list(messages),
        }
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        # Atomic file write via temporary file. The temporary name comes from the pid
        # plus a random suffix, not from `asyncio.get_running_loop().time()`: that call
        # raises `RuntimeError` when no loop is running, which made this synchronous
        # method unusable from synchronous callers — including tests of it. Matches
        # `SessionStore.save`, which never had the defect (#183).
        #
        # Adopted flush() and os.fsync() on temp descriptor for crash durability (#219, #257).
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record, indent=2))
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        # Only once the record is on disk does the view change. This assignment used to
        # come first, so an `OSError` from the write left the in-memory transcript
        # rewritten over a stale record — and the compaction path re-raises, answering
        # 5xx over a view it had already mutated. The two stores this method exists to
        # keep in step were then out of step precisely when the caller was told nothing
        # had happened (P6).
        self._session_messages[session_id] = list(messages)

    def get_session_history(self, agent_id: str, session_id: str) -> list[dict[str, Any]]:
        """Return persisted or in-memory conversation message history."""
        if session_id in self._session_messages:
            msgs = list(self._session_messages[session_id])
            _, typed_msgs = self.reconstruct_history(msgs, session_id=session_id)
            self._session_messages[session_id] = list(typed_msgs)
            return list(typed_msgs)
        record = self.load_session_record(session_id)
        if record is not None:
            raw_msgs: object = record.get("messages", [])
            if isinstance(raw_msgs, list):
                raw_list: list[object] = cast(list[object], raw_msgs)
                _, typed_msgs = self.reconstruct_history(raw_list, session_id=session_id)
                self._session_messages[session_id] = list(typed_msgs)
                return list(typed_msgs)
        state = self._core_store.load(session_id)
        if state is not None and state.messages:
            core_msgs = _core_messages_to_transcript(
                state.messages,
                agent_id=state.agent_id or agent_id,
                timestamp=state.updated_at,
            )
            self._session_messages[session_id] = list(core_msgs)
            return list(core_msgs)
        return []

    def _live_messages(
        self, agent_id: str, session_id: str, agent: BaseAgent | None = None
    ) -> tuple[list[ChatMessage], int] | None:
        """The Core's current messages and lifetime turn counter, or `None` if unknown.

        The live agent is preferred over the store because it is the writer: an agent
        holding an unpersisted turn is ahead of the record, never behind it.

        `agent` is for a caller that is already holding the writer and knows this manager
        cannot find it. A conversation seat's agent is built and cached by
        `RoomAgentResolver`, never registered here, so `get_agent` answers `None` for one and
        this would silently fall back to the persisted copy — behind the live agent by
        whatever it has not written yet, which is exactly the turn a reader is asking about.
        """
        agent = agent if agent is not None else self.get_agent(agent_id, session_id)
        if agent is not None:
            live = agent.get_session(session_id)
            return list(live.messages), live.turn_counter
        state = self._core_store.load(session_id)
        return (list(state.messages), state.turn_counter) if state is not None else None

    def active_turns(self, agent_id: str, session_id: str, agent: BaseAgent | None = None) -> int:
        """Turns held in this session's active context (#872).

        Falls back to counting agent messages in the transcript when the Core has nothing
        to say about the session, which is the same figure by a weaker route rather than
        a guess.

        `agent` names the live writer for a caller holding one this manager cannot find —
        see `_live_messages`. Without it a conversation seat reads its persisted copy, and a
        seat that has never been persisted reads zero: a saturation banner that stays quiet
        on the one conversation that needs it.
        """
        live = self._live_messages(agent_id, session_id, agent)
        if live is not None:
            return count_active_turns(live[0])
        transcript = self._session_messages.get(session_id)
        if transcript is None:
            return 0
        return _turns_taken(transcript)

    def truncate_session_history(
        self,
        agent_id: str,
        session_id: str,
        index: int,
    ) -> list[dict[str, Any]]:
        """Truncate UI transcript and Core session memory back to index (FR-13.7, P8).

        Parameters:
            agent_id: The agent ID associated with the session.
            session_id: The session ID to truncate.
            index: The 0-based message index to truncate to (retains messages 0..index-1).

        Raises:
            PathTraversalError: If session_id attempts path traversal.
            SessionIdCollisionError: If session record identifies another session.
            SessionMutationDuringTurnError: If target session has a turn in flight.
            ValueError: If index is negative.
        """
        if index < 0:
            raise ValueError("'index' must be non-negative")

        # Path resolution and ownership checks
        self.get_session_path(session_id)
        self.load_session_record(session_id)

        # Truncate UI presentation transcript
        transcript_msgs = self.get_session_history(agent_id, session_id)
        truncated_transcript = transcript_msgs[:index]
        # Counted over the transcript **raw**, failure rows included, and deliberately not
        # mapped like the Core index below: `turn_counter` is the lifetime figure for turns
        # this session has taken, and a turn that failed was still taken -- the Core's own
        # counter reads 4 after four sends of which one failed (#1023).
        new_turn_counter = _turns_taken(truncated_transcript)
        self._session_messages[session_id] = list(truncated_transcript)
        self.save_session_record(
            session_id=session_id,
            agent_id=agent_id,
            messages=truncated_transcript,
            turns=new_turn_counter,
        )

        # Truncate Core session memory. The index is an offset into the transcript, so it
        # is mapped against the very messages about to be cut rather than counted over the
        # rows -- the two lists do not have the same length (#1023).
        agent = self.get_agent(agent_id, session_id)
        state = (
            agent.get_session(session_id)
            if agent is not None
            else self._core_store.load(session_id)
        )
        core_messages = list(state.messages) if state is not None else []
        core_index = _core_index_for_transcript(truncated_transcript, core_messages)

        if agent is not None:
            new_msgs = _truncate_core_messages(core_messages, core_index)
            agent.load_history(new_msgs, turn_counter=new_turn_counter, session_id=session_id)
            try:
                agent.persist_session(session_id)
            except SessionStoreNotConfiguredError:
                pass
        elif state is not None:
            new_msgs = _truncate_core_messages(core_messages, core_index)
            new_state = state.with_messages(new_msgs, turn_counter=new_turn_counter)
            self._core_store.save(new_state)

        return list(truncated_transcript)

    def clear_session_history(
        self, agent_id: str, session_id: str, agent: BaseAgent | None = None
    ) -> None:
        """Clear the UI transcript and reset the Core session (#183, P8).

        The agent-side reset is delegated to `BaseAgent.reset_session`, which is the
        Core's single reset semantics. This method previously re-seeded the system
        prompt itself, in **two duplicated blocks** in this one body, and was one of
        three reset implementations in the tree that disagreed with each other — the
        other two being the CLI REPL's `/reset` and `BaseAgent.__init__`, which nothing
        else called.

        Path resolution runs first and is allowed to raise, so a traversal attempt is
        refused before anything is deleted.

        **Ownership is established before destruction too (#256).** Resolution alone was
        not enough: `"SESSA"` resolves legally to the one file `"SessA"` also resolves to
        on a case-insensitive filesystem, so the unlink at the end of this method would
        have destroyed another session's transcript, and the `_core_store.delete` in the
        no-agent branch its Core record. Both are now refused, by the read below and by
        `SessionStore.delete` respectively.

        **`agent` names the live writer when the caller is holding one this manager cannot
        find.** A conversation seat's agent is built and cached by `RoomAgentResolver` and is
        never registered in `_agents`, so `get_agent` answers `None` for one and the branch
        below would delete the stored record while that live agent went on holding the
        messages it had — and persisted them back over the deletion at its next turn, which
        presents as a clear that silently did nothing. Passing the agent takes the reset
        branch, so the in-memory copy and the record are cut by the same call.

        Raises:
            PathTraversalError: See `agent.session.resolve_session_path`.
            SessionIdCollisionError: If either artifact at this id's paths identifies a
                different session. Nothing is reset or unlinked.
        """
        # Path resolution first, so a traversal is refused before anything is touched.
        path = self.get_session_path(session_id)
        # Then ownership, before anything is reset or unlinked. `load_session_record`
        # raises on a transcript naming another session and returns `None` when there is
        # nothing there to own.
        self.load_session_record(session_id)

        # Core reset **before** the transcript unlink. The previous order deleted the
        # transcript first, so a Core reset that then failed left `transcript False /
        # core True` — the displayed history gone while the conversation the agent
        # actually reasons over was intact, which is the least recoverable of the four
        # possible outcomes because the user sees an empty pane and the model does not.
        # Resetting Core first means a failure leaves *both* sides untouched.
        agent = agent if agent is not None else self.get_agent(agent_id, session_id)
        if agent is not None:
            agent.reset_session(session_id)
        else:
            # No live agent, so the record is *deleted* rather than reset in place.
            # `SessionState.reset` needs the agent's `config.system_prompt` to re-seed,
            # and with no agent constructed there is nothing authoritative to read it
            # from — calling `reset()` with the empty default would persist a session
            # with no system prompt at all, which is strictly worse than no record.
            # Deleting makes the next `get_or_create_agent` seed correctly from config.
            self._core_store.delete(session_id)

        # Only now the transcript, once Core has definitely been reset.
        self._session_messages.pop(session_id, None)
        if path.exists():
            path.unlink(missing_ok=True)

    def get_agent(self, agent_id: str, session_id: str | None = None) -> BaseAgent | None:
        """Retrieve an active agent if already initialized."""
        if session_id:
            key = f"{agent_id}:{session_id}"
            if key in self._agents:
                return self._agents[key]
            if agent_id in self._agents and self._agents[agent_id].context.session_id == session_id:
                return self._agents[agent_id]
            return None
        if agent_id in self._agents:
            return self._agents[agent_id]
        def_key = f"{agent_id}:sess_{agent_id}"
        if def_key in self._agents:
            return self._agents[def_key]
        for ag in self._agents.values():
            if ag.agent_id == agent_id:
                return ag
        return None

    def list_agents(self, session_id: str | None = None) -> list[BaseAgent]:
        """Return list of unique active agents, optionally filtered by session_id."""
        if session_id:
            matching = [
                ag
                for ag in self._agents.values()
                if getattr(ag.context, "session_id", None) == session_id
            ]
            if matching:
                seen: set[str] = set()
                deduped: list[BaseAgent] = []
                for ag in matching:
                    if ag.agent_id not in seen:
                        seen.add(ag.agent_id)
                        deduped.append(ag)
                return deduped

        # Deduplicate by agent_id across all agents so multiple session instances don't duplicate
        seen_all: set[str] = set()
        deduped_all: list[BaseAgent] = []
        for ag in self._agents.values():
            if ag.agent_id not in seen_all:
                seen_all.add(ag.agent_id)
                deduped_all.append(ag)
        return deduped_all

    def reconstruct_history(
        self,
        raw_list: Sequence[object],
        system_prompt: str | None = None,
        session_id: str = "",
    ) -> tuple[list[ChatMessage], list[dict[str, Any]]]:
        """Reconstruct ChatMessage history and UI session transcript from a persisted message list.

        Preserves tool identity (`name`, `tool_call_id`, `tool_calls`) and distinguishes
        `content=None` from `content=""`.

        If a `TOOL` message lacks a tool name, this method attempts to repair it using
        positive evidence from preceding `tool_calls` or `tool_executions` matching `tool_call_id`.
        If the tool identity cannot be determined, it raises `SessionHistoryRehydrationError`
        rather than fabricating a name or handing an unmappable message to the model (P6).

        Parameters:
            raw_list: Raw list of message objects loaded from transcript JSON.
            system_prompt: Optional agent system prompt to prepend if not already seeded.
            session_id: Session identifier for diagnostic attribution on error.

        Returns:
            A tuple of (reconstructed ChatMessages for agent history, typed UI session message dicts).

        Raises:
            SessionHistoryRehydrationError: If a tool message cannot be given a valid name.
        """
        history_messages: list[ChatMessage] = []
        typed_session_msgs: list[dict[str, Any]] = []
        known_tool_calls: dict[str, str] = {}

        if system_prompt:
            history_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        for idx, item in enumerate(raw_list):
            if not isinstance(item, dict):
                continue
            item_dict: dict[str, Any] = dict(cast(dict[str, Any], item))
            typed_session_msgs.append(item_dict)

            role_str = str(item_dict.get("role", "")).strip().lower()
            if not role_str:
                raw_sender = str(item_dict.get("sender", "")).strip().lower()
                if raw_sender == "user":
                    role_str = "user"
                elif raw_sender in ("agent", "assistant"):
                    role_str = "assistant"
                elif raw_sender == "system":
                    role_str = "system"
                elif raw_sender == "tool":
                    role_str = "tool"
                elif not raw_sender:
                    raise SessionHistoryRehydrationError(
                        f"Cannot rehydrate session {session_id!r}: message at index {idx} "
                        "lacks both 'role' and 'sender' fields. Refusing the damaged "
                        "history rather than guessing who said it."
                    )
                else:
                    raise SessionHistoryRehydrationError(
                        f"Cannot rehydrate session {session_id!r}: message at index {idx} "
                        f"has unrecognized sender {raw_sender!r}. Refusing the damaged "
                        "history rather than guessing who said it."
                    )

            # A failed or cancelled turn is the page's record, not conversation: nothing
            # the agent said, so never handed back to the model as though it were (#969,
            # #1031). `cancelled` is no `MessageRole` either, so this is also what stops the
            # rebuild choking on a row that names no conversational role.
            if _records_a_turn_not_spoken(role_str, item_dict):
                continue

            # The same rule the outbound transcript is rendered with, reached through the
            # one predicate rather than restated here (#872).
            if not _is_presentable_role(role_str, bool(item_dict.get("compaction_ledger", False))):
                continue

            if role_str == "system":
                raw_c = item_dict.get("content")
                c_str = str(raw_c) if raw_c is not None else None
                history_messages.append(
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=c_str,
                        compaction_ledger=True,
                    )
                )
                continue

            try:
                role_enum = MessageRole(role_str)
            except ValueError as exc:
                raise SessionHistoryRehydrationError(
                    f"Cannot rehydrate session {session_id!r}: message at index {idx} "
                    f"has unrecognized role {role_str!r}. Refusing the damaged history "
                    "rather than guessing who said it."
                ) from exc

            # Preserve content fidelity: distinguish absent field / None from empty string ""
            if "content" not in item_dict or item_dict["content"] is None:
                content_str = None
            else:
                content_str = str(item_dict["content"])

            if role_enum == MessageRole.USER:
                user_name = item_dict.get("name")
                name_str = (
                    str(user_name).strip()
                    if user_name is not None and str(user_name).strip()
                    else None
                )
                history_messages.append(
                    ChatMessage(role=role_enum, content=content_str, name=name_str)
                )

            elif role_enum == MessageRole.ASSISTANT:
                reconstructed_tool_calls: list[ToolCallRequest] = []
                raw_tc_obj: object = item_dict.get("tool_calls")
                has_tc_list = isinstance(raw_tc_obj, list)
                if has_tc_list:
                    raw_tc_list: list[object] = cast(list[object], raw_tc_obj)
                    for tc_item in raw_tc_list:
                        if isinstance(tc_item, dict):
                            tc_dict: dict[str, Any] = cast(dict[str, Any], tc_item)
                            if "id" in tc_dict and "name" in tc_dict:
                                tc_id = str(tc_dict["id"])
                                tc_name = str(tc_dict["name"]).strip()
                                raw_args: object = tc_dict.get("arguments", {})
                                tc_args: dict[str, Any] = (
                                    cast(dict[str, Any], raw_args)
                                    if isinstance(raw_args, dict)
                                    else {}
                                )
                                reconstructed_tool_calls.append(
                                    ToolCallRequest(id=tc_id, name=tc_name, arguments=tc_args)
                                )
                                known_tool_calls[tc_id] = tc_name

                raw_te_obj: object = item_dict.get("tool_executions")
                if isinstance(raw_te_obj, list):
                    raw_te_list: list[object] = cast(list[object], raw_te_obj)
                    for te_item in raw_te_list:
                        if isinstance(te_item, dict):
                            te_dict: dict[str, Any] = cast(dict[str, Any], te_item)
                            te_id_val: object = te_dict.get("tool_call_id")
                            te_name_val: object = te_dict.get("tool_name")
                            if te_id_val is not None and te_name_val is not None:
                                te_id = str(te_id_val).strip()
                                te_name = str(te_name_val).strip()
                                if te_id and te_name:
                                    known_tool_calls[te_id] = te_name
                                if not has_tc_list and te_id and te_name:
                                    raw_te_args: object = te_dict.get("arguments", {})
                                    te_args: dict[str, Any] = (
                                        cast(dict[str, Any], raw_te_args)
                                        if isinstance(raw_te_args, dict)
                                        else {}
                                    )
                                    reconstructed_tool_calls.append(
                                        ToolCallRequest(
                                            id=te_id,
                                            name=te_name,
                                            arguments=te_args,
                                        )
                                    )

                history_messages.append(
                    ChatMessage(
                        role=role_enum,
                        content=content_str,
                        tool_calls=tuple(reconstructed_tool_calls),
                    )
                )

            elif role_enum == MessageRole.TOOL:
                raw_name_obj: object = item_dict.get("name") or item_dict.get("tool_name")
                tool_name: str | None = (
                    str(raw_name_obj).strip()
                    if raw_name_obj is not None and str(raw_name_obj).strip()
                    else None
                )

                raw_call_id_obj: object = item_dict.get("tool_call_id") or item_dict.get("tool_id")
                tool_call_id: str | None = (
                    str(raw_call_id_obj).strip()
                    if raw_call_id_obj is not None and str(raw_call_id_obj).strip()
                    else None
                )

                # Attempt repair if tool_name is absent/blank but tool_call_id is known
                is_inferred = False
                if not tool_name and tool_call_id and tool_call_id in known_tool_calls:
                    tool_name = known_tool_calls[tool_call_id]
                    is_inferred = True

                if not tool_name:
                    raise SessionHistoryRehydrationError(
                        f"Cannot rehydrate session {session_id!r}: message at index {idx} "
                        "has role 'tool' but lacks tool identity ('name' is missing or blank) "
                        "and cannot be repaired from preceding tool calls. Refusing damaged "
                        "history rather than fabricating a tool name or passing an unmappable "
                        "message to the model."
                    )

                item_dict["name"] = tool_name
                if is_inferred or bool(item_dict.get("name_inferred")):
                    item_dict["name_inferred"] = True

                history_messages.append(
                    ChatMessage(
                        role=role_enum,
                        content=content_str,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )

        return history_messages, typed_session_msgs

    def memory_for(self, agent_id: str) -> CrossSessionMemory:
        """The one cross-session memory store for `agent_id`, created on first use.

        Public, and the only such map on the manager that serves the head: chat sessions
        are not the only seat an agent takes. `RoomStack` seats the same ids in rooms and
        must reach *this* map rather than keep one of its own. `CrossSessionMemory.save()` rewrites
        the whole document, so a second map keyed the same way would be a second
        whole-document writer over one file, and each side would silently drop the facts
        the other recorded (P6). The ids collide by design: an install's agents are both
        its chat agents and the seats a conversation puts them in.

        Get-or-create has no `await` between the read and the write, so two coroutines
        cannot race a second store into being; `get_or_create_agent` additionally calls
        this under `self._lock`.
        """
        existing = self._agent_memories.get(agent_id)
        if existing is not None:
            return existing
        store = default_cross_session_memory(agent_id)
        self._agent_memories[agent_id] = store
        return store

    async def get_or_create_agent(
        self,
        agent_id: str,
        session_id: str | None = None,
        system_prompt: str | None = None,
        model_name: str | None = None,
        fallback_to_mock: bool | None = None,
    ) -> BaseAgent:
        """Get existing agent or instantiate and start a new BaseAgent hydrated from session state."""
        effective_session_id = session_id or f"sess_{agent_id}"
        agent_key = f"{agent_id}:{effective_session_id}"

        async with self._lock:
            if agent_key in self._agents:
                return self._agents[agent_key]
            if session_id is None and agent_id in self._agents:
                return self._agents[agent_id]
            if (
                agent_id in self._agents
                and self._agents[agent_id].context.session_id == effective_session_id
            ):
                return self._agents[agent_id]

            self._adopt_saved_choice()
            use_fallback = (
                fallback_to_mock if fallback_to_mock is not None else self._fallback_to_mock
            )
            llm = self._llm or create_llm_connector(
                provider=self._configured_provider or None,
                api_key=self._api_key_for(self._configured_provider),
                base_url=self._configured_base_url or None,
                fallback_to_mock=use_fallback,
            )
            from uclone_x.agent.bootstrap import agent_config_for_persona
            from uclone_x.agent.models import DEFAULT_SYSTEM_PROMPT
            from uclone_x.agent.persona_registry import get_default_persona_registry
            from uclone_x.tools.scoped_registry import ScopedToolRegistry

            effective_target_model = model_name or self._configured_model or None
            # The tool inventory is passed so `allowed_tools` is checked against what is
            # actually registered -- MCP-provided names included, which is why it is read
            # from the live registry here rather than from a constant.
            persona_reg = get_default_persona_registry(
                self._workspace_dir,
                tool_names=[tool.name for tool in self._tools.list_tools()],
            )
            persona_def = persona_reg.get_persona(agent_id)

            if persona_def is not None:
                # The persona's tools are resolved by the agent, not copied in here, so an
                # edit `apply_persona` hands it reaches them (#892); the rule is kept, with
                # the room's, in `agent_config_for_persona` (#1448).
                config = agent_config_for_persona(
                    persona_def,
                    agent_id=agent_id,
                    system_prompt=system_prompt or persona_def.system_prompt,
                )
            else:
                default_prompt = system_prompt or (
                    f"You are {agent_id}, a specialized UClone-X autonomous agent assistant. "
                    f"You collaborate with the user, execute tools, and maintain rigorous accuracy.\n\n{DEFAULT_SYSTEM_PROMPT}"
                )
                config = AgentConfig(
                    agent_id=agent_id,
                    name=agent_id,
                    system_prompt=default_prompt,
                    llm_config=AgentLLMConfig(
                        model_name=effective_target_model,
                        temperature=0.7,
                        max_tokens=2048,
                    ),
                )

            if (
                effective_target_model is not None
                and config.llm_config.model_name != effective_target_model
            ):
                updated_llm_cfg = config.llm_config.model_copy(
                    update={"model_name": effective_target_model}
                )
                config = config.model_copy(update={"llm_config": updated_llm_cfg})
            config = config.model_copy(update={"read_roots": self.read_roots})
            context = AgentContext(
                session_id=effective_session_id,
                agent_id=agent_id,
                current_state=AgentState.IDLE,
                # Every agent this manager creates was asked for directly, so none of
                # them has a parent. This read `"champion" if agent_id != "champion"`,
                # which drew an edge on `/api/agents/graph` from an agent that had
                # delegated nothing to an agent it had never heard of -- a hierarchy
                # invented from the id. A real subagent gets its parent from the agent
                # that spawned it (`BaseAgent`, `parent_agent_id=self.agent_id`).
                parent_agent_id=None,
                workspace_root=self._workspace_dir,
            )
            from uclone_x.agent.composition import HostDependencies, compose_agent

            # For a persona agent `config.allowed_tools` is now empty, so the agent's own
            # scope (advertised list, mid-turn refusal, direct-call refusal) is what enforces
            # the persona's tools; a proxy fixed here would outlive an edit to them.
            agent_tools = (
                ScopedToolRegistry(backing=self._tools, allowed_tools=config.allowed_tools)
                if config.allowed_tools
                else self._tools
            )
            host = HostDependencies(
                bus=self._bus,
                llm=llm,
                tools=agent_tools,
                tracer=self._tracer,
                store=self._core_store,
                budget=self._budget_tracker,
                skills=self._skill_registry,
                ontology=self._ontology_engine,
                # One store per agent id, shared across that agent's sessions: the point
                # of the facts is that they outlive the session that recorded them, and
                # two sessions holding two stores over one file lose each other's writes.
                memory=self.memory_for(agent_id),
            )
            agent = compose_agent(config=config, host=host, context=context)

            # P8: the Core session record is the primary source of the conversation. Try
            # it first; only fall back to reconstructing `ChatMessage`s from the UI's
            # presentation transcript, which is what this method used to do
            # unconditionally. That reconstruction is conversation-rehydration logic
            # living in the UI, which P8 forbids, and it is kept solely so installs with
            # an existing transcript and no Core record still resume. See #183 for the
            # follow-up that retires it.
            if agent.hydrate_session(effective_session_id) is not None:
                record = self.load_session_record(effective_session_id)
                if record is not None:
                    raw_cached: object = record.get("messages", [])
                    if isinstance(raw_cached, list):
                        cached_list: list[object] = cast(list[object], raw_cached)
                        self._session_messages[effective_session_id] = [
                            cast(dict[str, Any], item)
                            for item in cached_list
                            if isinstance(item, dict)
                        ]
                await agent.start()
                self._agents[agent_key] = agent
                if agent_id not in self._agents:
                    self._agents[agent_id] = agent
                return agent

            # Legacy path: reconstruct history from the UI transcript.
            record = self.load_session_record(effective_session_id)
            if record is not None and "messages" in record:
                raw_msgs: object = record.get("messages", [])
                if isinstance(raw_msgs, list):
                    raw_list: list[object] = cast(list[object], raw_msgs)
                    history_messages, typed_session_msgs = self.reconstruct_history(
                        raw_list=raw_list,
                        system_prompt=config.system_prompt,
                        session_id=effective_session_id,
                    )
                    turns = int(str(record.get("turns", 0)))
                    agent.load_history(history_messages, turn_counter=turns)
                    self._session_messages[effective_session_id] = list(typed_session_msgs)

            if effective_session_id not in self._session_messages:
                self._session_messages[effective_session_id] = []

            await agent.start()
            self._agents[agent_key] = agent
            if agent_id not in self._agents:
                self._agents[agent_id] = agent
            return agent

    async def stop_agent(self, agent_id: str, session_id: str | None = None) -> None:
        """Stop and remove a specific agent."""
        async with self._lock:
            if session_id:
                agent = self._agents.pop(f"{agent_id}:{session_id}", None)
                if agent is not None:
                    await agent.stop()
                if (
                    agent_id in self._agents
                    and self._agents[agent_id].context.session_id == session_id
                ):
                    self._agents.pop(agent_id, None)
            else:
                agent = self._agents.pop(agent_id, None)
                if agent is not None:
                    await agent.stop()
                for k in list(self._agents.keys()):
                    if k.startswith(f"{agent_id}:"):
                        ag = self._agents.pop(k)
                        await ag.stop()

    async def clear(self) -> None:
        """Stop and clear all active agents and in-memory session caches."""
        async with self._lock:
            for agent in list(dict.fromkeys(self._agents.values())):
                await agent.stop()
            self._agents.clear()
            self._session_messages.clear()

    def _eval_read_error(self, exc: Exception) -> str:
        """Name the reports directory and the exception, for a head to show in words."""
        return (
            f"Could not read evaluation reports from {self._eval_reports_dir}: "
            f"{type(exc).__name__}: {exc}"
        )

    def get_latest_evaluations(self) -> dict[str, Any]:
        """Load latest evaluation scorecard, suite reports, and aggregated metrics.

        `status` says why the scorecard is what it is, because an empty scorecard is
        three different facts (P6): `no_backend` (this runtime ships no evaluation
        suites), `empty` (suites exist and none has been run), and `error` (the reports
        could not be read, with `error` naming the directory and the exception). The
        failure is carried in a 200 body rather than an HTTP error because the dashboard
        drops a non-ok response and would render it as the empty scorecard again.
        """

        def empty_response(status: str, error: str | None = None) -> dict[str, Any]:
            return {
                "status": status,
                "error": error,
                "suites": [],
                "scorecard": {},
                "metrics": {
                    "total_suites": 0,
                    "total_probes": 0,
                    "passed_probes": 0,
                    "failed_probes": 0,
                    "pass_rate": 0.0,
                },
            }

        try:
            runner = create_eval_runner(reports_dir=self._eval_reports_dir)
            scorecard_map = runner.get_latest_scorecard()
        except EvalBackendUnavailableError:
            # A runtime shipped without suites has no evaluations, which is a
            # normal state for the dashboard rather than a failure to log loudly.
            logger.debug("No evaluation backend installed; reporting an empty scorecard.")
            return empty_response("no_backend")
        except Exception as exc:
            logger.warning(
                "Failed to load latest evaluations from %s: %s", self._eval_reports_dir, exc
            )
            return empty_response("error", self._eval_read_error(exc))

        if not scorecard_map:
            return empty_response("empty")

        suites: list[dict[str, Any]] = [
            report.model_dump() for _, report in sorted(scorecard_map.items())
        ]
        scorecard: dict[str, Any] = {
            suite_name: report.model_dump() for suite_name, report in sorted(scorecard_map.items())
        }

        total_probes = sum(int(s["summary"]["total_probes"]) for s in suites)
        passed_probes = sum(int(s["summary"]["passed_probes"]) for s in suites)
        failed_probes = sum(int(s["summary"]["failed_probes"]) for s in suites)
        pass_rate = (passed_probes / total_probes) if total_probes > 0 else 0.0

        return {
            "status": "ok",
            "error": None,
            "suites": suites,
            "scorecard": scorecard,
            "metrics": {
                "total_suites": len(suites),
                "total_probes": total_probes,
                "passed_probes": passed_probes,
                "failed_probes": failed_probes,
                "pass_rate": round(pass_rate, 4),
            },
        }

    def get_evaluation_history(
        self, suite: str | None = None, limit: int | None = None
    ) -> dict[str, Any]:
        """Load historical evaluation reports, optionally filtered by suite identifier.

        `status` and `error` follow `get_latest_evaluations`.
        """
        try:
            runner = create_eval_runner(reports_dir=self._eval_reports_dir)
            reports = runner.get_reports()
        except EvalBackendUnavailableError:
            logger.debug("No evaluation backend installed; reporting an empty history.")
            return {"status": "no_backend", "error": None, "reports": [], "total": 0}
        except Exception as exc:
            logger.warning(
                "Failed to load evaluation history from %s: %s", self._eval_reports_dir, exc
            )
            return {
                "status": "error",
                "error": self._eval_read_error(exc),
                "reports": [],
                "total": 0,
            }

        if suite is not None and suite.strip():
            target_suite = suite.strip()
            reports = [r for r in reports if r.suite == target_suite]

        reports.sort(key=lambda r: r.timestamp, reverse=True)

        if limit is not None and limit > 0:
            reports = reports[:limit]

        return {
            "status": "ok" if reports else "empty",
            "error": None,
            "reports": [r.model_dump() for r in reports],
            "total": len(reports),
        }

    def list_artifacts(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Enumerate the documents and images the clones generated (RFC §6.1).

        Only the tool-output directories are read. The workspace's own `docs/` and root-level
        Markdown are not artifacts: a clone did not produce them, and listing them put the
        repository's internal playbooks in the Docs & Artifacts tab as if a clone had.
        """
        results: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        root = self._workspace_dir
        if not root.is_dir():
            return results

        SUPPORTED_SUFFIXES = {".md", ".png", ".jpg", ".jpeg", ".webp", ".svg"}

        def _add_file(p: Path) -> None:
            if not p.is_file() or p.suffix.lower() not in SUPPORTED_SUFFIXES:
                return
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                return
            if rel in seen_paths:
                return
            seen_paths.add(rel)
            try:
                st = p.stat()
                is_img = p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
                title = p.name if is_img else _extract_markdown_title(p)
                results.append(
                    {
                        "id": f"art_{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:12]}",
                        "path": rel,
                        "name": p.name,
                        "title": title,
                        "type": "image" if is_img else "document",
                        "created_at": datetime.fromtimestamp(st.st_ctime, UTC).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "modified_at": datetime.fromtimestamp(st.st_mtime, UTC).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        ),
                        "size_bytes": st.st_size,
                    }
                )
            except Exception:
                pass

        # 1. Session-specific tool output artifacts if session_id provided
        if session_id and session_id.strip():
            clean_sid = session_id.strip()
            session_tool_dir = root / ".sandbox" / "tool_artifacts" / clean_sid
            if session_tool_dir.is_dir():
                for p in sorted(session_tool_dir.rglob("*")):
                    _add_file(p)
            session_custom_dir = root / "artifacts" / clean_sid
            if session_custom_dir.is_dir():
                for p in sorted(session_custom_dir.rglob("*")):
                    _add_file(p)

        # 2. Workspace artifacts directory. Not session-scoped yet: the image tool writes
        # `artifacts/images/img_<seed>_<sid[:6]>.png`, outside any per-session directory.
        artifacts_dir = root / "artifacts"
        if artifacts_dir.is_dir():
            for p in sorted(artifacts_dir.rglob("*")):
                _add_file(p)

        results.sort(key=lambda x: str(x.get("modified_at", "")), reverse=True)
        return results

    def get_artifact_file(self, path: str, session_id: str | None = None) -> tuple[Path, str]:
        """Safely retrieve artifact Path and detected MIME type from workspace (P6)."""
        if not path or not path.strip():
            raise PathTraversalError("Path must not be empty.")
        clean_path = path.strip()
        if "\0" in clean_path:
            raise PathTraversalError("Null byte detected in path.")

        validator = PathValidator()
        resolved = validator.resolve_safe_path(Path(clean_path), self._workspace_dir)

        if not resolved.is_file():
            raise FileNotFoundError(f"Artifact not found: {clean_path}")

        ext = resolved.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
            ".md": "text/markdown; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".json": "application/json",
        }
        mime = mime_map.get(ext, "application/octet-stream")
        return resolved, mime

    def get_artifact_content(self, path: str, session_id: str | None = None) -> str:
        """Safely retrieve raw markdown content from workspace, strictly rejecting path traversal (P6)."""
        resolved, _ = self.get_artifact_file(path=path, session_id=session_id)
        return resolved.read_text(encoding="utf-8")

    def get_knowledge_graph(
        self, session_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        """Return dynamic entity-relation triples (subject, predicate, object, provenance, tier) (RFC §6.1)."""
        engine = self._ontology_engine
        if agent_id and agent_id in self._agents:
            agent_inst = self._agents[agent_id]
            agent_onto = getattr(agent_inst, "ontology", None)
            if agent_onto is not None:
                engine = agent_onto  # pyright: ignore

        return knowledge_graph(engine, session_id=session_id, agent_id=agent_id)


_PERSONA_DRAFT_TIMEOUT_S = 120.0

_PERSONA_DRAFT_SYSTEM = (
    "You write the instructions (system prompt) for an AI assistant that a person is "
    "setting up. From the name, role, description and tools you are given, write clear "
    "second-person instructions: who the assistant is, what it helps with, how it should "
    "approach that work, its tone, and what it should avoid. Use short sections or bullet "
    "points, and keep it under 300 words. Write in the same language as the description "
    "(or the role, or the name, if there is no description). Do not invent facts about "
    "the person. Reply with the instructions only: no preamble, no explanation, no code fence."
)


def _persona_draft_request(name: str, role: str, description: str, tools: list[str]) -> str:
    """The user turn for a persona draft: only the fields the person filled in."""
    lines = [f"Name: {name}" if name else "", f"Role: {role}" if role else ""]
    lines.append(f"Description: {description}" if description else "")
    lines.append(f"Tools it can use: {', '.join(tools)}" if tools else "")
    return "\n".join(line for line in lines if line)


def _strip_code_fence(text: str) -> str:
    """Remove one enclosing Markdown code fence a model wraps its whole reply in."""
    stripped = text.strip()
    match = re.fullmatch(r"```[\w-]*\n(.*?)\n?```", stripped, flags=re.DOTALL)
    return (match.group(1) if match else stripped).strip()


def _template_persona_prompt(name: str, role: str, description: str, tools: list[str]) -> str:
    """The fixed draft used when no model answers: the fields slotted into a template."""
    title = name.replace("_", " ").replace("-", " ").title() if name else "Agent"
    role_text = role or "Specialized AI Assistant"

    lines = [
        f"You are {title}, the {role_text} in UClone-X.",
    ]
    if description:
        lines.append(f"- Mission & Scope: {description}")
    else:
        lines.append(f"- Mission & Scope: Faithfully fulfill duties as {role_text}.")

    lines.extend(
        [
            "- Direct Execution: Provide precise, proactive, and structured responses. Prioritize actionable outcomes and substantive analysis over generic boilerplate.",
            "- Transparency: Clearly explain reasoning and trade-offs when making recommendations.",
        ]
    )

    # Add domain-specific directives based on allowed tools
    tool_set = set(tools)
    if "generate_image" in tool_set:
        lines.append(
            "- Visual Generation: Translate visual concepts and scenes into descriptive image generation prompts and invoke generate_image directly."
        )
    if "file_edit" in tool_set or "file_write" in tool_set:
        lines.append(
            "- Safe File Operations: Ensure careful validation and atomic edits when modifying code or workspace files."
        )
    if "bash_run" in tool_set or "run_command" in tool_set:
        lines.append(
            "- Command Execution: Execute terminal commands cautiously and verify environment state before making irreversible changes."
        )
    if "web_search" in tool_set or "web_fetch" in tool_set:
        lines.append(
            "- Grounded Information: Search and cite reliable sources when researching recent or external information."
        )
    if "delegate_subagent" in tool_set:
        lines.append(
            "- Subagent Delegation: Decompose complex workflows into modular tasks and orchestrate subagents effectively."
        )

    lines.append(
        "- Language: Always communicate naturally in the language in which the user addresses you."
    )
    return "\n".join(lines)


def get_ui_session_manager(
    storage_dir: Path | None = None,
    eval_reports_dir: Path | None = None,
) -> AgentSessionManager:
    """Retrieve or lazily initialize the shared UI session manager."""
    global _ui_session_mgr
    if _ui_session_mgr is None:
        _ui_session_mgr = AgentSessionManager(
            storage_dir=storage_dir,
            eval_reports_dir=eval_reports_dir,
        )
    return _ui_session_mgr


def create_ui_app(
    static_dir: Path | None = None,
    bus: EventBus | None = None,
    llm: LLMProviderProtocol | None = None,
    tools: ToolRegistryProtocol | None = None,
    tracer: TelemetryTracer | None = None,
    session_manager: AgentSessionManager | None = None,
    fallback_to_mock: bool = False,
    storage_dir: Path | None = None,
    ontology_engine: OntologyEngine | None = None,
    skill_registry: SkillRegistry | None = None,
    budget_tracker: TokenBudgetManager | None = None,
    eval_reports_dir: Path | None = None,
    workspace_dir: Path | None = None,
    shutdown_event: asyncio.Event | None = None,
    bind_host: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI developer dashboard application.

    `bind_host` is the address the server listens on. While it is a loopback address,
    every request must name this machine by a loopback name (`LoopbackHostGuard`). When
    omitted it is read from `UI_BIND_HOST_ENV_VAR`, and failing that assumed to be
    `DEFAULT_UI_BIND_HOST`.
    """
    active_bus = bus if bus is not None else get_ui_event_bus()
    active_tracer = tracer if tracer is not None else get_ui_tracer()
    active_shutdown_event = shutdown_event if shutdown_event is not None else asyncio.Event()
    session_mgr = (
        session_manager
        if session_manager is not None
        else AgentSessionManager(
            bus=active_bus,
            llm=llm,
            tools=tools,
            tracer=active_tracer,
            fallback_to_mock=fallback_to_mock,
            storage_dir=storage_dir,
            ontology_engine=ontology_engine,
            skill_registry=skill_registry,
            budget_tracker=budget_tracker,
            eval_reports_dir=eval_reports_dir,
            workspace_dir=workspace_dir,
        )
    )
    if eval_reports_dir is not None and session_manager is not None:
        session_mgr.eval_reports_dir = eval_reports_dir

    # Per app, like the session manager: the user's external MCP servers, whose tools are
    # registered into the same registry every clone's turn reads its tools from.
    mcp_manager = MCPServerManager(
        registry=session_mgr.tools,
        config_path=session_mgr.storage_dir / "mcp_servers.json",
        workspace_root=session_mgr.workspace_dir,
    )

    @asynccontextmanager
    async def _app_lifespan(app_inst: FastAPI) -> AsyncGenerator[None, None]:
        # In the background: a local server launched through `npx` may spend a minute
        # downloading, and the dashboard must not wait on it to open. Until it answers,
        # the server reads as "connecting".
        mcp_start = asyncio.create_task(mcp_manager.start())
        yield
        mcp_start.cancel()
        with contextlib.suppress(BaseException):
            await mcp_start
        # Local servers are child processes; left running they outlive the app.
        await mcp_manager.close()
        evt: asyncio.Event | None = getattr(app_inst.state, "shutdown_event", None)
        if evt is not None:
            evt.set()
        stack = getattr(app_inst.state, "room_stack", None)
        if stack is not None:
            # Cancelled and awaited, not abandoned: a turn killed at interpreter teardown
            # between building its message and saving it spends the turn, pays for the
            # tokens, and records nothing.
            await stack.close()

    app = FastAPI(
        title="UClone-X Swarm Explorer API",
        version=__version__,
        description="Embedded real-time dashboard API for UClone-X event-driven agent runtime",
        lifespan=_app_lifespan,
    )
    app.state.session_manager = session_mgr
    app.state.bus = active_bus
    app.state.tracer = active_tracer
    app.state.eval_reports_dir = session_mgr.eval_reports_dir
    app.state.shutdown_event = active_shutdown_event
    app.state.mcp_manager = mcp_manager

    # Per app, not per module: two `create_ui_app` calls in one process — which is
    # every test session — must not share a model's in-flight pull (#1233). Also on
    # `app.state` so it can be read from outside the route's closure.
    pull_single_flight = SingleFlight()
    app.state.pull_single_flight = pull_single_flight

    # Enable CORS for local Vite dev server (HMR on port 5173 / 5180)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Added after CORS, so it is outermost: a refused request reaches nothing else.
    resolved_bind_host = (
        bind_host
        if bind_host is not None
        else os.environ.get(UI_BIND_HOST_ENV_VAR, DEFAULT_UI_BIND_HOST)
    )
    if _is_loopback_bind(resolved_bind_host):
        app.add_middleware(LoopbackHostGuard)

    from uclone_x.ui.rooms import RoomStack, register_room_routes

    room_stack = RoomStack(session_mgr)
    app.state.room_stack = room_stack
    register_room_routes(app, room_stack)

    from uclone_x.ui.room_dock import register_room_dock_routes

    # The dock's reads for the conversation on screen and its selected seat (#1353-#1357).
    register_room_dock_routes(app, room_stack)

    # `/api/agents` below lists live instances; this lists what is *installed*, which on a
    # fresh install is the only one of the two with anything in it (#1190). It is handed
    # the room stack as well as the session manager because a clone is running in either
    # seat, and under D1 the conversation seat is the ordinary one.
    from uclone_x.ui.clones import register_clone_routes

    register_clone_routes(app, session_mgr, room_stack)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Health check and absorbed failure accounting endpoint (#198)."""
        live_agents = session_mgr.list_agents()
        agent_errors: dict[str, list[str]] = {
            ag.agent_id: [str(err) for err in ag.processing_errors]
            for ag in live_agents
            if ag.processing_errors
        }
        total_agent_errors = sum(len(errs) for errs in agent_errors.values())

        return {
            "status": "healthy" if total_agent_errors == 0 else "degraded",
            "version": __version__,
            "git_commit": get_git_commit(),
            "started_at": SERVER_START_TIME,
            "runtime": "uclone_x",
            "protocol_version": "a2a-v1",
            "event_bus_mode": "in_memory_fastpath",
            "absorbed_failures": {
                "dropped_spans": {
                    "count": active_tracer.dropped_span_count,
                    "reasons": dict(active_tracer.drop_reasons),
                    "undelivered": active_tracer.undelivered_span_count,
                },
                "event_bus_drops": {
                    "count": active_bus.dropped_event_count,
                    "reasons": dict(active_bus.drop_reasons),
                },
                "agent_processing_errors": {
                    "count": total_agent_errors,
                    "agents": agent_errors,
                },
            },
        }

    @app.get("/api/diagnostics")
    async def diagnostics() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Operator diagnostics endpoint exposing system telemetry and absorbed failure metrics (#198)."""
        live_agents = session_mgr.list_agents()
        agent_errors: dict[str, list[str]] = {
            ag.agent_id: [str(err) for err in ag.processing_errors]
            for ag in live_agents
            if ag.processing_errors
        }
        total_agent_errors = sum(len(errs) for errs in agent_errors.values())

        return {
            "version": __version__,
            "git_commit": get_git_commit(),
            "started_at": SERVER_START_TIME,
            "runtime": "uclone_x",
            "agents": {
                "total": len(live_agents),
                "active_states": {ag.agent_id: ag.state.value for ag in live_agents},
            },
            "absorbed_failures": {
                "dropped_spans": {
                    "count": active_tracer.dropped_span_count,
                    "reasons": dict(active_tracer.drop_reasons),
                    "undelivered": active_tracer.undelivered_span_count,
                },
                "event_bus_drops": {
                    "count": active_bus.dropped_event_count,
                    "reasons": dict(active_bus.drop_reasons),
                },
                "agent_processing_errors": {
                    "count": total_agent_errors,
                    "agents": agent_errors,
                },
            },
            "event_bus": {
                "qsize": active_bus.qsize(),
                "maxsize": active_bus.maxsize,
                "backpressure_policy": active_bus.backpressure_policy.value,
                "error_count": active_bus.error_count,
            },
        }

    @app.get("/api/settings")
    async def get_settings(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Retrieve current active endpoints, configurations, and available providers."""
        _refuse_cross_origin(request)  # names the folders clones can read, and the workspace
        settings = session_mgr.get_settings()
        models = await fetch_available_models(
            provider=str(settings.get("llm_provider", "ollama")),
            base_url=cast(str | None, settings.get("llm_base_url")) or None,
        )
        settings["available_models"] = models
        return settings

    @app.get("/api/models")
    async def get_models() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Enumerate installed/available models for active provider (P0/Recognition over Recall)."""
        settings = session_mgr.get_settings()
        provider = str(settings.get("llm_provider", "ollama"))
        base_url = cast(str | None, settings.get("llm_base_url")) or None
        models = await fetch_available_models(provider=provider, base_url=base_url)
        return {
            "provider": provider,
            "models": models,
            "current_model": settings.get("llm_model", ""),
        }

    @app.post("/api/models/pull")
    async def pull_model_route(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Install a model onto the active Ollama daemon (blocking; UI shows a spinner).

        Refused cross-origin for the same reason the diagnostics routes are (see
        `_refuse_cross_origin`): this app is served with `allow_origins=["*"]` and
        `allow_credentials=True`, so without the check any page open in the same
        browser while `ucx ui` runs could make this host download weights of the
        page's choosing — Ollama's `/api/pull` accepts `hf.co/<owner>/<repo>:<quant>`
        — which then appear in the model picker as if the user had installed them.
        The frontend's confirmation is client-side and not in that path.

        `model` is still checked only for being non-empty, and deliberately so: it
        is JSON-encoded into the body of a request to the daemon, never a path
        segment or a shell word, so there is no injection sink to narrow. A
        registry allowlist would remove the working `hf.co/...` capability, which
        is a product decision rather than a fix for this hole.

        **The pull is bounded three ways (#1233, #1243).** `pull_model` consumes
        Ollama's NDJSON progress and gives up on *silence between lines*, so a slow
        download is never killed and a wedged one dies in a couple of minutes rather
        than a quarter of an hour; reaching that answers 504 rather than 502 so the
        surface can say which of "the daemon refused" and "the daemon went quiet"
        happened. `pull_single_flight` collapses concurrent requests for the same
        model onto one download, and `joined` says which side of that a caller landed
        on. Neither is the user's cancel button: that is the dashboard's
        `AbortController`, and a request it abandons leaves the shared run alone for
        whoever is still waiting on it.

        **The deadline belongs to the download, not to the waiter.** It is measured
        inside the shielded single-flight run, from when the download started — so a
        caller that joins a pull already 90 seconds into a stall inherits what is left
        of that stall's window rather than starting a fresh one, and a caller that
        aborts takes no deadline away with it. A per-caller deadline would be the
        other thing and would be wrong here: it would let a stalled download be kept
        alive indefinitely by a stream of new joiners.
        """
        _refuse_cross_origin(request)  # a page in another tab must not install weights
        model = str(req.get("model", "")).strip()
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        settings = session_mgr.get_settings()
        base_url = cast(str | None, settings.get("llm_base_url")) or None

        def pull() -> Coroutine[Any, Any, None]:
            return pull_model(model, base_url=base_url)

        try:
            started = await pull_single_flight.run(model, pull)
        except LLMTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"{exc} Stopped waiting; Ollama may still have work in flight. "
                    f"To watch it directly, with a progress bar and no ceiling of any "
                    f"kind, run: ucx llm pull {model}"
                ),
            ) from exc
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"status": "ok", "model": model, "joined": not started}

    @app.post("/api/models/delete")
    async def delete_model_route(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Remove a model from the active Ollama daemon.

        Refused cross-origin, as `pull_model_route` is and for the same reason:
        `ucx ui` runs on loopback with no authentication, so an unguarded route
        lets any page open in another tab delete the user's local weights with one
        `fetch`. `window.confirm` is client-side and not in that path.
        """
        _refuse_cross_origin(request)  # a page in another tab must not delete weights
        model = str(req.get("model", "")).strip()
        if not model:
            raise HTTPException(status_code=400, detail="model is required")
        settings = session_mgr.get_settings()
        base_url = cast(str | None, settings.get("llm_base_url")) or None
        try:
            await delete_model(model, base_url=base_url)
        except LLMProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"status": "ok", "model": model}

    @app.post("/api/settings")
    async def update_settings(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Update configurations, hot-reload runtime connectors, and broadcast settings.updated event."""
        _refuse_cross_origin(request)  # another tab must not widen read_roots
        try:
            updated = session_mgr.update_settings(
                llm_provider=req.get("llm_provider"),
                llm_base_url=req.get("llm_base_url"),
                llm_api_key=req.get("llm_api_key"),
                llm_model=req.get("llm_model"),
                comfyui_base_url=req.get("comfyui_base_url"),
                read_roots=req.get("read_roots"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Broadcast settings.updated event across EventBus (Principle 1 & 6)
        sys_pub = active_bus.register_publisher(
            sender_id="ui_settings",
            source=EventSource.SYSTEM,
        )
        await sys_pub.publish(
            AgentEvent(
                type=EventType.SETTINGS_UPDATED,
                recipient_id="*",
                topic="settings",
                payload=dict(updated),
            )
        )
        return updated

    def _mcp_payload() -> dict[str, Any]:
        return {
            "config_path": str(mcp_manager.config_path),
            "load_error": mcp_manager.load_error,
            "servers": mcp_manager.views(),
        }

    def _mcp_spec_from_request(req: dict[str, Any]) -> MCPServerSpec:
        try:
            return MCPServerSpec.model_validate(
                {
                    "name": req.get("name"),
                    "transport": req.get("transport"),
                    "url": req.get("url") or None,
                    "headers": req.get("headers") or {},
                    "command": req.get("command") or None,
                    "args": tuple(cast(list[str], req.get("args") or [])),
                    "env": req.get("env") or {},
                }
            )
        except ValidationError as exc:
            first = exc.errors()[0]
            detail = str(first.get("msg", exc)).removeprefix("Value error, ")
            raise HTTPException(status_code=400, detail=detail) from exc

    @app.get("/api/mcp/servers")
    async def list_mcp_servers(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The user's external MCP servers and their state. Credential values are never sent."""
        _refuse_unless_local(request)
        return _mcp_payload()

    @app.post("/api/mcp/servers", status_code=201)
    async def add_mcp_server(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Add a server and connect it. It is kept even when the first connection fails."""
        _refuse_unless_local(request)  # another tab must not start a local process
        spec = _mcp_spec_from_request(req)
        try:
            return await mcp_manager.add(spec)
        except DuplicateServerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/mcp/servers/import")
    async def import_mcp_servers(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Add every server in a pasted `mcpServers` JSON snippet."""
        _refuse_unless_local(request)
        try:
            added, skipped = await mcp_manager.import_json(str(req.get("json", "")))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"added": added, "skipped": skipped}

    @app.delete("/api/mcp/servers/{name}")
    async def remove_mcp_server(request: Request, name: str) -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        _refuse_unless_local(request)
        try:
            await mcp_manager.remove(name)
        except UnknownServerError as exc:
            raise HTTPException(status_code=404, detail=f"No server named '{name}'") from exc
        return {"status": "ok"}

    @app.post("/api/mcp/servers/{name}/reconnect")
    async def reconnect_mcp_server(request: Request, name: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _refuse_unless_local(request)
        try:
            return await mcp_manager.reconnect(name)
        except UnknownServerError as exc:
            raise HTTPException(status_code=404, detail=f"No server named '{name}'") from exc

    @app.post("/api/mcp/servers/{name}/enabled")
    async def set_mcp_server_enabled(  # pyright: ignore[reportUnusedFunction]
        request: Request, name: str, req: dict[str, Any]
    ) -> dict[str, Any]:
        _refuse_unless_local(request)
        enabled = req.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=400, detail="'enabled' must be true or false")
        try:
            return await mcp_manager.set_enabled(name, enabled)
        except UnknownServerError as exc:
            raise HTTPException(status_code=404, detail=f"No server named '{name}'") from exc

    @app.post("/api/settings/test")
    async def test_endpoint_connection(req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Test connectivity to specified LLM provider or ComfyUI endpoint."""
        target = str(req.get("target", "all")).lower()
        provider = req.get("llm_provider")
        base_url = req.get("llm_base_url")
        api_key = req.get("llm_api_key")
        comfy_url = req.get("comfyui_base_url")

        results: dict[str, Any] = {}

        if target in ("all", "comfyui"):
            target_comfy = str(comfy_url or session_mgr.get_settings()["comfyui_base_url"]).strip()
            if target_comfy:
                try:
                    client = ComfyClient(base_url=target_comfy, timeout=5.0)
                    is_alive = await client.alive()
                    stats: dict[str, Any] = {}
                    if is_alive:
                        try:
                            stats = await client.system_stats()
                        except Exception:
                            pass
                    results["comfyui"] = {
                        "status": "ok" if is_alive else "unreachable",
                        "online": is_alive,
                        "url": target_comfy,
                        "stats": stats,
                    }
                    await client.aclose()
                except Exception as exc:
                    results["comfyui"] = {
                        "status": "error",
                        "online": False,
                        "url": target_comfy,
                        "error": str(exc),
                    }

        if target in ("all", "llm"):
            eff_provider = (
                str(provider or session_mgr.get_settings()["llm_provider"]).strip().lower()
            )
            eff_base = (
                str(base_url).strip() if base_url is not None and str(base_url).strip() else None
            )
            eff_key = str(api_key).strip() if api_key is not None and str(api_key).strip() else None
            try:
                if eff_provider == "mock":
                    results["llm"] = {
                        "status": "ok",
                        "provider": "mock",
                        "message": "Mock provider ready",
                    }
                elif eff_provider == "ollama":
                    ollama_url = eff_base or resolve_ollama_base_url()
                    async with httpx.AsyncClient(timeout=5.0) as http_c:
                        resp = await http_c.get(f"{ollama_url.rstrip('/')}/api/tags")
                        if resp.status_code == 200:
                            data_obj: object = resp.json()
                            models: list[str] = []
                            if isinstance(data_obj, dict):
                                data_dict = cast(dict[str, Any], data_obj)
                                raw_models = data_dict.get("models")
                                if isinstance(raw_models, list):
                                    for item in cast(list[object], raw_models):
                                        if isinstance(item, dict):
                                            item_dict = cast(dict[str, Any], item)
                                            name_val = item_dict.get("name")
                                            if name_val is not None:
                                                models.append(str(name_val))
                            results["llm"] = {
                                "status": "ok",
                                "provider": "ollama",
                                "url": ollama_url,
                                "models": models,
                            }
                        else:
                            results["llm"] = {
                                "status": "error",
                                "provider": "ollama",
                                "url": ollama_url,
                                "error": f"Ollama HTTP {resp.status_code}",
                            }
                elif eff_provider == "vllm":
                    if not has_configured_vllm_endpoint(eff_base):
                        # The one provider whose test can fail before any request: there is
                        # no endpoint to reach. Saying so names what to fix, where a probe of
                        # a guessed port would report a refused connection instead (P6).
                        results["llm"] = {
                            "status": "error",
                            "provider": "vllm",
                            "error": "No vLLM endpoint is configured: set VLLM_BASE_URL or "
                            "enter the endpoint above (e.g. http://localhost:8000/v1).",
                        }
                    else:
                        vllm_url = resolve_vllm_base_url(eff_base)
                        async with httpx.AsyncClient(timeout=5.0) as http_c:
                            resp = await http_c.get(
                                f"{vllm_url.rstrip('/')}/models",
                                headers=vllm_request_headers(eff_key),
                            )
                            if resp.status_code == 200:
                                results["llm"] = {
                                    "status": "ok",
                                    "provider": "vllm",
                                    "url": vllm_url,
                                    "models": vllm_model_ids(resp.json()),
                                }
                            else:
                                results["llm"] = {
                                    "status": "error",
                                    "provider": "vllm",
                                    "url": vllm_url,
                                    "error": f"vLLM HTTP {resp.status_code}",
                                }
                elif eff_provider in ("openai", "anthropic", "gemini", "google"):
                    key_present = bool(
                        eff_key
                        or (eff_provider == "openai" and os.getenv("OPENAI_API_KEY"))
                        or (eff_provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"))
                        or (
                            eff_provider in ("gemini", "google")
                            and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
                        )
                    )
                    if key_present:
                        results["llm"] = {
                            "status": "ok",
                            "provider": eff_provider,
                            "message": f"{eff_provider.title()} credentials configured",
                        }
                    else:
                        results["llm"] = {
                            "status": "warning",
                            "provider": eff_provider,
                            "message": f"{eff_provider.title()} API key not set",
                        }
                else:
                    results["llm"] = {
                        "status": "error",
                        "provider": eff_provider,
                        "error": f"Unknown provider '{eff_provider}'",
                    }
            except Exception as exc:
                results["llm"] = {
                    "status": "error",
                    "provider": eff_provider,
                    "error": str(exc),
                }

        all_ok = all(v.get("status") == "ok" for v in results.values()) if results else True
        return {
            "status": "ok" if all_ok else "error",
            "results": results,
        }

    @app.get("/api/agents")
    async def list_agents(session_id: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Project the runtime agent topology for the dashboard from live session manager."""
        live_agents = session_mgr.list_agents(session_id=session_id)
        agents_data: list[dict[str, Any]] = []

        for ag in live_agents:
            agent_id = ag.agent_id
            state_val = ag.state.value
            depth = ag.context.depth
            parent_id = ag.context.parent_agent_id
            allowed_tools = list(ag.config.allowed_tools)
            # What the agent holds, not what it is permitted: a permitted name can have no
            # tool behind it (a memory tool on an agent with no store, #1431), and an empty
            # permission list permits every registered tool.
            capabilities = [tool.name for tool in ag.available_tools()]
            role = ag.config.role or "Agent"
            isolation_level = ag.config.isolation.level.value
            subagent_ids = [
                sub.agent_id for sub in live_agents if sub.context.parent_agent_id == agent_id
            ]
            current_turn = "Ready for instruction" if state_val == "IDLE" else f"State: {state_val}"
            agents_data.append(
                {
                    "id": agent_id,
                    "label": ag.config.name or agent_id,
                    "role": role,
                    "state": state_val,
                    "status": state_val,
                    "depth": depth,
                    "isolation_level": isolation_level,
                    "allowed_tools": allowed_tools,
                    "capabilities": capabilities,
                    "turn_index": ag.context.turn_index,
                    "current_turn": current_turn,
                    "current_task": current_turn,
                    "parent_agent_id": parent_id,
                    "parent_id": parent_id,
                    "subagents": subagent_ids,
                    "uptime_s": 0,
                    "max_steps": ag.config.max_steps,
                    # Deprecated alias, still emitted for clients pinned to the old key.
                    "max_turns": ag.config.max_steps,
                }
            )

        nodes: list[dict[str, Any]] = [
            {
                "id": ag["id"],
                "label": ag["label"],
                "role": ag["role"],
                "state": ag["state"],
                "status": ag["status"],
                "depth": ag["depth"],
                "isolation_level": ag["isolation_level"],
                "allowed_tools": ag["allowed_tools"],
                "capabilities": ag["capabilities"],
                "current_turn": ag["current_turn"],
                "current_task": ag["current_task"],
                "turn_index": ag["turn_index"],
                "max_steps": ag["max_steps"],
                "max_turns": ag["max_steps"],
                "position": {
                    "x": 100 + (idx % 3) * 200,
                    "y": 40 + int(ag["depth"]) * 140,
                },
            }
            for idx, ag in enumerate(agents_data)
        ]

        edges: list[dict[str, Any]] = [
            {
                "id": f"e-{ag['parent_agent_id']}-{ag['id']}",
                "source": ag["parent_agent_id"],
                "target": ag["id"],
                "type": "subagent_spawn",
                "label": "Spawn Sub-Agent",
                "animated": ag["state"] not in ("idle", "terminated", "error"),
            }
            for ag in agents_data
            if ag.get("parent_agent_id") is not None
        ]

        return {
            "data_source": "live",
            "agents": agents_data,
            "topology": {
                "nodes": nodes,
                "edges": edges,
            },
        }

    @app.get("/api/personas")
    async def list_personas() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return catalog of available declarative personas discovered by PersonaRegistry."""
        from uclone_x.agent.persona_registry import get_default_persona_registry

        # The inventory is passed here too, not only on the agent-creation path: this
        # route serves the dashboard on load and therefore usually reaches the cached
        # registry first, which would otherwise pin the process to an unvalidated one.
        registry = get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )
        data = [_persona_payload(registry, p) for p in registry.list_personas()]
        writable = registry.writable_dir()
        return {
            "status": "ok",
            "personas": data,
            "count": len(data),
            # What an editor offers: the tools a persona can name, and where a save lands.
            "available_tools": sorted(tool.name for tool in session_mgr.tools.list_tools()),
            "personas_dir": str(writable) if writable is not None else None,
        }

    def _save_persona(req: dict[str, Any], *, create: bool, name: str | None = None) -> Any:
        """Validate a persona payload, write it, and put it in force on live agents.

        Every refusal is a status with a sentence: 422 for a payload or name that cannot
        become a valid file where the loader reads, 409 for a write that would replace
        something it was not asked to, 404 for an edit of a persona no file defines.
        """
        from uclone_x.agent.persona_registry import (
            PersonaDraft,
            PersonaNotFound,
            PersonaWriteConflict,
            PersonaWriteRefused,
            get_default_persona_registry,
        )

        try:
            draft = PersonaDraft.model_validate(req)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=exc.errors(include_url=False, include_context=False)
            ) from exc
        if name is not None and draft.name != name:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the body names persona {draft.name!r} but the address names {name!r}. "
                    f"A persona cannot be renamed by an edit; create one under the new name."
                ),
            )
        registry = get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )
        try:
            persona = registry.save_persona(draft, create=create)
        except PersonaWriteRefused as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PersonaWriteConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PersonaNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(
            status_code=201 if create else 200,
            content={
                "status": "ok",
                "persona": _persona_payload(registry, persona),
                "live_agents_updated": session_mgr.apply_persona(persona),
            },
        )

    @app.get("/api/personas/{name}/avatar")
    async def get_persona_avatar(name: str) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Return one clone's picture, or refuse with what would put a picture there.

        A 404 here is an ordinary answer, not a fault: most clones have no picture, and the
        head draws its default when this refuses. The detail still names the remedy, because
        this is also what a reader sees who went looking for the file they thought they had
        installed.
        """
        from uclone_x.agent.persona_registry import get_default_persona_registry

        registry = get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )
        found = _persona_avatar(registry, name)
        if found is None:
            formats = ", ".join(suffix for suffix, _ in _AVATAR_FORMATS)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No picture is set for '{name}'. Put an image file beside its "
                    f"definition, named after it and ending in one of {formats}."
                ),
            )
        path, mime = found
        return Response(content=path.read_bytes(), media_type=mime)

    @app.post("/api/personas")
    async def create_persona(req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Create a persona as a YAML file in the workspace personas directory."""
        return _save_persona(req, create=True)

    @app.put("/api/personas/{name}")
    async def update_persona(name: str, req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Edit a persona; a built-in is edited by an override file in the workspace."""
        return _save_persona(req, create=False, name=name)

    @app.post("/api/personas/synthesize")
    async def synthesize_persona_prompt(req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Draft a clone's instructions from its name, role, description and tools.

        The connected model writes the draft. When no model answers, a fixed template
        fills in instead, and the reply says so (`source`, `fallback_reason`) so the
        editor never presents a template as the model's writing.
        """
        name = str(req.get("name", "")).strip()
        role = str(req.get("role", "")).strip()
        description = str(req.get("description", "")).strip()
        raw_tools: object = req.get("allowed_tools")
        tools: list[str] = []
        if isinstance(raw_tools, list):
            for item in cast(list[object], raw_tools):
                if isinstance(item, str):
                    tools.append(item)
                elif isinstance(item, (int, float, bool)):
                    tools.append(str(item))

        if not name and not role and not description:
            raise HTTPException(
                status_code=400,
                detail="Provide at least a name, role, or description to synthesize instructions.",
            )

        try:
            llm = session_mgr.default_llm or create_llm_connector(
                provider=session_mgr.configured_provider or None,
                api_key=session_mgr.configured_api_key or None,
                base_url=session_mgr.configured_base_url or None,
                fallback_to_mock=False,
            )
            response = await asyncio.wait_for(
                llm.generate(
                    LLMRequest(
                        model=session_mgr.configured_model or None,
                        messages=(
                            ChatMessage(role=MessageRole.SYSTEM, content=_PERSONA_DRAFT_SYSTEM),
                            ChatMessage(
                                role=MessageRole.USER,
                                content=_persona_draft_request(name, role, description, tools),
                            ),
                        ),
                        temperature=0.5,
                        max_tokens=1200,
                        auto_compact=False,
                    )
                ),
                timeout=_PERSONA_DRAFT_TIMEOUT_S,
            )
            drafted = _strip_code_fence(response.content or "")
            if not drafted:
                raise ValueError("the model returned an empty reply")
            return {"system_prompt": drafted, "source": "llm", "model": response.model_name}
        except Exception as exc:
            reason = (
                f"no reply within {_PERSONA_DRAFT_TIMEOUT_S:.0f} seconds"
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError))
                else str(exc) or type(exc).__name__
            )
            logger.warning("Persona draft fell back to the template: %s", reason)
            return {
                "system_prompt": _template_persona_prompt(name, role, description, tools),
                "source": "template",
                "fallback_reason": reason,
            }

    def _record_cancelled_turn(
        *,
        agent: BaseAgent | None,
        agent_id: str,
        session_id: str,
        message: str,
        client_turn_id: str | None,
        model: str | None,
        latency_ms: float,
        core_len_before: int | None,
    ) -> None:
        """Save the record a cancelled turn leaves, and keep the Core it left behind (#1031).

        Cancelling the turn task makes `execute_turn` re-raise, and
        `asyncio.CancelledError` is a `BaseException`: it walks straight past the handler
        that writes a failed turn's rows. So a stop used to write **nothing** while the Core
        kept the prompt, the `ASSISTANT` message carrying that step's tool calls and the
        `TOOL` result. The user saw no sign a tool had run, and the orphaned prompt -- a Core
        message no row could match and no rule could fold -- stalled `_core_index_for_transcript`
        permanently: measured end to end, a truncation that deleted *nothing* then cut the
        Core from 7 messages to 2, in memory and on disk, losing a turn still on the page and
        with it what the next request carried (#1031, PR #1033's probe).

        The repair is a record, not a rule in the index walk: with its own two rows the
        cancelled turn is shaped like every other turn, and the walk that already handles a
        failed one handles this one unchanged.

        Everything here is synchronous on purpose. The task is already cancelled, so the
        first `await` would raise `CancelledError` again and abandon the record half-written.
        Nothing raises out either: the caller re-raises the cancellation, and a failure to
        save must not become a different exception on the way out.
        """
        turn_messages: list[ChatMessage] = []
        turn_index = 0
        persist_error: str | None = None
        persist_error_type: str | None = None
        if agent is not None:
            try:
                state = agent.get_session(session_id)
                # Only what this turn appended. `None` means the turn was cancelled before
                # the length was taken, and a guess at the boundary would put an earlier
                # turn's tool calls on this row.
                if core_len_before is not None:
                    turn_messages = list(state.messages)[core_len_before:]
                turn_index = state.turn_counter
            except Exception as exc:
                persist_error = str(exc)
                persist_error_type = type(exc).__name__
                logger.warning(
                    "Could not read the Core session %s while recording a cancelled turn: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
            try:
                # The Core already holds the cancelled turn's messages, and this is what
                # puts them on disk: nothing else on this path writes the Core. Measured by
                # removing it -- a session whose first turn is stopped has **no** Core record
                # at all afterwards, and a session with earlier turns keeps only those, so
                # the saved row would describe a turn a restart could not find (#1031).
                # Spelled with the keyword because that makes this the one call site the
                # mutation below names.
                agent.persist_session(session_id=session_id)
            except SessionStoreNotConfiguredError:
                pass
            except Exception as exc:
                persist_error = str(exc)
                persist_error_type = type(exc).__name__
                logger.warning(
                    "Failed to persist Core session %s after a cancelled turn: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )

        try:
            current_history = session_mgr.get_session_history(agent_id, session_id)
        except Exception:
            logger.warning(
                "Failed to read session history for agent %s, session %s; the cancelled "
                "turn was not recorded",
                agent_id,
                session_id,
                exc_info=True,
            )
            return

        updated_history = [
            *current_history,
            *_cancelled_turn_rows(
                agent_id=agent_id,
                message=message,
                client_turn_id=client_turn_id,
                model=model,
                latency_ms=latency_ms,
                turn_index=turn_index,
                turn_messages=turn_messages,
                persist_error=persist_error,
                persist_error_type=persist_error_type,
            ),
        ]
        try:
            session_mgr.save_session_record(
                session_id=session_id,
                agent_id=agent_id,
                messages=updated_history,
                # A cancelled turn was still a turn taken -- `execute_turn` increments the
                # counter before it calls the model -- so the saved figure is the Core's own
                # counter, and `_turns_taken` over the rows agrees because the row is sent by
                # the agent (#1023).
                turns=turn_index or _turns_taken(updated_history),
            )
        except Exception:
            logger.warning(
                "Failed to save the transcript for session %s after a cancelled turn",
                session_id,
                exc_info=True,
            )

    async def _execute_turn_logic_impl(req: dict[str, Any]) -> dict[str, Any]:
        """Execute turn logic against live BaseAgent and return formatted turn result with P6 trace."""
        _refuse_a_reserved_session_id(
            req.get("session_id") if isinstance(req.get("session_id"), str) else None
        )
        client_turn_id = _client_turn_id(req)
        message = str(req.get("message", "")).strip()
        if not message:
            return {"error": "Message cannot be empty", "status": "error"}
        model_req = str(req.get("model") or req.get("llm_model") or "").strip() or None
        agent_id = _required_agent_id(req.get("agent_id"))
        session_id = str(req.get("session_id", f"sess_{agent_id}"))

        bus = session_mgr.bus
        start_time = asyncio.get_running_loop().time()

        agent: BaseAgent | None = None
        session_load_failed = False
        turn_booked: list[TokenUsage] | None = None
        # Read where this turn's own Core messages begin, so a cancellation can report the
        # calls it made and nothing an earlier turn made. `None` until the turn is about to
        # run: before that there is no boundary to report (#1031).
        core_len_before: int | None = None
        active_model: str | None = None
        try:
            # Obtain live BaseAgent instance from session manager
            try:
                agent = await session_mgr.get_or_create_agent(
                    agent_id=agent_id,
                    session_id=session_id,
                    model_name=model_req,
                )
            except Exception:
                session_load_failed = True
                raise

            assert agent is not None

            active_provider = (
                session_mgr.configured_provider
                or (getattr(agent.llm, "provider_name", None) if agent.llm else None)
                or (
                    getattr(session_mgr.default_llm, "provider_name", None)
                    if session_mgr.default_llm
                    else None
                )
                or os.getenv("LLM_PROVIDER")
                or "ollama"
            )

            # Ensure agent's active model matches requested model or configured model
            target_model = (
                model_req
                or session_mgr.configured_model
                or agent.config.llm_config.model_name
                or (resolve_ollama_model(None) if active_provider == "ollama" else None)
            )

            if target_model and agent.config.llm_config.model_name != target_model:
                agent.hot_reload_llm(session_mgr.default_llm or agent.llm, model_name=target_model)

            active_model = agent.config.llm_config.model_name or target_model or "default"

            logger.info(
                "💬 [UI Chat] Turn starting: agent=%s, session=%s, provider=%s, model=%s, prompt=%r",
                agent_id,
                session_id,
                active_provider,
                active_model,
                message[:60],
            )
            _console.print(
                f"[bold blue]💬 [UI Chat] Turn executing:[/bold blue] agent=[cyan]{agent_id}[/cyan] | "
                f"provider=[cyan]{active_provider}[/cyan] | model=[bold yellow]{active_model}[/bold yellow] | prompt={message[:50]!r}"
            )

            user_pub = bus.register_publisher(
                sender_id="user",
                source=EventSource.USER,
                capabilities={"user_source"},
            )
            await user_pub.publish(
                AgentEvent(
                    type=EventType.USER_INPUT,
                    recipient_id=agent_id,
                    topic="agent.chat.input",
                    payload={"message": message, "session_id": session_id},
                )
            )

            core_len_before = len(agent.get_session(session_id).messages)

            # The turn's figures are the steps booked from inside this block, not a slice of
            # the session's ledger: another turn on this session id, waiting on the agent's
            # turn lock or run by another agent, books into the same ledger (#982).
            with session_mgr.budget_tracker.collect_turn_usage(session_id) as turn_booked:
                # Delegate turn execution directly to BaseAgent (P8, Issue #175).
                # No `stream_callback` arm: `/api/chat/stream` was its only caller and
                # retired with the playground (#1208). A room streams through the room
                # orchestrator, which calls `BaseAgent.execute_turn` itself.
                try:
                    turn_result = await agent.execute_turn(message, stream_callback=None)
                except TypeError:
                    turn_result = await agent.execute_turn(message)

        except asyncio.CancelledError:
            # Not an `Exception`: Stop's cancellation walks past the handler below, which is
            # how a cancelled turn came to write no record at all (#1031). Recorded here and
            # re-raised unchanged, so a cancelled turn behaves exactly as before.
            _record_cancelled_turn(
                agent=agent,
                agent_id=agent_id,
                session_id=session_id,
                message=message,
                client_turn_id=client_turn_id,
                model=active_model,
                latency_ms=(asyncio.get_running_loop().time() - start_time) * 1000,
                core_len_before=core_len_before,
            )
            raise

        except Exception as exc:
            latency_ms = (asyncio.get_running_loop().time() - start_time) * 1000
            # The dashboard is where a beginner meets this failure, and where the
            # traceback never appears at all -- they see a red bubble. Recording it
            # is what makes `ucx report` and the dashboard's own report view able to
            # say anything later. Consent-gated; a no-op without it.
            record_failure(exc, context={"surface": "ui.turn"})
            logger.error("❌ [UI Chat] Turn execution failed: %s", exc)
            _console.print(f"[bold red]✖ [UI Chat] Turn failed:[/bold red] {exc}")
            is_offline = _is_offline_llm_error(exc)
            # `session_load_failed` is set by the broad handler wrapping `get_or_create_agent`,
            # which also builds the connector — so an LLM configuration fault arrives here
            # flagged as a session fault and gets `component=uclone_x.agent.session` with
            # `path=SESSION_LOAD_ERROR`. Naming the wrong subsystem is the mis-attribution P6
            # forbids, and it also produced a self-contradictory record
            # (`SESSION_LOAD_ERROR` carrying `served_by=offline_diagnostic`).
            # An unusable agent name arrives the same way: it is raised while the agent's
            # home directory is resolved, inside the same broad handler, and so inherits
            # `session_load_failed` from it. The session store is not involved and the
            # request is what is wrong, so naming that component would send the reader to
            # a subsystem with nothing to fix.
            is_session_err = (
                session_load_failed and not is_offline and not isinstance(exc, AgentHomeError)
            ) or isinstance(
                exc,
                (
                    SessionHistoryRehydrationError,
                    SessionIdCollisionError,
                    SessionMutationDuringTurnError,
                    SessionStoreNotConfiguredError,
                    PathTraversalError,
                ),
            )
            if isinstance(exc, LLMProviderNotConfiguredError) and "model" in str(exc).lower():
                reply_content = f"⚠️ {exc}"
                status = "warning"
            elif is_offline:
                reply_content = OFFLINE_LLM_DIAGNOSTIC_MESSAGE
                status = "warning"
            else:
                reply_content = f"Error: {exc}"
                status = "error"

            prov_hash = hashlib.sha256(f"{agent_id}:{message}:{reply_content}".encode()).hexdigest()
            provenance_data: dict[str, Any] = {
                "component": (
                    "uclone_x.agent.session" if is_session_err else "uclone_x.llm.orchestrator"
                ),
                "producer": agent_id,
                "content_hash": prov_hash,
                "degraded": True,
                "path": (
                    "SESSION_LOAD_ERROR"
                    if is_session_err
                    else ("OFFLINE_FALLBACK" if is_offline else "FAILOVER")
                ),
                "served_by": "offline_diagnostic" if is_offline else None,
            }
            if agent is not None and (agent.persona_name or agent.persona):
                provenance_data["persona"] = agent.persona_name or agent.persona

            # A turn that raised may still have booked steps first; they are reported.
            failed_tokens_used, failed_prompt_tokens_used, failed_count_source = (
                _turn_token_figures(turn_booked)
            )
            default_max_steps = cast(int, AgentConfig.model_fields["max_steps"].default or 50)
            budget_max = agent.config.max_steps if agent is not None else default_max_steps
            turn_idx = agent.turn_counter if agent is not None else 0
            run_steps = agent.run_steps if agent is not None else 0
            return {
                "status": status,
                "outcome": ChatTurnOutcome.FAILED,  # raised
                "error": str(exc),
                # Nothing that raised out of the turn is a refusal: a budget ceiling is
                # returned by `BaseAgent` as a result, never raised past it.
                "refusal": None,
                "agent_id": agent_id,
                "response": reply_content,
                "latency_ms": latency_ms,
                "turn_count": turn_idx,
                "run_turns": run_steps,
                "run_steps": run_steps,
                "turn_budget_max": budget_max,
                "turns_remaining": max(0, budget_max - run_steps),
                "step_budget_max": budget_max,
                "steps_remaining": max(0, budget_max - run_steps),
                "tokens_used": failed_tokens_used,
                "token_count_source": failed_count_source,
                "tool_calls": [],
                "tool_executions": [],
                "debug_info": {
                    "active_invariants": [],
                    "prompt_tokens_used": failed_prompt_tokens_used,
                    "system_prompt_excerpt": "",
                },
                "provenance": provenance_data,
                "durability": {
                    "persisted": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "stale_conflict": isinstance(exc, StaleSessionWriteError),
                },
                "persona": (agent.persona_name or agent.persona) if agent is not None else None,
            }

        latency_ms = (asyncio.get_running_loop().time() - start_time) * 1000
        tokens_used, prompt_tokens_used, token_count_source = _turn_token_figures(turn_booked)

        # Check if turn_result had an offline LLM error
        is_offline = bool(turn_result.error and _is_offline_llm_error(turn_result.error))

        if is_offline:
            reply_content = OFFLINE_LLM_DIAGNOSTIC_MESSAGE
            status = "warning"
        elif turn_result.error:
            reply_content = f"Error: {turn_result.error}"
            status = "error"
        else:
            reply_content = turn_result.content or ""
            status = "success" if turn_result.is_completed else "error"

        # Extract and serialize real tool call records and executions
        serialized_tool_calls: list[dict[str, Any]] = _serialized_tool_calls(turn_result.tool_calls)
        serialized_tool_executions: list[dict[str, Any]] = [
            {
                "tool_call_id": te.tool_call_id,
                "tool_name": te.tool_name,
                "arguments": cast(dict[str, Any], unwrap_immutable(te.arguments)),
                "output": unwrap_immutable(te.output),
                "status": te.status,
                "error": te.error,
                "duration_ms": te.duration_ms,
            }
            for te in turn_result.tool_executions
        ]

        active_invariants: list[str] = []
        if agent.ontology is not None:
            try:
                invariants = agent.ontology.get_active_invariants(tier_filter="all")
                active_invariants = [
                    f"{inv.name} ({inv.tier.value}): {inv.predicate or inv.description or inv.name}"
                    for inv in invariants
                ]
            except Exception:
                logger.warning(
                    "Failed to retrieve active ontology invariants for agent %s",
                    agent_id,
                    exc_info=True,
                )

        debug_info: dict[str, Any] = {
            "active_invariants": active_invariants,
            "prompt_tokens_used": prompt_tokens_used,
            "system_prompt_excerpt": (
                (
                    agent.config.system_prompt[:200]
                    + ("..." if len(agent.config.system_prompt) > 200 else "")
                )
                if agent.config.system_prompt
                else ""
            ),
        }

        # Publish tool trace events for SSE subscribers if tools were called
        for tc in turn_result.tool_calls:
            tool_arguments = cast(dict[str, Any], unwrap_immutable(tc.arguments))
            tool_pub = bus.register_publisher(
                sender_id=agent_id,
                source=EventSource.TOOL,
            )
            await tool_pub.publish(
                AgentEvent(
                    type=EventType.TOOL_CALL,
                    recipient_id=agent_id,
                    topic="agent.chat.tool",
                    payload={
                        "tool_id": tc.id,
                        "name": tc.name,
                        "arguments": tool_arguments,
                    },
                )
            )

        # Publish AGENT_REPLY event to bus for real-time subscribers (SSE)
        agent_pub = bus.register_publisher(
            sender_id=agent_id,
            source=EventSource.AGENT,
        )
        reply_payload: dict[str, Any] = {
            "message": reply_content,
            "model": active_model,
            "latency_ms": latency_ms,
            "tool_count": len(serialized_tool_calls),
            "turn_index": turn_result.turn_index,
            "is_completed": str(turn_result.is_completed),
            "tool_executions": serialized_tool_executions,
            "debug_info": debug_info,
        }
        if turn_result.error is not None:
            reply_payload["error"] = turn_result.error
        if turn_result.persona is not None:
            reply_payload["persona"] = turn_result.persona

        await agent_pub.publish(
            AgentEvent(
                type=EventType.AGENT_REPLY,
                recipient_id="user",
                topic="agent.chat.reply",
                priority=EventPriority.NORMAL,
                payload=reply_payload,
                # Forwarded verbatim from turn_result.provenance (Issue #117, #175)
                provenance=turn_result.provenance,
                trace_id=session_mgr.tracer.trace_id,
            )
        )

        # Build provenance record conforming to P6
        prov_hash = hashlib.sha256(f"{agent_id}:{message}:{reply_content}".encode()).hexdigest()
        if is_offline:
            served_model: str | None = "offline_diagnostic"
        elif turn_result.provenance and turn_result.provenance.served_by:
            srv = turn_result.provenance.served_by
            if srv.model:
                if srv.model.startswith(f"{srv.provider}:"):
                    served_model = srv.model
                else:
                    served_model = f"{srv.provider}:{srv.model}"
            else:
                served_model = srv.provider
        else:
            served_model = None

        prov_path = (
            "OFFLINE_FALLBACK"
            if is_offline
            else (
                turn_result.provenance.path.value
                if turn_result.provenance
                else ("PRIMARY" if turn_result.is_completed else "FAILOVER")
            )
        )

        is_degraded = (
            is_offline
            or not turn_result.is_completed
            or bool(turn_result.provenance and turn_result.provenance.degraded)
        )
        outcome = _answered_outcome(turn_result)
        # From the turn's stated stop reason, never from the wording of `error` (#969).
        refusal = turn_refusal(turn_result.stop_reason) if turn_result.error is not None else None

        provenance_data: dict[str, Any] = {
            "component": "uclone_x.llm.orchestrator",
            "producer": agent_id,
            "content_hash": prov_hash,
            "degraded": is_degraded,
            "path": prov_path,
            "served_by": served_model,
        }
        if turn_result.persona is not None:
            provenance_data["persona"] = turn_result.persona

        user_msg_entry = {
            "id": f"user-{int(start_time * 1000)}",
            "sender": "user",
            "role": "user",
            "content": message,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": active_model,
        }
        # The id a head sent this turn with, kept on the saved prompt so a head reloading the
        # conversation finds this turn's copy by identity rather than by prompt text, which a
        # retry repeats word for word (#1000). Optional: without it the entry is as it was.
        if client_turn_id is not None:
            user_msg_entry["client_turn_id"] = client_turn_id
        agent_msg_entry = {
            "id": f"agent-{int(asyncio.get_running_loop().time() * 1000)}",
            "sender": "agent",
            "role": "assistant",
            "content": reply_content,
            "timestamp": datetime.now(UTC).isoformat(),
            "latency_ms": latency_ms,
            "agent_id": agent_id,
            "model": active_model,
            # `turn_count` keeps its meaning: turns in this conversation. The budget
            # fields report the quantity `max_steps` actually bounds — a run of steps the
            # agent takes without returning to its caller. Reporting the lifetime count
            # against that ceiling filled a red bar to 51/50 during an ordinary
            # conversation and then refused the next message.
            "turn_count": turn_result.turn_index,
            "run_turns": agent.run_steps,
            "run_steps": agent.run_steps,
            "turn_budget_max": agent.config.max_steps,
            "turns_remaining": agent.steps_remaining,
            "step_budget_max": agent.config.max_steps,
            "steps_remaining": agent.steps_remaining,
            "tokens_used": tokens_used,
            "token_count_source": token_count_source,  # persisted, so a reload labels it
            "tool_calls": serialized_tool_calls,
            "tool_executions": serialized_tool_executions,
            "debug_info": debug_info,
            "provenance": provenance_data,
            "outcome": outcome,
        }
        if turn_result.persona is not None:
            agent_msg_entry["persona"] = turn_result.persona
        if turn_result.error is not None:
            # Saved as the failure it was, not as a reply (#969): `content` keeps the text the
            # page showed, `error` the turn's own reason.
            agent_msg_entry["role"] = TRANSCRIPT_FAILURE_ROLE
            agent_msg_entry["error"] = turn_result.error
            if refusal is not None:
                agent_msg_entry["refusal"] = refusal

        # Core session durability (Domain layer, #183, #229, #240, #247)
        persist_error: str | None = None
        persist_error_type: str | None = None
        is_stale_write = False
        try:
            agent.persist_session(session_id)
        except StaleSessionWriteError as exc:
            is_stale_write = True
            persist_error_type = "StaleSessionWriteError"
            persist_error = str(exc)
            logger.warning(
                "Stale write refusing Core session %s persist after turn: %s",
                session_id,
                exc,
            )
        except Exception as exc:
            persist_error_type = type(exc).__name__
            persist_error = str(exc)
            logger.warning(
                "Failed to persist Core session %s after a turn; the reply was returned "
                "but the conversation may not be durable: %s",
                session_id,
                exc,
                exc_info=True,
            )

        # Truth in provenance (P6, #247): If persistence failed, the turn outcome was
        # degraded, even if the model inference itself succeeded without error.
        if persist_error is not None:
            provenance_data["degraded"] = True
            if status == "success":
                status = "warning"

        durability_data: dict[str, Any] = {
            "persisted": persist_error is None,
            "error": persist_error,
            "error_type": persist_error_type,
            "stale_conflict": is_stale_write,
        }

        # Transcript history read & save (Presentation layer, #229, #247)
        current_history: list[dict[str, Any]] | None = None
        try:
            current_history = session_mgr.get_session_history(agent_id, session_id)
        except Exception:
            logger.warning(
                "Failed to read session history for agent %s, session %s; skipping transcript update",
                agent_id,
                session_id,
                exc_info=True,
            )

        agent_msg_entry["provenance"] = provenance_data
        agent_msg_entry["durability"] = durability_data

        if current_history is not None:
            updated_history = [*current_history, user_msg_entry, agent_msg_entry]
            try:
                session_mgr.save_session_record(
                    session_id=session_id,
                    agent_id=agent_id,
                    messages=updated_history,
                    turns=turn_result.turn_index,
                )
            except Exception:
                logger.warning(
                    "Failed to save transcript for session %s after turn",
                    session_id,
                    exc_info=True,
                )

        logger.info(
            "💬 [UI Chat] Turn finished: agent=%s, model=%s, latency_ms=%.1f, status=%s",
            agent_id,
            active_model,
            latency_ms,
            status,
        )
        _console.print(
            f"[bold green]✔ [UI Chat] Turn finished:[/bold green] agent=[cyan]{agent_id}[/cyan] | "
            f"model=[bold yellow]{active_model}[/bold yellow] | latency={latency_ms:.1f}ms | status={status}"
        )

        return {
            "status": status,
            "outcome": outcome,
            "error": turn_result.error,
            "refusal": refusal,
            "agent_id": agent_id,
            "model": active_model,
            "response": reply_content,
            "latency_ms": latency_ms,
            # `turn_count` keeps its meaning: turns in this conversation. The budget
            # fields report the quantity `max_steps` actually bounds — a run of steps the
            # agent takes without returning to its caller. Reporting the lifetime count
            # against that ceiling filled a red bar to 51/50 during an ordinary
            # conversation and then refused the next message.
            "turn_count": turn_result.turn_index,
            "run_turns": agent.run_steps,
            "run_steps": agent.run_steps,
            "turn_budget_max": agent.config.max_steps,
            "turns_remaining": agent.steps_remaining,
            "step_budget_max": agent.config.max_steps,
            "steps_remaining": agent.steps_remaining,
            "tokens_used": tokens_used,
            "token_count_source": token_count_source,
            "tool_calls": serialized_tool_calls,
            "tool_executions": serialized_tool_executions,
            "debug_info": debug_info,
            "provenance": provenance_data,
            "durability": durability_data,
            "persona": turn_result.persona,
        }

    @app.post("/api/turn")
    async def chat_with_agent(req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Send message prompt to live BaseAgent and receive execution turn result with P6 trace."""
        return await _execute_turn_logic_impl(req)

    @app.get("/api/session/history")
    async def get_chat_history(  # pyright: ignore[reportUnusedFunction]
        agent_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return persisted or in-memory conversation history for agent and session."""
        eff_session_id = session_id if session_id else f"sess_{agent_id}"
        try:
            messages = session_mgr.get_session_history(agent_id=agent_id, session_id=eff_session_id)
            active = session_mgr.active_turns(agent_id=agent_id, session_id=eff_session_id)
        except Exception as exc:
            raise _translate_session_error(exc) from exc
        return {
            "session_id": eff_session_id,
            "agent_id": agent_id,
            "messages": messages,
            # Reported here as well as on `/api/sessions` so a surface that has just
            # reloaded the conversation can settle the saturation notice from the same
            # response, rather than from a second request that may not have run yet.
            "active_turns": active,
            "is_saturated": active >= SATURATION_TURNS_THRESHOLD,
        }

    @app.post("/api/session/history/truncate")
    async def truncate_chat_history(  # pyright: ignore[reportUnusedFunction]
        req: dict[str, Any] | None = None,
        index: int | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Truncate conversation history and Core session state back to specified index (FR-13.7)."""
        body = req or {}
        raw_index = body.get("index") if "index" in body else index
        if raw_index is None:
            raise HTTPException(status_code=400, detail="Missing required 'index' parameter")
        try:
            eff_index = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail="'index' must be an integer") from exc
        if eff_index < 0:
            raise HTTPException(status_code=400, detail="'index' must be non-negative")

        eff_agent_id = _required_agent_id(body.get("agent_id") or agent_id)
        raw_session = body.get("session_id") if "session_id" in body else session_id
        eff_session_id = f"sess_{eff_agent_id}" if raw_session is None else str(raw_session)

        try:
            remaining_messages = session_mgr.truncate_session_history(
                agent_id=eff_agent_id,
                session_id=eff_session_id,
                index=eff_index,
            )
        except Exception as exc:
            raise _translate_session_error(exc) from exc

        return {
            "status": "truncated",
            "session_id": eff_session_id,
            "agent_id": eff_agent_id,
            "index": eff_index,
            "messages": remaining_messages,
        }

    @app.delete("/api/session/history")
    async def delete_chat_history(  # pyright: ignore[reportUnusedFunction]
        agent_id: str,
        session_id: str | None = None,
        index: int | None = None,
    ) -> dict[str, Any]:
        """Clear or truncate persisted conversation history and active in-memory session."""
        eff_session_id = session_id if session_id else f"sess_{agent_id}"
        try:
            if index is not None:
                if index < 0:
                    raise HTTPException(status_code=400, detail="'index' must be non-negative")
                remaining = session_mgr.truncate_session_history(
                    agent_id=agent_id,
                    session_id=eff_session_id,
                    index=index,
                )
                return {
                    "status": "truncated",
                    "session_id": eff_session_id,
                    "agent_id": agent_id,
                    "index": index,
                    "messages": remaining,
                }
            session_mgr.clear_session_history(agent_id=agent_id, session_id=eff_session_id)
        except Exception as exc:
            # A head clears history through this route, so an untranslated refusal
            # here is the one a user actually meets.
            raise _translate_session_error(exc) from exc
        return {"status": "cleared"}

    @app.get("/api/sessions")
    async def list_sessions() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return list of active and stored sessions from Core session store (FR-13.2)."""
        session_ids = session_mgr.core_store.list_session_ids()
        sessions: list[dict[str, Any]] = []
        for sid in session_ids:
            # A seated agent keeps its own session per room (G3). Those are a room's
            # internals: listed here they appear as one phantom conversation per agent per
            # room, which is what a user would have to tell apart from their own chats.
            # Filtered in the Core's own listing rather than in a component, because a
            # second head would need exactly the same filter.
            if sid.startswith(ROOM_SESSION_PREFIX):
                continue
            state = session_mgr.core_store.load(sid)
            if state is not None:
                # Saturation is a property of the *active* context, not of the session's
                # lifetime. Keyed to `turn_counter` it could never clear, because Core
                # retains that counter across compaction by design (P5) — so the button
                # the notice recommends appeared to do nothing (#872).
                active = count_active_turns(state.messages)
                sessions.append(
                    {
                        "session_id": sid,
                        "agent_id": state.agent_id,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                        "turn_counter": state.turn_counter,
                        "revision": state.revision,
                        "message_count": len(state.messages),
                        "active_turns": active,
                        "is_saturated": active >= SATURATION_TURNS_THRESHOLD,
                    }
                )
            else:
                sessions.append({"session_id": sid})
        sessions.sort(
            key=lambda s: str(s.get("updated_at") or s.get("created_at") or ""),
            reverse=True,
        )
        return {"sessions": sessions}

    @app.post("/api/dispatch")
    async def dispatch_task(req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Dispatch task prompt to swarm worker."""
        task_prompt = str(req.get("task", "")).strip()
        if not task_prompt:
            return {"error": "Task prompt cannot be empty", "status": "error"}
        role = str(req.get("role", "assistant"))
        now_ts = int(asyncio.get_running_loop().time())
        # A spawn needs a recipient. This used to name one after the clock, publish
        # `SUBAGENT_SPAWN` to it and answer `dispatched`, so the caller was told a task had
        # gone to an agent that has never existed (P6).
        recipient_id = _required_agent_id(req.get("agent_id"))
        task_id = f"task_{now_ts}"

        dispatch_bus = active_bus
        sys_pub = dispatch_bus.register_publisher(
            sender_id="ui_dispatcher",
            source=EventSource.SYSTEM,
        )
        await sys_pub.publish(
            AgentEvent(
                type=EventType.SUBAGENT_SPAWN,
                recipient_id=recipient_id,
                topic="swarm.dispatch",
                payload={"task_id": task_id, "task": task_prompt, "role": role},
            )
        )

        prov_hash = hashlib.sha256(f"{task_id}:{task_prompt}:{role}".encode()).hexdigest()
        return {
            "status": "dispatched",
            "task_id": task_id,
            "agent_id": recipient_id,
            "role": role,
            "timestamp": now_ts,
            "provenance": {
                "component": "uclone_x.ui.dispatcher",
                "producer": "ui_dispatcher",
                "content_hash": prov_hash,
                "degraded": False,
                "path": "primary",
                "served_by": "dispatcher",
            },
        }

    @app.get("/api/ontology")
    async def get_ontology() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return LinkML concept hierarchy, relation graph, and tier metadata from live Core Engine."""
        return session_mgr.ontology_engine.export_graph()

    @app.get("/api/artifacts")
    async def get_artifacts(session_id: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Enumerate the documents and images the clones generated (RFC §6.1)."""
        artifacts = session_mgr.list_artifacts(session_id=session_id)
        return {"artifacts": artifacts, "total": len(artifacts)}

    @app.get("/api/artifacts/content")
    async def get_artifact_content(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        path: str = "",
        session_id: str | None = None,
    ) -> Response:
        """Return artifact file content safely (markdown text or raw image bytes) (P6 security invariant, RFC §6.1)."""
        try:
            resolved_path, mime = session_mgr.get_artifact_file(path=path, session_id=session_id)
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if mime.startswith("image/"):
            return Response(
                content=resolved_path.read_bytes(),
                media_type=mime,
                headers={
                    "Cache-Control": "no-cache, must-revalidate",
                    "Pragma": "no-cache",
                },
            )

        content = resolved_path.read_text(encoding="utf-8")
        accept = request.headers.get("accept", "")
        if "application/json" in accept:
            return JSONResponse({"path": path, "content": content})
        return Response(content=content, media_type=mime)

    @app.get("/api/knowledge-graph")
    async def get_knowledge_graph(  # pyright: ignore[reportUnusedFunction]
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Return dynamic entity-relation triples (subject, predicate, object, provenance, tier) (RFC §6.1)."""
        return session_mgr.get_knowledge_graph(session_id=session_id, agent_id=agent_id)

    @app.get("/api/skills")
    async def get_skills() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return registered skills, P9 security audit reports, and quarantine statuses from live SkillRegistry."""
        return session_mgr.skill_registry.get_summary()

    @app.get("/api/acp/status")
    async def get_acp_status() -> AcpConformanceReport:  # pyright: ignore[reportUnusedFunction]
        """Report what this build answers of ACP, and whether anything is actually serving it.

        The presence block is measured, not declared, so "no shell is installed" and "a shell
        is running with no sessions" cannot render as the same empty state (P6). The method
        rows come from `uclone_x.acp.conformance`, which is the same source `initialize` will
        derive its capability response from — a second list assembled here is exactly the
        drift §3.3 of the specification describes.
        """
        return conformance_summary()

    @app.get("/api/budget")
    async def get_budget(session_id: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return token budget metrics, cost attribution, and compaction history from live budget tracker."""
        return session_mgr.budget_tracker.get_summary(session_id=session_id)

    # --- Diagnostics -------------------------------------------------------
    #
    # The dashboard is where a beginner meets a failure, so it is where the
    # question about recording one has to be asked. These endpoints expose the
    # same journal `ucx report` reads; neither they nor it send anything. The
    # report body is returned to the browser so the person can read it before
    # deciding, and the issue URL is opened by their browser, not by us.
    #
    # Cross-origin requests are refused on every diagnostics endpoint, reads
    # included. This app is served with `allow_origins=["*"]` and
    # `allow_credentials=True` -- a pre-existing setting that predates these
    # routes and that they should not quietly inherit. Without the check, any
    # page open in the same browser while `ucx ui` runs could turn recording on
    # and read the report back.
    #
    # `_refuse_cross_origin` is not diagnostics-only: the model-management
    # routes above (`POST /api/models/pull`, `POST /api/models/delete`) call it
    # too, for the same reason. It lives here because this is where it was first
    # needed, and a nested function is in scope for every route in this closure
    # regardless of the order they are written in.

    # `_LOOPBACK_HOSTS` is module-level: `LoopbackHostGuard` uses it too (#1413).

    def _refuse_cross_origin(request: Request) -> None:
        """Raise unless the request came from this machine or from no page at all.

        Applied to reads as well as writes. The first version left reads open,
        reasoning that they expose "what the user can already see" — which is
        the reasoning a same-origin policy exists to reject, and review said so.
        The report body is the failure record; a page that can read it
        cross-origin has taken it.

        Loopback origins are allowed at any port, and that is not a loophole:
        `vite.config.ts` proxies `/api` with `changeOrigin`, so the dashboard in
        development arrives with `Origin: http://localhost:5173` against a
        rewritten `Host` — a strict comparison would 403 the very UI this
        serves, while admitting nothing a remote page can produce.
        """
        origin = request.headers.get("origin")
        if origin is None:
            # curl, the CLI, a same-origin navigation: no `Origin` to check.
            return

        parsed = urlparse(origin)
        host = request.headers.get("host", "")
        if parsed.netloc == host or parsed.hostname in _LOOPBACK_HOSTS:
            return

        raise HTTPException(
            status_code=403,
            detail=(
                "Cross-origin requests to this endpoint are refused. "
                f"Origin {origin!r} is neither {host!r} nor a loopback address."
            ),
        )

    def _refuse_unless_local(request: Request) -> None:
        """`_refuse_cross_origin`, and also refuse a request addressed to a non-loopback name.

        For routes that start a program on this machine. The origin check alone passes a
        DNS-rebinding page: its `Origin` and `Host` both name the attacker's domain, which
        now resolves to 127.0.0.1, so they match. That page cannot make the browser send a
        `Host` of `localhost`, and this refuses every other name.

        On a loopback-bound server `LoopbackHostGuard` already refuses those names for every
        route (#1413). This check is what still holds on a server exposed with `ucx ui
        --host 0.0.0.0`: others on the network may use the dashboard, but only this computer
        may start programs on it.
        """
        _refuse_cross_origin(request)
        if _host_header_hostname(request.headers.get("host", "")) not in _LOOPBACK_HOSTS:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Tool servers can only be changed from this computer. "
                    "Open the app at http://localhost to change them."
                ),
            )

    @app.get("/api/diagnostics/consent")
    async def get_diagnostics_consent(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Whether failures may be recorded locally: granted, denied, or unasked."""
        _refuse_cross_origin(request)
        consent = read_consent()
        return {
            "state": consent.state,
            "error": consent.error,
            "journal": str(journal_path()),
        }

    @app.post("/api/diagnostics/consent")
    async def set_diagnostics_consent(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Record the answer. `collect` must be present: there is no default."""
        _refuse_cross_origin(request)
        payload: dict[str, Any] = await request.json()
        collect = payload.get("collect")
        if not isinstance(collect, bool):
            raise HTTPException(status_code=400, detail="`collect` must be true or false")
        try:
            set_consent(collect)
        except OSError as exc:
            # An unwritable diagnostics directory is a real state, and the
            # answer was not taken. Said as a sentence rather than raised as a
            # 500 with a traceback in the log and nothing on screen.
            raise HTTPException(
                status_code=500,
                detail=f"Your choice could not be saved to {consent_path()}: {exc}",
            ) from exc
        return {"state": consent_state()}

    @app.get("/api/diagnostics/report")
    async def get_diagnostics_report(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The report a person would send, plus where to send it.

        `issue_url` is `None` when the report is too long to carry in a URL; the
        client must offer the text for copying rather than opening a truncated
        form.
        """
        _refuse_cross_origin(request)
        read = read_journal()
        entries = list(read.entries)
        body = render_report(read)
        title = report_title(entries)
        return {
            "state": consent_state(),
            # `available` is not decoration: a client that sees `count: 0` with
            # no way to tell a read failure from an empty journal will render
            # "nothing to report" over a broken one.
            "available": read.is_available,
            "error": read.error,
            "unreadable_lines": read.unreadable_lines,
            "recording_blocked": read.recording_blocked,
            "count": len(entries),
            "distinct": len(summarise(entries)),
            "title": title,
            "body": body,
            "issue_url": issue_url(body, title),
            "search_url": search_url(entries[-1].fingerprint) if entries else None,
        }

    @app.delete("/api/diagnostics/report")
    async def clear_diagnostics_report(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Delete the recorded failures."""
        _refuse_cross_origin(request)
        outcome = clear_journal()
        if outcome.error is not None:
            # Answering 200 with `cleared: false` made a failed delete
            # indistinguishable from an empty journal to a client checking only
            # the status code -- which the dashboard now does.
            raise HTTPException(status_code=500, detail=outcome.error)
        return {"cleared": outcome.deleted}

    @app.get("/api/evaluations/latest")
    async def get_latest_evaluations() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return latest evaluation scorecard, suite summaries, and probe breakdowns."""
        return session_mgr.get_latest_evaluations()

    @app.get("/api/evaluations/history")
    async def get_evaluation_history(  # pyright: ignore[reportUnusedFunction]
        suite: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return historical evaluation reports, optionally filtered by suite name."""
        return session_mgr.get_evaluation_history(suite=suite, limit=limit)

    @app.get("/api/stream")
    async def event_stream(  # pyright: ignore[reportUnusedFunction]
        request: Request, max_events: int | None = None
    ) -> Response:
        """Server-Sent Events (SSE) endpoint streaming live A2A and engine events."""

        async def event_generator() -> AsyncIterator[str]:
            # Initial connection notification
            seq = _next_sequence_number()
            init_event = {
                "seq": seq,
                "type": "SYSTEM_CONNECTED",
                "priority": "P1",
                "version": __version__,
                "timestamp": asyncio.get_running_loop().time(),
            }
            yield f"data: {json.dumps(init_event)}\n\n"

            bus = active_bus
            sub = bus.subscribe("*")
            count = 0
            shutdown_event: asyncio.Event | None = getattr(
                request.app.state, "shutdown_event", None
            )
            try:
                while max_events is None or count < max_events:
                    if shutdown_event is not None and shutdown_event.is_set():
                        break
                    if _ui_shutdown_event is not None and _ui_shutdown_event.is_set():
                        break
                    if await request.is_disconnected():
                        break
                    try:
                        wait_timeout = 1.0 if max_events is None else 0.5
                        effective_shutdown = shutdown_event or _ui_shutdown_event
                        if effective_shutdown is not None:
                            get_task = asyncio.create_task(sub.get())
                            shutdown_task = asyncio.create_task(effective_shutdown.wait())
                            try:
                                done, pending = await asyncio.wait(
                                    [get_task, shutdown_task],
                                    timeout=wait_timeout,
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                            except BaseException:
                                # The client closing the stream arrives as a
                                # `CancelledError` thrown in at this `await`, and
                                # `GeneratorExit` at `aclose()` the same way. Neither
                                # reaches the drain below, so both waiters are disposed
                                # of here before `sub.close()` in the `finally` wakes
                                # one of them unread (#1038).
                                _discard_waiter(get_task)
                                _discard_waiter(shutdown_task)
                                raise
                            await _cancel_and_drain(pending)
                            if effective_shutdown.is_set():
                                # This drain, not the one above, is what makes the `break`
                                # safe. A subscription closed in this same window wakes
                                # `get_task` with `SubscriptionClosedError`, which lands
                                # it in `done` -- so the `pending` cancellation above
                                # never touched it and the `break` walked away without
                                # reading it (#872).
                                await _cancel_and_drain([get_task])
                                if get_task.done() and not get_task.cancelled():
                                    try:
                                        res: AgentEvent = get_task.result()
                                        seq = _next_sequence_number()
                                        event_payload: dict[str, Any] = {
                                            "seq": seq,
                                            "type": "AGENT_EVENT",
                                            "priority": "P2",
                                            "event_id": res.event_id,
                                            "event_type": res.type.value,
                                            "source": res.source.value,
                                            "sender_id": res.sender_id,
                                            "recipient_id": res.recipient_id,
                                            "topic": res.topic,
                                            "payload": unwrap_immutable(res.payload),
                                            "timestamp": res.timestamp,
                                            "provenance": _sse_provenance_block(res),
                                        }
                                        yield f"data: {json.dumps(event_payload)}\n\n"
                                    except Exception:
                                        pass
                                break
                            if get_task in done:
                                event = get_task.result()
                            else:
                                raise TimeoutError
                        else:
                            event = await asyncio.wait_for(sub.get(), timeout=wait_timeout)

                        seq = _next_sequence_number()
                        event_payload: dict[str, Any] = {
                            "seq": seq,
                            "type": "AGENT_EVENT",
                            "priority": "P2",
                            "event_id": event.event_id,
                            "event_type": event.type.value,
                            "source": event.source.value,
                            "sender_id": event.sender_id,
                            "recipient_id": event.recipient_id,
                            "topic": event.topic,
                            "payload": unwrap_immutable(event.payload),
                            "timestamp": event.timestamp,
                            "provenance": _sse_provenance_block(event),
                        }
                        yield f"data: {json.dumps(event_payload)}\n\n"
                        count += 1
                    except TimeoutError:
                        if shutdown_event is not None and shutdown_event.is_set():
                            break
                        if _ui_shutdown_event is not None and _ui_shutdown_event.is_set():
                            break
                        if await request.is_disconnected():
                            break
                        seq = _next_sequence_number()
                        heartbeat = {
                            "seq": seq,
                            "type": "HEARTBEAT",
                            "priority": "P3",
                            "timestamp": asyncio.get_running_loop().time(),
                        }
                        yield f"data: {json.dumps(heartbeat)}\n\n"
                        count += 1
                    except SubscriptionClosedError:
                        # End of stream, not an error: the bus closed this subscription
                        # (shutdown, or a reconnect that superseded it). Previously
                        # nothing caught it and it escaped `event_generator` mid-response
                        # (#872).
                        break
                    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
                        break
            finally:
                sub.close()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Mount static files if available
    target_static = static_dir or Path(__file__).parent.parent / "ui_static"
    if target_static.is_dir() and (target_static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(target_static), html=True), name="static")
    else:

        @app.get("/")
        async def fallback_index() -> JSONResponse:  # pyright: ignore[reportUnusedFunction]
            return JSONResponse(
                {
                    "message": "UClone-X Backend API running. Run 'cd frontend && npm run dev' for live Vite UI or build static UI with 'npm run build'.",
                    "version": __version__,
                    "endpoints": [
                        "/api/health",
                        "/api/agents",
                        "/api/sessions",
                        "/api/session/history",
                        "/api/session/history/truncate",
                        "/api/ontology",
                        "/api/skills",
                        "/api/budget",
                        "/api/evaluations/latest",
                        "/api/evaluations/history",
                        "/api/stream",
                        "/docs",
                    ],
                }
            )

    return app
