"""FastAPI backend application for UClone-X developer UI dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import hashlib
import ipaddress
import json
import logging
import os
import re
import subprocess
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

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
from uclone_x.agent.clone_builder import AppScope, ontology_map, provider_tool_binder
from uclone_x.agent.models import (
    PersonaDefinition,
)
from uclone_x.agent.persona_avatar import (
    AVATAR_FORMATS,
    MAX_AVATAR_BYTES,
    AvatarChange,
    AvatarPersonaNotFound,
    AvatarRefused,
    AvatarStaleChange,
    PersonaAvatarStore,
    avatar_url,
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
from uclone_x.core.agent_home import seat_id_for
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
    set_consent,
)
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.models import BASE_PERSONA_TOOLS
from uclone_x.core.remote_worker import (
    COMFYUI_SERVICE_NAME,
    SSHTunnelManager,
    find_configured_ssh_hosts,
    is_valid_host,
    probe_remote_host,
)
from uclone_x.core.session_diagnostics import (
    DEFAULT_MAX_CONVERSATION_TURNS,
    count_active_turns,
)
from uclone_x.core.set_aside import expire_set_aside, set_aside_unreadable
from uclone_x.engine.event_bus import (
    AgentEvent,
    EventBus,
    EventSource,
    EventType,
    SubscriptionClosedError,
)
from uclone_x.errors import (
    LLMProviderError,
    LLMTimeoutError,
    PathTraversalError,
    PlainRefusalError,
    RoomNotFoundError,
)
from uclone_x.evaluation import (
    EvalBackendUnavailableError,
    create_eval_runner,
    default_reports_dir,
)
from uclone_x.i18n import DEFAULT_UI_LANGUAGE, UI_LANGUAGES, UiLanguage, is_ui_language
from uclone_x.link.uclone2.supervisor import LinkSupervisor
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.connections import (
    Connection,
    ConnectionError_,
    ModelRef,
    ModelRefError,
    image_ref_problem,
    kind_rows,
    saved_default_models,
)
from uclone_x.llm.connectors.factory import bind_image_engine_settings
from uclone_x.llm.connectors.ollama import (
    delete_model,
    pull_model,
)
from uclone_x.llm.connectors.saved_choice import (
    SETTINGS_FILE_NAME,
    add_connection,
    remove_connection,
    save_default_models,
    settings_data,
    update_connection,
    update_settings_file,
)
from uclone_x.llm.gateway import (
    Capability,
    DefaultBinding,
    ModelGateway,
    connection_paid,
    unsupported_listing,
)
from uclone_x.llm.model_listing import (
    fetch_available_models as fetch_available_models,
)
from uclone_x.llm.model_listing import (
    list_local_models as list_local_models,
)
from uclone_x.llm.model_listing import (
    list_ollama_entries as list_ollama_entries,
)
from uclone_x.llm.model_listing import (
    read_provider_catalog as read_provider_catalog,
)
from uclone_x.llm.model_listing import (
    vllm_model_ids as vllm_model_ids,
)
from uclone_x.llm.model_listing import (
    vllm_request_headers as vllm_request_headers,
)
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.llm.usage.gate import UsageGate
from uclone_x.llm.usage.store import USAGE_FILE_NAME
from uclone_x.memory.store import CrossSessionMemory, default_cross_session_memory
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.room.service import (
    SESSION_ID_PREFIX as _ROOM_SESSION_PREFIX,
)
from uclone_x.room.service import (
    SESSION_ID_SEPARATOR as _ROOM_SESSION_SEPARATOR,
)
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.shells.ui_process import UI_BIND_HOST_ENV_VAR
from uclone_x.skills.approvals import SkillApprovalLedger
from uclone_x.skills.auditor import (
    SkillRegistry,
    load_approved_skills,
    runtime_skill_store_dir,
)
from uclone_x.skills.models import skill_hidden_from
from uclone_x.skills.proposals import (
    SETTINGS_PERSON,
    SkillDecisionCode,
    SkillProposalChangedError,
    SkillProposalError,
    SkillProposalStore,
)
from uclone_x.skills.shipped_pins import SHIPPED_SKILL_PINS
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import in_app_state_dir, replace_file
from uclone_x.tools.mcp_manager import (
    DuplicateServerError,
    MCPServerManager,
    MCPServerSpec,
    UnknownServerError,
)
from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.registry import create_default_registry
from uclone_x.tools.tool_binder import ToolBinder
from uclone_x.ui.knowledge import knowledge_graph
from uclone_x.ui.single_flight import SingleFlight

logger = logging.getLogger(__name__)
_console = Console()

_T = TypeVar("_T")

#: The connection the remote-GPU tunnel's Ollama is added as (model-gateway G1).
REMOTE_GPU_CONNECTION_LABEL = "Remote GPU"
REMOTE_GPU_CONNECTION_ID = "remote-gpu"
#: The connection the remote-GPU tunnel's ComfyUI is added as (model-gateway §3.5): a
#: `comfyui` connection beside the others, so its models join the picture set.
REMOTE_GPU_PICTURES_LABEL = "Remote GPU pictures"
REMOTE_GPU_PICTURES_ID = "remote-gpu-pictures"


def _optional_text(value: object) -> str | None:
    """A request field as stripped text, or `None` when it is absent or blank."""
    return value.strip() if isinstance(value, str) and value.strip() else None


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


def _absorbed_agent_errors(agents: Sequence[BaseAgent]) -> dict[str, list[str]]:
    """Clone id -> the failures its live seats absorbed, for the health reads (#198).

    Keyed by clone, as before rooms: one clone seated in two conversations is one entry
    holding both seats' failures.
    """
    errors: dict[str, list[str]] = {}
    for agent in agents:
        if agent.processing_errors:
            errors.setdefault(agent.agent_id, []).extend(
                str(err) for err in agent.processing_errors
            )
    return errors


def _persona_payload(registry: PersonaRegistry, persona: PersonaDefinition) -> dict[str, Any]:
    """One persona as the dashboard reads it, and as its editor sends it back.

    `system_prompt` is the persona's own text with the appended default prompt split off
    and reported as `append_default_prompt`, so an editor that saves what it was shown
    writes the flag back rather than a frozen copy of the default prompt.
    """
    own_prompt, appended = split_appended_default_prompt(persona.system_prompt)
    builtin = registry.is_builtin(persona.name)
    clone_id = registry.id_of(persona.name)
    # A persona with no clone directory (registered in-process) has no id; its handle
    # stands in, as it does everywhere such a persona is addressed.
    address = clone_id or persona.name
    store = PersonaAvatarStore(registry)
    return {
        # The clone's id: what seats, chats and every route address it by (§4 step 2).
        "id": address,
        # The name typed after @ to address it; unique, and changes only by a rename.
        "handle": persona.name,
        # The handle again, under the key a persona draft names it by when it is saved.
        "name": persona.name,
        # What a person reads as its name, per locale.
        "display_name": dict(persona.display_name),
        "role": persona.role,
        "description": persona.description,
        "system_prompt": own_prompt,
        "append_default_prompt": appended,
        "allowed_tools": list(persona.allowed_tools),
        "base_tools": [*BASE_PERSONA_TOOLS],
        # Model refs (`<connection id>/<model id>`) or null for the system default
        # (model-gateway §3.7.1), flat as a draft names them and grouped as `llm_config`.
        "model_name": persona.llm_config.model_name,
        "fast_model": persona.llm_config.fast_model,
        "image_model": persona.llm_config.image_model,
        "llm_config": {
            "model_name": persona.llm_config.model_name,
            "fast_model": persona.llm_config.fast_model,
            "image_model": persona.llm_config.image_model,
        },
        "model_tier": str(persona.llm_config.model_tier),
        "temperature": persona.llm_config.temperature,
        "max_tokens": persona.llm_config.max_tokens,
        "enable_write_tools": persona.enable_write_tools,
        "enable_subagent_tools": persona.enable_subagent_tools,
        # Clone ids, as stored; a head shows each by the peer's own entry in this list.
        "a2a_peers": list(persona.a2a_peers),
        "builtin": builtin,
        "overrides_builtin": not builtin and registry.has_builtin(persona.name),
        # The address to show its picture from; the `?v=` changes whenever the picture does.
        "avatar_url": avatar_url(address, store.find(persona.name)),
        # Whether that picture was chosen here rather than shipped: only a chosen one resets.
        "avatar_chosen": store.chosen(persona.name) is not None,
        # The id of the latest change to that picture, from any tab or the clone itself: a
        # head hides an Undo whose change is no longer the latest.
        "avatar_change_id": store.latest_change(persona.name),
    }


def _required_agent_id(raw: object) -> str:
    """Read the agent a request names, refusing the request when it names none.

    Read by `/api/dispatch`, which, like the retired chat routes (#1731), used to fall
    back to a built-in persona name when the request carried no `agent_id`. On an install whose agents are its own -- which is every
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
                "there is no default agent. `GET /api/clones` lists the ones this "
                "install has."
            ),
        )
    return raw.strip()


def _next_sequence_number() -> int:
    """Increment and return global SSE sequence counter."""
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


def _turns_taken(transcript: Sequence[dict[str, Any]]) -> int:
    """Turns this transcript records the agent as having taken, failed ones included (#1023).

    A turn that failed was still a turn taken -- the Core's own `turn_counter` reads 4 after
    four sends of which one failed -- so a failure row counts here, and `turn_counter` stays
    the lifetime figure it is (P5).

    Read by `active_turns` only when the Core has nothing to say about the session.
    """
    return sum(
        1
        for entry in transcript
        if entry.get("role") in ("assistant", "agent") or entry.get("sender") == "agent"
    )


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


#: No key held for a provider: no key, no source, no variable.
class AgentSessionManager:
    """The app's clone scope, settings and session records, for the UI layer.

    It holds no agent. Every conversation is a room (D1 Rev 23), and a room's seats are
    built and cached by that room's `RoomAgentResolver` (`RoomStack`); this class hands
    them the scope they are built from (`app_scope`).
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        llm: LLMProviderProtocol | None = None,
        tools: ToolRegistryProtocol | None = None,
        tracer: TelemetryTracer | None = None,
        fallback_to_mock: bool = False,
        storage_dir: Path | None = None,
        skill_registry: SkillRegistry | None = None,
        budget_tracker: TokenBudgetManager | None = None,
        eval_reports_dir: Path | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        self._bus = bus if bus is not None else get_ui_event_bus()
        #: A connector the embedding head handed in (a test's fake, a demo): it answers every
        #: clone that follows the system default. `None` in the product, where every model
        #: comes from the gateway's connections.
        self._given_llm = llm
        #: Told when the connections or default models change. A conversation's seats are
        #: built and cached by the room stack, which this class cannot see; without this
        #: they kept the connector they were built with (#1446).
        self._models_listeners: list[Callable[[], None]] = []
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
        # A transcript kept aside when its conversation was cleared is kept 30 days once no
        # transcript has been written under its name again, as a Core record's copy is (#1877).
        expire_set_aside(self._transcript_dir, suffix=".json")
        # Over the runtime skill store by default, which `ucx skill approve` writes; the
        # approved skills in it are loaded at app startup (`create_ui_app`'s lifespan).
        # A registry built with no directory is inert: `reload_approved` loads nothing,
        # every `load_skill` is refused and the prompt lists no skill (P9).
        self._skill_registry = (
            skill_registry
            if skill_registry is not None
            else SkillRegistry(skills_dir=runtime_skill_store_dir())
        )
        self._budget_tracker = (
            budget_tracker if budget_tracker is not None else TokenBudgetManager()
        )
        self._eval_reports_dir = (
            eval_reports_dir.resolve() if eval_reports_dir is not None else default_reports_dir()
        )
        # One memory store instance per agent id, not per session. Two live sessions of the
        # same agent constructing their own stores over the same file would each hold the
        # whole fact set in memory and each `save()` the whole of it, so the second writer
        # drops whatever the first recorded after it loaded (#1097).
        self._agent_memories: dict[str, CrossSessionMemory] = {}
        #: Clone id -> that clone's one rules engine, as `_agent_memories` is for memory
        #: (clone-knowledge-graph step 6). Every head this manager serves builds its clones
        #: from this map, so a clone reasons under one set of rules in chat and in rooms.
        self._clone_ontologies = ontology_map()
        #: (provider, base URL) -> the host binder for it, or `None` where every tool is
        #: pinned. Shared by every clone, so a tool description is embedded once.
        self._tool_binders: dict[tuple[str, str], ToolBinder | None] = {}
        self._session_messages: dict[str, list[dict[str, Any]]] = {}
        #: Folders outside the workspace that clones may read, as the user entered them.
        self._configured_read_roots: tuple[str, ...] = ()
        #: The language the heads and the CLI write in; `"system"` follows the OS or browser.
        self._configured_ui_language: UiLanguage = DEFAULT_UI_LANGUAGE
        # The same file `ucx run` and `ucx install` read and seed (`saved_choice.py`).
        self._settings_file: Path = self._storage_dir / SETTINGS_FILE_NAME
        #: Whether a Settings save of this dashboard found the file unreadable and moved it
        #: aside (#1860). Settings says so from then on, for as long as this dashboard runs:
        #: the keys the person saved are in that copy, not in the file Settings now shows.
        self._settings_set_aside = False
        # This dashboard's paid calls are held to the limits in its own settings file and
        # booked in its own storage directory, which its Usage panel reads.
        self._usage_file: Path = self._storage_dir / USAGE_FILE_NAME
        self._usage_gate = UsageGate.for_storage(self._settings_file, self._usage_file)
        #: Every model comes from here: the connections and default models in this
        #: dashboard's settings file, a connector per ref (model-gateway §3.3).
        self._gateway = ModelGateway(
            self._settings_file,
            usage_gate=self._usage_gate,
            default_binding=DefaultBinding(llm) if llm is not None else None,
        )
        self._load_persisted_settings()
        #: The local port of the remote-GPU tunnel's ComfyUI while it is connected. The
        #: tunnel belongs to the app (`create_ui_app`), which binds its own answer here.
        self._gpu_tunnel_comfy_port: Callable[[], int | None] = lambda: None
        # Pictures follow this dashboard's settings file -- the image connections and the
        # picture models -- and the remote-GPU tunnel, read again on every draw, so a change
        # in Settings applies to the next one (model-gateway §3.5).
        bind_image_engine_settings(
            self._tools.get("generate_image"),
            self._settings_file,
            self.gpu_tunnel_comfy_port,
        )
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
    def settings_file(self) -> Path:
        """This dashboard's `settings.json`, in its storage directory."""
        return self._settings_file

    @property
    def usage_file(self) -> Path:
        """This dashboard's paid-model usage store, in its storage directory."""
        return self._usage_file

    @property
    def gateway(self) -> ModelGateway:
        """The model gateway every clone, room and Settings route here asks for a model."""
        return self._gateway

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def tools(self) -> ToolRegistryProtocol:
        return self._tools

    @property
    def tracer(self) -> TelemetryTracer:
        return self._tracer

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

    def gpu_tunnel_comfy_port(self) -> int | None:
        """The local port the connected remote-GPU tunnel forwards to its ComfyUI, if any."""
        return self._gpu_tunnel_comfy_port()

    def bind_gpu_tunnel(self, comfy_port: Callable[[], int | None]) -> None:
        """Answer `gpu_tunnel_comfy_port` from ``comfy_port``, read on every call."""
        self._gpu_tunnel_comfy_port = comfy_port

    def _default_deep_connection(self) -> Connection | None:
        ref = self._gateway.resolve(None, "deep")
        return self._gateway.connection(ref.connection_id) if ref is not None else None

    @staticmethod
    def _mask_key(key: str) -> str:
        """``AQ.…abcd``-style: enough to recognise a key, never enough to use it."""
        trimmed = key.strip()
        return f"{trimmed[:3]}...{trimmed[-4:]}" if len(trimmed) > 8 else "***"

    def on_models_changed(self, listener: Callable[[], None]) -> None:
        """Call `listener` whenever the connections or default models change."""
        self._models_listeners.append(listener)

    def models_changed(self) -> None:
        """Tell every open conversation that the connections or default models changed."""
        for listener in self._models_listeners:
            listener()

    # -- Connections and default models (model-gateway §3.7.1) --

    def _write(self, change: Callable[[], _T]) -> _T:
        """Run ``change`` (a write through the one writer), keeping an unreadable file aside.

        A file this build cannot read is refused by the writer with a sentence naming its
        path, which must not reach the page (#1860). As a Settings save always has, the file
        is kept aside, unchanged, and replaced with what this dashboard holds; the change is
        then made in the new file and Settings says the earlier one was kept (#1877).

        Raises:
            ConnectionError_ | ModelRefError: the change itself is refused, in plain words.
            OSError: the file could not be kept aside or written.
        """
        try:
            return change()
        except PlainRefusalError:
            raise
        except ValueError as refusal:
            aside = update_settings_file(
                {}, path=self._settings_file, replace_unreadable_with=self._persisted_settings()
            )
            if aside is None:
                raise
            logger.warning("Settings could not be read before a change was saved: %s", refusal)
            self._settings_set_aside = True  # and Settings says the earlier file was kept
        try:
            return change()
        except PlainRefusalError:
            raise
        except ValueError as exc:  # unreadable again: another writer since it was kept aside
            raise OSError("the settings file could not be read, so nothing was saved") from exc

    def connection_payload(self, conn: Connection) -> dict[str, Any]:
        """One connection as `GET /api/connections` reports it (§3.7.1)."""
        listing = self._gateway.last_listing(conn)
        if listing is None and conn.unsupported:
            # Said at once, with no check: a kind this build does not know has nothing to ask.
            listing = unsupported_listing(conn)
        status = "unchecked" if listing is None else listing.status
        if listing is None and conn.kind in ("gemini", "openai", "anthropic") and not conn.key:
            status = "no_key"
        return {
            "id": conn.id,
            "kind": conn.kind,
            "label": conn.display_label,
            "base_url": conn.base_url,
            "key_set": bool(conn.key),
            "key_masked": self._mask_key(conn.key) if conn.key else None,
            "source": conn.source,
            "env_var": conn.env_var,
            "key_env_var": conn.key_env_var,
            "paid": connection_paid(conn),
            "status": status,
            "detail": None if listing is None else listing.detail,
            "model_count": None
            if listing is None or listing.status != "connected"
            else len([e for e in listing.entries if e.chat_capable]),
        }

    def connections_payload(self) -> dict[str, Any]:
        return {
            "connections": [self.connection_payload(c) for c in self._gateway.connections()],
            "kinds": kind_rows(),
        }

    def _writable_connection(self, conn_id: str) -> Connection:
        """The saved row ``conn_id`` names; refused when it is unknown or set by a variable."""
        conn = self._gateway.connection(conn_id)
        if conn is None:
            raise ConnectionError_(f"There is no connection called {conn_id!r}.")
        if conn.source == "env":
            variable = conn.env_var or "an environment variable"
            raise ConnectionError_(
                f"The connection {conn.id} is set by {variable}, so it cannot be changed "
                f"here. Change or unset the variable instead."
            )
        return conn

    async def check_connection(self, conn_id: str) -> dict[str, Any]:
        """Ask the connection for its models now; return it with what it said."""
        conn = self._gateway.connection(conn_id)
        if conn is None:
            raise ConnectionError_(f"There is no connection called {conn_id!r}.")
        await self._gateway.listing(conn, refresh=True)
        return self.connection_payload(conn)

    async def add_connection(
        self, kind: str, *, label: str | None, base_url: str | None, key: str | None
    ) -> dict[str, Any]:
        conn = self._write(
            lambda: add_connection(
                kind, label=label, base_url=base_url, key=key, path=self._settings_file
            )
        )
        self.models_changed()
        return await self.check_connection(conn.id)

    async def patch_connection(self, conn_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        self._writable_connection(conn_id)
        fields = {name: changes[name] for name in ("label", "base_url", "key") if name in changes}
        self._write(lambda: update_connection(conn_id, path=self._settings_file, **fields))
        self.models_changed()
        return await self.check_connection(conn_id)

    def remove_connection(self, conn_id: str) -> None:
        self._writable_connection(conn_id)
        self._write(lambda: remove_connection(conn_id, path=self._settings_file))
        self.models_changed()

    def connection_dependents(self, conn_id: str) -> dict[str, Any]:
        """The clones and defaults that name ``conn_id`` (§3.6: listed before a removal)."""
        from uclone_x.agent.persona_registry import get_default_persona_registry

        prefix = f"{conn_id}/"
        registry = get_default_persona_registry(self._workspace_dir)
        clones: list[dict[str, Any]] = []
        for persona in registry.list_personas():
            cfg = persona.llm_config
            slots = [
                slot
                for slot, value in (
                    ("model_name", cfg.model_name),
                    ("fast_model", cfg.fast_model),
                    ("image_model", cfg.image_model),
                )
                if value and value.startswith(prefix)
            ]
            if slots:
                clones.append(
                    {
                        "id": registry.id_of(persona.name) or persona.name,
                        "name": persona.name,
                        "slots": slots,
                    }
                )
        saved = saved_default_models(settings_data(self._settings_file))
        defaults = [
            slot
            for slot, value in (("deep", saved.deep), ("fast", saved.fast), ("image", saved.image))
            if value and value.startswith(prefix)
        ]
        return {"clones": clones, "defaults": defaults}

    def defaults_payload(self) -> dict[str, Any]:
        defaults = self._gateway.defaults()
        return {"deep": defaults.deep, "fast": defaults.fast, "image": defaults.image}

    def save_default_models(self, changes: Mapping[str, str | None]) -> None:
        """Save the default refs ``changes`` names (S1: the one writer)."""
        self._write(lambda: save_default_models(changes, path=self._settings_file))
        self.models_changed()

    def _load_persisted_settings(self) -> None:
        """Load persisted settings from storage directory if available.

        Nothing read here is copied into `os.environ`: the connector is built from explicit
        arguments, and the environment stays what the operator set.
        """
        if not self._settings_file.is_file():
            return
        try:
            data = json.loads(self._settings_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read settings file %s: %s", self._settings_file, exc)
            return
        if not isinstance(data, dict):
            return
        cfg = cast(dict[str, Any], data)
        # The connections and default models are not held here: the gateway reads them
        # from the file on every call (S1), and the pre-gateway `llm_*` keys are never read.
        raw_language: object = cfg.get("ui_language")
        if is_ui_language(raw_language):
            self._configured_ui_language = raw_language
        raw_roots: object = cfg.get("read_roots")
        if isinstance(raw_roots, list):
            self._configured_read_roots = tuple(
                entry.strip()
                for entry in cast(list[object], raw_roots)
                if isinstance(entry, str) and entry.strip()
            )

    def _persisted_settings(self) -> dict[str, Any]:
        """Every setting this dashboard holds, as a settings file written whole holds them.

        The connections and default models are not held in memory, so a file replaced
        after it could not be read starts without them (#1860).
        """
        return {
            "read_roots": list(self._configured_read_roots),
            "ui_language": self._configured_ui_language,
        }

    def _save_persisted_settings(self, changes: dict[str, Any]) -> None:
        """Merge the settings this save changed into the settings file.

        Only `changes` are written: the file is shared with setup, `ucx key` and
        `ucx llm use`. When the file cannot be read, it is replaced with everything this
        dashboard holds, as a Settings save always did.
        """
        everything = self._persisted_settings()
        try:
            aside = update_settings_file(
                changes, path=self._settings_file, replace_unreadable_with=everything
            )
            if aside is not None:
                self._settings_set_aside = True  # a Settings save kept the file aside
        except Exception as exc:
            logger.warning("Failed to write settings file %s: %s", self._settings_file, exc)

    def get_settings(self) -> dict[str, Any]:
        """The settings Settings shows, other than the connections and models.

        Those have their own routes (`GET /api/connections`, `GET /api/models`), and the
        pre-gateway `llm_*` fields are gone (model-gateway step 3).
        """
        return {
            # The words are the page's, in the person's language: this says only that it
            # happened, never where the copy is or why it could not be read (#1860).
            "settings_set_aside": self._settings_set_aside,
            "default_models": self.defaults_payload(),
            "workspace_dir": str(self._workspace_dir),
            "ui_language": self._configured_ui_language,
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

    def update_settings(
        self,
        read_roots: list[str] | None = None,
        ui_language: object = None,
        default_models: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Save the fields this request names, each on its own; return the settings.

        Nothing is written into `os.environ`. ``default_models`` is checked against the
        model set by the route before it reaches here (§3.7.1); a malformed ref is refused
        here, before anything is changed.
        """
        if ui_language is not None and not is_ui_language(ui_language):
            raise ValueError(
                f"ui_language must be one of {', '.join(UI_LANGUAGES)}, got {ui_language!r}"
            )
        clean_roots = (
            _validate_read_roots(read_roots, self._storage_dir, self._configured_read_roots)
            if read_roots is not None
            else None
        )
        if default_models is not None:
            self.save_default_models(default_models)

        if clean_roots is not None:
            # Every conversation seat reads the list again at its next turn
            # (`RoomAgentResolver.resolve`), so nothing is pushed to a live agent here.
            self._configured_read_roots = clean_roots

        changes: dict[str, Any] = {}
        if clean_roots is not None:
            changes["read_roots"] = list(self._configured_read_roots)
        if ui_language is not None:
            self._configured_ui_language = ui_language
            changes["ui_language"] = ui_language
        if changes:
            self._save_persisted_settings(changes)
        return self.get_settings()

    def get_session_path(self, session_id: str) -> Path:
        """Resolve the UI transcript path for `session_id`, refusing escapes (P3).

        Under `<root>/ui/`, not `<root>/` — the root path belongs to the Core store and
        sharing it destroyed conversations in both directions.

        Delegates to `uclone_x.core.session.resolve_session_path`, which is the single
        implementation of this guard. It previously existed twice — here and in the Core
        store — and a duplicated security control is one that gets fixed in a single copy.

        Raises:
            PathTraversalError: See `resolve_session_path`.
        """
        return resolve_session_path(self._transcript_dir, session_id)

    def load_session_record(self, session_id: str) -> dict[str, Any] | None:
        """Load the UI transcript at `<root>/ui/<id>.json`, or `None` if there is none.

        Only the namespaced path is read. A file at the family root is not a transcript
        this build writes, so it is not consulted.

        **A transcript belonging to a different session is refused (#256).** The Core store
        is not the only artifact keyed by a session id in a filename, so it is not the only
        one the case- and normalization-insensitive filesystem folds. Measured on
        `e8b3e2f`: after a transcript was saved for `"SessA"`,
        `load_session_record("SESSA")` returned `SessA`'s transcript — including its
        `session_id` — and the history reader cached it under `"SESSA"`, so the
        dashboard displayed one session's conversation as another's. `#256` names only the
        Core store; this door was found by enumerating the id-bearing surface. The rule is
        `agent.session.verify_record_identity`, the same one the Core store uses, not a
        second copy.

        The check is conditional on the record actually carrying a `session_id` string,
        which is deliberate: it refuses on **positive evidence** of a different owner and
        stays silent about a file too damaged to name its session, which is already
        handled by the shape mismatch below.

        Raises:
            PathTraversalError: See `core.session.resolve_session_path`.
            SessionIdCollisionError: If the transcript found identifies another session.
        """
        path = self.get_session_path(session_id)
        if not path.is_file():
            return None
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

    def _live_messages(
        self, agent_id: str, session_id: str, agent: BaseAgent | None = None
    ) -> tuple[list[ChatMessage], int] | None:
        """The Core's current messages and lifetime turn counter, or `None` if unknown.

        The live agent is preferred over the store because it is the writer: an agent
        holding an unpersisted turn is ahead of the record, never behind it.

        `agent` is the writer, when the caller holds one. A conversation seat's agent is
        built and cached by `RoomAgentResolver` and this manager holds none, so without it
        this reads the persisted copy -- behind the live agent by whatever it has not
        written yet, which is exactly the turn a reader is asking about.
        """
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

        **`agent` names the live writer when the caller holds one.** A conversation seat's
        agent is built and cached by `RoomAgentResolver`, and this manager holds no agent,
        so without it the branch below would delete the stored record while that live agent
        went on holding the messages it had — and persisted them back over the deletion at its next turn, which
        presents as a clear that silently did nothing. Passing the agent takes the reset
        branch, so the in-memory copy and the record are cut by the same call.

        Raises:
            PathTraversalError: See `core.session.resolve_session_path`.
            SessionIdCollisionError: If either artifact at this id's paths identifies a
                different session. Nothing is reset or unlinked.
        """
        # Path resolution first, so a traversal is refused before anything is touched.
        path = self.get_session_path(session_id)
        # Then ownership, before anything is reset or unlinked. `load_session_record`
        # raises on a transcript naming another session and returns `None` when there is
        # nothing there to own.
        transcript = self.load_session_record(session_id)

        # Core reset **before** the transcript unlink. The previous order deleted the
        # transcript first, so a Core reset that then failed left `transcript False /
        # core True` — the displayed history gone while the conversation the agent
        # actually reasons over was intact, which is the least recoverable of the four
        # possible outcomes because the user sees an empty pane and the model does not.
        # Resetting Core first means a failure leaves *both* sides untouched.
        if agent is not None:
            agent.reset_session(session_id)
        else:
            # No live agent, so the record is *deleted* rather than reset in place.
            # `SessionState.reset` needs the agent's `config.system_prompt` to re-seed,
            # and with no agent constructed there is nothing authoritative to read it
            # from — calling `reset()` with the empty default would persist a session
            # with no system prompt at all, which is strictly worse than no record.
            # Deleting makes the next clone built for it seed correctly from config.
            self._core_store.delete(session_id)

        # Only now the transcript, once Core has definitely been reset.
        self._session_messages.pop(session_id, None)
        if transcript is None and path.is_file():
            # There and unreadable -- damaged, or written by a build this one cannot
            # parse: kept beside its name, not unlinked, like the Core record above
            # (#1860). If it cannot be moved, it is left where it is.
            try:
                aside = set_aside_unreadable(path)
            except OSError as exc:
                logger.warning(
                    "Transcript at %s cannot be read and could not be moved aside (%s); "
                    "it was left where it is",
                    path,
                    exc,
                )
            else:
                logger.warning(
                    "Transcript at %s cannot be read; it was moved aside, unchanged, to %s",
                    path,
                    aside,
                )
        elif path.exists():
            path.unlink(missing_ok=True)

    def app_scope(self, llm: LLMProviderProtocol | None = None) -> AppScope:
        """The app scope every clone this manager serves is built from (§5.9.2).

        One for a 1:1 chat and one per room, over the same parts: the chat and a room seat
        of one clone differ only in what the room adds (owner ruling 2026-09-27). Each clone
        gets its own connector from its resolved model ref through the gateway
        (model-gateway §3.4); `llm` answers clones that follow the default when no default
        model is saved and none was given at construction.

        The persona registry is read afresh, against the live tool inventory -- MCP names
        included -- so `allowed_tools` is checked against what is registered now. The
        binder is kept per (provider, base URL) of the default deep model's connection, so
        each tool description is embedded once for every clone; the grow-only bound set
        itself is per session.
        """
        from uclone_x.agent.persona_registry import get_default_persona_registry

        if llm is not None and self._gateway.default_binding is None:
            self._gateway.set_default_binding(DefaultBinding(llm))
        default_conn = self._default_deep_connection()
        binder_key = (
            default_conn.kind if default_conn is not None else "",
            (default_conn.base_url or "") if default_conn is not None else "",
        )
        if binder_key not in self._tool_binders:
            self._tool_binders[binder_key] = provider_tool_binder(*binder_key)

        def default_llm() -> LLMProviderProtocol | None:
            return self._gateway.default_deep()[0]

        scope = AppScope.create(
            workspace_root=self._workspace_dir,
            persona_registry=get_default_persona_registry(
                self._workspace_dir,
                tool_names=[tool.name for tool in self._tools.list_tools()],
            ),
            memory_for=self.memory_for,
            gateway=self._gateway,
            read_roots=lambda: self.read_roots,
            bus=self._bus,
            llm=default_llm(),
            tools=self._tools,
            tracer=self._tracer,
            store=self._core_store,
            budget=self._budget_tracker,
            skills=self._skill_registry,
            # Each clone's own engine, never the manager's (step 6): one engine on the app
            # scope was every clone's, and what one clone was taught reached them all.
            ontology_for=self.ontology_for,
            tool_binder=self._tool_binders[binder_key],
            # The desktop app has no approval prompt in a conversation, so a call that
            # needs a person is refused at once and names where to decide instead (the
            # story view for codex proposals; owner decision 2026-09-26).
            approvals_answered=False,
        )
        # A peer a chat clone calls starts from the default connector in effect at the call;
        # the gateway then binds its own model (`PersonaTaskHandler`).
        return dataclasses.replace(
            scope, live_host=lambda: dataclasses.replace(scope.host, llm=default_llm())
        )

    def ontology_for(self, agent_id: str) -> OntologyEngineProtocol:
        """The one rules engine of clone `agent_id`, created on first use (step 6).

        Public for the same reason as `memory_for`: `RoomStack` seats the same ids and its
        knowledge read reports the rules this map holds, not a copy of its own.
        """
        return self._clone_ontologies(agent_id)

    def memory_for(self, agent_id: str) -> CrossSessionMemory:
        """The one cross-session memory store for `agent_id`, created on first use.

        Public, and the only such map on the manager that serves the head: chat sessions
        are not the only seat an agent takes. `RoomStack` seats the same ids in rooms and
        must reach *this* map rather than keep one of its own. Writes no longer race (each is one
        sqlite transaction), but a store also holds the clone's embedding index, built from
        the facts once per object, so a second map keyed the same way would build and
        refresh a second index over one file. The ids collide by design: an install's agents are both
        its chat agents and the seats a conversation puts them in.

        Get-or-create has no `await` between the read and the write, so two coroutines
        cannot race a second store into being.
        """
        # Keyed by clone id: a handle is resolved first, so one clone never has two stores.
        key = seat_id_for(agent_id)
        existing = self._agent_memories.get(key)
        if existing is not None:
            return existing
        store = default_cross_session_memory(key)
        self._agent_memories[key] = store
        return store

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

        Only the `artifacts/` directories are read. The workspace's own `docs/` and root-level
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

        # 1. The session's own artifacts directory, if session_id provided. Tool results
        # are not listed: they are kept in the session store (#1848), not the workspace,
        # and are not documents a clone produced.
        if session_id and session_id.strip():
            clean_sid = session_id.strip()
            session_custom_dir = root / "artifacts" / clean_sid
            if session_custom_dir.is_dir():
                for p in sorted(session_custom_dir.rglob("*")):
                    _add_file(p)

        # 2. Workspace artifacts directory. Not session-scoped yet: the image tools now save
        # under `artifacts/<session id>/images/` (#1390), but pictures saved before that, and
        # those given an explicit `output_path`, sit outside any per-session directory.
        artifacts_dir = root / "artifacts"
        if artifacts_dir.is_dir():
            for p in sorted(artifacts_dir.rglob("*")):
                _add_file(p)

        results.sort(key=lambda x: str(x.get("modified_at", "")), reverse=True)
        return results

    def get_artifact_file(
        self, path: str, session_id: str | None = None, *, root: Path | None = None
    ) -> tuple[Path, str]:
        """Safely retrieve artifact Path and detected MIME type from workspace (P6).

        `root` is the workspace to read from (a conversation's own, clone-data-scopes §3.6);
        `None` is the server's.
        """
        if not path or not path.strip():
            raise PathTraversalError("Path must not be empty.")
        clean_path = path.strip()
        if "\0" in clean_path:
            raise PathTraversalError("Null byte detected in path.")

        validator = PathValidator()
        resolved = validator.resolve_safe_path(Path(clean_path), root or self._workspace_dir)

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

    def get_knowledge_graph(self, agent_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Return dynamic entity-relation triples (subject, predicate, object, provenance, tier) (RFC §6.1).

        From clone `agent_id`'s one rules engine (`ontology_for`), the one its seats and
        chats reason with: the manager keeps no shared engine (clone-knowledge-graph §3.8,
        #1869).
        """
        return knowledge_graph(self.ontology_for(agent_id), session_id=session_id)


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
    skill_registry: SkillRegistry | None = None,
    budget_tracker: TokenBudgetManager | None = None,
    eval_reports_dir: Path | None = None,
    workspace_dir: Path | None = None,
    shutdown_event: asyncio.Event | None = None,
    bind_host: str | None = None,
    link_supervisor: LinkSupervisor | None = None,
) -> FastAPI:
    """Create and configure the FastAPI developer dashboard application.

    `link_supervisor` runs the uClone2 link sessions while the app is up; by default, one
    over the links file (`UCLONE_LINKS_DIR`). With no link stored, nothing is dialled.

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
            skill_registry=skill_registry,
            budget_tracker=budget_tracker,
            eval_reports_dir=eval_reports_dir,
            workspace_dir=workspace_dir,
        )
    )
    if eval_reports_dir is not None and session_manager is not None:
        session_mgr.eval_reports_dir = eval_reports_dir

    # Before any route reads a clone (clone-data-scopes §3.8): migrate the agents root,
    # import the launch directory's personas, install the builtins. A no-op once done.
    # The package directory is read through the registry module, as the registry reads it.
    from uclone_x.agent import persona_registry as _registry_module
    from uclone_x.agent.clone_store import ensure_clone_store

    _package_dir = _registry_module.BUILTIN_PERSONAS_DIR
    ensure_clone_store(
        session_mgr.workspace_dir, builtin_dir=_package_dir if _package_dir.is_dir() else None
    )

    # Per app, like the session manager: the user's external MCP servers, whose tools are
    # registered into the same registry every clone's turn reads its tools from.
    mcp_manager = MCPServerManager(
        registry=session_mgr.tools,
        config_path=session_mgr.storage_dir / "mcp_servers.json",
        workspace_root=session_mgr.workspace_dir,
    )
    tunnel_manager = SSHTunnelManager()
    # Where a picture is drawn reads the tunnel: ComfyUI on its local port is the GPU server.
    session_mgr.bind_gpu_tunnel(lambda: tunnel_manager.get_status().comfyui_local_port)
    links = link_supervisor if link_supervisor is not None else LinkSupervisor()
    # What a remote-GPU connect overwrote, kept on disk rather than in memory: the tunnel is
    # a child process that dies with this one (or on its own), and a restore held only in
    # memory left `127.0.0.1:11435` saved as the LLM address after a restart, pointing at
    # nothing. The record holds, per field the connect changed, the value it replaced
    # ("original") and the value it wrote ("applied"). A field is put back only while it
    # still holds the applied value, so an address the user changed by hand in between
    # is left alone.
    remote_restore_path = session_mgr.storage_dir / "remote_gpu_restore.json"

    def _str_map(raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        items = cast("dict[object, object]", raw).items()
        return {k: v for k, v in items if isinstance(k, str) and isinstance(v, str)}

    def _read_remote_restore() -> tuple[dict[str, str], dict[str, str]]:
        """The (original, applied) maps of the saved record; empty when there is none."""
        try:
            raw: object = json.loads(remote_restore_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, {}
        if not isinstance(raw, dict):
            return {}, {}
        record = cast("dict[object, object]", raw)
        return _str_map(record.get("original")), _str_map(record.get("applied"))

    def _restore_remote_settings() -> dict[str, str]:
        """Put back what the last connect overwrote and is still in place, then forget it.

        Returns only the fields actually put back.
        """
        _original, applied = _read_remote_restore()
        restored: dict[str, str] = {}
        # The tunnel's Ollama and ComfyUI were added as connections of their own; each goes
        # with the tunnel, unless its address was changed by hand since.
        for conn_id, field in (
            (REMOTE_GPU_CONNECTION_ID, "connection"),
            (REMOTE_GPU_PICTURES_ID, "picture_connection"),
        ):
            tunnel_conn = session_mgr.gateway.connection(conn_id)
            if (
                conn_id in applied
                and tunnel_conn is not None
                and tunnel_conn.source == "settings"
                and tunnel_conn.base_url == applied[conn_id]
            ):
                session_mgr.remove_connection(conn_id)
                restored[field] = conn_id
        with contextlib.suppress(FileNotFoundError):
            remote_restore_path.unlink()
        return restored

    tunnel_manager.add_disconnect_listener(_restore_remote_settings)

    # What the last connect asked for, kept until Disconnect is pressed: a restart (or a dev
    # reload) ends the tunnel with the process, and the next start makes the same connect.
    remote_session_path = session_mgr.storage_dir / "remote_gpu_session.json"

    async def _connect_remote_gpu(
        host: str,
        *,
        apply_settings: bool,
        auto_start_comfyui: bool,
        sync_llm: bool,
        timeout: float,
        ports: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Open the tunnel, point the saved addresses at it, and remember the request."""
        ports = ports or {}
        tunnel_status = await tunnel_manager.connect(
            host=host,
            preferred_local_ollama_port=ports.get("ollama"),
            preferred_local_comfyui_port=ports.get("comfyui"),
            auto_start_comfyui=auto_start_comfyui,
            timeout=timeout,
        )
        if not tunnel_status.connected:
            return {
                "status": "error",
                "connected": False,
                "error": tunnel_status.error or "Failed to connect tunnel",
                "tunnel": tunnel_status.to_dict(),
                "llm_on_remote": False,
                "images_on_remote": False,
            }

        changes: dict[str, Any] = {}
        llm_skipped: str | None = None  # kept in the reply's shape; a connection never skips
        if apply_settings:
            # A reconnect keeps the first record: the current values are the tunnel's own.
            original, applied = _read_remote_restore()

            for m in tunnel_status.mappings:
                if m.service_name == "ollama":
                    if not sync_llm:
                        continue
                    # The tunnel's Ollama becomes a connection of its own beside the others
                    # (model-gateway G1), so its models join the set; nothing else is
                    # disconnected, and no default model is changed behind the person.
                    address = f"http://127.0.0.1:{m.local_port}"
                    existing = session_mgr.gateway.connection(REMOTE_GPU_CONNECTION_ID)
                    if existing is None:
                        add_connection(
                            "ollama",
                            label=REMOTE_GPU_CONNECTION_LABEL,
                            base_url=address,
                            path=session_mgr.settings_file,
                        )
                    elif existing.source == "settings":
                        update_connection(
                            REMOTE_GPU_CONNECTION_ID,
                            base_url=address,
                            path=session_mgr.settings_file,
                        )
                    session_mgr.models_changed()
                    applied[REMOTE_GPU_CONNECTION_ID] = address
                    original.setdefault(REMOTE_GPU_CONNECTION_ID, "")
                    changes["connection"] = REMOTE_GPU_CONNECTION_ID
                elif m.service_name == "comfyui":
                    # The tunnel's ComfyUI becomes a picture connection of its own (§3.5),
                    # so its models join the picture set; the person's own ComfyUI, and the
                    # picture models chosen, are left as they are.
                    address = f"http://127.0.0.1:{m.local_port}"
                    existing = session_mgr.gateway.connection(REMOTE_GPU_PICTURES_ID)
                    if existing is None or existing.source == "settings":
                        update_connection(
                            REMOTE_GPU_PICTURES_ID,
                            kind="comfyui",
                            create=True,
                            label=REMOTE_GPU_PICTURES_LABEL,
                            base_url=address,
                            path=session_mgr.settings_file,
                        )
                    session_mgr.models_changed()
                    applied[REMOTE_GPU_PICTURES_ID] = address
                    original.setdefault(REMOTE_GPU_PICTURES_ID, "")
                    changes["picture_connection"] = REMOTE_GPU_PICTURES_ID
            if original:
                remote_restore_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"original": original, "applied": applied}
                replace_file(remote_restore_path, json.dumps(record, indent=2).encode())

        # The local ports go in too, so the next start asks for the same ones and the
        # addresses it writes match the ones this run wrote.
        session_record = {
            "host": host,
            "apply_settings": apply_settings,
            "auto_start_comfyui": auto_start_comfyui,
            "sync_llm": sync_llm,
            "ports": {m.service_name: m.local_port for m in tunnel_status.mappings},
        }
        remote_session_path.parent.mkdir(parents=True, exist_ok=True)
        replace_file(remote_session_path, json.dumps(session_record, indent=2).encode())

        images_on_remote = bool(
            tunnel_status.connected
            and any(m.service_name == COMFYUI_SERVICE_NAME for m in tunnel_status.mappings)
        )
        llm_on_remote = bool(
            tunnel_status.connected
            and any(m.service_name == "ollama" for m in tunnel_status.mappings)
            and changes.get("connection")
        )

        return {
            "status": "ok",
            "connected": True,
            "tunnel": tunnel_status.to_dict(),
            "applied_changes": changes,
            "llm_skipped": llm_skipped,
            "llm_on_remote": llm_on_remote,
            "images_on_remote": images_on_remote,
        }

    async def _reconnect_remote_gpu() -> None:
        """Make again the connect the previous run left open; a failure is only logged.

        The record stays when the worker does not answer, so the start after it tries again;
        only Disconnect forgets it.
        """
        try:
            raw: object = json.loads(remote_session_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        record = cast("dict[str, object]", raw)
        host = record.get("host")
        if not isinstance(host, str) or not is_valid_host(host):
            return
        raw_ports = record.get("ports")
        ports: dict[str, int] = {}
        if isinstance(raw_ports, dict):
            for k, v in cast("dict[object, object]", raw_ports).items():
                if isinstance(k, str) and isinstance(v, int):
                    ports[k] = v
        try:
            result = await _connect_remote_gpu(
                host,
                apply_settings=record.get("apply_settings") is not False,
                auto_start_comfyui=record.get("auto_start_comfyui") is not False,
                sync_llm=record.get("sync_llm") is True,
                timeout=15.0,
                ports=ports,
            )
        except Exception:
            logger.warning("Remote-GPU reconnect to %s failed", host, exc_info=True)
            return
        if result["connected"]:
            logger.info("Remote-GPU tunnel to %s reconnected at startup", host)
        else:
            logger.warning("Remote-GPU reconnect to %s failed: %s", host, result["error"])

    @asynccontextmanager
    async def _app_lifespan(app_inst: FastAPI) -> AsyncGenerator[None, None]:
        # Before the first turn: an approved skill that is not loaded is one no clone is
        # told about and `load_skill` refuses. A missing store loads nothing.
        await load_approved_skills(session_mgr.skill_registry)
        # In the background: a local server launched through `npx` may spend a minute
        # downloading, and the dashboard must not wait on it to open. Until it answers,
        # the server reads as "connecting".
        mcp_start = asyncio.create_task(mcp_manager.start())
        # A record left on disk means the previous run was connected when it stopped. Its
        # tunnel died with it, so the addresses it saved lead nowhere.
        if tunnel_manager.get_status().connected is False and _read_remote_restore()[0]:
            with contextlib.suppress(Exception):
                restored = _restore_remote_settings()
                logger.info("Restored settings a remote-GPU tunnel had replaced: %s", restored)
        # A connect that was never disconnected is made again, in the background: the probe
        # and ssh take seconds, and an unreachable worker must not hold the dashboard up.
        # Until it answers, the addresses just put back are the ones in use.
        reconnect = asyncio.create_task(_reconnect_remote_gpu())
        # The uClone2 links dial out in the background; a stored link that cannot reach
        # uClone2 retries on its own and never holds the dashboard up.
        await links.start()
        # A `/loop` the previous process was running is on its room's record; `--dev`
        # restarts the worker on every change under `src/`, a `git pull` included (#1936).
        loop_stack = getattr(app_inst.state, "room_stack", None)
        if loop_stack is not None:
            try:
                resumed = loop_stack.resume_room_loops()
            except Exception:
                logger.warning("Repeating tasks could not be continued", exc_info=True)
            else:
                if resumed:
                    logger.info("Continued %d repeating task(s) from the room records", resumed)
        yield
        # First, while the event loop is healthy: each session sends `bye{logout}` so
        # uClone2 shows the clone offline at once (bounded at 2 s per session, in parallel).
        with contextlib.suppress(Exception):
            await links.shutdown()
        mcp_start.cancel()
        with contextlib.suppress(BaseException):
            await mcp_start
        reconnect.cancel()
        with contextlib.suppress(BaseException):
            await reconnect
        # Local servers are child processes; left running they outlive the app.
        await mcp_manager.close()
        with contextlib.suppress(BaseException):
            await tunnel_manager.close()
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
    app.state.tunnel_manager = tunnel_manager
    app.state.link_supervisor = links

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
    # At start, after the clone store: every stored room still seating clones by handle is
    # rewritten to their ids, once, as loading it would (clone-data-scopes §4 step 3).
    room_stack.service.survey_rooms()
    app.state.room_stack = room_stack
    register_room_routes(
        app,
        room_stack,
        # Late-bound: the check is defined further down this function, read per request.
        refuse_cross_origin=lambda request: _refuse_cross_origin(request),
    )

    from uclone_x.ui.room_dock import register_room_dock_routes

    # The dock's reads for the conversation on screen and its selected seat (#1353-#1357).
    register_room_dock_routes(
        app,
        room_stack,
        refuse_cross_origin=lambda request: _refuse_cross_origin(request),
    )

    # This lists what is *installed*, not what is running (#1190; the live-instance
    # `/api/agents` was removed 2026-09-27, #1775). It is handed
    # the room stack as well as the session manager because a clone is running in either
    # seat, and under D1 the conversation seat is the ordinary one.
    from uclone_x.ui.clones import CloneCatalog, register_clone_routes

    def _clone_catalog() -> CloneCatalog:
        """Every clone's persona fields, and what an editor offers beside the list.

        `/api/clones` absorbed `/api/personas` (clone-data-scopes §3.7): one resource,
        whose rows carry both what a clone is and whether it is running.
        """
        from uclone_x.agent.persona_registry import get_default_persona_registry
        from uclone_x.core.models import BASE_PERSONA_TOOLS

        # The inventory is passed here too, not only on the agent-creation path: this
        # route serves the dashboard on load and therefore usually reaches the cached
        # registry first, which would otherwise pin the process to an unvalidated one.
        from uclone_x.tools.base import tool_writes_files

        all_tools = session_mgr.tools.list_tools()
        tool_names = [tool.name for tool in all_tools]
        write_tools = [tool.name for tool in all_tools if tool_writes_files(tool)]
        registry = get_default_persona_registry(session_mgr.workspace_dir, tool_names=tool_names)
        entries = [_persona_payload(registry, p) for p in registry.list_personas()]
        writable = registry.writable_dir()
        return entries, {
            # What an editor offers: the tools a clone can name, and where a save lands.
            "available_tools": sorted(tool_names),
            "personas_dir": str(writable) if writable is not None else None,
            "base_tools": list(BASE_PERSONA_TOOLS),
            "write_tools": sorted(write_tools),
        }

    register_clone_routes(
        app,
        session_mgr,
        room_stack,
        _clone_catalog,
        refuse_cross_origin=lambda request: _refuse_cross_origin(request),
    )

    # The Files screen: the artifact folders across every conversation, including deleted
    # ones, apart from the room-scoped Docs dock (#1554).
    from uclone_x.ui.artifacts import register_artifact_routes
    from uclone_x.ui.person import PersonGate, register_person_routes

    # The secret that tells a person's decision from a program's (#1589 item 6). Made here,
    # per app, and kept in memory only: never in `os.environ`, where every tool subprocess
    # would inherit it.
    person_gate = PersonGate()
    app.state.person_gate = person_gate
    register_person_routes(app, person_gate)
    register_artifact_routes(
        app,
        room_stack,
        person_gate,
        refuse_cross_origin=lambda request: _refuse_cross_origin(request),
    )
    # What extensions add to the head -- the story view, for the story extension (#2205).
    from uclone_x.ui.extension_routes import register_extension_routes

    register_extension_routes(
        app,
        room_stack,
        person_gate,
        refuse_cross_origin=lambda request: _refuse_cross_origin(request),
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Health check and absorbed failure accounting endpoint (#198)."""
        live_agents = room_stack.live_agents()
        agent_errors = _absorbed_agent_errors(live_agents)
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
                    "count": active_tracer.buffer_evicted_span_count,
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
        live_agents = room_stack.live_agents()
        agent_errors = _absorbed_agent_errors(live_agents)
        total_agent_errors = sum(len(errs) for errs in agent_errors.values())

        return {
            "version": __version__,
            "git_commit": get_git_commit(),
            "started_at": SERVER_START_TIME,
            "runtime": "uclone_x",
            "agents": {
                "total": len({ag.agent_id for ag in live_agents}),
                "active_states": {ag.agent_id: ag.state.value for ag in live_agents},
            },
            "absorbed_failures": {
                "dropped_spans": {
                    "count": active_tracer.buffer_evicted_span_count,
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

    @app.get("/api/media/status")
    async def media_status(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Which picture model would draw now under the default picture model, and why.

        The probe `ucx media status` prints, run off the event loop: it makes two local
        HTTP probes. The cloud is judged by whether its connection has a key, never by a
        request. A chosen model that cannot be used is said in ``resolved.refusal``.
        """
        _refuse_cross_origin(request)
        from uclone_x.tools.builtin.image_status import media_status_payload, probe_image_engines

        try:
            report = await asyncio.to_thread(
                functools.partial(probe_image_engines, detect_comfyui=True),
                session_mgr.settings_file,
                session_mgr.gpu_tunnel_comfy_port(),
            )
        except PlainRefusalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return media_status_payload(report)

    @app.get("/api/settings")
    async def get_settings(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The settings other than connections and models (those have their own routes)."""
        _refuse_cross_origin(request)  # names the folders clones can read, and the workspace
        return session_mgr.get_settings()

    #: What a write that failed on the disk says: never the path or the cause (#1860).
    settings_not_saved = "The settings could not be saved. The reason is in the log."

    def _connection_refusal(exc: Exception, status_code: int = 400) -> JSONResponse:
        # The plain sentence the refusal was written with; never a class name or a path.
        if not isinstance(exc, PlainRefusalError):
            logger.warning("A connection change was not saved: %s", exc)
            return JSONResponse({"detail": settings_not_saved}, status_code=500)
        return JSONResponse({"detail": str(exc)}, status_code=status_code)

    @app.get("/api/connections")
    async def list_connections(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Every connection and the kinds that can be added (model-gateway §3.7.1)."""
        _refuse_cross_origin(request)  # names the hosts the user reaches and masked keys
        return session_mgr.connections_payload()

    async def _settings_event(updated: dict[str, Any]) -> None:
        sys_pub = active_bus.register_publisher(sender_id="ui_settings", source=EventSource.SYSTEM)
        await sys_pub.publish(
            AgentEvent(
                type=EventType.SETTINGS_UPDATED,
                recipient_id="*",
                topic="settings",
                payload=dict(updated),
            )
        )

    @app.post("/api/connections")
    async def add_connection_route(request: Request, req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Add a connection and check it at once; answers it as `GET` lists it."""
        _refuse_cross_origin(request)  # another tab must not add a host to send keys to
        try:
            added = await session_mgr.add_connection(
                str(req.get("kind") or ""),
                label=_optional_text(req.get("label")),
                base_url=_optional_text(req.get("base_url")),
                key=_optional_text(req.get("key")),
            )
        except (ConnectionError_, ValueError, OSError) as exc:
            return _connection_refusal(exc)
        await _settings_event({"connections": "added", "id": added["id"]})
        return added

    @app.patch("/api/connections/{conn_id}")
    async def patch_connection_route(request: Request, conn_id: str, req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Change a connection's label, address or key; an empty key removes it."""
        _refuse_cross_origin(request)
        try:
            changed = await session_mgr.patch_connection(conn_id, req)
        except (ConnectionError_, ValueError, OSError) as exc:
            return _connection_refusal(exc)
        await _settings_event({"connections": "changed", "id": conn_id})
        return changed

    @app.post("/api/connections/{conn_id}/check")
    async def check_connection_route(request: Request, conn_id: str) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Ask the connection for its models now (Check connection)."""
        _refuse_cross_origin(request)  # a check sends the key to the connection's host
        try:
            return await session_mgr.check_connection(conn_id)
        except ConnectionError_ as exc:
            return _connection_refusal(exc, 404)

    @app.get("/api/connections/{conn_id}/dependents")
    async def connection_dependents_route(request: Request, conn_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The clones and defaults that name this connection, listed before a removal."""
        _refuse_cross_origin(request)
        return session_mgr.connection_dependents(conn_id)

    @app.delete("/api/connections/{conn_id}")
    async def remove_connection_route(request: Request, conn_id: str) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Remove a saved connection. One set by an environment variable is refused."""
        _refuse_cross_origin(request)
        try:
            session_mgr.remove_connection(conn_id)
        except (ConnectionError_, ValueError, OSError) as exc:
            return _connection_refusal(exc)
        await _settings_event({"connections": "removed", "id": conn_id})
        return {"removed": conn_id}

    @app.get("/api/models")
    async def get_models(  # pyright: ignore[reportUnusedFunction]
        request: Request, capability: str = "chat", refresh: bool = False
    ) -> Any:
        """The model set grouped by connection, with the defaults (model-gateway §3.7.1).

        `refresh=1` forgets the kept listings first, for the picker's "Refresh list".
        """
        _refuse_cross_origin(request)  # it asks every connection with its key
        wanted: Capability
        if capability == "chat":
            wanted = "chat"
        elif capability == "image":
            wanted = "image"
        else:
            return JSONResponse({"detail": "Choose chat or image models."}, status_code=400)
        gateway = session_mgr.gateway
        models = await gateway.model_set(wanted, refresh=refresh)
        return {
            "groups": [group.as_json() for group in models.groups],
            "defaults": session_mgr.defaults_payload(),
            "recommended": gateway.recommended(models)
            if capability == "chat"
            else {"deep": None, "fast": None},
        }

    def _ollama_address(req: dict[str, Any]) -> str | None:
        """The address of the Ollama connection ``connection_id`` names, else the first one.

        With no Ollama connection and none named, `None`: the local daemon's own address
        (`resolve_ollama_base_url`), as `ucx llm pull` uses.
        """
        wanted = _optional_text(req.get("connection_id"))
        ollamas = [c for c in session_mgr.gateway.connections() if c.kind == "ollama"]
        if wanted is None:
            return ollamas[0].base_url if ollamas else None
        chosen = next((c for c in ollamas if c.id == wanted), None)
        if chosen is None:
            raise HTTPException(
                status_code=400, detail=f"There is no Ollama connection called {wanted}."
            )
        return chosen.base_url

    @app.get("/api/models/installed")
    async def installed_models_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, connection_id: str | None = None
    ) -> Any:
        """What one Ollama connection has installed, for Settings' install and remove (#2167).

        Every model, embedders included, each saying whether it can hold a conversation
        (Ollama's own capabilities, `list_ollama_entries`): an embedder is not offered to
        chat but can still be removed, so it is listed here and said to be one.
        `connection_id` names the connection as `POST /api/models/pull` does.
        """
        _refuse_cross_origin(request)  # names what the user has installed
        base_url = _ollama_address({"connection_id": connection_id})
        entries = await list_ollama_entries(base_url)
        if entries is None:
            return JSONResponse(
                {
                    "detail": "Couldn't get an answer from Ollama. Check that it is running "
                    "and that the address is right.",
                    "code": "unreachable",
                },
                status_code=502,
            )
        return {
            "connection_id": connection_id,
            "models": [{"id": e.id, "chat": e.chat_capable} for e in entries],
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
        base_url = _ollama_address(req)

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
            raise HTTPException(
                status_code=502, detail=f"Could not pull {model} from Ollama."
            ) from exc
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
        try:
            await delete_model(model, base_url=_ollama_address(req))
        except LLMProviderError as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not delete {model} from Ollama."
            ) from exc
        return {"status": "ok", "model": model}

    async def _refuse_unlisted_defaults(changes: dict[str, Any]) -> str | None:
        """Why a default model may not be saved, in plain words; `None` when it may (§3.7.1)."""
        gateway = session_mgr.gateway
        for slot, value in changes.items():
            if value is None or (slot == "image" and value == "auto"):
                continue
            if not isinstance(value, str):
                return "A default model must be a model name."
            try:
                ref = ModelRef.parse(value)
            except ModelRefError as exc:
                return str(exc)
            conn = gateway.connection(ref.connection_id)
            if conn is None:
                return f"There is no connection called {ref.connection_id}."
            if slot == "image" and (problem := image_ref_problem(ref, conn)) is not None:
                return problem
            capability: Capability = "image" if slot == "image" else "chat"
            group = (await gateway.model_set(capability)).group(conn.id)
            if group is None or group.status != "connected":
                detail = group.detail if group is not None and group.detail else ""
                return f"{conn.display_label} cannot list its models right now. {detail}".strip()
            if not any(entry.ref == str(ref) for entry in group.models):
                return f"{conn.display_label} does not list the model {ref.model}."
        return None

    @app.post("/api/settings")
    async def update_settings(request: Request, req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Save the settings the request names, and broadcast settings.updated.

        ``default_models`` (`{deep?, fast?, image?}`) is saved only when each ref is in the
        current model set; otherwise the refusal names the connection's state (§3.7.1).
        """
        _refuse_cross_origin(request)  # another tab must not widen read_roots
        raw_defaults = req.get("default_models")
        defaults: dict[str, Any] | None = None
        if raw_defaults is not None:
            if not isinstance(raw_defaults, dict):
                return JSONResponse(
                    {"detail": "default_models must name deep, fast or image."}, status_code=400
                )
            defaults = {str(k): v for k, v in cast(dict[object, object], raw_defaults).items()}
            unknown = [slot for slot in defaults if slot not in ("deep", "fast", "image")]
            if unknown:
                return JSONResponse(
                    {"detail": "default_models must name deep, fast or image."}, status_code=400
                )
            refusal = await _refuse_unlisted_defaults(defaults)
            if refusal is not None:
                return JSONResponse({"detail": refusal}, status_code=400)
        try:
            updated = session_mgr.update_settings(
                read_roots=req.get("read_roots"),
                ui_language=req.get("ui_language"),
                default_models=cast("dict[str, str | None] | None", defaults),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="The settings could not be saved because the configuration is invalid.",
            ) from exc
        await _settings_event(updated)
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
        _refuse_unless_local(request, action="read")
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

    @app.post("/api/settings/remote-gpu/probe")
    async def probe_remote_gpu_endpoint(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Probe an SSH remote host for GPU capabilities and AI services."""
        _refuse_unless_local(request)
        host = str(req.get("host", "")).strip()
        timeout = float(req.get("timeout", 6.0))
        if not host:
            raise HTTPException(
                status_code=400,
                detail="Field 'host' is required",
            )
        inspection = await probe_remote_host(host, timeout=timeout)
        return inspection.to_dict()

    @app.post("/api/settings/remote-gpu/connect")
    async def connect_remote_gpu_endpoint(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Establish an SSH tunnel to the remote GPU worker and optionally activate endpoints."""
        _refuse_unless_local(request)
        host = str(req.get("host", "")).strip()
        apply_settings = bool(req.get("apply_settings", True))
        auto_start_comfyui = bool(req.get("auto_start_comfyui", True))
        # The LLM moves to the remote worker only when asked. It used to follow whenever the
        # remote Ollama held a model of the same name, so a worker meant for images also
        # answered every turn, and nothing on the Settings card said so.
        sync_llm = req.get("sync_llm") is True
        timeout = float(req.get("timeout", 15.0))
        if not host:
            raise HTTPException(
                status_code=400,
                detail="Field 'host' is required",
            )
        return await _connect_remote_gpu(
            host,
            apply_settings=apply_settings,
            auto_start_comfyui=auto_start_comfyui,
            sync_llm=sync_llm,
            timeout=timeout,
        )

    @app.post("/api/settings/remote-gpu/disconnect")
    async def disconnect_remote_gpu_endpoint(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> dict[str, Any]:
        """Disconnect the active SSH tunnel session and restore previous endpoints."""
        _refuse_unless_local(request)
        await tunnel_manager.disconnect()
        # Pressed by hand: the next start leaves the worker alone.
        with contextlib.suppress(FileNotFoundError):
            remote_session_path.unlink()
        restored: dict[str, Any] = dict(_restore_remote_settings())

        return {
            "status": "ok",
            "connected": False,
            "restored_settings": restored,
            "tunnel": tunnel_manager.get_status().to_dict(),
        }

    @app.get("/api/settings/remote-gpu/status")
    async def get_remote_gpu_status(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> dict[str, Any]:
        """Return the current status of the remote GPU tunnel.

        A tunnel whose ssh process has exited reads as disconnected here, and the addresses
        it saved are put back at the same moment, so a dead tunnel does not stay the LLM or
        ComfyUI address until someone presses Disconnect.
        """
        _refuse_unless_local(request)
        status = tunnel_manager.get_status()
        payload = status.to_dict()
        if not status.connected and _read_remote_restore()[0]:
            payload["restored_settings"] = _restore_remote_settings()
        _, applied_changes = _read_remote_restore()
        payload["images_on_remote"] = bool(
            status.connected
            and any(m.service_name == COMFYUI_SERVICE_NAME for m in status.mappings)
        )
        payload["llm_on_remote"] = bool(
            status.connected
            and any(m.service_name == "ollama" for m in status.mappings)
            and applied_changes.get(REMOTE_GPU_CONNECTION_ID)
        )
        return payload

    @app.get("/api/settings/remote-gpu/hosts")
    async def get_remote_gpu_hosts(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> dict[str, Any]:
        """Return configured SSH hosts along with the last-connected host."""
        _refuse_unless_local(request)
        last_host: str | None = None
        try:
            raw: object = json.loads(remote_session_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                h = cast("dict[str, object]", raw).get("host")
                if isinstance(h, str) and is_valid_host(h):
                    last_host = h.strip()
        except (OSError, ValueError):
            pass

        cfg_hosts = find_configured_ssh_hosts()
        hosts: list[str] = []
        if last_host:
            hosts.append(last_host)
        for h in cfg_hosts:
            if h not in hosts:
                hosts.append(h)
        return {"hosts": hosts}

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

        # `id` and `handle` are what the entry was listed with; a head may send the entry
        # back whole. The handle is the draft's `name`, and the id is never written.
        fields = {
            key: value for key, value in req.items() if key not in ("id", "handle", "llm_config")
        }
        if "name" not in fields and isinstance(req.get("handle"), str):
            fields["name"] = req["handle"]
        # The model slots may come grouped, as `llm_config` (model-gateway §3.7.1); a slot
        # named there wins over the flat field of the same name.
        grouped = req.get("llm_config")
        if isinstance(grouped, dict):
            for slot in ("model_name", "fast_model", "image_model"):
                if slot in grouped:
                    fields[slot] = cast(dict[str, Any], grouped)[slot]
        try:
            draft = PersonaDraft.model_validate(fields)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=exc.errors(include_url=False, include_context=False)
            ) from exc
        # A ref naming a connection that does not exist is refused; one whose connection
        # exists but does not list the model is saved (it may be offline) (§3.7.1).
        for slot in ("model_name", "fast_model", "image_model"):
            value = getattr(draft, slot)
            if not value or (slot == "image_model" and value.strip() == "auto"):
                continue
            ref = ModelRef.parse(value)
            conn_id = ref.connection_id
            if (conn := session_mgr.gateway.connection(conn_id)) is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"There is no connection called {conn_id}. Add it in Settings, "
                    f"or choose another model.",
                )
            # A picture model must be on a connection that draws, and the GPU server takes
            # only `<id>/auto`: it cannot be told which model to use (§3.5).
            if slot == "image_model" and (problem := image_ref_problem(ref, conn)) is not None:
                raise HTTPException(status_code=422, detail=problem)
        registry = get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )
        # The address names the clone by id or by handle; the body by handle.
        addressed = registry.handle_for(name) if name is not None else None
        if addressed is not None and draft.name != addressed:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"the body names clone {draft.name!r} but the address names {name!r}. "
                    f"A clone's handle cannot be changed by this edit; create one under the "
                    f"new handle."
                ),
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
                # How many open conversations take the edit at their next turn.
                "live_agents_updated": room_stack.persona_edited(persona),
            },
        )

    def _avatar_registry() -> PersonaRegistry:
        from uclone_x.agent.persona_registry import get_default_persona_registry

        return get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )

    def _registry_for_a_picture_change(request: Request) -> PersonaRegistry:
        """The registry, after refusing a request another site's page sent.

        The app answers every origin, so without this an unrelated open tab could replace
        a clone's face.
        """
        _refuse_cross_origin(request)  # before anything is read or written
        return _avatar_registry()

    @app.get("/api/clones/{ref}/avatar")
    async def get_clone_avatar(ref: str) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Return one clone's picture, or refuse with what would put a picture there.

        A 404 here is an ordinary answer, not a fault: most clones have no picture, and the
        head draws its default when this refuses. The detail still names the remedy, because
        this is also what a reader sees who went looking for the file they thought they had
        installed. `no-cache` because the persona's `avatar_url` carries a `?v=` that
        changes with the picture, and a cached old face under a new `?v=` would hide that.
        """
        registry = _avatar_registry()
        name = registry.handle_for(ref)  # the clone's id or its handle
        found = PersonaAvatarStore(registry).find(name)
        if found is None:
            formats = ", ".join(suffix for suffix, _ in AVATAR_FORMATS)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No picture is set for '{name}'. Choose one from its profile: an image "
                    f"file ending in one of {formats}."
                ),
            )
        return Response(
            content=found.path.read_bytes(),
            media_type=found.mime,
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
        )

    def _avatar_answer(
        registry: PersonaRegistry, name: str, change: AvatarChange
    ) -> dict[str, Any]:
        """The changed clone, the kept picture that would undo the change, and its id.

        `previous_path` is the path to `PUT` back to undo it -- the kept picture in the
        clone's directory, which the `PUT` accepts by name; `null` means the clone had no
        chosen picture before, so undoing is a `DELETE`. Either way the undo carries
        `change_id` as `undo_of`, and is refused if a later change was made.
        """
        persona = registry.get_persona(name)
        if persona is None:
            raise HTTPException(status_code=404, detail=f"There is no clone '{name}' here.")
        return {
            "status": "ok",
            "persona": _persona_payload(registry, persona),
            "previous_path": str(change.previous) if change.previous is not None else None,
            "change_id": change.change_id,
        }

    def _avatar_refusal(exc: AvatarRefused) -> JSONResponse:
        """The refusal as `{"detail": <plain words>, "code": <which reason>}`.

        The head words its own message from `code` (a path outside the workspace and a file
        in the wrong format ask for different things); `detail` is for everything else.
        """
        status = (
            404
            if isinstance(exc, AvatarPersonaNotFound)
            else 409
            if isinstance(exc, AvatarStaleChange)
            else 422
        )
        return JSONResponse({"detail": str(exc), "code": exc.reason_code}, status_code=status)

    async def _avatar_upload(request: Request) -> bytes:
        """The uploaded picture, read no further than the size cap.

        `content-length` is checked first, but a chunked upload carries none, so the body
        is also counted as it arrives and reading stops once it passes the cap.
        """
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_AVATAR_BYTES:
            raise _avatar_too_large()
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_AVATAR_BYTES:
                raise _avatar_too_large()
        return bytes(body)

    def _avatar_too_large() -> AvatarRefused:
        return AvatarRefused(
            f"That picture is larger than {MAX_AVATAR_BYTES // (1024 * 1024)} MB, "
            "the most a clone's picture can be. Choose a smaller one.",
            reason_code="too_large",
        )

    async def _avatar_json(request: Request) -> object:
        """The JSON body, or a plain refusal when it is not JSON at all."""
        try:
            return await request.json()
        except ValueError as exc:
            raise _avatar_no_source() from exc

    def _avatar_no_source() -> AvatarRefused:
        return AvatarRefused(
            'Name the picture to use as {"source_path": "<path in the workspace>"}.',
            reason_code="no_source",
        )

    @app.put("/api/clones/{ref}/avatar")
    async def put_clone_avatar(ref: str, request: Request) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Set one clone's picture from a workspace file or from the image in the body.

        The body is either JSON `{"source_path": "<path in the workspace>"}`, normally an
        image a clone just drew, or the picture itself with an `image/*` content type, for
        an upload. Either way the bytes must be a PNG, JPEG, WebP or GIF image; the picture
        it replaces is kept as `avatar.prev.<ext>` in the clone's directory. A JSON body may
        add `"undo_of": <id>`, the `change_id` of the change it undoes; it is then made only
        while that change is still the latest, and refused with 409 `stale_change` otherwise.
        """
        registry = _registry_for_a_picture_change(request)
        name = registry.handle_for(ref)  # the clone's id or its handle
        store = PersonaAvatarStore(registry)
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        try:
            if content_type == "application/json":
                body = await _avatar_json(request)
                undo_of = _undo_of_body(body)
                source = _avatar_source(store, name, body)
                change = store.set_from_path(name, source, undo_of=undo_of)
            elif content_type.startswith("image/"):
                change = store.set(name, await _avatar_upload(request))
            else:
                return JSONResponse(
                    {
                        "detail": (
                            "Send the picture itself as an image, or name a picture in the "
                            'workspace as {"source_path": "..."}.'
                        ),
                        "code": "not_an_image",
                    },
                    status_code=415,
                )
        except AvatarRefused as exc:
            return _avatar_refusal(exc)
        return _avatar_answer(registry, name, change)

    def _undo_of_body(body: object) -> int | None:
        """The `undo_of` a JSON body names; `0`, which no change has, for one that is not a count."""
        raw = cast(dict[str, object], body).get("undo_of") if isinstance(body, dict) else None
        if raw is None:
            return None
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else 0

    def _undo_of_query(request: Request) -> int | None:
        """The `?undo_of=` a `DELETE` names, read as `_undo_of_body` reads a body."""
        raw = request.query_params.get("undo_of")
        if raw is None:
            return None
        return int(raw) if raw.isascii() and raw.isdigit() else 0

    def _avatar_source(store: PersonaAvatarStore, name: str, body: object) -> Path:
        """The file a `PUT` names: the clone's kept picture, or one in the workspace."""
        raw = cast(dict[str, object], body).get("source_path") if isinstance(body, dict) else None
        if not isinstance(raw, str) or raw.strip() == "":
            raise _avatar_no_source()
        kept = store.named_previous(name, raw)
        if kept is not None:
            return kept
        try:
            return PathValidator().resolve_safe_path(Path(raw), session_mgr.workspace_dir)
        except PathTraversalError as exc:
            raise AvatarRefused(
                "That picture is outside the workspace, so it was not used. Choose one in the "
                "workspace, or upload it instead.",
                reason_code="outside_workspace",
            ) from exc

    @app.delete("/api/clones/{ref}/avatar")
    async def delete_clone_avatar(ref: str, request: Request) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Put a clone's chosen picture aside, so it shows its shipped one or the default.

        `?undo_of=<id>` makes it the undo of that change, refused as `PUT`'s is.
        """
        registry = _registry_for_a_picture_change(request)
        name = registry.handle_for(ref)  # the clone's id or its handle
        try:
            change = PersonaAvatarStore(registry).reset(name, undo_of=_undo_of_query(request))
        except AvatarRefused as exc:
            return _avatar_refusal(exc)
        return _avatar_answer(registry, name, change)

    @app.post("/api/clones")
    async def create_clone(req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Create a clone: its handle, display name and persona fields, in one directory."""
        return _save_persona(req, create=True)

    # Declared before `/api/clones/{ref}`, which would otherwise read `synthesize` as a
    # clone's handle for every method it serves.
    @app.post("/api/clones/synthesize")
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
            # On the person's behalf, owned by no clone: the default deep model (§3.4).
            llm, draft_model = session_mgr.gateway.default_deep()
            if llm is None:
                raise ValueError("no default model is saved")
            response = await asyncio.wait_for(
                llm.generate(
                    LLMRequest(
                        model=draft_model,
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

    @app.get("/api/clones/{ref}")
    async def get_clone(ref: str, request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """One clone's entry, addressed by its id or its handle."""
        _refuse_cross_origin(request)
        registry = _avatar_registry()
        persona = registry.get_persona(ref)
        if persona is None:
            raise HTTPException(status_code=404, detail=f"There is no clone '{ref}' here.")
        return {"status": "ok", "persona": _persona_payload(registry, persona)}

    @app.put("/api/clones/{ref}")
    async def update_clone(ref: str, req: dict[str, Any]) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Edit a clone, addressed by its id or its handle; a builtin's file is rewritten too."""
        return _save_persona(req, create=False, name=ref)

    @app.get("/api/sessions")
    async def list_sessions(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return list of active and stored sessions from Core session store (FR-13.2)."""
        _refuse_cross_origin(request)  # another site must not list sessions (#2146)
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

    def _developer_graph_clone(agent_id: str | None) -> str:
        """The clone a developer-graph route reads, refused in plain words when unusable.

        There is no shared engine to fall back on (clone-knowledge-graph §3.8, #1869): a
        request names the clone, and a name that is not one of the clones listed here --
        installed or running, as `GET /api/clones` lists them -- is refused before an engine
        is made for it. Without that, every unknown name read would leave an empty engine in
        the manager's map for the app's lifetime and answer 200 with an empty graph. A row
        the listing marks unreadable is not a clone that can be read, so it is refused the
        same way (#1879).
        """
        from uclone_x.ui.clones import CloneStatus, clone_listing

        if not agent_id:
            raise HTTPException(status_code=400, detail="Name the clone to read with agent_id.")
        # The listing does show a folder whose name no clone can have (`Bad Name`), marked
        # unreadable; skipping unreadable rows refuses that name as well as a damaged home.
        # A head names the clone by its id (clone-data-scopes §4 step 2); a handle is read
        # too. The engines are keyed by id, so either reaches the same one.
        for clone in clone_listing(session_mgr, room_stack).clones:
            if clone.status is CloneStatus.UNREADABLE:
                continue
            if agent_id in (clone.id, clone.name):
                return clone.id or clone.name
        raise HTTPException(status_code=404, detail="There is no clone with that name here.")

    @app.get("/api/ontology")
    async def get_ontology(request: Request, agent_id: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The concepts, relations, axioms and tier counts of clone `agent_id`'s rules engine."""
        _refuse_cross_origin(request)
        return session_mgr.ontology_for(_developer_graph_clone(agent_id)).export_graph()

    @app.get("/api/artifacts")
    async def get_artifacts(request: Request, session_id: str | None = None) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Enumerate the documents and images the clones generated (RFC §6.1)."""
        _refuse_cross_origin(request)  # another site must not list artifacts (#2146)
        artifacts = session_mgr.list_artifacts(session_id=session_id)
        return {"artifacts": artifacts, "total": len(artifacts)}

    @app.get("/api/artifacts/content")
    async def get_artifact_content(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        path: str = "",
        session_id: str | None = None,
        room_id: str | None = None,
    ) -> Response:
        """Return artifact file content safely (markdown text or raw image bytes) (P6 security invariant, RFC §6.1).

        With `room_id`, `path` is read in that conversation's workspace, where its seats
        wrote it (clone-data-scopes §3.6); without, in the server's. Either workspace can be
        any folder a person picked, so a read is refused to another site's page, and never
        reaches into the app's own state folders (#2143).
        """
        _refuse_cross_origin(request)  # another site must not read a workspace's files
        try:
            root = room_stack.workspace_of(room_id) if room_id else None
            resolved_path, mime = session_mgr.get_artifact_file(
                path=path, session_id=session_id, root=root
            )
        except RoomNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PathTraversalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if in_app_state_dir(resolved_path):
            raise HTTPException(
                status_code=403,
                detail="That file is in this app's own clones and conversations, "
                "which a workspace folder does not show.",
            )

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
        request: Request,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Clone `agent_id`'s triples, nodes and edges, optionally one session's (RFC §6.1)."""
        _refuse_cross_origin(request)
        return session_mgr.get_knowledge_graph(
            _developer_graph_clone(agent_id), session_id=session_id
        )

    @app.get("/api/skills")
    async def get_skills() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Return registered skills, P9 security audit reports, and quarantine statuses from live SkillRegistry.

        Each skill also carries `hidden_from`: the clones whose tool scope lacks a tool the
        skill requires, and which tools (#1826), so the panel can say why a clone is not
        offered it rather than letting the skill vanish from that clone in silence.
        """
        from uclone_x.agent.bootstrap import seat_tool_scope
        from uclone_x.agent.persona_registry import get_default_persona_registry

        summary = session_mgr.skill_registry.get_summary()
        registry = get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )
        # The scope a seated clone runs with, by the agent's own rule (#1865), not the
        # persona's list read separately: the two agree only while no operator list is set.
        scopes = {p.name: seat_tool_scope(p) for p in registry.list_personas()}
        for entry in summary["skills"]:
            entry["hidden_from"] = skill_hidden_from(entry.get("requires_tools", []), scopes)
            # Settings offers Revoke only for a skill that did not ship (#1827).
            entry["shipped"] = entry.get("name") in SHIPPED_SKILL_PINS
        summary["proposals"] = await _skill_proposals()
        return summary

    # --- Skill proposals (#1827) --------------------------------------------------------
    #
    # A clone proposes a skill with `propose_skill`; it waits, unloaded, in the store's
    # `.pending/` area. Only a person may approve, reject or revoke, so each route refuses a
    # cross-origin request and then requires a window this server confirmed (#1589) before
    # the body is read: the model's shell cannot approve its own proposal. A refusal is the
    # store's plain sentence, or one fixed sentence, never an exception's text.

    _SKILLS_NO_STORE = "There is no skill folder for this project, so there is nothing to change."
    _SKILLS_NOT_CHANGED = "The skill could not be changed. Try again."

    class _SkillDecisionRefused(Exception):
        """A refused skill decision: the English sentence, and the code the head localizes."""

        def __init__(self, status: int, detail: str, code: SkillDecisionCode) -> None:
            super().__init__(detail)
            self.status = status
            self.detail = detail
            self.code = code

    def _refused(exc: _SkillDecisionRefused) -> JSONResponse:
        # `detail` is the plain English sentence; `code` names the reason, so a window in
        # another language shows the same refusal in its own words (#1865). The code is
        # never put into the sentence.
        return JSONResponse({"detail": exc.detail, "code": exc.code}, status_code=exc.status)

    def _proposal_store() -> SkillProposalStore:
        root = session_mgr.skill_registry.store_root
        if root is None or not root.is_dir():
            raise _SkillDecisionRefused(404, _SKILLS_NO_STORE, "no_store")
        return SkillProposalStore(root)

    def _decision_refusal(exc: SkillProposalError) -> _SkillDecisionRefused:
        # 412 when the proposal is not the one shown, so Settings can show it again.
        status = 409 if not isinstance(exc, SkillProposalChangedError) else 412
        return _SkillDecisionRefused(status, str(exc), exc.decision_code or "not_changed")

    async def _skill_proposals() -> list[dict[str, Any]]:
        root = session_mgr.skill_registry.store_root
        if root is None or not root.is_dir():
            return []
        try:
            proposals = await asyncio.to_thread(SkillProposalStore(root).list_proposals)
        except OSError as exc:
            logger.warning("Skill proposals could not be listed: %s", exc)
            return []
        return [proposal.to_dict() for proposal in proposals]

    async def _skill_body_text(
        request: Request, key: str, missing: str, code: SkillDecisionCode
    ) -> str:
        try:
            payload: object = await request.json()
        except ValueError:
            payload = None
        value = payload.get(key) if isinstance(payload, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(value, str) or not value:
            # Coded like every other refusal, so a Korean window does not show English.
            raise _SkillDecisionRefused(400, missing, code)
        return value

    async def _skill_version(request: Request) -> str:
        return await _skill_body_text(
            request, "version", "Say which version of the skill.", "no_version"
        )

    async def _after_skill_change() -> dict[str, Any]:
        # The registry reloads from the store, so `load_skill` and new sessions see the
        # change now; a session already running keeps the catalog it started with.
        registry = session_mgr.skill_registry
        await load_approved_skills(registry)
        return {"ok": True}

    def _a_person_decides(decision: Request) -> None:
        # A skill decision is a person's (#1589): the model's shell reaches these routes
        # too, so a window the server did not confirm is refused before anything changes.
        _refuse_cross_origin(decision)
        person_gate.require(decision)

    @app.post("/api/skills/{name}/approve", response_model=None)
    async def approve_skill_proposal(name: str, request: Request) -> dict[str, Any] | JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Approve a clone's proposal: audit, install as the active version, pin its digest."""
        _a_person_decides(request)
        try:
            version = await _skill_version(request)
            # The digest of the proposal as Settings showed it: approval installs exactly
            # that text or nothing, even if a clone's file tools rewrote it since (#1827).
            seen_digest = await _skill_body_text(
                request, "seen_digest", "Look at the proposal again, then approve it.", "not_seen"
            )
            await _proposal_store().approve(
                name,
                version,
                seen_digest=seen_digest,
                approver=SETTINGS_PERSON,
                ledger=SkillApprovalLedger(),
            )
        except _SkillDecisionRefused as exc:
            return _refused(exc)
        except SkillProposalError as exc:
            return _refused(_decision_refusal(exc))
        except Exception as exc:  # the person gets one plain sentence; the log gets the rest
            logger.warning("Approving the skill proposal '%s' failed: %s", name, exc)
            return _refused(_SkillDecisionRefused(500, _SKILLS_NOT_CHANGED, "not_changed"))
        return await _after_skill_change()

    @app.post("/api/skills/{name}/reject", response_model=None)
    async def reject_skill_proposal(name: str, request: Request) -> dict[str, Any] | JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Turn down a clone's proposal; it is kept under `.rejected/`.

        Bound to the digest Settings showed, as approve is (#1865): a proposal that changed
        since is refused with 412 and stays pending.
        """
        _a_person_decides(request)
        try:
            version = await _skill_version(request)
            seen_digest = await _skill_body_text(
                request, "seen_digest", "Look at the proposal again, then decide.", "not_seen"
            )
            store = _proposal_store()
            await asyncio.to_thread(
                store.reject,
                name,
                version,
                seen_digest=seen_digest,
                rejecter=SETTINGS_PERSON,
                reason=None,
            )
        except _SkillDecisionRefused as exc:
            return _refused(exc)
        except SkillProposalError as exc:
            return _refused(_decision_refusal(exc))
        except Exception as exc:
            logger.warning("Turning down the skill proposal '%s' failed: %s", name, exc)
            return _refused(_SkillDecisionRefused(500, _SKILLS_NOT_CHANGED, "not_changed"))
        return await _after_skill_change()

    @app.post("/api/skills/{name}/revoke", response_model=None)
    async def revoke_skill(name: str, request: Request) -> dict[str, Any] | JSONResponse:  # pyright: ignore[reportUnusedFunction]
        """Stop using an approved skill: remove its pin and mark it rejected."""
        _a_person_decides(request)
        try:
            store = _proposal_store()
            await asyncio.to_thread(
                store.revoke, name, revoker=SETTINGS_PERSON, ledger=SkillApprovalLedger()
            )
        except _SkillDecisionRefused as exc:
            return _refused(exc)
        except SkillProposalError as exc:
            return _refused(_decision_refusal(exc))
        except Exception as exc:
            logger.warning("Revoking the skill '%s' failed: %s", name, exc)
            return _refused(_SkillDecisionRefused(500, _SKILLS_NOT_CHANGED, "not_changed"))
        return await _after_skill_change()

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

    def _refuse_unless_local(request: Request, *, action: str = "change") -> None:
        """`_refuse_cross_origin`, and also refuse a request addressed to a non-loopback name.

        For routes that inspect or start a program on this machine (#1462). The origin check
        alone passes a DNS-rebinding page: its `Origin` and `Host` both name the attacker's
        domain, which now resolves to 127.0.0.1, so they match. That page cannot make the
        browser send a `Host` of `localhost`, and this refuses every other name.

        On a loopback-bound server `LoopbackHostGuard` already refuses those names for every
        route (#1413). This check is what still holds on a server exposed with `ucx ui
        --host 0.0.0.0`: others on the network may use the dashboard, but only this computer
        may inspect or start programs on it.
        """
        _refuse_cross_origin(request)
        if _host_header_hostname(request.headers.get("host", "")) not in _LOOPBACK_HOSTS:
            if action == "read":
                detail = "Tool servers can only be viewed on the machine running the app."
            else:
                detail = (
                    "Tool servers can only be changed from this computer. "
                    "Open the app at http://localhost to change them."
                )
            raise HTTPException(
                status_code=403,
                detail=detail,
            )

    from uclone_x.ui.usage import register_usage_routes

    # Settings → Usage (`llm-token-gateway.md` §4.5): the dashboard's own settings file and
    # the usage store beside it, the same two its paid connectors' gate reads.
    register_usage_routes(
        app,
        settings_file=session_mgr.settings_file,
        usage_file=session_mgr.usage_file,
        refuse_cross_origin=_refuse_cross_origin,
    )

    from uclone_x.ui.links import register_link_routes

    def _local_clone_names() -> list[str]:
        from uclone_x.agent.persona_registry import get_default_persona_registry

        registry = get_default_persona_registry(
            session_mgr.workspace_dir,
            tool_names=[tool.name for tool in session_mgr.tools.list_tools()],
        )
        return [p.name for p in registry.list_personas()]

    # Settings → 연결 → uClone2 (`uclone2-link.md` §3.6): the same supervisor the lifespan
    # starts, so a link made here starts its session at once.
    register_link_routes(
        app,
        supervisor=links,
        local_clone_names=_local_clone_names,
        refuse_cross_origin=_refuse_cross_origin,
    )

    # Settings ▸ Browser and the extension's link (`browser-agent.md` §3.6): the process's
    # hub, so the browser tool's tabs go to the Chrome paired here.
    from uclone_x.browser.extension import default_extension_hub
    from uclone_x.ui.browser import register_browser_routes, saved_pairing_token

    browser_settings = session_mgr.settings_file
    extension_hub = default_extension_hub()
    extension_hub.use_token(lambda: saved_pairing_token(browser_settings))
    register_browser_routes(
        app,
        hub=extension_hub,
        settings_file=browser_settings,
        refuse_cross_origin=_refuse_cross_origin,
        on_stop=room_stack.stop,
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
        """Record the answer. `collect` must be present: there is no default.

        The answer is the person's consent, so it is taken only from a window this server
        confirmed (#1589, `uclone_x.ui.person`), before the body is read: a program on this
        computer, the model's shell included, cannot give it on their behalf.
        """
        _refuse_cross_origin(request)
        person_gate.require(request)
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
        _refuse_cross_origin(request)  # another site must not tap the event stream (#2146)

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
                        "/api/sessions",
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
