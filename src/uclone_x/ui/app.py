"""FastAPI backend application for UClone-X developer UI dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
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
    Sequence,
)
from contextlib import asynccontextmanager
from datetime import UTC, datetime
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
from uclone_x.agent.clone_builder import AppScope, build_clone, provider_tool_binder
from uclone_x.agent.models import (
    PersonaDefinition,
)
from uclone_x.agent.persona_avatar import (
    AVATAR_FORMATS,
    MAX_AVATAR_BYTES,
    AvatarPersonaNotFound,
    AvatarRefused,
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
from uclone_x.core.remote_worker import (
    SSHTunnelManager,
    probe_remote_host,
)
from uclone_x.core.session_diagnostics import (
    DEFAULT_MAX_CONVERSATION_TURNS,
    count_active_turns,
)
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
    SessionHistoryRehydrationError,
)
from uclone_x.evaluation import (
    EvalBackendUnavailableError,
    create_eval_runner,
    default_reports_dir,
)
from uclone_x.i18n import DEFAULT_UI_LANGUAGE, UI_LANGUAGES, UiLanguage, is_ui_language
from uclone_x.llm import create_llm_connector
from uclone_x.llm.budget import TokenBudgetManager
from uclone_x.llm.catalog import (
    CatalogCache,
    CatalogEntry,
    CatalogResult,
    Lister,
    key_fingerprint,
    read_catalog,
)
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.connectors.factory import bind_image_engine_settings, saved_choice_in_effect
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.ollama import (
    delete_model,
    pull_model,
    resolve_ollama_base_url,
)
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.saved_choice import (
    LLM_MODEL_FAST_KEY,
    SETTINGS_FILE_NAME,
    api_key_for,
    delete_api_key,
    same_provider,
    save_api_key,
    settings_data,
    update_settings_file,
)
from uclone_x.llm.connectors.vllm import (
    has_configured_vllm_endpoint,
    resolve_vllm_base_url,
)
from uclone_x.llm.model_policy import recommend
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ToolCallRequest,
)
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.llm.providers import (
    PROVIDERS,
    canonical_provider,
    env_key,
    env_model,
    spec_for,
)
from uclone_x.llm.usage.gate import UsageGate
from uclone_x.llm.usage.store import USAGE_FILE_NAME
from uclone_x.memory.store import CrossSessionMemory, default_cross_session_memory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.room.service import (
    SESSION_ID_PREFIX as _ROOM_SESSION_PREFIX,
)
from uclone_x.room.service import (
    SESSION_ID_SEPARATOR as _ROOM_SESSION_SEPARATOR,
)
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.shells.ui_process import UI_BIND_HOST_ENV_VAR
from uclone_x.skills.auditor import (
    SkillRegistry,
    load_approved_skills,
    runtime_skill_store_dir,
)
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import replace_file
from uclone_x.tools.builtin.comfy_client import (
    DEFAULT_COMFYUI_BASE_URL,
    ComfyClient,
)
from uclone_x.tools.builtin.comfy_image_tool import ComfyImageGenTool
from uclone_x.tools.builtin.image import (
    IMAGE_ENGINE_KEY,
    IMAGE_MODEL_KEY,
    parse_image_engine_setting,
    parse_image_model,
)
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


def vllm_request_headers(
    api_key: str | None = None, *, env_fallback: bool = True
) -> dict[str, str]:
    """`Authorization` only when a key exists, because `vllm serve --api-key` is optional.

    An empty bearer is not the same as no header: a server started without `--api-key`
    accepts the request either way, but one behind a proxy that reads the header rejects
    `Bearer ` with a 401 that describes a credential nobody configured (#385).

    `env_fallback=False` sends only the key passed in: for an address typed on the form, which
    must not receive the key held for the saved server (#1666).
    """
    env_key = os.getenv("VLLM_API_KEY") if env_fallback else None
    key = (api_key or env_key or "").strip()
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


class LocalKeyRefusedError(Exception):
    """A local server answered 401 or 403: it is running, and it refused the key (#1672).

    Raised by `list_local_models` only when asked to, so the form can say "the server refused
    the key" rather than "no model list came back", which sends the user to look for a
    stopped server that is not stopped.
    """


#: The statuses that mean the server is there and turned the key away.
_KEY_REFUSED_STATUSES = frozenset({401, 403})


async def list_local_models(
    provider: str,
    base_url: str | None = None,
    timeout: float = 3.0,
    *,
    vllm_headers: dict[str, str] | None = None,
    raise_on_refused_key: bool = False,
) -> list[str] | None:
    """The models a local server has installed, or None when no listing came back (#1666).

    None and `[]` are different answers, and Settings says different things for them: no
    listing (nothing answered, or something that is not this server did) names a stopped
    server or a wrong address, where an empty list names an install
    the user has not done yet. `fetch_available_models` folds the two together.

    `vllm_headers` replaces the default ones, which carry `VLLM_API_KEY`; see `preview_catalog`.
    With `raise_on_refused_key`, a vLLM server's 401 or 403 raises `LocalKeyRefusedError`
    instead of reading as no listing.
    """
    clean_provider = provider.strip().lower()
    if clean_provider == "ollama":
        ollama_url = (base_url or "").strip() or resolve_ollama_base_url()
        try:
            async with httpx.AsyncClient(timeout=timeout) as http_c:
                resp = await http_c.get(f"{ollama_url.rstrip('/')}/api/tags")
                if resp.status_code != 200:
                    return None  # something answered, but not Ollama's listing
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
            return None
    if clean_provider == "vllm":
        if not has_configured_vllm_endpoint(base_url):
            # An unconfigured endpoint is not an empty inventory. Probing vLLM's documented
            # default port to fill the dropdown would be this module guessing where the
            # operator's server is, and reporting a refused connection as "no models" (P6).
            return None
        vllm_url = resolve_vllm_base_url(base_url)
        listing_url = f"{vllm_url.rstrip('/')}/models"
        refused = False
        try:
            async with httpx.AsyncClient(timeout=timeout) as http_c:
                resp = await http_c.get(
                    listing_url,
                    headers=vllm_request_headers() if vllm_headers is None else vllm_headers,
                )
                if resp.status_code == 200:
                    return vllm_model_ids(resp.json())
                refused = resp.status_code in _KEY_REFUSED_STATUSES
        except Exception as exc:
            logger.debug("Failed to query vLLM models from %s: %s", vllm_url, exc)
            return None
        if refused and raise_on_refused_key:
            raise LocalKeyRefusedError(vllm_url)
        return None
    if clean_provider == "mock":
        return ["mock-gpt-4o", "mock-llm"]
    return None


async def fetch_available_models(
    provider: str,
    base_url: str | None = None,
    timeout: float = 3.0,
) -> list[str]:
    """Enumerate the models a local server has installed (P0/Recognition over Recall).

    The cloud providers are not answered here: their models come from each provider's own
    listing, through `read_provider_catalog` (#1631). The remembered lists this function
    returned for them had retired models in them.
    """
    return await list_local_models(provider, base_url, timeout) or []


#: The providers whose installed models Settings lists from the server itself (#1666).
LOCAL_LISTING_PROVIDERS = frozenset({"ollama", "vllm"})

CatalogConnector = type[OpenAIConnector] | type[AnthropicConnector] | type[GeminiConnector]

#: The cloud providers whose own model listing Settings shows (#1631): the settings id, the
#: connector that reads the listing, and the name the user holds the key with.
CATALOG_PROVIDERS: dict[str, tuple[str, CatalogConnector, str]] = {
    "openai": ("openai", OpenAIConnector, "OpenAI"),
    "anthropic": ("anthropic", AnthropicConnector, "Anthropic"),
    "gemini": ("gemini", GeminiConnector, "Google"),
    "google": ("gemini", GeminiConnector, "Google"),
}

#: A listing takes one request per page; Settings waits for it before the picker fills.
_CATALOG_TIMEOUT_SECONDS = 10.0


async def read_provider_catalog(
    provider: str,
    *,
    base_url: str | None,
    api_key: str | None,
    cache: CatalogCache,
) -> CatalogResult | None:
    """The provider's own model listing for this key and endpoint, or None for a local server.

    A live listing is reused from `cache` until it expires; anything else is asked again, so
    a fixed key or network shows on the next open (`CatalogCache`).
    """
    spec = CATALOG_PROVIDERS.get(provider.strip().lower())
    if spec is None:
        return None
    settings_id, connector_cls, display_provider = spec
    endpoint = (base_url or "").strip() or None
    cache_key = (settings_id, endpoint or "", key_fingerprint(api_key))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    lister: Lister | None = None
    if api_key and api_key.strip():
        key = api_key.strip()

        async def list_models() -> list[CatalogEntry]:
            connector = connector_cls(
                api_key=key, base_url=endpoint, timeout=_CATALOG_TIMEOUT_SECONDS
            )
            return await connector.list_models()

        lister = list_models

    result = await read_catalog(
        provider=settings_id,
        display_provider=display_provider,
        lister=lister,
        recommend=recommend,
    )
    cache.put(cache_key, result)
    return result


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
        "a2a_peers": list(persona.a2a_peers),
        "builtin": builtin,
        "overrides_builtin": not builtin and registry.has_builtin(persona.name),
        # The address to show its picture from; the `?v=` changes whenever the picture does.
        "avatar_url": avatar_url(persona.name, PersonaAvatarStore(registry).find(persona.name)),
        # Whether that picture was chosen here rather than shipped: only a chosen one resets.
        "avatar_chosen": PersonaAvatarStore(registry).chosen(persona.name) is not None,
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
                "there is no default agent. `GET /api/personas` lists the ones this "
                "install has."
            ),
        )
    return raw.strip()


def _next_sequence_number() -> int:
    """Increment and return global SSE sequence counter."""
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


TRANSCRIPT_FAILURE_ROLE = "failure"
"""The transcript role of a turn that failed (#969).

A failure is not something the agent said. The chat head saved one as `role: "assistant"`
with `Error: ...` content, so the saved conversation claimed a reply and the only mark on it
was that wording. A record with this role keeps the text the page showed in `content` and
the turn's error in `error`; it is shown to the user and never rebuilt into model context.
"""


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


def _records_a_turn_not_spoken(role: str, entry: dict[str, Any]) -> bool:
    """Whether this row records how a turn ended rather than something that was said (#1031).

    `reconstruct_history` must keep both out of rebuilt model context: neither is anything
    the agent said. Such rows exist only in transcripts the retired single-agent chat path
    (`/api/turn`, removed in #1731) saved before every conversation became a room.
    """
    return role == TRANSCRIPT_CANCELLED_ROLE or _is_failure_entry(role, entry)


def _is_presentable_role(role: str, compaction_ledger: bool) -> bool:
    """Whether a message with this role belongs in the transcript the user reads.

    Stated over the two fields it actually depends on, because `reconstruct_history` asks
    it of a persisted transcript dict rather than of a typed `ChatMessage` (#872).

    A system message is an anchor, not conversation, *unless* it is a compaction ledger.
    """
    return role != MessageRole.SYSTEM.value or compaction_ledger


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
_NO_KEY: tuple[None, None, None] = (None, None, None)


class KeyRemovalRefused(ValueError):
    """A key removal Settings refuses, with a stable `code` a screen can translate.

    `code` is ``"unknown_provider"`` or ``"key_in_use"``; the message is the English
    sentence for a caller that shows text as it comes.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
        self._agents: dict[str, BaseAgent] = {}
        # One memory store instance per agent id, not per session. Two live sessions of the
        # same agent constructing their own stores over the same file would each hold the
        # whole fact set in memory and each `save()` the whole of it, so the second writer
        # drops whatever the first recorded after it loaded (#1097).
        self._agent_memories: dict[str, CrossSessionMemory] = {}
        #: (provider, base URL) -> the host binder for it, or `None` where every tool is
        #: pinned. Shared by every clone, so a tool description is embedded once.
        self._tool_binders: dict[tuple[str, str], ToolBinder | None] = {}
        self._session_messages: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._configured_provider: str | None = None
        self._configured_base_url: str | None = None
        #: The deep model: what a clone's turns run on unless its persona names its own.
        self._configured_model: str | None = None
        #: The fast model, for auxiliary calls (room routing); `None` means the deep one.
        self._configured_model_fast: str | None = None
        #: Chat agent -> whether its (deep, fast) model follows Settings, i.e. its persona
        #: named none. A Settings save moves only those; a persona's own model stays.
        self._follows_settings: dict[str, tuple[bool, bool]] = {}
        self._configured_comfyui_url: str | None = os.getenv(
            "COMFYUI_BASE_URL", DEFAULT_COMFYUI_BASE_URL
        )
        #: Folders outside the workspace that clones may read, as the user entered them.
        self._configured_read_roots: tuple[str, ...] = ()
        #: The language the heads and the CLI write in; `"system"` follows the OS or browser.
        self._configured_ui_language: UiLanguage = DEFAULT_UI_LANGUAGE
        # The same file `ucx run` and `ucx install` read and seed (`saved_choice.py`).
        self._settings_file: Path = self._storage_dir / SETTINGS_FILE_NAME
        # This dashboard's paid calls are held to the limits in its own settings file and
        # booked in its own storage directory, which its Usage panel reads.
        self._usage_file: Path = self._storage_dir / USAGE_FILE_NAME
        self._usage_gate = UsageGate.for_storage(self._settings_file, self._usage_file)
        self._load_persisted_settings()
        # Pictures follow this dashboard's settings file, the chat provider in effect and
        # the Gemini address the chat uses, all read again on every draw, so a change in
        # Settings applies to the next one.
        bind_image_engine_settings(
            self._tools.get("generate_image"),
            self._settings_file,
            lambda: self.provider_in_effect,
            self.gemini_base_url_in_effect,
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

    def build_llm(self, **kwargs: Any) -> BaseLLMConnector:
        """A connector for this dashboard: `create_llm_connector` bound to its storage.

        Every connector the dashboard builds comes from here, so a paid one is held to the
        limits its Usage panel shows, and a connector resolved from a saved choice uses the
        choice this dashboard's Settings saved, even when `storage_dir` is not the session
        root.
        """
        return create_llm_connector(
            usage_gate=self._usage_gate, saved_choice_file=self._settings_file, **kwargs
        )

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
        """The deep model saved in Settings, or `None` when none was."""
        return self._configured_model

    @property
    def configured_model_fast(self) -> str | None:
        """The fast model saved in Settings, or `None` when fast follows deep."""
        return self._configured_model_fast

    @property
    def deep_model(self) -> str | None:
        """The model a clone's turn runs on when its persona names none.

        The one saved in Settings, else the active provider's model variable. `None` when
        neither names one: the request is then refused in plain words rather than sent with
        a model id written in source.
        """
        return self._model_state(self.provider_in_effect)[0]

    @property
    def fast_model(self) -> str | None:
        """The model auxiliary calls (room routing) run on: the fast one, else deep."""
        return self._saved_fast_model(self.provider_in_effect) or self.deep_model

    # -- What is in effect: an explicit argument, then the environment, then the file. --
    #
    # A variable the operator set wins over a choice saved in Settings, for the provider,
    # its model and its endpoint as for its key. A saved model or endpoint belongs to the
    # saved provider: when the environment names another provider, they are not carried
    # over to it.

    def _provider_state(self) -> tuple[str | None, str, str]:
        """``(provider, source, variable)``: `LLM_PROVIDER` when it names one, else the file."""
        from_env = canonical_provider(os.getenv("LLM_PROVIDER"))
        if from_env is not None:
            return from_env, "env", "LLM_PROVIDER"
        if self._configured_provider:
            return self._configured_provider, "settings", ""
        return None, "", ""

    def _saved_applies(self, provider: str | None) -> bool:
        """Whether the model and endpoint saved in the file are ``provider``'s."""
        return (
            self._configured_provider is None
            or provider is None
            or same_provider(self._configured_provider, provider)
        )

    def _model_state(self, provider: str | None) -> tuple[str | None, str, str]:
        """``(model, source, variable)`` for ``provider``: its model variable, else the file."""
        from_env = env_model(provider)
        if from_env is not None:
            return from_env[0], "env", from_env[1]
        if self._configured_model and self._saved_applies(provider):
            return self._configured_model, "settings", ""
        return None, "", ""

    def _saved_fast_model(self, provider: str | None) -> str | None:
        return self._configured_model_fast if self._saved_applies(provider) else None

    def _base_url_state(self, provider: str | None) -> tuple[str | None, str, str]:
        """``(endpoint, source, variable)`` for ``provider``: its endpoint variable, else the file."""
        spec = spec_for(provider)
        if spec is not None and spec.base_url_env:
            value = (os.getenv(spec.base_url_env) or "").strip()
            if value:
                return value, "env", spec.base_url_env
        if self._configured_base_url and self._saved_applies(provider):
            return self._configured_base_url, "settings", ""
        return None, "", ""

    @property
    def provider_in_effect(self) -> str | None:
        """The provider a turn goes to: `LLM_PROVIDER`, else the one saved; `None` if neither."""
        return self._provider_state()[0]

    def base_url_in_effect(self, provider: str | None) -> str | None:
        """The endpoint ``provider``'s requests go to, when one is set for it."""
        return self._base_url_state(provider)[0]

    def gemini_base_url_in_effect(self) -> str | None:
        """The Gemini address a Gemini chat here goes to, for pictures to reuse (#1769).

        `None` unless Gemini is the chat provider in effect or the one saved: a saved
        address with no saved provider may belong to another provider's server, and a
        picture request sent there would carry the Gemini key to it.
        """
        if not (
            same_provider(self.provider_in_effect, "gemini")
            or same_provider(self._configured_provider, "gemini")
        ):
            return None
        return self.base_url_in_effect("gemini")

    def _build_llm_in_effect(self) -> BaseLLMConnector:
        provider = self.provider_in_effect
        return self._build_configured_llm(provider, self.base_url_in_effect(provider))

    def global_models(self) -> tuple[str | None, str | None]:
        """``(deep, fast)`` as Settings holds them now; read per build, never cached."""
        return self.deep_model, self.fast_model

    @property
    def configured_provider(self) -> str | None:
        """Configured provider override for the UI session manager."""
        return self._configured_provider

    @property
    def configured_api_key(self) -> str | None:
        """The key a request to the configured provider carries, when one is held for it."""
        return self._api_key_for(self.provider_in_effect)

    def _key_state(self, provider: str | None) -> tuple[str | None, str | None, str | None]:
        """``(key, source, variable)`` for ``provider``: the environment first, then the file.

        `source` is ``"env"`` or ``"settings"``, and `variable` names the environment
        variable when that is where the key came from. Read from the file on every call, so
        a key saved by `ucx key set` while this dashboard runs is seen at once.
        """
        from_env = env_key(provider)
        if from_env is not None:
            return from_env[0], "env", from_env[1]
        canonical = canonical_provider(provider)
        if canonical is None:
            return _NO_KEY
        saved = api_key_for(settings_data(self._settings_file), canonical)
        if saved:
            return saved, "settings", None
        return _NO_KEY

    def _api_key_for(self, provider: str | None) -> str | None:
        """The key held for ``provider`` and only for it, or `None`."""
        return self._key_state(provider)[0]

    def resolved_api_key(self, provider: str | None) -> str | None:
        """The key a request to ``provider`` would carry: its own, else the live connector's.

        The live connector's key counts only when that connector is ``provider``'s, so a key
        is never sent to a provider it was not saved for.
        """
        own = self._api_key_for(provider)
        if own:
            return own
        live = self._llm
        if live is not None and same_provider(getattr(live, "provider_name", None), provider):
            return cast(str | None, getattr(live, "api_key", None))
        return None

    def stored_api_key_for(self, provider: str, base_url: str | None = None) -> str | None:
        """A key already held for ``provider``, and only for it: saved for it, or its env var.

        For a provider the user has picked but not saved (#1657). The live connector's key
        is not used: it may belong to the provider in use, and must not be sent to another
        one to preview its models.

        A held key goes only to the provider's own server, or to the endpoint saved with it.
        An endpoint left on the form from another provider (a vLLM box, a proxy) gets none.
        """
        if base_url is not None and not (
            self._configured_provider is not None
            and same_provider(self._configured_provider, provider)
            and base_url.rstrip("/") == (self._configured_base_url or "").rstrip("/")
        ):
            return None
        return self._api_key_for(provider)

    @staticmethod
    def _mask_key(key: str) -> str:
        """``AQ.…abcd``-style: enough to recognise a key, never enough to use it."""
        trimmed = key.strip()
        return f"{trimmed[:3]}...{trimmed[-4:]}" if len(trimmed) > 8 else "***"

    def _provider_key_states(self) -> list[dict[str, Any]]:
        """Every keyed provider's key state, whichever provider is active."""
        states: list[dict[str, Any]] = []
        for spec in PROVIDERS.values():
            if not spec.key_env_vars:
                continue
            key, source, variable = self._key_state(spec.id)
            states.append(
                {
                    "id": spec.id,
                    "key_set": bool(key),
                    "key_masked": self._mask_key(key) if key else "",
                    "key_source": source or "",
                    "key_env_var": variable or "",
                }
            )
        return states

    @property
    def configured_base_url(self) -> str | None:
        """Configured provider base URL override for the UI session manager."""
        return self._configured_base_url

    @property
    def default_llm(self) -> LLMProviderProtocol | None:
        """Default or configured LLM provider connector."""
        return self._llm

    def _build_configured_llm(self, provider: str | None, base_url: str | None) -> BaseLLMConnector:
        """A connector for ``provider`` from explicit values, the deep model as its default.

        Everything is passed as an argument: nothing is exported into the environment for
        the connector to find there.
        """
        extra: dict[str, Any] = {}
        deep = self.deep_model
        if deep:
            extra["model"] = deep
        return self.build_llm(
            provider=provider or None,
            api_key=self._api_key_for(provider),
            base_url=base_url or None,
            fallback_to_mock=self._fallback_to_mock,
            **extra,
        )

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
        if isinstance(cfg.get("llm_provider"), str):
            raw_provider = cast(str, cfg["llm_provider"]).strip().lower()
            self._configured_provider = canonical_provider(raw_provider) or raw_provider or None
        if isinstance(cfg.get("llm_base_url"), str):
            self._configured_base_url = cast(str, cfg["llm_base_url"]).strip()
        if isinstance(cfg.get("llm_model"), str):
            self._configured_model = cast(str, cfg["llm_model"]).strip() or None
        if isinstance(cfg.get(LLM_MODEL_FAST_KEY), str):
            self._configured_model_fast = cast(str, cfg[LLM_MODEL_FAST_KEY]).strip() or None
        if isinstance(cfg.get("comfyui_base_url"), str):
            self._configured_comfyui_url = cast(str, cfg["comfyui_base_url"]).strip()
            img_tool = self._tools.get("generate_image")
            if self._configured_comfyui_url and isinstance(img_tool, ComfyImageGenTool):
                img_tool.update_base_url(self._configured_comfyui_url)
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

        if self._llm is None and (self._configured_provider or self._configured_base_url):
            try:
                self._llm = self._build_llm_in_effect()
            except Exception as exc:
                logger.warning(
                    "Failed to initialize active LLM connector from persisted settings %s: %s",
                    self._settings_file,
                    exc,
                )

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
        self._configured_provider = canonical_provider(saved.provider) or saved.provider
        self._configured_model = self._configured_model or saved.model
        self._configured_model_fast = self._configured_model_fast or saved.model_fast
        self._configured_base_url = self._configured_base_url or saved.base_url
        if self._llm is None:
            try:
                adopted = self._build_llm_in_effect()
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
        opened before either picks the new connector up the same way. An agent takes the
        Settings models only in the slots its persona left empty; a persona's own model is
        never replaced by a Settings save.
        """
        self._llm = new_llm
        deep, fast = self.global_models()
        seen: set[int] = set()
        for key, agent in self._agents.items():
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            deep_follows, fast_follows = self._follows_settings.get(key, (True, True))
            agent.hot_reload_llm(
                new_llm,
                model_name=deep if deep_follows else None,
                fast_model=fast if fast_follows else None,
            )
        for listener in self._llm_listeners:
            listener(new_llm)

    def _save_persisted_settings(self, changes: dict[str, Any]) -> None:
        """Merge the settings this save changed into the settings file.

        Only `changes` are written: the file is shared with setup and `ucx llm use`, and
        rewriting it whole from memory put `"llm_provider": null` back over a model saved
        after this dashboard started. When the file cannot be read, it is replaced with
        everything this dashboard holds, as a Settings save always did. Keys are not held
        here, so they are written by `save_api_key` alone.
        """
        everything: dict[str, Any] = {
            "llm_provider": self._configured_provider,
            "llm_base_url": self._configured_base_url,
            "llm_model": self._configured_model,
            LLM_MODEL_FAST_KEY: self._configured_model_fast,
            "comfyui_base_url": self._configured_comfyui_url,
            "read_roots": list(self._configured_read_roots),
            "ui_language": self._configured_ui_language,
        }
        try:
            update_settings_file(
                changes, path=self._settings_file, replace_unreadable_with=everything
            )
        except Exception as exc:
            logger.warning("Failed to write settings file %s: %s", self._settings_file, exc)

    def _active_provider(self) -> str:
        """The provider Settings reports: `LLM_PROVIDER`, saved, the live connector's, or ollama."""
        return (
            self.provider_in_effect
            or (getattr(self._llm, "provider_name", None) if self._llm else None)
            or "ollama"
        )

    def _env_overrides(self, provider: str) -> list[dict[str, str]]:
        """Each setting an environment variable decides instead of the file, in plain words.

        Saving Settings still writes the file; these say why the saved choice is not the
        one in use while the variable stays set.
        """
        overrides: list[dict[str, str]] = []
        for field, label, (_, source, variable) in (
            ("llm_provider", "provider", self._provider_state()),
            ("llm_model", "model", self._model_state(provider)),
            ("llm_base_url", "endpoint", self._base_url_state(provider)),
        ):
            if source == "env" and variable:
                overrides.append(
                    {
                        "field": field,
                        "env_var": variable,
                        "message": (
                            f"The environment variable {variable} is set, so it is used "
                            f"instead of the {label} chosen here. Saving still keeps your "
                            f"choice for when the variable is removed."
                        ),
                    }
                )
        return overrides

    def get_settings(self) -> dict[str, Any]:
        """Return active endpoints, configurations, and masked credentials."""
        self._adopt_saved_choice()
        active_provider = self._active_provider()
        spec = spec_for(active_provider)

        active_base_url, base_url_source, base_url_var = self._base_url_state(active_provider)
        if not active_base_url and self._llm and hasattr(self._llm, "base_url"):
            active_base_url = cast(str | None, getattr(self._llm, "base_url", None))
        if not active_base_url:
            if active_provider == "ollama":
                active_base_url = resolve_ollama_base_url()
            elif active_provider == "vllm":
                # `resolve_vllm_base_url` refuses rather than defaults, and the refusal
                # belongs on a turn, not on opening the panel where the endpoint is typed.
                active_base_url = resolve_vllm_base_url() if has_configured_vllm_endpoint() else ""
            elif spec is not None and spec.base_url_env:
                active_base_url = os.getenv(spec.base_url_env) or ""

        # A cloud model is the one the user or `*_MODEL` named, or none: an unset model is
        # filled from the provider's listing (`catalog.recommended`), never from a
        # remembered id, which is how Settings came to show a retired one (#1631).
        active_model, model_source, model_var = self._model_state(active_provider)
        _, provider_source, provider_var = self._provider_state()

        key, source, variable = self._key_state(active_provider)
        if not key:
            live_key = self.resolved_api_key(active_provider)
            if live_key:
                key, source = live_key, "settings"

        comfy_url = self._configured_comfyui_url or os.getenv(
            "COMFYUI_BASE_URL", DEFAULT_COMFYUI_BASE_URL
        )

        available_providers = [pid for pid in PROVIDERS if pid != "mock"]
        if active_provider == "mock":
            available_providers.append("mock")

        return {
            "llm_provider": active_provider,
            "llm_base_url": active_base_url or "",
            "llm_model": active_model or "",
            "llm_model_fast": self._saved_fast_model(active_provider) or "",
            "llm_provider_source": provider_source,
            "llm_provider_env_var": provider_var,
            "llm_model_source": model_source,
            "llm_model_env_var": model_var,
            "llm_base_url_source": base_url_source,
            "llm_base_url_env_var": base_url_var,
            "env_overrides": self._env_overrides(active_provider),
            "llm_api_key_set": bool(key),
            "llm_api_key_masked": self._mask_key(key) if key else "",
            "llm_api_key_source": source or "",
            "llm_api_key_env_var": variable or "",
            "providers": self._provider_key_states(),
            "comfyui_base_url": comfy_url,
            **self._image_settings(),
            "providers_available": available_providers,
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

    def _image_settings(self) -> dict[str, str]:
        """``image_engine`` and ``image_model`` as saved, read from the file each time.

        The file is their only home: the dispatcher reads it on every draw, so a copy held
        here could only disagree with it. A stored value the dispatcher would refuse is
        shown as saved, with the refusal in ``image_settings_problem``.
        """
        data = settings_data(self._settings_file)
        raw_engine: object = data.get(IMAGE_ENGINE_KEY)
        raw_model: object = data.get(IMAGE_MODEL_KEY)
        problem = ""
        try:
            engine: str = parse_image_engine_setting(raw_engine)
        except PlainRefusalError as exc:
            engine, problem = str(raw_engine), str(exc)
        try:
            model = parse_image_model(raw_model)
        except PlainRefusalError as exc:
            model, problem = str(raw_model), problem or str(exc)
        return {"image_engine": engine, "image_model": model, "image_settings_problem": problem}

    def on_llm_replaced(self, listener: Callable[[LLMProviderProtocol | None], None]) -> None:
        """Call `listener` with the new connector whenever Settings replaces it."""
        self._llm_listeners.append(listener)

    def set_agent_model(self, agent: BaseAgent, model_name: str) -> None:
        """Give ``agent`` its own deep model; a later Settings save no longer moves it."""
        agent.hot_reload_llm(self._llm or agent.llm, model_name=model_name)
        for key, held in self._agents.items():
            if held is agent:
                _, fast_follows = self._follows_settings.get(key, (True, True))
                self._follows_settings[key] = (False, fast_follows)

    def remove_api_key(self, provider: str) -> dict[str, Any]:
        """Remove the key saved for ``provider``; return the settings as they now are.

        The key the active connector is using is not removed: the next turn would fail with
        no key. The person switches provider first, or pastes a new key over it.

        Raises:
            KeyRemovalRefused: an unknown provider, or the active provider's key.
        """
        canonical = canonical_provider(provider)
        if canonical is None:
            raise KeyRemovalRefused("unknown_provider", "That provider is not one Settings knows.")
        if same_provider(self._active_provider(), canonical) and self._llm is not None:
            raise KeyRemovalRefused(
                "key_in_use",
                "This key is in use. Switch to another provider first, "
                "or paste a new key to replace it.",
            )
        delete_api_key(canonical, path=self._settings_file)
        return self.get_settings()

    def update_settings(
        self,
        llm_provider: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        comfyui_base_url: str | None = None,
        read_roots: list[str] | None = None,
        ui_language: object = None,
        llm_model_fast: str | None = None,
        llm_api_key_provider: str | None = None,
        image_engine: object = None,
        image_model: object = None,
    ) -> dict[str, Any]:
        """Update configurations, hot-reload LLM connectors and tools across active agents.

        Nothing is written into `os.environ`. A key is saved for the provider it names
        (`llm_api_key_provider`, else the provider this save leaves active) and for no other.
        """
        self._adopt_saved_choice()
        if ui_language is not None and not is_ui_language(ui_language):
            raise ValueError(
                f"ui_language must be one of {', '.join(UI_LANGUAGES)}, got {ui_language!r}"
            )
        # Refused before anything is changed, like `ui_language`: a save carrying an
        # unknown picture engine changes nothing else either.
        image_changes: dict[str, str] = {}
        try:
            if image_engine is not None:
                image_changes[IMAGE_ENGINE_KEY] = parse_image_engine_setting(image_engine)
            if image_model is not None:
                image_changes[IMAGE_MODEL_KEY] = parse_image_model(image_model)
        except PlainRefusalError as exc:
            raise ValueError(str(exc)) from exc
        clean_roots = (
            _validate_read_roots(read_roots, self._storage_dir, self._configured_read_roots)
            if read_roots is not None
            else None
        )
        state_updates: dict[str, str | None] = {}

        if llm_provider is not None and llm_provider.strip():
            prov_clean = canonical_provider(llm_provider)
            if prov_clean is None:
                raise ValueError(f"Unsupported LLM provider: {llm_provider.strip().lower()}")
            state_updates["_configured_provider"] = prov_clean

        eff_provider = state_updates.get("_configured_provider", self._configured_provider)

        if llm_base_url is not None:
            state_updates["_configured_base_url"] = llm_base_url.strip()

        if llm_model is not None and llm_model.strip():
            state_updates["_configured_model"] = llm_model.strip()

        if llm_model_fast is not None:
            # Empty is a choice here, not an omission: fast follows deep again.
            state_updates["_configured_model_fast"] = llm_model_fast.strip() or None

        new_key: tuple[str, str] | None = None
        if llm_api_key is not None:
            clean_key = llm_api_key.strip()
            if clean_key and not clean_key.startswith("***") and "..." not in clean_key:
                key_for = canonical_provider(llm_api_key_provider) or canonical_provider(
                    eff_provider
                    or (getattr(self._llm, "provider_name", None) if self._llm else None)
                    or os.getenv("LLM_PROVIDER")
                )
                if key_for is None:
                    raise ValueError("Choose which provider this key is for, then save again.")
                new_key = (key_for, clean_key)

        # A save that names no LLM field (the language control's) keeps the connector: rebuilding
        # it re-probes the provider, and a failed probe would refuse a language save.
        reload_llm = any(
            v is not None
            for v in (llm_provider, llm_base_url, llm_api_key, llm_model, llm_model_fast)
        )
        previous = {attr: getattr(self, attr) for attr in state_updates}
        for k, v in state_updates.items():
            setattr(self, k, v)
        if new_key is not None:
            # The key is saved before the rebuild reads it, and stays saved if the rebuild
            # fails: it is the person's key for that provider whatever happens to the probe.
            save_api_key(new_key[0], new_key[1], path=self._settings_file)
        new_llm = None
        if reload_llm:
            # The provider in effect after this save: `LLM_PROVIDER` still wins over the
            # one just saved, which is written to the file all the same.
            eff_provider_for_llm = str(
                self.provider_in_effect
                or (getattr(self._llm, "provider_name", None) if self._llm else None)
                or ""
            )
            try:
                new_llm = self._build_configured_llm(
                    # Empty means "resolve from configuration". The literal "ollama" here made
                    # the settings path build a localhost connector for a user who had
                    # configured nothing, which is the case #533's refusal exists to report.
                    eff_provider_for_llm or None,
                    self.base_url_in_effect(eff_provider_for_llm or None),
                )
            except Exception:
                for k, v in previous.items():
                    setattr(self, k, v)
                raise

        if new_llm is not None:
            self._install_llm(new_llm)

            logger.info(
                "⚙️ [UI Settings] Model/Settings updated: provider=%s, model=%s, fast=%s, base_url=%s",
                eff_provider,
                self._configured_model,
                self._configured_model_fast,
                self._configured_base_url,
            )
            _console.print(
                f"[bold green]⚙️ [UI Settings] Active model updated:[/bold green] [bold yellow]{self._configured_model}[/bold yellow] "
                f"(provider: [cyan]{eff_provider}[/cyan])"
            )

        if comfyui_base_url is not None and comfyui_base_url.strip():
            clean_comfy = comfyui_base_url.strip()
            self._configured_comfyui_url = clean_comfy
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
                (LLM_MODEL_FAST_KEY, "_configured_model_fast"),
            )
            if attr in state_updates
        }
        if comfyui_base_url is not None and comfyui_base_url.strip():
            changes["comfyui_base_url"] = self._configured_comfyui_url
        if clean_roots is not None:
            changes["read_roots"] = list(self._configured_read_roots)
        if ui_language is not None:
            self._configured_ui_language = ui_language
            changes["ui_language"] = ui_language
        changes.update(image_changes)
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
        `e8b3e2f`: after a transcript was saved for `"SessA"`,
        `load_session_record("SESSA")` returned `SessA`'s transcript — including its
        `session_id` — and the history reader cached it under `"SESSA"`, so the
        dashboard displayed one session's conversation as another's. `#256` names only the
        Core store; this door was found by enumerating the id-bearing surface. The rule is
        `agent.session.verify_record_identity`, the same one the Core store uses, not a
        second copy.

        The check is conditional on the record actually carrying a `session_id` string,
        which is deliberate: it refuses on **positive evidence** of a different owner and
        stays silent about a legacy file too damaged to name its session, which is already
        handled by the shape mismatch below.

        Raises:
            PathTraversalError: See `core.session.resolve_session_path`.
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
            PathTraversalError: See `core.session.resolve_session_path`.
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

    def app_scope(self, llm: LLMProviderProtocol | None = None) -> AppScope:
        """The app scope every clone this manager serves is built from (§5.9.2).

        One for a 1:1 chat and one per room, over the same parts: the chat and a room seat
        of one clone differ only in what the room adds (owner ruling 2026-09-27). `llm` is
        the connector for a chat agent built before Settings installed one.

        The persona registry is read afresh, against the live tool inventory -- MCP names
        included -- so `allowed_tools` is checked against what is registered now. The
        binder is kept per (provider, base URL), so each tool description is embedded once
        for every clone; the grow-only bound set itself is per session.
        """
        from uclone_x.agent.persona_registry import get_default_persona_registry

        settings = self.get_settings()
        binder_key = (
            str(settings.get("llm_provider") or ""),
            str(settings.get("llm_base_url") or ""),
        )
        if binder_key not in self._tool_binders:
            self._tool_binders[binder_key] = provider_tool_binder(*binder_key)
        scope = AppScope.create(
            workspace_root=self._workspace_dir,
            persona_registry=get_default_persona_registry(
                self._workspace_dir,
                tool_names=[tool.name for tool in self._tools.list_tools()],
            ),
            memory_for=self.memory_for,
            global_models=self.global_models,
            read_roots=lambda: self.read_roots,
            bus=self._bus,
            llm=self._llm or llm,
            tools=self._tools,
            tracer=self._tracer,
            store=self._core_store,
            budget=self._budget_tracker,
            skills=self._skill_registry,
            ontology=self._ontology_engine,
            tool_binder=self._tool_binders[binder_key],
            # The desktop app has no approval prompt in a conversation, so a call that
            # needs a person is refused at once and names where to decide instead (the
            # story view for codex proposals; owner decision 2026-09-26).
            approvals_answered=False,
        )
        # A peer a chat clone calls is answered on the connector in effect at the call,
        # so a model chosen in Settings since reaches the callee too.
        return dataclasses.replace(
            scope, live_host=lambda: dataclasses.replace(scope.host, llm=self._llm or llm)
        )

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
            if self._llm is not None:
                llm = self._llm
            else:
                deep = self.deep_model
                provider = self.provider_in_effect
                llm = self.build_llm(
                    provider=provider,
                    api_key=self._api_key_for(provider),
                    base_url=self.base_url_in_effect(provider),
                    fallback_to_mock=use_fallback,
                    **({"model": deep} if deep else {}),
                )

            # Built as a room seat of this clone is built (owner ruling 2026-09-27); a chat
            # adds nothing a room adds. A model asked for with this request wins, else the
            # persona's own, else Settings'; only the slots left empty follow Settings.
            built = build_clone(
                self.app_scope(llm),
                clone_id=agent_id,
                session_id=effective_session_id,
                model_name=model_name,
                fallback_prompt=system_prompt or None,
            )
            agent = built.agent
            config = agent.config
            follows = built.follows

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
                self._follows_settings[agent_key] = follows
                if agent_id not in self._agents:
                    self._agents[agent_id] = agent
                    self._follows_settings[agent_id] = follows
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
            self._follows_settings[agent_key] = follows
            if agent_id not in self._agents:
                self._agents[agent_id] = agent
                self._follows_settings[agent_id] = follows
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
    tunnel_manager = SSHTunnelManager()
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
        original, applied = _read_remote_restore()
        cur = session_mgr.get_settings()
        restored: dict[str, str] = {}
        for field in ("llm_base_url", "comfyui_base_url"):
            if field in original and str(cur.get(field) or "") == applied.get(field):
                restored[field] = original[field]
        # The provider was switched to reach the tunnel's Ollama; it goes back with the
        # address, never on its own.
        if "llm_base_url" in restored and "llm_provider" in original:
            restored["llm_provider"] = original["llm_provider"]
        if restored:
            session_mgr.update_settings(
                llm_provider=restored.get("llm_provider"),
                llm_base_url=restored.get("llm_base_url"),
                comfyui_base_url=restored.get("comfyui_base_url"),
            )
        with contextlib.suppress(FileNotFoundError):
            remote_restore_path.unlink()
        return restored

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
        yield
        mcp_start.cancel()
        with contextlib.suppress(BaseException):
            await mcp_start
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

    # This lists what is *installed*, not what is running (#1190; the live-instance
    # `/api/agents` was removed 2026-09-27, #1775). It is handed
    # the room stack as well as the session manager because a clone is running in either
    # seat, and under D1 the conversation seat is the ordinary one.
    from uclone_x.ui.clones import register_clone_routes

    register_clone_routes(app, session_mgr, room_stack)

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
    register_artifact_routes(app, room_stack, person_gate)

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

    catalog_cache = CatalogCache()

    async def _catalog_for(provider: str) -> CatalogResult | None:
        # Only an endpoint the user set is sent: the listing goes where the turns would.
        return await read_provider_catalog(
            provider,
            base_url=session_mgr.base_url_in_effect(provider),
            api_key=session_mgr.resolved_api_key(provider),
            cache=catalog_cache,
        )

    async def _model_ids(
        provider: str, settings: dict[str, Any], catalog: CatalogResult | None
    ) -> list[str]:
        if catalog is not None:
            return [entry.id for entry in catalog.entries if entry.chat_capable]
        return await fetch_available_models(
            provider=provider,
            base_url=cast(str | None, settings.get("llm_base_url")) or None,
        )

    @app.get("/api/media/status")
    async def media_status(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Which image engine would draw now, under the `image_engine` setting, and why.

        The probe `ucx media status` prints, run off the event loop: it makes two local
        HTTP probes. Gemini is judged by whether its key is there, never by a request.
        A stored setting the dispatcher would refuse answers 409 with that refusal.
        """
        _refuse_cross_origin(request)
        from uclone_x.tools.builtin.image_status import probe_image_engines

        try:
            report = await asyncio.to_thread(
                probe_image_engines, session_mgr.settings_file, session_mgr.provider_in_effect
            )
        except PlainRefusalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ready": report.ready,
            "engine": report.engine,
            "setting": report.image_engine,
            "engines": [
                {"name": name, "ready": ready, "reason_code": code}
                for name, ready, code in report.engine_states()
            ],
        }

    @app.get("/api/settings")
    async def get_settings(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Retrieve current active endpoints, configurations, and available providers."""
        _refuse_cross_origin(request)  # names the folders clones can read, and the workspace
        settings = session_mgr.get_settings()
        provider = str(settings.get("llm_provider", "ollama"))
        catalog = await _catalog_for(provider)
        settings["catalog"] = catalog.model_dump(mode="json") if catalog else None
        settings["available_models"] = await _model_ids(provider, settings, catalog)
        return settings

    @app.get("/api/models")
    async def get_models(refresh: bool = False) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """Enumerate installed/available models for active provider (P0/Recognition over Recall).

        `refresh=1` forgets the kept listings first, for the picker's "Refresh list".
        """
        if refresh:
            catalog_cache.clear()  # the saved provider's "Refresh list"
        settings = session_mgr.get_settings()
        provider = str(settings.get("llm_provider", "ollama"))
        catalog = await _catalog_for(provider)
        return {
            "provider": provider,
            "models": await _model_ids(provider, settings, catalog),
            "current_model": settings.get("llm_model", ""),
            "catalog": catalog.model_dump(mode="json") if catalog else None,
        }

    @app.post("/api/models/catalog")
    async def preview_catalog(request: Request, req: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        """The model listing for a provider picked on the form, before it is saved (#1657).

        The key and endpoint come from the form, in the body, never the URL. With no key typed,
        only a key already held for that provider is used (`stored_api_key_for`). A local
        provider (Ollama, vLLM) answers `catalog: null`, the models its server has installed,
        and whether anything answered at all (`reachable`, #1666).
        """
        _refuse_cross_origin(request)  # a page in another tab must not spend the user's key
        provider = str(req.get("provider", "")).strip().lower()
        if not provider:
            raise HTTPException(status_code=400, detail="provider is required")
        if req.get("refresh"):
            catalog_cache.clear()
        typed_key = str(req.get("api_key") or "").strip()
        base_url = str(req.get("base_url") or "").strip() or None
        api_key = typed_key or session_mgr.stored_api_key_for(provider, base_url)
        catalog = await read_provider_catalog(
            provider, base_url=base_url, api_key=api_key, cache=catalog_cache
        )
        if catalog is None and provider in LOCAL_LISTING_PROVIDERS:
            # The server's installed models, at the address on the form (#1666). `reachable`
            # tells a stopped server apart from one with nothing installed.
            # The address is whatever is typed, fetched with no click: only the typed key, or
            # one held for this exact saved endpoint, goes with it, never `VLLM_API_KEY`.
            headers = vllm_request_headers(api_key, env_fallback=False)
            try:
                local = await list_local_models(
                    provider, base_url, vllm_headers=headers, raise_on_refused_key=True
                )
            except LocalKeyRefusedError:
                # Running, and it turned the key away: not "nothing answered" (#1672).
                return {
                    "provider": provider,
                    "models": [],
                    "reachable": True,
                    "key_refused": True,
                    "catalog": None,
                }
            return {
                "provider": provider,
                "models": local or [],
                "reachable": local is not None,
                "catalog": None,
            }
        models = [entry.id for entry in catalog.entries if entry.chat_capable] if catalog else []
        return {
            "provider": provider,
            "models": models,
            "catalog": catalog.model_dump(mode="json") if catalog else None,
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
                ui_language=req.get("ui_language"),
                llm_model_fast=req.get("llm_model_fast"),
                llm_api_key_provider=req.get("llm_api_key_provider"),
                image_engine=req.get("image_engine"),
                image_model=req.get("image_model"),
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

    @app.delete("/api/settings/api-keys/{provider}")
    async def remove_api_key(request: Request, provider: str) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Remove the key saved for one provider; every other provider's key stays.

        A refusal answers 400 with ``detail`` (English) and ``code``, which the screen
        translates rather than showing the English text.
        """
        _refuse_cross_origin(request)
        try:
            return session_mgr.remove_api_key(provider)
        except KeyRemovalRefused as exc:
            return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=400)

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
                        # The address may be one just typed: it gets the typed key, or one
                        # held for this exact saved endpoint, never `VLLM_API_KEY` (#1672).
                        vllm_key = eff_key or session_mgr.stored_api_key_for("vllm", eff_base)
                        async with httpx.AsyncClient(timeout=5.0) as http_c:
                            resp = await http_c.get(
                                f"{vllm_url.rstrip('/')}/models",
                                headers=vllm_request_headers(vllm_key, env_fallback=False),
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
                                    "key_refused": resp.status_code in _KEY_REFUSED_STATUSES,
                                }
                elif (keyed := spec_for(eff_provider)) is not None and keyed.requires_key:
                    key_present = bool(eff_key or session_mgr.stored_api_key_for(keyed.id))
                    if key_present:
                        results["llm"] = {
                            "status": "ok",
                            "provider": eff_provider,
                            "message": f"{keyed.display_name} key saved",
                        }
                    else:
                        results["llm"] = {
                            "status": "warning",
                            "provider": eff_provider,
                            "message": f"No {keyed.display_name} key is saved yet",
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
        tunnel_status = await tunnel_manager.connect(
            host=host,
            auto_start_comfyui=auto_start_comfyui,
            timeout=timeout,
        )
        if not tunnel_status.connected:
            return {
                "status": "error",
                "connected": False,
                "error": tunnel_status.error or "Failed to connect tunnel",
                "tunnel": tunnel_status.to_dict(),
            }

        changes: dict[str, Any] = {}
        llm_skipped: str | None = None
        if apply_settings:
            cur = session_mgr.get_settings()
            # A reconnect keeps the first record: the current values are the tunnel's own.
            original, applied = _read_remote_restore()
            current_model = str(cur.get("llm_model") or "").strip()

            for m in tunnel_status.mappings:
                if m.service_name == "ollama":
                    if not sync_llm:
                        continue
                    remote_models = tunnel_status.ollama_models
                    current_bare = current_model.split(":")[0].lower()
                    model_on_remote = (
                        not current_model
                        or not remote_models
                        or any(
                            rm.lower() == current_model.lower()
                            or rm.lower().startswith(current_bare + ":")
                            for rm in remote_models
                        )
                    )
                    if not model_on_remote:
                        # Switching would 404 every turn with "model not found".
                        llm_skipped = "model_not_on_remote"
                        logger.info(
                            "Preserving local llm_base_url; active model %r not found on remote worker models %r",
                            current_model,
                            remote_models,
                        )
                        continue
                    original.setdefault("llm_provider", str(cur.get("llm_provider") or ""))
                    original.setdefault("llm_base_url", str(cur.get("llm_base_url") or ""))
                    applied["llm_provider"] = "ollama"
                    applied["llm_base_url"] = f"http://127.0.0.1:{m.local_port}"
                    session_mgr.update_settings(
                        llm_provider="ollama",
                        llm_base_url=f"http://127.0.0.1:{m.local_port}",
                    )
                    changes["llm_provider"] = "ollama"
                    changes["llm_base_url"] = f"http://127.0.0.1:{m.local_port}"
                elif m.service_name == "comfyui":
                    original.setdefault("comfyui_base_url", str(cur.get("comfyui_base_url") or ""))
                    applied["comfyui_base_url"] = f"http://127.0.0.1:{m.local_port}"
                    session_mgr.update_settings(
                        comfyui_base_url=f"http://127.0.0.1:{m.local_port}",
                    )
                    changes["comfyui_base_url"] = f"http://127.0.0.1:{m.local_port}"
            if original:
                remote_restore_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"original": original, "applied": applied}
                replace_file(remote_restore_path, json.dumps(record, indent=2).encode())

        return {
            "status": "ok",
            "connected": True,
            "tunnel": tunnel_status.to_dict(),
            "applied_changes": changes,
            "llm_skipped": llm_skipped,
        }

    @app.post("/api/settings/remote-gpu/disconnect")
    async def disconnect_remote_gpu_endpoint(  # pyright: ignore[reportUnusedFunction]
        request: Request,
    ) -> dict[str, Any]:
        """Disconnect the active SSH tunnel session and restore previous endpoints."""
        _refuse_unless_local(request)
        await tunnel_manager.disconnect()
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
        return payload

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

    @app.get("/api/personas/{name}/avatar")
    async def get_persona_avatar(name: str) -> Response:  # pyright: ignore[reportUnusedFunction]
        """Return one clone's picture, or refuse with what would put a picture there.

        A 404 here is an ordinary answer, not a fault: most clones have no picture, and the
        head draws its default when this refuses. The detail still names the remedy, because
        this is also what a reader sees who went looking for the file they thought they had
        installed. `no-cache` because the persona's `avatar_url` carries a `?v=` that
        changes with the picture, and a cached old face under a new `?v=` would hide that.
        """
        found = PersonaAvatarStore(_avatar_registry()).find(name)
        if found is None:
            formats = ", ".join(suffix for suffix, _ in AVATAR_FORMATS)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No picture is set for '{name}'. Choose one from its profile, or put an "
                    f"image file beside its definition, named after it and ending in one of "
                    f"{formats}."
                ),
            )
        return Response(
            content=found.path.read_bytes(),
            media_type=found.mime,
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"},
        )

    def _avatar_answer(
        registry: PersonaRegistry, name: str, *, undo_with: Path | None = None
    ) -> dict[str, Any]:
        """The changed clone, and the kept picture that would undo the change.

        `previous_path` is the workspace path to `PUT` back to undo it; `null` means the
        clone had no chosen picture before, so undoing is a `DELETE`.
        """
        persona = registry.get_persona(name)
        if persona is None:
            raise HTTPException(status_code=404, detail=f"There is no clone named '{name}' here.")
        return {
            "status": "ok",
            "persona": _persona_payload(registry, persona),
            "previous_path": _workspace_relative(undo_with),
        }

    def _workspace_relative(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(Path(session_mgr.workspace_dir).resolve()).as_posix()
        except ValueError:
            return None

    def _avatar_refusal(exc: AvatarRefused) -> JSONResponse:
        """The refusal as `{"detail": <plain words>, "code": <which reason>}`.

        The head words its own message from `code` (a path outside the workspace and a file
        in the wrong format ask for different things); `detail` is for everything else.
        """
        status = 404 if isinstance(exc, AvatarPersonaNotFound) else 422
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

    @app.put("/api/personas/{name}/avatar")
    async def put_persona_avatar(name: str, request: Request) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Set one clone's picture from a workspace file or from the image in the body.

        The body is either JSON `{"source_path": "<path in the workspace>"}`, normally an
        image a clone just drew, or the picture itself with an `image/*` content type, for
        an upload. Either way the bytes must be a PNG, JPEG, WebP or GIF image; the picture
        it replaces is kept as `<name>.prev.<ext>`.
        """
        registry = _registry_for_a_picture_change(request)
        store = PersonaAvatarStore(registry)
        had_chosen = store.chosen(name) is not None
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        try:
            if content_type == "application/json":
                store.set_from_path(name, _avatar_source(await _avatar_json(request)))
            elif content_type.startswith("image/"):
                store.set(name, await _avatar_upload(request))
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
        return _avatar_answer(
            registry, name, undo_with=store.previous(name) if had_chosen else None
        )

    def _avatar_source(body: object) -> Path:
        """The workspace file a `PUT` names, resolved as tools resolve a path."""
        raw = cast(dict[str, object], body).get("source_path") if isinstance(body, dict) else None
        if not isinstance(raw, str) or raw.strip() == "":
            raise _avatar_no_source()
        try:
            return PathValidator().resolve_safe_path(Path(raw), session_mgr.workspace_dir)
        except PathTraversalError as exc:
            raise AvatarRefused(
                "That picture is outside the workspace, so it was not used. Choose one in the "
                "workspace, or upload it instead.",
                reason_code="outside_workspace",
            ) from exc

    @app.delete("/api/personas/{name}/avatar")
    async def delete_persona_avatar(name: str, request: Request) -> Any:  # pyright: ignore[reportUnusedFunction]
        """Put a clone's chosen picture aside, so it shows its shipped one or the default."""
        registry = _registry_for_a_picture_change(request)
        try:
            kept = PersonaAvatarStore(registry).reset(name)
        except AvatarRefused as exc:
            return _avatar_refusal(exc)
        return _avatar_answer(registry, name, undo_with=kept)

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
            draft_provider = session_mgr.provider_in_effect
            llm = session_mgr.default_llm or session_mgr.build_llm(
                provider=draft_provider,
                api_key=session_mgr.resolved_api_key(draft_provider),
                base_url=session_mgr.base_url_in_effect(draft_provider),
                fallback_to_mock=False,
                **({"model": session_mgr.deep_model} if session_mgr.deep_model else {}),
            )
            response = await asyncio.wait_for(
                llm.generate(
                    LLMRequest(
                        model=session_mgr.deep_model,
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

    from uclone_x.ui.usage import register_usage_routes

    # Settings → Usage (`llm-token-gateway.md` §4.5): the dashboard's own settings file and
    # the usage store beside it, the same two its paid connectors' gate reads.
    register_usage_routes(
        app,
        settings_file=session_mgr.settings_file,
        usage_file=session_mgr.usage_file,
        refuse_cross_origin=_refuse_cross_origin,
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
