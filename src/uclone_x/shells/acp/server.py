"""ACP (Agent Client Protocol) inbound shell adapter over stdio JSON-RPC."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Final, cast

import uclone_x
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.session import SessionStore, validate_session_id
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.engine.event_bus import AgentEvent, EventBus, EventSubscription, EventType
from uclone_x.shells.acp.models import (
    ACP_PROTOCOL_VERSION,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SERVER_NAME,
    SESSION_NOT_FOUND,
    UNSUPPORTED_MCP_TRANSPORT,
    ACPCapabilities,
    ACPSessionState,
    make_jsonrpc_error,
    make_jsonrpc_notification,
    make_jsonrpc_response,
)

logger = logging.getLogger(__name__)

#: Builds the agent that serves one ACP session. It is called with the session id and
#: must return an agent whose active session is that id, composed over the same
#: `SessionStore` the server reads -- so what a turn persists is what `load_session`
#: finds. The CLI builds it from a persona (`uclone_x.cli.commands.acp`).
SessionAgentFactory = Callable[[str], BaseAgent]

#: How many per-session agents a server holds at once (#1454). See `ACPServer`.
DEFAULT_MAX_LIVE_AGENTS: Final[int] = 8

# What a client is told when a request cannot be served. An editor shows these to the
# person using it, so they say what happened and what to do, and nothing about how the
# server is built; the cause goes to the log.
UNKNOWN_SESSION_MESSAGE: Final[str] = (
    "This conversation isn't open. Start a new one, or reopen it first."
)
NO_AGENT_MESSAGE: Final[str] = "There is no agent on this connection to answer."
SESSION_EXISTS_MESSAGE: Final[str] = (
    "A conversation with this id already exists. Reopen it instead of starting a new one."
)
BAD_SESSION_ID_MESSAGE: Final[str] = "That conversation id can't be used. Choose another one."
TOO_MANY_BUSY_MESSAGE: Final[str] = (
    "Too many conversations are open and unsaved or still answering. Wait for one to "
    "finish, or cancel it, and try again."
)
CANNOT_START_MESSAGE: Final[str] = "This conversation could not be started. Please try again."
CANNOT_OPEN_MESSAGE: Final[str] = "This conversation could not be opened."
NOT_SAVED_MESSAGE: Final[str] = (
    "The reply above could not be saved, so it will be missing when this conversation is reopened."
)
TURN_FAILED_MESSAGE: Final[str] = "Something went wrong while answering. Please try again."
USAGE_LIMIT_MESSAGE: Final[str] = (
    "The usage limit has been reached, so this reply was stopped. Trying again will stop "
    "at the same limit until the limit is raised."
)
TOO_MANY_STEPS_MESSAGE: Final[str] = (
    "This request needed more steps than one reply is allowed, so it was stopped before "
    "it finished. Asking for less at a time may help."
)
BLOCKED_MESSAGE: Final[str] = (
    "A rule set up for this agent stopped this message before it was answered. Sending "
    "it again is likely to be stopped the same way."
)
_PLAIN_ERROR_STOP_REASONS: Final[frozenset[str]] = frozenset({"step_results_over_window"})
"""Stop reasons whose `TurnResult.error` is written for the user and is sent as is."""
_STOP_REASON_MESSAGES: Final[dict[str, str]] = {
    "budget_exceeded": USAGE_LIMIT_MESSAGE,
    "step_budget_exceeded": TOO_MANY_STEPS_MESSAGE,
    "blocked_by_hook": BLOCKED_MESSAGE,
}
"""Turn ends told in plain words of their own. Their `error` is not sent: it can name a
limit's internals or a hook's own text. Only the usage limit is one a retry is sure to
meet again; the step budget resets every turn (#1509 review). Any other turn error can carry a provider's
text, a path or a traceback line, so the client gets `TURN_FAILED_MESSAGE`. Every error
not sent as is goes to the log (#1509).
"""
TURN_IN_FLIGHT_MESSAGE: Final[str] = (
    "This conversation is still answering. Wait for the reply, or cancel it, and try again."
)
REQUEST_FAILED_MESSAGE: Final[str] = (
    "Something went wrong while handling this request. Please try again."
)


class _NoRoomForAgentError(Exception):
    """Every held agent is mid-turn or holds history the store does not."""


class _SessionUnopenableError(Exception):
    """A session's saved history could not be read back into a new agent."""


class _AgentBuildError(Exception):
    """The agent factory failed, or built an agent for another session."""


class ACPServer:
    """Inbound ACP shell adapter implementing ACP over stdio JSON-RPC.

    Provides client-to-agent interoperability for editors (Zed, Neovim, external harnesses)
    per decision #575 and Issue #649.

    **One agent per session (#1454).** Each ACP `sessionId` is served by an agent of its
    own, built by `agent_factory` when the session is started or reopened. The server
    used to hold one agent for every session, so a second session's turn was sent with
    the first session's history; an agent's `execute_turn` runs against its *active*
    session, and giving each session its own agent is how the rest of the codebase keeps
    conversations apart (the chat head keys agents by `(agent_id, session_id)`).

    **Every turn is saved** through `BaseAgent.persist_session` -- the call the chat head
    and room seats make -- so the record and the durable event log (#1442) hold each ACP
    turn, and anything recorded on save reaches ACP sessions without a second call site.

    **Idle agents are bounded, not kept forever.** At most `max_live_agents` are held. When
    one more is needed, the least recently used agent that has no turn in flight *and*
    whose last save succeeded is released. Releasing loses nothing, because its session
    is on disk; its next prompt rebuilds an agent from the store, as `load_session` does.
    An agent whose save failed holds history the store does not, so it is never released
    to make room; if no agent can be released the request is refused, in words, rather
    than dropping a conversation. The agents are not started (no bus loop): the server
    drives each turn itself, so releasing one is dropping the reference. The release
    happens only once the new agent is built and its record saved or restored, so a
    request that fails on the way releases nothing (#1494).

    **One turn at a time per session (#1494).** A `prompt` for a session whose turn is
    still in flight is refused, in words, rather than started beside it: two turns on one
    agent would interleave their messages, and the record of in-flight turns holds one
    task per session. A turn stays in that record until its task has actually finished --
    `cancel` asks the task to stop but does not remove it -- because a turn that is
    cancelled, or ignores the cancel for a while, is still using its agent, and an agent
    in that record is never released.
    """

    def __init__(
        self,
        agent_factory: SessionAgentFactory | None = None,
        bus: EventBus | None = None,
        store: SessionStoreProtocol | None = None,
        reader: asyncio.StreamReader | None = None,
        writer: asyncio.StreamWriter | None = None,
        max_live_agents: int = DEFAULT_MAX_LIVE_AGENTS,
    ) -> None:
        if max_live_agents < 1:
            raise ValueError(f"max_live_agents must be at least 1, got {max_live_agents}")
        self._agent_factory = agent_factory
        self._bus = bus
        self._store = store or SessionStore()
        self._reader = reader
        self._writer = writer
        self._max_live_agents = max_live_agents

        self._sessions: dict[str, ACPSessionState] = {}
        # Least recently used first. Only sessions with an agent built are here.
        self._agents: OrderedDict[str, BaseAgent] = OrderedDict()
        # Sessions whose last save failed: their agent holds what the store does not.
        self._unsaved: set[str] = set()
        self._in_flight_tasks: dict[str, asyncio.Task[None]] = {}
        self._pending_client_requests: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._client_req_counter: int = 0
        self._running: bool = False
        self._write_lock = asyncio.Lock()

    def session_agent(self, session_id: str) -> BaseAgent | None:
        """The agent currently serving `session_id`, or `None` if none is held."""
        return self._agents.get(session_id)

    @property
    def capabilities(self) -> ACPCapabilities:
        """Return declared ACP capabilities."""
        return ACPCapabilities()

    def get_session(self, session_id: str) -> ACPSessionState | None:
        """Get session state if present."""
        return self._sessions.get(session_id)

    def is_turn_in_flight(self, session_id: str) -> bool:
        """Check whether a turn task is in flight for session."""
        task = self._in_flight_tasks.get(session_id)
        return task is not None and not task.done()

    def get_in_flight_task(self, session_id: str) -> asyncio.Task[None] | None:
        """Return active in-flight task for session if any."""
        return self._in_flight_tasks.get(session_id)

    async def run_stdio(self) -> None:
        """Run the stdio JSON-RPC loop until EOF or cancelled."""
        self._running = True
        loop = asyncio.get_running_loop()

        if self._reader is None or self._writer is None:
            # Set up async streams for stdin and stdout
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

            write_transport, write_protocol = await loop.connect_write_pipe(
                asyncio.streams.FlowControlMixin, sys.stdout.buffer
            )
            writer = asyncio.StreamWriter(write_transport, write_protocol, reader, loop)
            self._reader = reader
            self._writer = writer

        try:
            while self._running:
                message = await self._read_message()
                if message is None:
                    # EOF reached
                    break
                await self._process_raw_message(message)
        finally:
            self._running = False
            # Cancel any in-flight turns
            for task in list(self._in_flight_tasks.values()):
                if not task.done():
                    task.cancel()

    async def _read_message(self) -> str | None:
        """Read a single JSON-RPC message, supporting both Content-Length and newline framing."""
        if self._reader is None:
            return None

        line_bytes = await self._reader.readline()
        if not line_bytes:
            return None

        line = line_bytes.decode("utf-8")

        # Check for HTTP/LSP-style Content-Length header framing
        if line.lower().startswith("content-length:"):
            length_str = line.split(":", 1)[1].strip()
            try:
                content_length = int(length_str)
            except ValueError:
                return None

            # Consume subsequent empty line (CRLF or LF)
            while True:
                header_line = await self._reader.readline()
                if not header_line or header_line in (b"\r\n", b"\n", b""):
                    break

            body_bytes = await self._reader.readexactly(content_length)
            return body_bytes.decode("utf-8")

        # Otherwise newline-delimited JSON line
        content = line.strip()
        while not content:
            # Skip empty lines
            line_bytes = await self._reader.readline()
            if not line_bytes:
                return None
            content = line_bytes.decode("utf-8").strip()

        return content

    async def send_response(self, response_dict: dict[str, Any]) -> None:
        """Serialize and send a JSON-RPC response or notification to stdout."""
        data = json.dumps(response_dict) + "\n"
        if self._writer is not None:
            async with self._write_lock:
                self._writer.write(data.encode("utf-8"))
                await self._writer.drain()

    async def process_raw_message(self, raw_str: str) -> None:
        """Public entrypoint to process a raw JSON-RPC message string."""
        await self._process_raw_message(raw_str)

    async def _process_raw_message(self, raw_str: str) -> None:
        """Parse raw JSON and dispatch request, response, or notification."""
        try:
            parsed = json.loads(raw_str)
        except Exception as exc:
            await self.send_response(make_jsonrpc_error(None, PARSE_ERROR, f"Parse error: {exc}"))
            return

        if not isinstance(parsed, dict):
            await self.send_response(
                make_jsonrpc_error(None, PARSE_ERROR, "Invalid JSON-RPC payload: expected object")
            )
            return

        parsed_dict: dict[str, Any] = cast(dict[str, Any], parsed)

        # Check if this is a response from the client to a request we sent (e.g. request_permission)
        raw_id: object = parsed_dict.get("id")
        req_id: int | str | None = None
        if isinstance(raw_id, (int, str)):
            req_id = raw_id

        if req_id is not None and req_id in self._pending_client_requests:
            future = self._pending_client_requests.pop(req_id)
            if not future.done():
                future.set_result(parsed_dict)
            return

        # Otherwise it is an inbound request or notification from the client
        method: object = parsed_dict.get("method")
        if not isinstance(method, str):
            if req_id is not None:
                await self.send_response(
                    make_jsonrpc_error(req_id, PARSE_ERROR, "Missing or invalid method")
                )
            return

        params: object = parsed_dict.get("params")
        params_dict: dict[str, Any] = (
            cast(dict[str, Any], params) if isinstance(params, dict) else {}
        )

        # Notification vs request
        is_notification = req_id is None

        try:
            result = await self.dispatch_method(method, params_dict, req_id)
            if not is_notification and result is not None:
                await self.send_response(result)
        except Exception:
            # The cause goes to the log. The reply is shown to the person in the editor, and
            # the exception text can name paths and internals (#1494).
            logger.exception("Error handling ACP method %s", method)
            if not is_notification:
                await self.send_response(
                    make_jsonrpc_error(req_id, INTERNAL_ERROR, REQUEST_FAILED_MESSAGE)
                )

    async def dispatch_method(
        self,
        method: str,
        params: dict[str, Any],
        req_id: int | str | None,
    ) -> dict[str, Any] | None:
        """Dispatch an ACP protocol method."""
        if method == "initialize":
            return await self._handle_initialize(params, req_id)
        if method == "new_session":
            return await self._handle_new_session(params, req_id)
        if method == "load_session":
            return await self._handle_load_session(params, req_id)
        if method == "set_session_mode":
            return await self._handle_set_session_mode(params, req_id)
        if method == "set_config_option":
            return await self._handle_set_config_option(params, req_id)
        if method == "prompt":
            return await self._handle_prompt(params, req_id)
        if method == "cancel":
            return await self._handle_cancel(params, req_id)

        if req_id is not None:
            return make_jsonrpc_error(
                req_id,
                METHOD_NOT_FOUND,
                f"Method '{method}' not found",
            )
        return None

    async def _handle_initialize(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any]:
        """Handle 'initialize': negotiate protocol version and capabilities (§2, §3.3).

        `serverInfo.version` is the package version, read from `uclone_x.__version__`
        through the module rather than imported by name. A `from uclone_x import
        __version__` would bind the string once at import time, which reports the same
        value but is no longer traceable to the declaration -- and is untestable, since
        nothing a test can set would move it. Reading the attribute per call keeps the
        report following whatever `src/uclone_x/__init__.py` declares, and lets
        `test_acp_initialize_reports_package_version` prove that by moving it.

        Note this is a *reporting* site, not a fourth declaration:
        `test_version_is_declared_once_in_effect` covers the three declaration files and
        deliberately does not list this one.
        """
        result = {
            "protocolVersion": ACP_PROTOCOL_VERSION,
            "capabilities": self.capabilities.model_dump(),
            "serverInfo": {
                "name": SERVER_NAME,
                "version": uclone_x.__version__,
            },
        }
        return make_jsonrpc_response(req_id, result)

    async def _handle_new_session(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any]:
        """Handle 'new_session': validate MCP servers and create runtime session."""
        mcp_servers: object = params.get("mcpServers") or params.get("mcp_servers") or []
        if isinstance(mcp_servers, list):
            for srv in cast(list[object], mcp_servers):
                if isinstance(srv, dict):
                    srv_dict = cast(dict[str, Any], srv)
                    transport = srv_dict.get("transport")
                    # Check for AcpMcpServer (transport="acp" or serverId present)
                    if transport == "acp" or "serverId" in srv_dict:
                        return make_jsonrpc_error(
                            req_id,
                            UNSUPPORTED_MCP_TRANSPORT,
                            "AcpMcpServer is not implementable against today's model: "
                            "reverse client communication unsupported (§5)",
                        )

        session_id = str(
            params.get("sessionId") or params.get("session_id") or uuid.uuid4().hex[:12]
        )
        cwd = cast(str | None, params.get("cwd"))
        try:
            validate_session_id(session_id)
        except Exception:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, BAD_SESSION_ID_MESSAGE)

        if self._agent_factory is not None:
            refusal = self._start_session_agent(session_id, req_id)
            if refusal is not None:
                return refusal

        state = ACPSessionState(
            session_id=session_id,
            mode="auto",
            cwd=cwd,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._sessions[session_id] = state

        return make_jsonrpc_response(
            req_id,
            {"sessionId": session_id, "mode": state.mode},
        )

    async def _handle_load_session(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any]:
        """Handle 'load_session': wire to SessionStore (§2, §3.2)."""
        session_id = params.get("sessionId") or params.get("session_id")
        if not session_id:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, "Missing sessionId")

        session_id = str(session_id)
        if session_id in self._sessions:
            state = self._sessions[session_id]
            return make_jsonrpc_response(req_id, {"sessionId": session_id, "mode": state.mode})

        # A record that cannot be read is not an absent one: reporting it as "not found"
        # would tell the person their conversation does not exist (P6).
        try:
            data = self._store.load(session_id)
        except Exception:
            logger.exception("Could not read the saved ACP session %s", session_id)
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, CANNOT_OPEN_MESSAGE)
        if data is None:
            return make_jsonrpc_error(
                req_id,
                SESSION_NOT_FOUND,
                f"Session not found: {session_id}",
            )

        if self._agent_factory is not None:
            # The saved history goes into this session's own agent now, so the next
            # prompt continues the conversation instead of starting a blank one.
            try:
                self._revive_session_agent(session_id)
            except _NoRoomForAgentError:
                return make_jsonrpc_error(req_id, INTERNAL_ERROR, TOO_MANY_BUSY_MESSAGE)
            except _SessionUnopenableError:
                return make_jsonrpc_error(req_id, INTERNAL_ERROR, CANNOT_OPEN_MESSAGE)

        mode = "auto"
        state = ACPSessionState(
            session_id=session_id,
            mode=mode,
            created_at=time.time(),
            updated_at=time.time(),
        )
        self._sessions[session_id] = state
        return make_jsonrpc_response(req_id, {"sessionId": session_id, "mode": mode})

    async def _handle_set_session_mode(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any]:
        """Handle 'set_session_mode': configure session permission mode."""
        session_id = params.get("sessionId") or params.get("session_id")
        mode = params.get("mode")

        if not session_id or not mode:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, "Missing sessionId or mode")

        session_id = str(session_id)
        if session_id not in self._sessions:
            return make_jsonrpc_error(req_id, SESSION_NOT_FOUND, f"Session not found: {session_id}")

        valid_modes = {"auto", "ask", "readonly"}
        if mode not in valid_modes:
            return make_jsonrpc_error(
                req_id,
                INVALID_PARAMS,
                f"Invalid mode '{mode}'. Valid modes are: {sorted(valid_modes)}",
            )

        self._sessions[session_id].mode = mode
        self._sessions[session_id].updated_at = time.time()
        return make_jsonrpc_response(req_id, {"sessionId": session_id, "mode": mode})

    async def _handle_set_config_option(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any]:
        """Handle 'set_config_option': configure session-specific parameters."""
        session_id = params.get("sessionId") or params.get("session_id")
        name = params.get("name")
        value = params.get("value")

        if not session_id or name is None:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, "Missing sessionId or option name")

        session_id = str(session_id)
        if session_id not in self._sessions:
            return make_jsonrpc_error(req_id, SESSION_NOT_FOUND, f"Session not found: {session_id}")

        self._sessions[session_id].config[str(name)] = value
        self._sessions[session_id].updated_at = time.time()
        return make_jsonrpc_response(
            req_id, {"sessionId": session_id, "name": name, "value": value}
        )

    async def _handle_prompt(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any] | None:
        """Handle 'prompt': execute turn and stream semantic updates to client."""
        session_id = params.get("sessionId") or params.get("session_id")
        raw_prompt = params.get("prompt") or params.get("content") or ""

        if not session_id:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, "Missing sessionId")

        session_id = str(session_id)
        # An id no `new_session` or `load_session` opened is refused. This used to open
        # one on the spot, answering from whatever agent the server held -- a turn in a
        # conversation the client never started, and never saved (P6).
        if session_id not in self._sessions:
            return make_jsonrpc_error(req_id, SESSION_NOT_FOUND, UNKNOWN_SESSION_MESSAGE)
        if self._agent_factory is None:
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, NO_AGENT_MESSAGE)
        # A second turn beside a live one is refused, not started: it would replace the
        # first in `_in_flight_tasks`, and the first -- still answering -- could then be
        # released to make room (#1494).
        if self.is_turn_in_flight(session_id):
            return make_jsonrpc_error(req_id, INVALID_PARAMS, TURN_IN_FLIGHT_MESSAGE)
        try:
            agent = self._agent_for_turn(session_id)
        except _NoRoomForAgentError:
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, TOO_MANY_BUSY_MESSAGE)
        except _SessionUnopenableError:
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, CANNOT_OPEN_MESSAGE)

        # Parse prompt text from string or structured content items
        prompt_text = ""
        if isinstance(raw_prompt, str):
            prompt_text = raw_prompt
        elif isinstance(raw_prompt, list):
            parts: list[str] = []
            for item in cast(list[object], raw_prompt):
                if isinstance(item, dict):
                    item_dict = cast(dict[str, Any], item)
                    parts.append(str(item_dict.get("text", "")))
                else:
                    parts.append(str(item))
            prompt_text = "\n".join(parts)

        # Create turn execution task
        turn_task = asyncio.create_task(self._execute_turn(session_id, agent, prompt_text, req_id))
        self._in_flight_tasks[session_id] = turn_task

        # Return None here because response will be sent when turn completes or errors
        return None

    # ----------------------------------------------------------------------------------
    # Per-session agents (#1454)
    # ----------------------------------------------------------------------------------

    def _build_session_agent(self, session_id: str) -> BaseAgent:
        """Build `session_id`'s agent, not yet held; `_hold_session_agent` holds it.

        Room is checked first, so nothing is built for a server that could not hold it,
        but nothing is released here: the caller holds the agent only after its record is
        saved or restored, and the release happens then. A build, save or restore that
        fails has released nobody's agent for nothing (#1494).

        Raises:
            _NoRoomForAgentError: Every held agent is mid-turn or unsaved.
            _AgentBuildError: The factory raised, or built an agent for another id. The
                cause is logged here; callers answer the client in words, since the
                exception text can name paths and internals (the old catch-all sent it).
        """
        factory = self._agent_factory
        if factory is None:  # callers check first; kept so the type narrows here
            raise RuntimeError("ACPServer has no agent factory")
        self._check_room()
        try:
            agent = factory(session_id)
        except Exception as exc:
            logger.exception("Could not build the agent for ACP session %s", session_id)
            raise _AgentBuildError() from exc
        if agent.session_id != session_id:
            # A turn runs against the agent's active session, so an agent built for
            # another id would answer -- and save -- in the wrong conversation.
            logger.error(
                "The ACP agent factory built an agent for session %r when asked for %r",
                agent.session_id,
                session_id,
            )
            raise _AgentBuildError()
        return agent

    def _check_room(self) -> None:
        """Raise `_NoRoomForAgentError` if the server is full and no agent can be released."""
        if len(self._agents) >= self._max_live_agents:
            self._idle_agent_to_release()

    def _hold_session_agent(self, session_id: str, agent: BaseAgent) -> None:
        """Hold `agent` for `session_id`, releasing an idle agent first if the server is full.

        Called with no `await` since `_build_session_agent` checked for room, so the agent
        it found releasable still is.
        """
        if len(self._agents) >= self._max_live_agents:
            victim = self._idle_agent_to_release()
            del self._agents[victim]
            logger.debug("Released the idle agent of ACP session %s", victim)
        self._agents[session_id] = agent

    def _idle_agent_to_release(self) -> str:
        """The least recently used session whose agent can be rebuilt from the store.

        Raises:
            _NoRoomForAgentError: Every held agent is mid-turn or unsaved.
        """
        for held_id in self._agents:
            if held_id in self._unsaved or self.is_turn_in_flight(held_id):
                continue
            return held_id
        raise _NoRoomForAgentError()

    def _start_session_agent(
        self, session_id: str, req_id: int | str | None
    ) -> dict[str, Any] | None:
        """Build a new session's agent and save its first record; an error response or None.

        The record is written with the agent's own `persist_session`, the call every turn
        makes, so the first save and every later one share one revision chain: a record
        written some other way would carry a revision the agent does not hold, and its
        first turn's save would be refused as stale.
        """
        if session_id in self._sessions:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, SESSION_EXISTS_MESSAGE)
        try:
            existing = self._store.load(session_id)
        except Exception:
            logger.exception("Could not check for a saved ACP session %s", session_id)
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, CANNOT_START_MESSAGE)
        if existing is not None:
            return make_jsonrpc_error(req_id, INVALID_PARAMS, SESSION_EXISTS_MESSAGE)
        try:
            agent = self._build_session_agent(session_id)
        except _NoRoomForAgentError:
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, TOO_MANY_BUSY_MESSAGE)
        except _AgentBuildError:
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, CANNOT_START_MESSAGE)
        try:
            agent.persist_session(session_id)
        except Exception:
            logger.exception("Could not save the new ACP session %s", session_id)
            return make_jsonrpc_error(req_id, INTERNAL_ERROR, CANNOT_START_MESSAGE)
        self._hold_session_agent(session_id, agent)
        return None

    def _revive_session_agent(self, session_id: str) -> BaseAgent:
        """Build `session_id`'s agent and restore its saved history into it.

        Raises:
            _NoRoomForAgentError: Every held agent is mid-turn or unsaved.
            _SessionUnopenableError: The record is gone or could not be read back, or no
                agent could be built to hold it.
        """
        try:
            agent = self._build_session_agent(session_id)
        except _AgentBuildError as exc:
            raise _SessionUnopenableError() from exc
        try:
            restored = agent.hydrate_session(session_id)
        except Exception as exc:
            logger.exception("Could not restore the saved ACP session %s", session_id)
            raise _SessionUnopenableError() from exc
        if restored is None:
            # An agent left on a blank seeded session would answer as if the conversation
            # had just begun, and its save would then be refused -- or worse, accepted
            # over a record written later. Refused here instead.
            logger.error("The saved ACP session %s is no longer in the store", session_id)
            raise _SessionUnopenableError()
        self._hold_session_agent(session_id, agent)
        self._unsaved.discard(session_id)
        return agent

    def _agent_for_turn(self, session_id: str) -> BaseAgent:
        """The agent to answer `session_id` with, rebuilt from the store if released."""
        agent = self._agents.get(session_id)
        if agent is not None:
            self._agents.move_to_end(session_id)
            return agent
        return self._revive_session_agent(session_id)

    def _save_turn(self, session_id: str, agent: BaseAgent) -> bool:
        """Persist `session_id` after a turn; whether it was saved.

        Synchronous on purpose: the cancelled path calls it before its first `await`, which
        would otherwise raise the cancellation again and abandon the save.
        """
        try:
            agent.persist_session(session_id=session_id)
        except Exception:
            logger.exception("Could not save ACP session %s after its turn", session_id)
            self._unsaved.add(session_id)
            return False
        self._unsaved.discard(session_id)
        return True

    async def _execute_turn(
        self,
        session_id: str,
        agent: BaseAgent,
        prompt_text: str,
        req_id: int | str | None,
    ) -> None:
        """Execute turn asynchronously, stream updates, and send final prompt response."""
        sub: EventSubscription | None = None
        listener_task: asyncio.Task[None] | None = None

        try:
            # Subscribe to bus events for session-scoped updates
            if self._bus is not None:
                sub = self._bus.subscribe({f"session.{session_id}"})
                listener_task = asyncio.create_task(self._forward_session_events(session_id, sub))

            # Emit initial semantic update: turn started
            await self.send_response(
                make_jsonrpc_notification(
                    "session_update",
                    {
                        "sessionId": session_id,
                        "update": {"type": "state", "state": "working"},
                    },
                )
            )

            turn_result = await agent.execute_turn(prompt_text)
            turn_content = turn_result.content or ""
            saved = self._save_turn(session_id, agent)

            # Emit completion semantic update. A failed turn with nothing to show sends
            # no empty text; its reason goes in the error below.
            if turn_result.error is None or turn_content:
                await self.send_response(
                    make_jsonrpc_notification(
                        "session_update",
                        {
                            "sessionId": session_id,
                            "update": {"type": "text", "content": turn_content},
                        },
                    )
                )

            # A turn that ended in an error is not reported as completed, with empty
            # output (#1509): the client is told what happened, in plain words.
            if turn_result.error is not None:
                stop_reason = turn_result.stop_reason or ""
                if stop_reason in _PLAIN_ERROR_STOP_REASONS:
                    failure = turn_result.error
                else:
                    logger.error(
                        "Turn for ACP session %s failed (%s): %s",
                        session_id,
                        stop_reason,
                        turn_result.error,
                    )
                    failure = _STOP_REASON_MESSAGES.get(stop_reason, TURN_FAILED_MESSAGE)
                if req_id is not None:
                    await self.send_response(make_jsonrpc_error(req_id, INTERNAL_ERROR, failure))
            # The reply is shown either way; an unsaved one is not reported as completed,
            # because it would be missing when the conversation is reopened (P6).
            elif req_id is not None and not saved:
                await self.send_response(
                    make_jsonrpc_error(req_id, INTERNAL_ERROR, NOT_SAVED_MESSAGE)
                )
            elif req_id is not None:
                await self.send_response(
                    make_jsonrpc_response(
                        req_id,
                        {
                            "sessionId": session_id,
                            "status": "completed",
                            "output": turn_content,
                        },
                    )
                )
        except asyncio.CancelledError:
            # Turn was cancelled (§3.1). Saved first, before anything awaits: what the
            # turn already added to the session is kept, as the chat head keeps it (#1031).
            self._save_turn(session_id, agent)
            logger.info("Turn for session %s was cancelled", session_id)
            await self.send_response(
                make_jsonrpc_notification(
                    "session_update",
                    {
                        "sessionId": session_id,
                        "update": {"type": "state", "state": "cancelled"},
                    },
                )
            )
            if req_id is not None:
                await self.send_response(
                    make_jsonrpc_response(
                        req_id,
                        {
                            "sessionId": session_id,
                            "status": "cancelled",
                            "message": "Turn execution cancelled",
                        },
                    )
                )
            raise
        except Exception as exc:
            logger.exception("Error executing turn for session %s: %s", session_id, exc)
            self._save_turn(session_id, agent)
            if req_id is not None:
                await self.send_response(
                    make_jsonrpc_error(req_id, INTERNAL_ERROR, TURN_FAILED_MESSAGE)
                )
        finally:
            if listener_task is not None and not listener_task.done():
                listener_task.cancel()
            if sub is not None:
                try:
                    sub.close()
                except Exception:
                    pass
            # The only task recorded for this session is this one: a prompt is refused while
            # one is in flight, and `cancel` leaves the record to the task (#1494).
            self._in_flight_tasks.pop(session_id, None)

    async def _forward_session_events(self, session_id: str, sub: EventSubscription) -> None:
        """Forward bus events as ACP semantic updates and handle permission requests."""
        while True:
            try:
                evt = await sub.get()
                if evt.type == EventType.TOOL_CALL:
                    payload = evt.payload
                    await self.send_response(
                        make_jsonrpc_notification(
                            "session_update",
                            {
                                "sessionId": session_id,
                                "update": {
                                    "type": "tool_call",
                                    "name": payload.get("name") or payload.get("tool_name"),
                                    "arguments": payload.get("arguments"),
                                },
                            },
                        )
                    )
                elif evt.type == EventType.TOOL_RESULT:
                    payload = evt.payload
                    await self.send_response(
                        make_jsonrpc_notification(
                            "session_update",
                            {
                                "sessionId": session_id,
                                "update": {
                                    "type": "tool_result",
                                    "output": payload.get("result") or payload.get("output"),
                                },
                            },
                        )
                    )
                elif evt.type == EventType.TOOL_APPROVAL_REQUEST:
                    # First-class permission prompt mechanism (#474)
                    asyncio.create_task(self._prompt_client_permission(session_id, evt))
                elif evt.type == EventType.AGENT_REPLY:
                    payload = evt.payload
                    await self.send_response(
                        make_jsonrpc_notification(
                            "session_update",
                            {
                                "sessionId": session_id,
                                "update": {
                                    "type": "text_delta",
                                    "content": payload.get("content", ""),
                                },
                            },
                        )
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Error in _forward_session_events: %s", exc)

    async def prompt_client_permission(self, session_id: str, evt: AgentEvent) -> None:
        """Public entrypoint to issue permission prompt to client."""
        await self._prompt_client_permission(session_id, evt)

    async def _prompt_client_permission(self, session_id: str, evt: AgentEvent) -> None:
        """Issue request_permission JSON-RPC call to client and publish approval decision."""
        payload = evt.payload
        request_id = payload.get("request_id")
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        reason = payload.get("reason", "Tool execution requires confirmation")

        self._client_req_counter += 1
        client_req_id = f"perm_{self._client_req_counter}"

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_client_requests[client_req_id] = future

        perm_request = {
            "jsonrpc": "2.0",
            "id": client_req_id,
            "method": "request_permission",
            "params": {
                "sessionId": session_id,
                "tool": tool_name,
                "arguments": arguments,
                "reason": reason,
                "requestId": request_id,
            },
        }
        await self.send_response(perm_request)

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            result: object = response.get("result") or {}
            result_dict = cast(dict[str, Any], result) if isinstance(result, dict) else {}
            allowed = bool(result_dict.get("allowed") or result_dict.get("action") == "allow")
            action = "allow" if allowed else "block"
        except Exception:
            action = "block"
        finally:
            self._pending_client_requests.pop(client_req_id, None)

        if self._bus is not None:
            approval_resp = AgentEvent(
                type=EventType.TOOL_APPROVAL_RESPONSE,
                topic=f"session.{session_id}",
                sender_id="acp_server",
                payload={
                    "request_id": request_id,
                    "tool_call_id": payload.get("tool_call_id"),
                    "action": action,
                    "reason": f"Client permission response ({action})",
                },
            )
            await self._bus.publish(approval_resp)

    async def _handle_cancel(
        self, params: dict[str, Any], req_id: int | str | None
    ) -> dict[str, Any] | None:
        """Handle 'cancel': turn-scoped cancellation addressing by sessionId (§3.1)."""
        session_id = params.get("sessionId") or params.get("session_id")
        if not session_id:
            if req_id is not None:
                return make_jsonrpc_error(req_id, INVALID_PARAMS, "Missing sessionId")
            return None

        session_id = str(session_id)
        # Read, not removed: a cancelled turn is still running until its task ends (it
        # saves, reports, and may be inside a call that ignores the cancel for a while),
        # and a turn missing from this record could have its agent released or a second
        # turn started beside it. The task removes itself when it finishes (#1494).
        task = self.get_in_flight_task(session_id)
        if task is not None and not task.done():
            task.cancel()
            status = "canceled"
        else:
            status = "no_active_turn"

        if req_id is not None:
            return make_jsonrpc_response(req_id, {"sessionId": session_id, "status": status})
        return None
