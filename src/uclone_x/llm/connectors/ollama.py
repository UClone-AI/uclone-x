"""Ollama local model connector (Principle 5 & Principle 6)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    LLMProviderError,
    LLMTimeoutError,
    ModelLacksToolSupportError,
    UnmappableChatMessageError,
)
from uclone_x.llm.connectors.base import (
    BaseLLMConnector,
    parse_dict_payload,
    reported_count,
    resolve_token_counts,
)
from uclone_x.llm.context_window import (
    OLLAMA_CONTEXT_WINDOWS,
    OllamaContextWindows,
    default_ollama_num_ctx,
    ollama_model_key,
)
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)


def describe_transport_error(exc: BaseException) -> str:
    """Render a transport exception so the message is never empty (P6, #212).

    Several `httpx` transport exceptions carry no arguments — `httpx.ReadTimeout()` is
    the one that mattered — so an `f"...: {exc}"` message stringifies to nothing and the
    resulting `LLMProviderError` names no cause at all. P6 requires a failure to
    "surface immediately with its precise root cause"; a blank message does not. Fall
    back to the exception's type name, which is the root cause when there is no text.
    """
    text = str(exc).strip()
    return text if text else type(exc).__name__


def _chat_timeout_message(seconds: float, exc: BaseException) -> str:
    """Say that *the caller's ceiling* cut the generation off, and name the ceiling.

    `generate` and `stream` reported a read timeout with the same
    `Failed to connect to Ollama: ...` wording they use for a refused connection, so a
    request the daemon accepted and answered too slowly read as a daemon that was never
    there. The two facts have opposite remedies — raise the ceiling, or start Ollama —
    and #1277 is what the conflation costs: seven `frontier_live` probes ended
    `UNREACHABLE` on a 60 s read timeout, and the only evidence that they were cut off
    rather than unreachable was that none of their durations exceeded 60 s.

    `pull_model` already draws this line (#1233); this is the same line on the chat path.
    A `ConnectTimeout` stays on the unreachable side, because failing to *reach* the
    daemon is the same fact as a refused connection and not a truncated generation.
    """
    return (
        f"Ollama did not answer within {seconds:g}s ({describe_transport_error(exc)}). "
        "The request was accepted and cut off part-way, so this is a ceiling the caller "
        "set expiring, not an unreachable daemon: the remedy is a longer timeout, not a "
        "running Ollama."
    )


def _chat_transport_failure(
    seconds: float, exc: httpx.RequestError, *, unreachable_prefix: str
) -> LLMProviderError:
    """The error to raise for a transport failure on the chat path.

    One function rather than a pair of `except` clauses in each of `generate` and
    `stream`, because the distinction it draws is the whole point of #1277 and a copy of
    it is a copy that can be repaired in one place and left wrong in the other. The two
    callers differ only in how they name an unreachable daemon, which is wording they
    already had and which existing tests pin.

    The `ConnectTimeout` exclusion is not a special case; it is the definition. A connect
    timeout means the daemon was never reached, which is the same fact as a refused
    connection and the opposite of a generation cut off part-way -- and because
    `httpx.ConnectTimeout` is a subclass of `httpx.TimeoutException`, a check that did not
    exclude it would report every unreachable host as a ceiling that needed raising.
    """
    if isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout):
        return LLMTimeoutError(_chat_timeout_message(seconds, exc), seconds=seconds)
    return LLMProviderError(f"{unreachable_prefix}: {describe_transport_error(exc)}")


#: The phrase Ollama's 400 body uses when the model's template has no tool support, as in
#: `{"error":"registry.ollama.ai/library/deepseek-r1:14b does not support tools"}`.
_NO_TOOL_SUPPORT_PHRASE = "does not support tools"


def _chat_status_failure(
    model: str, status_code: int, body: str, *, prefix: str
) -> LLMProviderError:
    """The error to raise when `/api/chat` answers with a status other than 200.

    Shared by `generate` and `stream` for the reason `_chat_transport_failure` is. A model
    that cannot take tools gets an error of its own, worded for the person who chose the
    model, because a clone turn always sends tools and so the status and body would
    otherwise be all they see of a problem only a different model fixes. Any other status
    keeps the diagnostic wording, which existing tests pin.
    """
    if status_code == 400 and _NO_TOOL_SUPPORT_PHRASE in body:
        return ModelLacksToolSupportError(model)
    return LLMProviderError(f"{prefix} {status_code}: {body}")


def normalize_ollama_base_url(url: str) -> str:
    """Normalize an Ollama base URL: whitespace, trailing slashes, trailing '/v1', and a missing scheme.

    The scheme is the part that is easy to miss. `OLLAMA_HOST` is Ollama's own variable and
    its documented form is a bare `host:port` — `127.0.0.1:41234`, which is exactly what
    `scripts/verify_online_install.sh` exports to isolate the daemon it starts. That value
    reaches here through `resolve_ollama_base_url`, and handed to httpx unchanged it raises
    `UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol` at the
    first request — a crash during a turn rather than a diagnosis at resolution time.

    `http` and not `https`: the values this repairs are loopback and LAN addresses of a
    daemon that serves plain HTTP, and a wrong guess of `https` would fail the handshake
    rather than the URL parse. Anything already carrying a scheme is left alone, and an
    empty string stays empty — `http://` on its own names nothing.
    """
    cleaned = url.strip().rstrip("/")
    while cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    if cleaned and not cleaned.lower().startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned


OLLAMA_ENDPOINT_ENV_VARS: tuple[str, ...] = (
    "OLLAMA_BASE_URL",
    "OLLAMA_INDEPTH_BASE_URL",
    "OLLAMA_FAST_BASE_URL",
    "LOCAL_LLM_BASE_URL",
    "OLLAMA_HOST",
)
"""Environment variables that name an Ollama endpoint, in precedence order.

`OLLAMA_INDEPTH_BASE_URL` sits second because model resolution already reads
`OLLAMA_MODEL`, `OLLAMA_INDEPTH_MODEL`, `OLLAMA_FAST_MODEL` in that order, and a tier
should not mean one thing for a model and another for an endpoint. It was absent when this
tuple was first written (#539), even though `docs/local-development-guide.md` documents it
as part of the pre-configured 2-Tier setup and `cli/commands/llm.py` reads it — so the
refusal that tuple feeds fired on an environment this repository tells people to create.
The extraction preserved a pre-existing omission and then documented the result as
completeness, which is the drift the list exists to prevent.

Declared once so that a caller asking *whether* an endpoint is configured and a caller
asking *which* endpoint it is read the same list. `create_llm_connector` needs the first
question and `resolve_ollama_base_url` answers the second; duplicating the names in the
factory is how the two drift apart.
"""


def has_configured_ollama_endpoint(base_url: str | None = None) -> bool:
    """Report whether an Ollama endpoint was named by argument or environment.

    False means the environment says nothing about Ollama — it does **not** mean Ollama is
    unreachable, which is a question only a request can answer. The distinction matters to
    `create_llm_connector`: an unconfigured environment must not be answered with a default
    endpoint, while a configured one is an explicit choice to honour.
    """
    if base_url and base_url.strip():
        return True
    for name in OLLAMA_ENDPOINT_ENV_VARS:
        # Deliberately not `(os.getenv(name) or "").strip()`: substituting a literal for a
        # missing value is the shape `test_no_connector_substitutes_a_literal_for_a_missing_value`
        # sweeps this package for, and the sweep is right to refuse it even where the
        # substitution is immediately discarded.
        value = os.getenv(name)
        if value is not None and value.strip():
            return True
    return False


def resolve_ollama_base_url(base_url: str | None = None) -> str:
    """Resolve the Ollama base URL from arguments or environment fallback hierarchy.

    Fallback hierarchy:
    1. Explicit base_url argument
    2. OLLAMA_BASE_URL
    3. OLLAMA_INDEPTH_BASE_URL
    4. OLLAMA_FAST_BASE_URL
    5. LOCAL_LLM_BASE_URL
    6. OLLAMA_HOST
    7. Default: http://localhost:11434
    """
    candidates = (base_url, *(os.getenv(name) for name in OLLAMA_ENDPOINT_ENV_VARS))
    for candidate in candidates:
        if candidate and candidate.strip():
            return normalize_ollama_base_url(candidate)
    return "http://localhost:11434"


def resolve_ollama_model(model: str | None = None) -> str:
    """Resolve model name from arguments or environment fallback hierarchy.

    Fallback hierarchy:
    1. Explicit model argument (if not None, empty string, or 'default')
    2. OLLAMA_MODEL
    3. OLLAMA_INDEPTH_MODEL
    4. OLLAMA_FAST_MODEL
    5. Default: qwen3:8b
    """
    if model is not None:
        stripped = model.strip()
        if stripped and stripped != "default":
            return stripped

    candidates = (
        os.getenv("OLLAMA_MODEL"),
        os.getenv("OLLAMA_INDEPTH_MODEL"),
        os.getenv("OLLAMA_FAST_MODEL"),
    )
    for candidate in candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    return "qwen3:8b"


#: How long the daemon keeps a model and its KV cache loaded after a request, when neither
#: the caller nor `OLLAMA_KEEP_ALIVE` says (#1372). The daemon's own default is five
#: minutes, shorter than an ordinary pause between turns, and a model unloaded in the pause
#: re-reads the whole conversation cold on the next turn. Thirty minutes covers a pause
#: without holding memory for a conversation nobody came back to.
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"


def resolve_ollama_keep_alive(value: str | int | float | None = None) -> str | int | float:
    """The `keep_alive` sent with every chat request (#1372).

    Precedence: `value`, then `OLLAMA_KEEP_ALIVE`, then `DEFAULT_OLLAMA_KEEP_ALIVE`. The
    environment variable is the daemon's own, read here in the same format (`-1`, `3600`,
    `30m`), because a request's `keep_alive` overrides the daemon's setting: without this,
    an operator who set `OLLAMA_KEEP_ALIVE=-1` for the daemon would have it replaced by
    the default on every request. A bare number is sent as a number of seconds -- the
    daemon reads a JSON string as a duration, and `"-1"` has no unit.
    """
    raw: str | int | float | None = value
    if raw is None:
        env = os.getenv("OLLAMA_KEEP_ALIVE")
        raw = env.strip() if env and env.strip() else None
    if raw is None:
        return DEFAULT_OLLAMA_KEEP_ALIVE
    if isinstance(raw, str):
        text = raw.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text
    return raw


DEFAULT_OLLAMA_TIMEOUT_SECONDS: float = 180.0


def resolve_ollama_timeout(timeout: float | None = None) -> float:
    """Resolve timeout from arguments or environment variable hierarchy.

    Fallback hierarchy:
    1. Explicit timeout argument (if not None and not DEFAULT_OLLAMA_TIMEOUT_SECONDS)
    2. OLLAMA_TIMEOUT environment variable
    3. LLM_TIMEOUT environment variable
    4. Default: DEFAULT_OLLAMA_TIMEOUT_SECONDS (180.0s)
    """
    if timeout is not None and timeout != DEFAULT_OLLAMA_TIMEOUT_SECONDS:
        return timeout

    for env_name in ("OLLAMA_TIMEOUT", "LLM_TIMEOUT"):
        val = os.getenv(env_name)
        if val and val.strip():
            try:
                parsed = float(val.strip())
                if parsed > 0:
                    return parsed
            except ValueError:
                pass

    return DEFAULT_OLLAMA_TIMEOUT_SECONDS


class OllamaConnector(BaseLLMConnector):
    """Local LLM connector for Ollama endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        http_client: httpx.AsyncClient | None = None,
        keep_alive: str | int | float | None = None,
        context_windows: OllamaContextWindows | None = None,
    ) -> None:
        resolved_base = resolve_ollama_base_url(base_url)
        resolved_timeout = resolve_ollama_timeout(timeout)
        super().__init__(
            api_key=api_key,
            base_url=resolved_base,
            timeout=resolved_timeout,
            http_client=http_client,
        )
        # The resolved endpoint as a `str`: `resolve_ollama_base_url` always returns one,
        # and the window store is keyed by it.
        self._endpoint = resolved_base
        self.keep_alive = resolve_ollama_keep_alive(keep_alive)
        self._windows = context_windows if context_windows is not None else OLLAMA_CONTEXT_WINDOWS
        # The `num_ctx` last sent for each model. A request that names no window is sent
        # the same one: Ollama reloads a model whose options change, so a request without
        # it (a summary, a nudge) would reload the model at another window and the
        # next turn would reload it back, re-reading the conversation cold both times.
        self._num_ctx: dict[str, int] = {}
        # Models sent a `num_ctx` the daemon has not been asked about since.
        self._unconfirmed: set[str] = set()

    async def observe_context_window(self, model: str | None = None) -> int | None:
        """The window the daemon serves `model` at, reading `/api/ps` when it is not known.

        Read when no figure is held for the model, or when a `num_ctx` was sent since the
        last read: the daemon reloads the model at the sent window, or clamps it to the
        model's trained window, and only the daemon can say which. Otherwise the held
        figure is returned without a request. `None` when the daemon has not loaded the
        model -- never a guess (P6).
        """
        # Keyed as the daemon names the model, so `llama3.2` and `llama3.2:latest` are one.
        name = ollama_model_key(resolve_ollama_model(model))
        base = self._endpoint
        if self._windows.get(base, name) is None or name in self._unconfirmed:
            await self._windows.refresh(
                base, timeout=min(self.timeout, 2.0), http_client=self._http_client
            )
            served = self._windows.get(base, name)
            if served is not None:
                self._unconfirmed.discard(name)
                sent = self._num_ctx.get(name)
                if sent is not None and served < sent:
                    logger.warning(
                        "Ollama serves %s at %d tokens, not the %d requested; "
                        "compaction counts against %d",
                        name,
                        served,
                        sent,
                        served,
                    )
        return self._windows.get(base, name)

    @property
    def context_windows(self) -> OllamaContextWindows:
        """The store this connector records the daemon's windows in, for the agent to read."""
        return self._windows

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def _default_model(self) -> str:
        """The model a request naming none is sent to, resolved when read (#1447).

        `BaseAgent` reads this to name what a streamed step *asked for* when the request
        named no model. Without it the agent had nothing to read and recorded the literal
        `"default"` -- a model nobody asked Ollama for -- while `OLLAMA_MODEL` answered.
        A property, not a stored value, because the resolution reads the environment on
        every request and this must agree with what `_build_payload` sends.
        """
        return resolve_ollama_model(None)

    def _map_finish_reason(self, done_reason: str | None) -> FinishReason:
        """Map Ollama's `done_reason` onto `FinishReason`, or report it as unknown.

        Only the values enumerated here are claimed. Everything else — a `done_reason`
        Ollama names that this mapper does not, and the field being absent — becomes
        `FinishReason.UNKNOWN` rather than `STOP` (P6, #385).

        This replaces two separate inline expressions. The one in `generate` read
        `FinishReason.LENGTH if data.get("done_reason") == "length" else
        FinishReason.STOP`, so every unrecognised reason arrived as a clean completion;
        the one in `stream` did not read `done_reason` **at all** and reported `STOP`
        for every non-tool-call finish, which is the same defect with no input. Both now
        call this, so the two paths cannot disagree about what a reason means.

        `tool_calls` is not decided here: it is a property of the parsed response body
        rather than of `done_reason`, and it is applied by the caller.
        """
        if done_reason == "stop":
            return FinishReason.STOP
        if done_reason == "length":
            return FinishReason.LENGTH
        return FinishReason.UNKNOWN

    def _build_payload(self, request: LLMRequest, stream: bool = False) -> dict[str, Any]:
        """Translate a provider-neutral `LLMRequest` into an Ollama `/api/chat` body.

        Every field is emitted as the caller wrote it, or the message is refused with
        `UnmappableChatMessageError` — the shape `GeminiConnector._build_payload`
        established in #380/PR #384, for the reason given there: an outbound request has
        no result envelope, so P6's declared-recovery exemption (which requires in-band
        attribution via `provenance` on a result) is structurally unavailable, leaving
        emit-faithfully or fail as the only two options.

        The previous `"content": msg.content or ""` was unconditional and applied to
        **every** role, so on every path a message carrying no recorded content and one
        carrying the empty string reached the model as the same bytes. That a model runs
        locally changes who is billed, not whether the substitution is observable
        downstream — it is not (P6, #385).

        **The refusal that replaced it was also unconditional, and that was too wide.**
        `openai.py`, `anthropic.py` and `gemini.py` all scope theirs to the roles where
        the provider has no representation for an absent value — `tool`, `system`, and a
        `user`/assistant turn carrying nothing at all. This connector refused every role,
        including the one case where absent content is the *correct* encoding: an
        assistant turn whose entire content is a tool call. `base.py` stores exactly that
        as `content=resp_content or None` with `tool_calls` set, so one tool call poisoned
        the session — every later turn replayed that message and was refused here, before
        any request was made. The turn cost 15ms and returned
        `served_by=agent.core:error_handler`.

        Absent is still not `""`. Following `openai.py`, the key is **omitted** rather
        than coerced, so the distinction survives on the wire: a message with no recorded
        content sends no `content` field, and one holding `""` sends `"content": ""`.

        **Ollama `think` parameter (#695)**: sent only when the caller sets
        `LLMRequest.thinking`, as that value. A request that leaves it `None` -- every agent
        turn -- sends no `think` key, so Ollama and the model template decide whether a
        reasoning model thinks, and the resulting `thinking` channel is captured in
        `ModelResponse.thinking` and `StreamChunk.delta_thinking`. Callers that need a bare
        answer set it to `False` explicitly (the room's speaker selector does), which sends
        `"think": false`. The connector never chooses a value itself: disabling reasoning by
        default would degrade complex turns, and forcing it on can be rejected by
        non-reasoning models.

        Raises:
            UnmappableChatMessageError: a message has no faithful Ollama
                representation. The offending value is named in the message.
        """
        model = resolve_ollama_model(request.model)
        messages_payload: list[dict[str, Any]] = []

        for msg in request.messages:
            m_dict: dict[str, Any] = {"role": msg.role.value}
            if msg.content is not None:
                m_dict["content"] = msg.content
            elif not (msg.role is MessageRole.ASSISTANT and msg.tool_calls):
                raise UnmappableChatMessageError(
                    f"ChatMessage(role={msg.role.value!r}, name={msg.name!r}) has "
                    "content=None, so there is nothing to send as this message's content. "
                    "It is not coerced to '', because a message with no recorded content "
                    "and one holding the empty string are different inputs to the model "
                    "and would arrive identically (P6, #385). An assistant turn carrying "
                    "tool_calls is the one exception: there the absence is the encoding, "
                    "not a missing value, and the content key is omitted rather than "
                    "emitted as ''."
                )
            if msg.tool_calls:
                m_dict["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
                        }
                    }
                    for tc in msg.tool_calls
                ]
            messages_payload.append(m_dict)

        options: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        # The window, always sent, so the daemon serves the window the compaction trigger
        # counts against instead of choosing its own (#1372): the caller's when it names
        # one, else the one last sent for this model, else `default_ollama_num_ctx()` --
        # `OLLAMA_CONTEXT_LENGTH` when set, then `DEFAULT_OLLAMA_NUM_CTX`. The daemon's
        # own choice is 4096 under 24 GB of VRAM, which one persona's request alone
        # outgrew on a 16 GB GPU.
        window_key = ollama_model_key(model)
        num_ctx = request.context_window
        if num_ctx is None:
            num_ctx = self._num_ctx.get(window_key, default_ollama_num_ctx())
        if self._num_ctx.get(window_key) != num_ctx:
            self._unconfirmed.add(window_key)
        self._num_ctx[window_key] = num_ctx
        options["num_ctx"] = num_ctx

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_payload,
            "stream": stream,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if request.thinking is not None:
            payload["think"] = request.thinking

        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": cast(dict[str, Any], unwrap_immutable(t.parameters)),
                    },
                }
                for t in request.tools
            ]

        return payload

    async def generate(self, request: LLMRequest) -> ModelResponse:
        """Generate complete response via Ollama /api/chat."""
        payload = self._build_payload(request, stream=False)
        url = f"{self.base_url}/api/chat"
        client = self._get_client()

        should_close = self._http_client is None
        try:
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise _chat_status_failure(
                    str(payload["model"]),
                    resp.status_code,
                    resp.text,
                    prefix="Ollama provider returned status",
                )
            data: dict[str, Any] = resp.json()
        except httpx.RequestError as exc:
            raise _chat_transport_failure(
                self.timeout, exc, unreachable_prefix="Failed to connect to Ollama"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"Invalid JSON from Ollama: {describe_transport_error(exc)}"
            ) from exc
        except asyncio.CancelledError:
            logger.debug("Ollama generate call cancelled by caller")
            raise
        finally:
            if should_close:
                await client.aclose()

        msg_data: dict[str, Any] = data.get("message", {})
        content = msg_data.get("content")
        raw_thinking = msg_data.get("thinking")
        thinking = str(raw_thinking) if raw_thinking is not None else None

        tool_calls_raw: list[dict[str, Any]] = msg_data.get("tool_calls", [])
        tool_calls: list[ToolCallRequest] = []
        for i, tc in enumerate(tool_calls_raw):
            fn: dict[str, Any] = tc.get("function", {})
            name = str(fn.get("name", ""))
            args: object = fn.get("arguments", {})
            typed_args = parse_dict_payload(args)
            tool_calls.append(
                ToolCallRequest(
                    id=f"call_{i}",
                    name=name,
                    arguments=typed_args,
                )
            )

        # A count Ollama left out is estimated and labelled, never read as 0 (#939). The
        # output estimate includes the thinking text, which `eval_count` counts too.
        in_tokens = reported_count(data, "prompt_eval_count")
        out_tokens = reported_count(data, "eval_count")
        in_tokens, out_tokens, count_source = resolve_token_counts(
            request,
            in_tokens,
            out_tokens,
            reply="".join(part for part in (thinking, content) if part),
            tool_calls=tool_calls,
        )

        usage = TokenUsage(
            provider="ollama",
            model=resolve_ollama_model(request.model),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            count_source=count_source,
        )

        raw_done_reason = data.get("done_reason")
        finish_reason = (
            FinishReason.TOOL_CALLS
            if tool_calls
            else self._map_finish_reason(
                str(raw_done_reason) if raw_done_reason is not None else None
            )
        )

        requested_model = resolve_ollama_model(request.model)
        model_name = str(data.get("model", requested_model))
        # P6: `requested` is the model actually sent to the server — the resolved
        # name, since nobody asks Ollama for a model called "default" — and
        # `served_by` is what the response reports (#149).
        provenance = Provenance.primary(
            provider="ollama", model=requested_model, served_model=model_name
        )

        return ModelResponse(
            content=content,
            thinking=thinking,
            tool_calls=tuple(tool_calls),
            usage=usage,
            finish_reason=finish_reason,
            model_name=model_name,
            provenance=provenance,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream chunks from Ollama /api/chat."""
        payload = self._build_payload(request, stream=True)
        url = f"{self.base_url}/api/chat"
        client = self._get_client()
        should_close = self._http_client is None
        # What the stream said, for estimating a count the `done` line leaves out (#939).
        streamed: list[str] = []
        streamed_calls: list[ToolCallRequest] = []

        thinking_tokens_count = 0
        content_tokens_count = 0
        model_name = str(payload["model"])
        logger.debug("Ollama stream started: url=%s, model=%s", url, model_name)

        try:
            async with client.stream("POST", url, json=payload, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    raise _chat_status_failure(
                        model_name,
                        resp.status_code,
                        err_body.decode("utf-8", errors="replace"),
                        prefix="Ollama streaming returned status",
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data: dict[str, Any] = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # What the server says it ran, on every line of a real stream. Read,
                    # never defaulted: a line naming no model carries `None` (#1447).
                    raw_served: object = data.get("model")
                    served_model = (
                        str(raw_served)
                        if raw_served is not None and str(raw_served).strip()
                        else None
                    )

                    msg_data: dict[str, Any] = data.get("message", {})
                    delta_content = msg_data.get("content")
                    raw_delta_thinking = msg_data.get("thinking")
                    delta_thinking = (
                        str(raw_delta_thinking) if raw_delta_thinking is not None else None
                    )

                    if delta_thinking:
                        if thinking_tokens_count == 0:
                            logger.info(
                                "🧠 [Ollama] Model %s started reasoning/thinking", model_name
                            )
                        thinking_tokens_count += 1
                        if thinking_tokens_count % 50 == 0:
                            logger.debug(
                                "🧠 [Ollama] Still thinking... (%d tokens so far)",
                                thinking_tokens_count,
                            )

                    if delta_content:
                        if thinking_tokens_count > 0 and content_tokens_count == 0:
                            logger.info(
                                "💬 [Ollama] Model %s finished thinking (%d tokens), generating response",
                                model_name,
                                thinking_tokens_count,
                            )
                        content_tokens_count += 1

                    tool_calls_raw: list[dict[str, Any]] = msg_data.get("tool_calls", [])
                    tool_calls: list[ToolCallRequest] = []
                    for i, tc in enumerate(tool_calls_raw):
                        fn: dict[str, Any] = tc.get("function", {})
                        name = str(fn.get("name", ""))
                        args: object = fn.get("arguments", {})
                        typed_args = parse_dict_payload(args)
                        tool_calls.append(
                            ToolCallRequest(
                                id=f"call_{i}",
                                name=name,
                                arguments=typed_args,
                            )
                        )

                    if delta_thinking:
                        streamed.append(delta_thinking)
                    if delta_content:
                        streamed.append(delta_content)
                    streamed_calls.extend(tool_calls)

                    is_done = bool(data.get("done", False))
                    usage: TokenUsage | None = None
                    finish_reason: FinishReason | None = None

                    if is_done:
                        in_tok = reported_count(data, "prompt_eval_count")
                        out_tok = reported_count(data, "eval_count")
                        in_tok, out_tok, count_source = resolve_token_counts(
                            request,
                            in_tok,
                            out_tok,
                            reply="".join(streamed),
                            tool_calls=streamed_calls,
                        )
                        usage = TokenUsage(
                            provider="ollama",
                            model=resolve_ollama_model(request.model),
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=in_tok + out_tok,
                            count_source=count_source,
                        )
                        stream_done_reason = data.get("done_reason")
                        finish_reason = (
                            FinishReason.TOOL_CALLS
                            if tool_calls
                            else self._map_finish_reason(
                                str(stream_done_reason) if stream_done_reason is not None else None
                            )
                        )
                        logger.info(
                            "✔ [Ollama] Stream finished: model=%s, thinking_tokens=%d, content_tokens=%d, finish_reason=%s",
                            model_name,
                            thinking_tokens_count,
                            content_tokens_count,
                            finish_reason,
                        )

                    if delta_content or delta_thinking or tool_calls or usage or finish_reason:
                        yield StreamChunk(
                            delta_content=delta_content,
                            delta_thinking=delta_thinking,
                            tool_calls=tuple(tool_calls),
                            usage=usage,
                            finish_reason=finish_reason,
                            model=served_model,
                        )
        except httpx.RequestError as exc:
            raise _chat_transport_failure(
                self.timeout, exc, unreachable_prefix="Ollama stream connection error"
            ) from exc
        except asyncio.CancelledError:
            logger.debug("Ollama stream cancelled by caller")
            raise
        finally:
            if should_close:
                await client.aclose()


#: How long `pull_model`/`delete_model` will wait to *reach* the daemon (#1233).
#:
#: `httpx` applies a bare float to connect, read, write and pool alike, so the
#: pull's own ceiling below was also the connect ceiling. On loopback that
#: is invisible — a daemon that is not listening refuses at once — but
#: `resolve_ollama_base_url` also resolves the README's 2-tier topology, where
#: `OLLAMA_HOST` names another machine. A host that is asleep or firewalled
#: blackholes the SYN rather than refusing it, and the caller then waited the
#: whole pull ceiling to learn something that takes seconds to establish.
#: Connecting is not downloading and does not get the downloading budget.
MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS = 5.0


def _model_management_timeout(timeout: float) -> httpx.Timeout:
    """Spend `timeout` on the transfer and `MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS` on reaching it."""
    return httpx.Timeout(timeout, connect=MODEL_MANAGEMENT_CONNECT_TIMEOUT_SECONDS)


#: How long `pull_model` will wait for the *next* line of Ollama's pull stream (#1243).
#:
#: This is the quantity the pull actually turns on, and it replaces a ceiling on
#: total elapsed time. Elapsed time cannot tell a slow pull from a dead one — a 70B
#: model on a hotel link legitimately runs for hours, and a daemon that wedged in the
#: first second looks identical to a client that is not reading the stream. Silence
#: between NDJSON lines tells them apart, because a pull making progress is never
#: silent.
#:
#: **Measured, on Ollama 0.31.2 against a local daemon on 2026-09-20.** The progress
#: stream is a *time-driven ticker*, not a byte-driven one: across three complete
#: pulls it emitted a line every ~60 ms (p50 0.060 s, p90 0.061 s over 100+ lines),
#: and it kept ticking at that rate through the six lines before any byte had been
#: transferred. The first line (`pulling manifest`) arrived 13 ms after the request.
#: The only gap worth the name was manifest resolution against the registry on a cold
#: cache — **0.744 s**, one round trip — and it was also the largest gap seen in any
#: run.
#:
#: **Why 120 s and not something near the measurement.** 120 s is ~160x the largest
#: gap observed and ~2000x the tick, and the headroom is deliberately not sized from
#: the happy path. Two quiet phases were *not* measured and are the reason for it: a
#: cold manifest resolution over a slow or distant link is an unbounded multiple of
#: the 0.744 s seen on a fast one, and `verifying sha256 digest` over a 40 GB model
#: completed instantly on the 46 MB model measured here, so whether it ticks at all
#: on a large one is unknown. Being wrong low kills a working pull, which is the bug
#: this replaces; being wrong high costs a wedged pull an extra minute or two of a
#: held single-flight entry. The asymmetry decides the number.
PULL_SILENCE_TIMEOUT_SECONDS: float = 120.0

#: A backstop on a daemon that talks forever and never finishes (#1243).
#:
#: A silence ceiling alone would be unbounded against one failure the measurement
#: above makes concrete rather than hypothetical: because the ticker is driven by
#: time and not by bytes, a transfer whose bytes stop while the daemon stays healthy
#: keeps emitting lines at 60 ms with `completed` frozen. That stream never goes
#: silent, so nothing below would ever end it, and it would hold this model's
#: single-flight entry and an `httpx.AsyncClient` for as long as the daemon lives.
#: The old wall-clock ceiling did bound that case, and dropping it entirely would be
#: a regression rather than a simplification.
#:
#: It is a backstop and is sized to never be the thing that decides a real pull: two
#: hours is ~40 GB at ~45 Mbit/s, several times over any model the README recommends.
#: A pull it does cut short is one that ran for two hours *while reporting progress*,
#: and the refusal names `ucx llm pull`, which has no ceiling and a real progress bar.
PULL_TOTAL_BACKSTOP_SECONDS: float = 7200.0


def _raise_for_pull_line(model: str, line: str) -> str | None:
    """Return the `status` this NDJSON line reports, raising if it reports a failure.

    `None` means the line said nothing readable — unparseable, not an object, or
    carrying no `status`. That is not evidence either way, and turning "I could not
    read this" into "the pull failed" would invent a failure.
    """
    try:
        payload: Any = json.loads(line)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    body = cast(dict[str, Any], payload)

    error = body.get("error")
    if error:
        raise LLMProviderError(f"Ollama could not pull {model!r}: {error}")

    status = body.get("status")
    return None if status is None else str(status)


def _raise_for_pull_outcome(model: str, last_status: str | None) -> None:
    """Raise unless the last thing Ollama said about this pull was `success`.

    A stream that ends on `pulling 797b70c4edf8` ended early — the connection went
    away mid-transfer — and a stream that never said anything readable at all is not
    evidence of a failure, so only the first is reported as one.
    """
    if last_status is None:
        return
    if last_status != "success":
        raise LLMProviderError(
            f"Ollama did not finish pulling {model!r}: last status was {last_status!r}"
        )


def _stalled_message(model: str, silence_timeout: float, last_status: str | None) -> str:
    """Say that the *stream went quiet*, and what it last said, not that time ran out.

    An operator reading `did not finish within 900s` learns only that a number they
    did not choose has elapsed. The fact worth reporting is that Ollama stopped
    talking, and the last status is where it stopped — `pulling manifest` is a
    registry problem and `pulling <digest>` is a transfer one.
    """
    where = f"last status was {last_status!r}" if last_status else "it never sent a first line"
    return (
        f"Ollama stopped sending pull progress for {model!r}: "
        f"nothing for {silence_timeout:g}s, and {where}. "
        f"This is a stalled pull, not a slow one — a pull that is merely slow keeps reporting."
    )


async def pull_model(
    model: str,
    base_url: str | None = None,
    *,
    silence_timeout: float = PULL_SILENCE_TIMEOUT_SECONDS,
    total_timeout: float = PULL_TOTAL_BACKSTOP_SECONDS,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Install `model` onto the Ollama daemon, consuming `POST /api/pull`'s NDJSON (#1243).

    **The ceiling is on silence, not on duration.** This used to ask for
    `{"stream": false}` and hold a single ceiling over the whole pull, which measured
    elapsed time — a quantity that answers "has this taken long?" when the question is
    "is this still working?". The two are not the same question and no number makes
    them the same: a 70B model on a bad link takes longer than any ceiling anyone is
    willing to set, and a daemon that wedged in the first second sits under every one
    of them until it expires. With `{"stream": true}` the daemon reports a line per
    progress update, so the gap between lines is measurable and it is what
    distinguishes the two. A pull making progress, however slowly, is never killed
    here; a pull that has gone quiet dies in `silence_timeout`.

    `total_timeout` is a backstop and not a second answer to the same question — see
    `PULL_TOTAL_BACKSTOP_SECONDS` for the one failure it covers that silence cannot.

    **Both are per *download*, not per waiter.** `POST /api/models/pull` runs this
    inside a `SingleFlight` entry that every concurrent caller awaits through
    `asyncio.shield`, so a browser that aborts does not cancel the download for a tab
    still waiting. These deadlines belong to the shielded run: they are measured from
    when *the download* started, not from when any particular caller began waiting,
    and a caller that goes away neither extends nor shortens them.

    **The status line is not the whole answer.** Ollama reports the outcome in the
    body, and a pull that fails part-way still comes back 200 — carrying
    `{"error": "..."}` on some line, or ending on a `status` that is not `"success"`.
    Returning on the status code alone made the UI say `Installed model "X".` for a
    model that was never fetched (P6: a failure must surface as a failure).
    """
    resolved_base = resolve_ollama_base_url(base_url)
    url = f"{resolved_base}/api/pull"
    client = http_client if http_client is not None else httpx.AsyncClient()
    should_close = http_client is None
    loop = asyncio.get_running_loop()
    backstop_at = loop.time() + total_timeout
    last_status: str | None = None
    try:
        async with client.stream(
            "POST",
            url,
            json={"model": model, "stream": True},
            timeout=_model_management_timeout(silence_timeout),
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                raise LLMProviderError(
                    f"Ollama provider returned status {resp.status_code}: {resp.text}"
                )
            lines = resp.aiter_lines()
            while True:
                remaining = backstop_at - loop.time()
                if remaining <= 0:
                    raise LLMTimeoutError(
                        f"Ollama is still streaming progress for {model!r} after "
                        f"{total_timeout:g}s without finishing; giving up on waiting for it",
                        seconds=total_timeout,
                    )
                try:
                    line = await asyncio.wait_for(anext(lines), min(silence_timeout, remaining))
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    # The wait that expired is the shorter of the two, so which fact
                    # this is depends on which one it was — and a backstop reported as
                    # a stall would name a silence that never happened.
                    if loop.time() >= backstop_at:
                        raise LLMTimeoutError(
                            f"Ollama is still streaming progress for {model!r} after "
                            f"{total_timeout:g}s without finishing; giving up on waiting for it",
                            seconds=total_timeout,
                        ) from exc
                    raise LLMTimeoutError(
                        _stalled_message(model, silence_timeout, last_status),
                        seconds=silence_timeout,
                    ) from exc
                status = _raise_for_pull_line(model, line)
                if status is not None:
                    last_status = status
    except httpx.ConnectTimeout as exc:
        # Reaching the daemon is not downloading from it. This is the connect
        # ceiling expiring, which is the same fact as a refused connection and
        # not a stalled stream; it must not be reported as the latter.
        raise LLMProviderError(
            f"Failed to connect to Ollama: {describe_transport_error(exc)}"
        ) from exc
    except httpx.TimeoutException as exc:
        # `httpx`'s read timeout measures the same quantity one level down — time
        # between *bytes* rather than between lines — so whichever of the two fires
        # first, the fact being reported is that the stream went quiet.
        raise LLMTimeoutError(
            _stalled_message(model, silence_timeout, last_status),
            seconds=silence_timeout,
        ) from exc
    except httpx.RequestError as exc:
        raise LLMProviderError(
            f"Failed to connect to Ollama: {describe_transport_error(exc)}"
        ) from exc
    finally:
        if should_close:
            await client.aclose()
    _raise_for_pull_outcome(model, last_status)


async def delete_model(
    model: str,
    base_url: str | None = None,
    timeout: float = 30.0,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Remove `model` from the Ollama daemon via `DELETE /api/delete`."""
    resolved_base = resolve_ollama_base_url(base_url)
    url = f"{resolved_base}/api/delete"
    client = http_client if http_client is not None else httpx.AsyncClient()
    should_close = http_client is None
    try:
        resp = await client.request(
            "DELETE", url, json={"model": model}, timeout=_model_management_timeout(timeout)
        )
        if resp.status_code != 200:
            raise LLMProviderError(
                f"Ollama provider returned status {resp.status_code}: {resp.text}"
            )
    except httpx.RequestError as exc:
        raise LLMProviderError(
            f"Failed to connect to Ollama: {describe_transport_error(exc)}"
        ) from exc
    finally:
        if should_close:
            await client.aclose()
