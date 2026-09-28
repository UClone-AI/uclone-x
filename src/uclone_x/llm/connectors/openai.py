"""OpenAI provider connector (Principle 5 & Principle 6)."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, ClassVar, cast

import httpx

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    UnmappableChatMessageError,
)
from uclone_x.llm.catalog import CatalogEntry
from uclone_x.llm.connectors.base import (
    BaseLLMConnector,
    is_local_endpoint,
    parse_dict_payload,
    refuse_response_schema,
    reported_count,
    resolve_model,
    resolve_token_counts,
)
from uclone_x.llm.connectors.base import named_model as _named
from uclone_x.llm.connectors.failures import failed_request, failed_status, unusable_response
from uclone_x.llm.connectors.listing import from_unix, get_listing_page, listed_items, optional_str
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)

#: Id fragments of the models an OpenAI-compatible listing mixes in with the chat models:
#: embeddings, speech, transcription, images, moderation and the legacy completion bases.
#: Their ids are the only thing OpenAI's listing says about them.
_NOT_CHAT_FRAGMENTS = (
    "embedding",
    "tts",
    "whisper",
    "transcribe",
    "dall-e",
    "image",
    "moderation",
    "davinci",
    "babbage",
    "sora",
)


def cached_prompt_tokens(usage: dict[str, Any] | None) -> int | None:
    """`usage.prompt_tokens_details.cached_tokens`, or `None` when not reported (#1371).

    OpenAI counts them inside `prompt_tokens`, as `TokenUsage.cache_read_input_tokens`
    expects. An OpenAI-compatible server that leaves the field out (a vLLM server may,
    depending on how it was started) records no count rather than a zero.
    """
    details: object = usage.get("prompt_tokens_details") if usage else None
    if not isinstance(details, dict):
        return None
    return reported_count(cast("dict[str, Any]", details), "cached_tokens")


def named_model(request: LLMRequest) -> str | None:
    """The model the caller named, or `None` when they left the choice to the connector.

    Public so that `VLLMConnector._resolve_request_model` does not reach into another
    module for a private helper (#1304).

    `"default"` is treated as naming nothing, because the UI sends that string for "whatever
    this provider uses" and a provider asked for a model literally called `default` answers
    404.
    """
    return _named(request.model)


class OpenAIConnector(BaseLLMConnector):
    """OpenAI API connector, and the base for every OpenAI-compatible endpoint.

    **What a subclass varies, and why these are declared rather than restated (#1304).**
    A local vLLM or LM Studio server speaks `/v1/chat/completions`, so subclassing is how
    that wire format is shared. Four things then differ: the provider's identity, its
    display name in a failure message, the model it serves by default, and whether it
    authenticates at all. Everything else — the payload builder, the finish-reason mapper,
    the usage completion — is identical, and duplicating it into a sibling module would
    leave two copies to drift.

    `provider_name` was already a property, but `generate` and `stream` did not read it:
    `TokenUsage.provider` and `Provenance.primary` each carried `"openai"` written out. A
    subclass overriding the property therefore changed nothing a caller could see, and a
    self-hosted endpoint was attributed to OpenAI — a substituted figure of the kind P6
    forbids, and #958's third item. The literals are
    gone; each site reads `self.provider_name`.
    """

    _display_name: ClassVar[str] = "OpenAI"
    """How this endpoint is named in a failure message the operator reads.

    A connection error against `http://localhost:8000/v1` that reads `OpenAI connection
    error` sends the reader to check an API key and a status page for a service that was
    never involved.
    """

    _api_key_env_var: ClassVar[str] = "OPENAI_API_KEY"
    _base_url_env_var: ClassVar[str] = "OPENAI_BASE_URL"
    _default_base_url: ClassVar[str] = "https://api.openai.com/v1"
    _requires_api_key: ClassVar[bool] = True
    """Whether a missing credential is refused at construction.

    `True` here and for every hosted provider. An endpoint a person runs themselves may
    have no credential at all, and for it a demanded key is a value the operator has to
    invent — which is why this is a declared property of the endpoint rather than a `try`
    around the check.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ) -> None:
        """``model`` is what a request naming no model is sent to; see ``resolve_model``."""
        #: The model a request naming none asks for, or ``None``: then such a request is
        #: refused before the network rather than sent to a model id written here.
        self._default_model: str | None = _named(model)
        key_env_var = self._api_key_env_var
        resolved_key = api_key if api_key is not None else os.getenv(key_env_var)
        if self._requires_api_key and (resolved_key is None or not resolved_key.strip()):
            raise LLMCredentialsNotConfiguredError(
                f"{type(self).__name__} requires an API key: pass api_key= or set "
                f"{key_env_var} (got api_key={api_key!r}, "
                f"{key_env_var}={os.getenv(key_env_var)!r}). It is not defaulted "
                "to '', because an `Authorization: Bearer ` header with nothing after it "
                "turns a configuration defect into a provider-side 401 on the first "
                "billed call — a retryable-looking transport fault whose real cause is "
                "here (P6, #385)."
            )
        resolved_base = base_url or os.getenv(self._base_url_env_var) or self._default_base_url
        super().__init__(
            api_key=resolved_key,
            base_url=resolved_base.rstrip("/"),
            timeout=timeout,
            http_client=http_client,
        )

    @property
    def paid(self) -> bool:
        """Paid unless the endpoint is on this machine or a private network.

        The same connector reaches OpenAI and any OpenAI-compatible server (vLLM, a local
        gateway), so the endpoint decides, not the class.
        """
        return not is_local_endpoint(self.base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def list_models(self) -> list[CatalogEntry]:
        """The models this key can use, from `GET /models` (#1631).

        OpenAI's listing reports an id, an owner and a publish time, and no context window;
        whether a model can chat is read from its id (`_NOT_CHAT_FRAGMENTS`).
        """
        page = await get_listing_page(
            self,
            provider=self._display_name,
            url=f"{self.base_url}/models",
            headers=self._auth_headers(),
        )
        entries: list[CatalogEntry] = []
        for item in listed_items(page, "data"):
            model_id = optional_str(item.get("id"))
            if model_id is None:
                continue
            lowered = model_id.lower()
            entries.append(
                CatalogEntry(
                    id=model_id,
                    chat_capable=not any(fragment in lowered for fragment in _NOT_CHAT_FRAGMENTS),
                    created_at=from_unix(item.get("created")),
                )
            )
        return entries

    def _auth_headers(self) -> dict[str, str]:
        """The credential headers for this endpoint, or none when it has no credential.

        `_require_api_key()` is still what produces the header when there is a key, so the
        empty-bearer defect #385 closed stays closed: nothing here substitutes `""` for a
        missing credential. What a subclass may say is that the endpoint needs no
        credential — and then no `Authorization` header is sent at all, which is different
        from sending an empty one.
        """
        if not self._requires_api_key and (self.api_key is None or not self.api_key.strip()):
            return {}
        return {"Authorization": f"Bearer {self._require_api_key()}"}

    def _resolve_request_model(self, request: LLMRequest) -> str:
        """The model this connector asks its endpoint for: the request's, else its own.

        With neither, the request is refused before the network (``resolve_model``). A
        method, so `VLLMConnector` can also consult `VLLM_MODEL` before refusing (#1304).
        """
        return resolve_model(request.model, self._default_model, self._display_name)

    def _map_finish_reason(self, reason: str | None) -> FinishReason:
        """Map OpenAI's `finish_reason` onto `FinishReason`, or report it as unknown.

        Only the values enumerated here are claimed. Everything else — a
        `finish_reason` OpenAI names that this mapper does not, and the field being
        absent — becomes `FinishReason.UNKNOWN` rather than `STOP`. The
        `return FinishReason.STOP` fall-through this replaces reported every
        unrecognised value as a clean, complete generation, which is a substitution in
        the direction of success (P6, #385).
        """
        if reason == "stop":
            return FinishReason.STOP
        if reason == "length":
            return FinishReason.LENGTH
        if reason == "tool_calls":
            return FinishReason.TOOL_CALLS
        if reason == "content_filter":
            return FinishReason.CONTENT_FILTER
        return FinishReason.UNKNOWN

    def _build_payload(self, request: LLMRequest, stream: bool = False) -> dict[str, Any]:
        """Translate a provider-neutral `LLMRequest` into an OpenAI `chat/completions` body.

        Every field is emitted as the caller wrote it, or the message is refused with
        `UnmappableChatMessageError` — the shape `GeminiConnector._build_payload`
        established in #380/PR #384, for the reason given there: an outbound request has
        no result envelope, so P6's declared-recovery exemption (which requires in-band
        attribution via `provenance` on a result) is structurally unavailable, leaving
        emit-faithfully or fail as the only two options.

        **The contradiction this function used to contain, and which of the two forms was
        intended (#385).** `if msg.content is not None` and
        `m_dict["content"] = msg.content or ""` sat eight lines apart, so a reader who
        checked the guard concluded the connector was safe while the `TOOL` branch
        overwrote its result. `git log -L` shows both arrived in the *same* commit
        (`62412fd`), so this was never drift between two authors or two eras — one
        constraint was encoded two ways.

        The `is not None` guard is the intended general rule and is kept. The `TOOL`
        override existed because that branch carries a real *additional* constraint the
        general rule does not express: OpenAI requires `content` on a `tool` message, so
        omitting the key is not an option there the way it is elsewhere. The constraint
        is correct; expressing it as `or ""` was not, because it satisfies "the key is
        present" by inventing the value, reporting "no return value was recorded" as
        "the tool returned the empty string". A required field with no faithful value is
        exactly the case `UnmappableChatMessageError` exists for, so the branch now
        refuses `None` and emits `""` as `""`.

        `tool_call_id` is refused when absent for the same reason it is refused in
        `anthropic.py`: it is the key OpenAI correlates the result to the call by, and
        silently dropping it sent a `tool` message with no association at all.

        Raises:
            UnmappableChatMessageError: a message has no faithful OpenAI
                representation. The offending value is named in the message.
        """
        refuse_response_schema(request, self.provider_name)
        model = self._resolve_request_model(request)
        messages_payload: list[dict[str, Any]] = []

        for msg in request.messages:
            m_dict: dict[str, Any] = {"role": msg.role.value}
            if msg.content is not None:
                m_dict["content"] = msg.content
            if msg.name is not None:
                m_dict["name"] = msg.name
            if msg.role == MessageRole.TOOL:
                if msg.tool_call_id is None or not msg.tool_call_id.strip():
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', name={msg.name!r}) has "
                        f"tool_call_id={msg.tool_call_id!r}, and OpenAI's "
                        "tool_call_id is the key it correlates the result to the tool call "
                        "that requested it. It is not silently omitted, because a tool "
                        "message with no association is not a result with a missing label — "
                        "it is a result attached to nothing (P6, #385)."
                    )
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', tool_call_id={msg.tool_call_id!r}) has "
                        "content=None. OpenAI requires content on a tool message, so it "
                        "cannot be omitted the way it can on other roles — but it is not "
                        "coerced to '' either, because that reports 'no return value was "
                        "recorded' as 'the tool returned the empty string'. This is the "
                        "site where `if msg.content is not None` above was un-done eight "
                        "lines later by `msg.content or ''` (P6, #385). Set content "
                        "explicitly at the call site."
                    )
                m_dict["tool_call_id"] = msg.tool_call_id
                m_dict["content"] = msg.content
            if msg.tool_calls:
                m_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(unwrap_immutable(tc.arguments)),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages_payload.append(m_dict)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_payload,
            "temperature": request.temperature,
            "stream": stream,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

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

        if stream:
            payload["stream_options"] = {"include_usage": True}

        return payload

    async def generate(self, request: LLMRequest) -> ModelResponse:
        """Generate response from OpenAI endpoint."""
        model = self._resolve_request_model(request)
        payload = self._build_payload(request, stream=False)
        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None

        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise failed_status(
                    provider=self._display_name,
                    model=model,
                    status_code=resp.status_code,
                    body=resp.text,
                )
            data: dict[str, Any] = resp.json()
        except httpx.RequestError as exc:
            raise failed_request(provider=self._display_name, model=model, exc=exc) from exc
        except json.JSONDecodeError as exc:
            raise unusable_response(
                provider=self._display_name, model=model, detail=str(exc)
            ) from exc
        finally:
            if should_close:
                await client.aclose()

        choices: list[dict[str, Any]] = data.get("choices", [])
        if not choices:
            raise unusable_response(provider=self._display_name, model=model, detail="no choices")

        choice = choices[0]
        msg: dict[str, Any] = choice.get("message", {})
        content = msg.get("content")

        tool_calls: list[ToolCallRequest] = []
        raw_tool_calls: list[dict[str, Any]] = msg.get("tool_calls", [])
        for tc in raw_tool_calls:
            fn: dict[str, Any] = tc.get("function", {})
            name = str(fn.get("name", ""))
            raw_args: object = fn.get("arguments", "{}")
            typed_args = parse_dict_payload(raw_args)
            tool_calls.append(
                ToolCallRequest(
                    id=str(tc.get("id", "")),
                    name=name,
                    arguments=typed_args,
                )
            )

        # A count OpenAI (or an OpenAI-compatible server) left out is estimated and
        # labelled, never read as 0 (#939).
        usage_data: dict[str, Any] | None = data.get("usage")
        in_tokens = reported_count(usage_data, "prompt_tokens")
        out_tokens = reported_count(usage_data, "completion_tokens")
        in_tokens, out_tokens, count_source = resolve_token_counts(
            request, in_tokens, out_tokens, reply=content, tool_calls=tool_calls
        )
        model_name = str(data.get("model", model))

        usage = TokenUsage(
            provider=self.provider_name,
            model=model_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            count_source=count_source,
            cache_read_input_tokens=cached_prompt_tokens(usage_data),
        )

        finish_reason = self._map_finish_reason(choice.get("finish_reason"))
        # P6: `requested` is what was asked of OpenAI, `served_by` what the response
        # says ran. Collapsing both onto the served name erased a provider-side alias
        # and reported `degraded=False` for a model the caller never named (#149).
        provenance = Provenance.primary(
            provider=self.provider_name, model=model, served_model=model_name
        )

        return ModelResponse(
            content=content,
            tool_calls=tuple(tool_calls),
            usage=usage,
            finish_reason=finish_reason,
            model_name=model_name,
            provenance=provenance,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks from OpenAI SSE endpoint."""
        model = self._resolve_request_model(request)
        payload = self._build_payload(request, stream=True)
        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None
        # What the stream said, for estimating an output count its usage leaves out.
        streamed: list[str] = []

        try:
            async with client.stream(
                "POST", url, json=payload, headers=headers, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    raise failed_status(
                        provider=self._display_name,
                        model=model,
                        status_code=resp.status_code,
                        body=err_body.decode("utf-8", errors="replace"),
                    )

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json: dict[str, Any] = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # What the endpoint says it ran; OpenAI and vLLM name it on every
                    # chunk. Read, never defaulted: a chunk naming none carries `None`,
                    # and the agent attributes the streamed step from it (#1447).
                    raw_served: object = chunk_json.get("model")
                    served_model = (
                        str(raw_served)
                        if raw_served is not None and str(raw_served).strip()
                        else None
                    )

                    choices: list[dict[str, Any]] = chunk_json.get("choices", [])
                    delta_content: str | None = None
                    tool_calls: list[ToolCallRequest] = []
                    finish_reason: FinishReason | None = None

                    if choices:
                        delta: dict[str, Any] = choices[0].get("delta", {})
                        delta_content = delta.get("content")
                        if delta_content:
                            streamed.append(delta_content)
                        raw_tc: list[dict[str, Any]] = delta.get("tool_calls", [])
                        for tc in raw_tc:
                            fn: dict[str, Any] = tc.get("function", {})
                            streamed.extend(
                                str(part)
                                for part in (fn.get("name"), fn.get("arguments"))
                                if part is not None
                            )
                            raw_args: object = fn.get("arguments", "{}")
                            typed_args = parse_dict_payload(raw_args)
                            tool_calls.append(
                                ToolCallRequest(
                                    id=str(tc.get("id", "")),
                                    name=str(fn.get("name", "")),
                                    arguments=typed_args,
                                )
                            )
                        # `is not None`, not truthiness: OpenAI sends
                        # `"finish_reason": null` on every intermediate chunk, which
                        # carries no claim and must stay `None`, but `""` is a value the
                        # provider reported and `generate` maps it to `UNKNOWN`. Under
                        # `if raw_fr:` the two paths disagreed about the same response
                        # property, and the stream path was the one that reverted to
                        # reporting nothing (P6, #397).
                        raw_fr: object = choices[0].get("finish_reason")
                        if raw_fr is not None:
                            finish_reason = self._map_finish_reason(str(raw_fr))

                    stream_usage: dict[str, Any] | None = chunk_json.get("usage")
                    usage: TokenUsage | None = None
                    if stream_usage:
                        # No usage chunk at all yields no usage, and the agent labels its
                        # own estimate (#935); a usage chunk missing a count is completed
                        # here, from what the stream said (#939).
                        in_tok = reported_count(stream_usage, "prompt_tokens")
                        out_tok = reported_count(stream_usage, "completion_tokens")
                        in_tok, out_tok, count_source = resolve_token_counts(
                            request, in_tok, out_tok, reply="".join(streamed)
                        )
                        model_name = str(
                            chunk_json.get("model", self._resolve_request_model(request))
                        )
                        usage = TokenUsage(
                            provider=self.provider_name,
                            model=model_name,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=in_tok + out_tok,
                            count_source=count_source,
                            cache_read_input_tokens=cached_prompt_tokens(stream_usage),
                        )

                    if delta_content or tool_calls or usage or finish_reason:
                        yield StreamChunk(
                            delta_content=delta_content,
                            tool_calls=tuple(tool_calls),
                            usage=usage,
                            finish_reason=finish_reason,
                            model=served_model,
                        )
        except httpx.RequestError as exc:
            raise failed_request(provider=self._display_name, model=model, exc=exc) from exc
        finally:
            if should_close:
                await client.aclose()
