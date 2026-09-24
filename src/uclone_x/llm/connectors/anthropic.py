"""Anthropic Claude provider connector (Principle 5 & Principle 6)."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    LLMCredentialsNotConfiguredError,
    LLMProviderError,
    UnmappableChatMessageError,
)
from uclone_x.llm.connectors.base import (
    BaseLLMConnector,
    parse_dict_payload,
    reported_count,
    resolve_token_counts,
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

_DEFAULT_MODEL = "claude-3-5-sonnet"

ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT = 4096
"""The ceiling sent when a caller names none, because Anthropic requires the field.

Not a P6 substitution: nothing failed, and no value produced by a failed operation is
being replaced. It is this connector's declared policy for a **required request field**
the caller left unset, and `0` is passed through as `0` — `request.max_tokens or 4096`
billed a caller who asked for zero against 4096, which is the defect #385/PR #393 fixed
and `test_anthropic_sends_max_tokens_zero_as_zero_rather_than_4096` pins.

It is a named constant so that the site is reachable by a symbol search. As a bare literal
inside a conditional expression it was invisible to PR #393's own `or <literal>` search
pattern, so re-running that search reported the class absent while the site stood — the
case where the evidence and the fact diverge (#397). `test_no_connector_substitutes_a_
literal_for_a_missing_value` is the in-repo, control-bearing replacement for that search.
"""


def _requested_model(request: LLMRequest) -> str:
    """The model this connector asks Anthropic for, defaulted when the caller named none.

    One expression, used by the request builder, the streaming path and the
    `Provenance.requested` it reports. It was written out three times with the same
    literal; if one copy were ever changed and not the others, provenance would name
    a `requested` model that was never sent and `degraded` would flip — in the exact
    field #149 exists to make trustworthy.
    """
    if request.model is not None:
        stripped = request.model.strip()
        if stripped and stripped != "default":
            return stripped
    return _DEFAULT_MODEL


class AnthropicConnector(BaseLLMConnector):
    """Anthropic Claude API connector."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        if resolved_key is None or not resolved_key.strip():
            raise LLMCredentialsNotConfiguredError(
                "AnthropicConnector requires an API key: pass api_key= or set "
                f"ANTHROPIC_API_KEY (got api_key={api_key!r}, "
                f"ANTHROPIC_API_KEY={os.getenv('ANTHROPIC_API_KEY')!r}). It is not "
                "defaulted to '', because an empty x-api-key header turns a "
                "configuration defect into a provider-side 401 on the first billed "
                "call — a retryable-looking transport fault whose real cause is here "
                "(P6, #385)."
            )
        resolved_base = (
            base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"
        )
        super().__init__(
            api_key=resolved_key,
            base_url=resolved_base.rstrip("/"),
            timeout=timeout,
            http_client=http_client,
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _map_finish_reason(self, stop_reason: str | None) -> FinishReason:
        """Map Anthropic's `stop_reason` onto `FinishReason`, or report it as unknown.

        Only the values enumerated here are claimed. Everything else — a `stop_reason`
        Anthropic names that this mapper does not, and the field being absent — becomes
        `FinishReason.UNKNOWN` rather than `STOP`. The `return FinishReason.STOP`
        fall-through this replaces reported every unrecognised value as a clean,
        complete generation, which is a substitution in the direction of success
        (P6, #385).
        """
        if stop_reason in ("end_turn", "stop_sequence"):
            return FinishReason.STOP
        if stop_reason == "max_tokens":
            return FinishReason.LENGTH
        if stop_reason == "tool_use":
            return FinishReason.TOOL_CALLS
        return FinishReason.UNKNOWN

    def _build_payload(self, request: LLMRequest, stream: bool = False) -> dict[str, Any]:
        """Translate a provider-neutral `LLMRequest` into an Anthropic `messages` body.

        Every field is emitted as the caller wrote it, or the message is refused with
        `UnmappableChatMessageError`. This mirrors `GeminiConnector._build_payload`
        (#380/PR #384) and exists for the same reason: an outbound request has **no
        result envelope**, so P6's declared-recovery exemption — which requires in-band
        attribution via `provenance` on a result — is structurally unavailable on this
        path. That leaves exactly two options, emit faithfully or fail, and no third one
        in which a default is acceptable because it is well chosen.

        The site that differs in kind from the rest (#385):

        * `tool_use_id` is a **correlation key**, not a label. Anthropic matches a
          `tool_result` block to the `tool_use` block that requested it *by id*. The
          previous `msg.tool_call_id or ""` therefore did not mislabel a result — it
          fabricated the key the association is made on, so the result is matched
          against a block that does not exist, or against whichever block an empty id
          collides with. Nothing in the payload text or the response reveals it; the
          damage is in the association. It is refused.

        The rest are the ordinary conflation: `content=None` ("no value recorded") and
        `content=""` ("the empty value") are different facts and are never collapsed
        onto the same bytes. `max_tokens` is passed through as written — `or 4096` made
        a caller that asked for 0 be billed against 4096 — while `None` still takes the
        connector's declared 4096, because Anthropic requires the field.

        Raises:
            UnmappableChatMessageError: a message has no faithful Anthropic
                representation. The offending value is named in the message.
        """
        model = _requested_model(request)
        system_prompts: list[str] = []
        messages_payload: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        "ChatMessage(role='system') has content=None, so there is no "
                        "instruction to send. It is not skipped, because filtering it made "
                        "three distinct requests — a system message holding '', one holding "
                        "None, and no system message at all — emit a byte-identical "
                        "`system` field (P6, #385). An empty instruction is joined like any "
                        "other, so ['A', ''] sends 'A\\n\\n' rather than 'A'."
                    )
                system_prompts.append(msg.content)
            elif msg.role == MessageRole.TOOL:
                if msg.tool_call_id is None or not msg.tool_call_id.strip():
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', name={msg.name!r}) has "
                        f"tool_call_id={msg.tool_call_id!r}, and Anthropic's "
                        "tool_result.tool_use_id is the key it correlates the result to the "
                        "tool_use block that requested it. It is not defaulted to '', "
                        "because an empty id does not mislabel the result — it fabricates "
                        "the correlation key, so the result is matched against a block that "
                        "does not exist or against whichever block an empty id collides "
                        "with, and neither the payload text nor the response shows it "
                        "(P6, #385)."
                    )
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', tool_call_id={msg.tool_call_id!r}) has "
                        "content=None, and a tool result with no recorded return value has "
                        "no established Anthropic representation. It is not coerced to '', "
                        "because that reports 'the tool returned nothing' as 'the tool "
                        "returned the empty string' (P6, #385). Set content explicitly at "
                        "the call site."
                    )
                messages_payload.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
            elif msg.role == MessageRole.ASSISTANT:
                content_blocks: list[dict[str, Any]] = []
                if msg.content is not None:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
                        }
                    )
                if not content_blocks:
                    raise UnmappableChatMessageError(
                        "ChatMessage(role='assistant') has content=None and no tool_calls, "
                        "so there is no block to send. The previous "
                        "`content_blocks if content_blocks else (msg.content or '')` "
                        "defaulted twice over — the empty block list fell back to the same "
                        "content that produced it, and that content was then coerced to '' "
                        "— sending an assistant turn that said nothing as one that said the "
                        "empty string (P6, #385). Omit the message or give it content."
                    )
                messages_payload.append({"role": "assistant", "content": content_blocks})
            else:
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role={msg.role.value!r}) has content=None, so there is "
                        "no text to send. It is not coerced to '', because an absent turn "
                        "and an empty turn are different inputs to the model and would "
                        "arrive as the same bytes (P6, #385)."
                    )
                messages_payload.append({"role": "user", "content": msg.content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_payload,
            "max_tokens": (
                request.max_tokens
                if request.max_tokens is not None
                else ANTHROPIC_REQUIRED_MAX_TOKENS_DEFAULT
            ),
            "temperature": request.temperature,
            "stream": stream,
        }

        if system_prompts:
            payload["system"] = "\n\n".join(system_prompts)

        if request.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": cast(dict[str, Any], unwrap_immutable(t.parameters)),
                }
                for t in request.tools
            ]

        return payload

    async def generate(self, request: LLMRequest) -> ModelResponse:
        """Generate response from Anthropic API."""
        payload = self._build_payload(request, stream=False)
        url = f"{self.base_url}/messages"
        headers: dict[str, str] = {
            "x-api-key": self._require_api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None

        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise LLMProviderError(f"Anthropic error {resp.status_code}: {resp.text}")
            data: dict[str, Any] = resp.json()
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Anthropic connection error: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"Invalid JSON from Anthropic: {exc}") from exc
        finally:
            if should_close:
                await client.aclose()

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=parse_dict_payload(block.get("input", {})),
                    )
                )

        content = "".join(text_parts) if text_parts else None
        # A count Anthropic left out is estimated and labelled, never read as 0 (#939).
        usage_data: dict[str, Any] | None = data.get("usage")
        in_tokens = reported_count(usage_data, "input_tokens")
        out_tokens = reported_count(usage_data, "output_tokens")
        in_tokens, out_tokens, count_source = resolve_token_counts(
            request, in_tokens, out_tokens, reply=content, tool_calls=tool_calls
        )
        requested_model = _requested_model(request)
        model_name = data.get("model", requested_model)

        usage = TokenUsage(
            provider="anthropic",
            model=model_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=in_tokens + out_tokens,
            count_source=count_source,
        )

        finish_reason = self._map_finish_reason(data.get("stop_reason"))
        # P6: see the note in `openai.py` — the alias stays visible (#149).
        provenance = Provenance.primary(
            provider="anthropic", model=requested_model, served_model=model_name
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
        """Stream response chunks from Anthropic SSE API."""
        payload = self._build_payload(request, stream=True)
        url = f"{self.base_url}/messages"
        headers: dict[str, str] = {
            "x-api-key": self._require_api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None

        # `None` until the stream reports a count; a count it never reports is estimated
        # from the request and from what the stream said, and labelled (#939).
        input_tokens: int | None = None
        streamed: list[str] = []

        try:
            async with client.stream(
                "POST", url, json=payload, headers=headers, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    raise LLMProviderError(
                        f"Anthropic stream error {resp.status_code}: {err_body.decode('utf-8', errors='replace')}"
                    )

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        event_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event_data.get("type")
                    delta_content: str | None = None
                    tool_calls: list[ToolCallRequest] = []
                    finish_reason: FinishReason | None = None
                    usage: TokenUsage | None = None

                    if event_type == "message_start":
                        msg_data = event_data.get("message", {})
                        u = msg_data.get("usage")
                        input_tokens = reported_count(u, "input_tokens")
                    elif event_type == "content_block_delta":
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            delta_content = delta.get("text")
                            if delta_content:
                                streamed.append(delta_content)
                    elif event_type == "message_delta":
                        delta = event_data.get("delta", {})
                        # `is not None`, not truthiness: an absent or null
                        # `stop_reason` on a `message_delta` carries no claim, but `""`
                        # is a value Anthropic reported and `generate` maps it to
                        # `UNKNOWN`. See the same change in `openai.py` (P6, #397).
                        stop_reason: object = delta.get("stop_reason")
                        if stop_reason is not None:
                            finish_reason = self._map_finish_reason(str(stop_reason))
                        u = event_data.get("usage")
                        in_tok, out_tok, count_source = resolve_token_counts(
                            request,
                            input_tokens,
                            reported_count(u, "output_tokens"),
                            reply="".join(streamed),
                        )
                        model_name = _requested_model(request)
                        usage = TokenUsage(
                            provider="anthropic",
                            model=model_name,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=in_tok + out_tok,
                            count_source=count_source,
                        )

                    if delta_content or tool_calls or usage or finish_reason:
                        yield StreamChunk(
                            delta_content=delta_content,
                            tool_calls=tuple(tool_calls),
                            usage=usage,
                            finish_reason=finish_reason,
                        )
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Anthropic stream connection error: {exc}") from exc
        finally:
            if should_close:
                await client.aclose()
