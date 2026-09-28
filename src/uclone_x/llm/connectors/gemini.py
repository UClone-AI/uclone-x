"""Google Gemini provider connector (Principle 5 & Principle 6)."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.provenance import Provenance
from uclone_x.errors import (
    ImageNotReturnedError,
    LLMCredentialsNotConfiguredError,
    UnmappableChatMessageError,
)
from uclone_x.llm.catalog import CatalogEntry
from uclone_x.llm.connectors.base import (
    BaseLLMConnector,
    named_model,
    parse_dict_payload,
    refuse_response_schema,
    resolve_model,
    resolve_token_counts,
)
from uclone_x.llm.connectors.failures import failed_request, failed_status, unusable_response
from uclone_x.llm.connectors.listing import (
    MAX_LISTING_PAGES,
    get_listing_page,
    listed_items,
    optional_int,
    optional_str,
)
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenCountSource,
    TokenUsage,
    ToolCallRequest,
)

_logger = logging.getLogger(__name__)

#: Who the person using the app holds the key with, as a failure names it (#1630).
_PROVIDER = "Google"


def _usage_counts(usage_meta: dict[str, Any]) -> tuple[int, int, int]:
    """Input, output and total tokens from a Gemini `usageMetadata`.

    `totalTokenCount` is `promptTokenCount + candidatesTokenCount` plus two counts that
    sit beside them: `thoughtsTokenCount` (a thinking model's reasoning, billed as
    output) and `toolUsePromptTokenCount` (tool-use context Gemini adds to the prompt).
    Reading only the first two made every `gemini-2.5-*` reply fail `TokenUsage`'s sum
    check -- thinking is on by default there -- so the call errored after it had been
    billed. Each count is folded into the side it is charged on, which is also where
    OpenAI's `completion_tokens` already puts reasoning tokens.

    Inside a `usageMetadata` Gemini did send, an absent field is a zero: its JSON omits
    zero-valued fields, e.g. `candidatesTokenCount` for a blocked prompt.
    """
    in_tokens = int(usage_meta.get("promptTokenCount", 0)) + int(
        usage_meta.get("toolUsePromptTokenCount", 0)
    )
    out_tokens = int(usage_meta.get("candidatesTokenCount", 0)) + int(
        usage_meta.get("thoughtsTokenCount", 0)
    )
    return in_tokens, out_tokens, int(usage_meta.get("totalTokenCount", in_tokens + out_tokens))


def _first_inline_image(data: object, *, model: str) -> tuple[bytes, str]:
    """The first candidate's first ``inlineData`` part, decoded, or `ImageNotReturnedError`.

    The reason a reply held no picture (its ``finishReason``, any text the model wrote
    instead) is logged below WARNING, as `failures` logs a raw response; the error itself
    carries none of it.
    """
    candidates: list[object] = []
    if isinstance(data, dict):
        found = cast(dict[str, object], data).get("candidates")
        if isinstance(found, list):
            candidates = cast(list[object], found)
    finish: object = None
    for candidate in candidates[:1]:
        if not isinstance(candidate, dict):
            break
        cand = cast(dict[str, object], candidate)
        finish = cand.get("finishReason")
        content = cand.get("content")
        parts = cast(dict[str, object], content).get("parts") if isinstance(content, dict) else None
        for part in cast(list[object], parts) if isinstance(parts, list) else []:
            inline = (
                cast(dict[str, object], part).get("inlineData") if isinstance(part, dict) else None
            )
            if not isinstance(inline, dict):
                continue
            blob = cast(dict[str, object], inline)
            encoded, mime = blob.get("data"), blob.get("mimeType")
            if not isinstance(encoded, str) or not encoded:
                continue
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                continue
            return raw, mime if isinstance(mime, str) and mime else "application/octet-stream"
    _logger.info(
        "%s answered image model %s with no image part (finishReason=%r)", _PROVIDER, model, finish
    )
    raise ImageNotReturnedError(provider=_PROVIDER, model=model)


class GeminiConnector(BaseLLMConnector):
    """Google Gemini API connector."""

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
        self._default_model: str | None = named_model(model)
        resolved_key = (
            api_key
            if api_key is not None
            else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        )
        if resolved_key is None or not resolved_key.strip():
            raise LLMCredentialsNotConfiguredError(
                "GeminiConnector requires an API key: pass api_key= or set GEMINI_API_KEY "
                f"or GOOGLE_API_KEY (got api_key={api_key!r}, "
                f"GEMINI_API_KEY={os.getenv('GEMINI_API_KEY')!r}, "
                f"GOOGLE_API_KEY={os.getenv('GOOGLE_API_KEY')!r}). It is not defaulted to "
                "'', because an empty x-goog-api-key header turns a configuration defect "
                "into a provider-side 401 on the first billed call — a retryable-looking "
                "transport fault whose real cause is here (P6, #385)."
            )
        resolved_base = (
            base_url
            or os.getenv("GEMINI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta"
        )
        super().__init__(
            api_key=resolved_key,
            base_url=resolved_base.rstrip("/"),
            timeout=timeout,
            http_client=http_client,
        )

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _requested_model(self, request: LLMRequest) -> str:
        """The model this connector asks Gemini for: the request's, else its own, else refused.

        One expression, used by the request builder, the streaming path and the
        `Provenance.requested` it reports, so provenance cannot name a `requested` model
        that was never sent (#149).
        """
        return resolve_model(request.model, self._default_model, _PROVIDER)

    async def list_models(self) -> list[CatalogEntry]:
        """The models this key can use, from Gemini's `models.list` (#1631).

        Gemini reports each model's input and output token limits and the methods it
        supports; a model without `generateContent` (an embedding model) cannot chat.
        """
        entries: list[CatalogEntry] = []
        params: dict[str, str] = {"pageSize": "1000"}
        for _ in range(MAX_LISTING_PAGES):
            page = await get_listing_page(
                self,
                provider=_PROVIDER,
                url=f"{self.base_url}/models",
                headers={"x-goog-api-key": self._require_api_key()},
                params=params,
            )
            for item in listed_items(page, "models"):
                name = optional_str(item.get("name"))
                if name is None:
                    continue
                methods = item.get("supportedGenerationMethods")
                entries.append(
                    CatalogEntry(
                        id=name.removeprefix("models/"),
                        display_name=optional_str(item.get("displayName")),
                        context_window=optional_int(item.get("inputTokenLimit")),
                        max_output_tokens=optional_int(item.get("outputTokenLimit")),
                        chat_capable=isinstance(methods, list)
                        and "generateContent" in cast(list[object], methods),
                    )
                )
            token = optional_str(page.get("nextPageToken"))
            if token is None:
                break
            params = {"pageSize": "1000", "pageToken": token}
        return entries

    async def generate_image(self, prompt: str, aspect_ratio: str, model: str) -> tuple[bytes, str]:
        """One picture from a Gemini image model: ``(image bytes, MIME type)``.

        ``generateContent`` with ``responseModalities: ["IMAGE"]`` and the aspect ratio in
        ``imageConfig``; the first ``inlineData`` part of the first candidate is the picture.
        A reply with no such part -- text only, or no candidates, which is how the model
        declines a description -- raises `ImageNotReturnedError`; nothing stands in for the
        picture (P6). Every other failure is mapped by `failures`, like a chat call's.
        """
        chosen = resolve_model(model, self._default_model, _PROVIDER)
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": aspect_ratio},
            },
        }
        url = f"{self.base_url}/models/{chosen}:generateContent"
        headers: dict[str, str] = {
            "x-goog-api-key": self._require_api_key(),
            "Content-Type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise failed_status(
                    provider=_PROVIDER, model=chosen, status_code=resp.status_code, body=resp.text
                )
            data: object = resp.json()
        except httpx.RequestError as exc:
            raise failed_request(provider=_PROVIDER, model=chosen, exc=exc) from exc
        except json.JSONDecodeError as exc:
            raise unusable_response(provider=_PROVIDER, model=chosen, detail=str(exc)) from exc
        finally:
            if should_close:
                await client.aclose()
        return _first_inline_image(data, model=chosen)

    def _map_finish_reason(self, reason: str | None) -> FinishReason:
        """Map Gemini's `finishReason` onto `FinishReason`, or report it as unknown.

        Only the values enumerated here are claimed. Everything else — a `finishReason`
        Gemini names that this mapper does not, and the field being absent — becomes
        `FinishReason.UNKNOWN` rather than `STOP`. PR #384 fixed this connector's
        request path and left this mapper untouched, so the same substitution survived
        on the response side: `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT` and every
        other reason arrived at the caller as a clean, complete generation (P6, #385).
        """
        if reason == "STOP":
            return FinishReason.STOP
        if reason == "MAX_TOKENS":
            return FinishReason.LENGTH
        if reason == "SAFETY":
            return FinishReason.CONTENT_FILTER
        return FinishReason.UNKNOWN

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Translate a provider-neutral `LLMRequest` into a Gemini `generateContent` body.

        Every field this reads is emitted as the caller wrote it, or the message is
        refused with `UnmappableChatMessageError`. Nothing is defaulted, because the
        result of this method is what a paid provider is billed for and reasons over,
        and a substitution made here is invisible in the response (#380, P6):

        * A `TOOL` message with no usable `name` is refused rather than sent under a
          placeholder. `functionResponse.name` is how Gemini correlates a result with
          the `functionCall` it issued, so a placeholder does not lose the name — it
          **fabricates** one, and the model then reasons about a function that never
          ran under a name nothing can be traced back to.
        * `content=None` and `content=""` are different facts, so they are never
          collapsed. Where `None` has a faithful Gemini representation it is emitted
          (an `ASSISTANT` turn that is tool calls and no prose emits no text part);
          where it does not, the message is refused. `{"result": null}` is *not*
          emitted as the faithful form for a `TOOL` result: whether Gemini
          distinguishes it from `{"result": ""}` is unmeasured here, and emitting it
          on that basis would move the silent fallback from this function to the
          provider rather than remove it.
        * `parts` is never backfilled with `[{"text": ""}]`, and an empty `SYSTEM`
          text is joined rather than filtered. Both fallbacks made a message that
          carried nothing indistinguishable from one that carried the empty string,
          in the one direction the caller cannot inspect.

        Why refusal rather than a better-chosen default, stated as P6 requires it:
        P6 permits a declared fallback only when it is attributable in-band, via
        `provenance` on the result envelope. An **outbound request has no result
        envelope**, so on this path the exemption is structurally unavailable and the
        only two options P6 leaves are to emit the caller's value faithfully or to
        fail. There is no third option in which a default is acceptable because it is
        well chosen.

        A caller needing `None` carried onto the wire rather than refused wants a
        component whose contract is round-trip identity against an in-process model,
        not one whose output's reception is unobservable. No such component exists at
        this revision; `ADKContentAdapter` arrives with #367/PR #378 and does carry
        `None` through as `{"result": None}`, which is correct there and would be a
        silent substitution here.

        Raises:
            UnmappableChatMessageError: a message has no faithful Gemini
                representation. The offending value is named in the message.
        """
        refuse_response_schema(request, self.provider_name)
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        "ChatMessage(role='system') has content=None, so there is no "
                        "instruction to send. It is not skipped, because filtering it made "
                        "three distinct requests — a system message holding '', one holding "
                        "None, and no system message at all — emit byte-identical "
                        "systemInstruction (P6, #380). An empty instruction is joined like "
                        "any other, so [ 'A', '' ] sends 'A\\n\\n' rather than 'A'."
                    )
                system_texts.append(msg.content)
            elif msg.role == MessageRole.TOOL:
                if msg.name is None or not msg.name.strip():
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', tool_call_id={msg.tool_call_id!r}) has "
                        f"name={msg.name!r}, and Gemini functionResponse.name is required. "
                        "It is not defaulted to a placeholder such as 'tool', because a "
                        "placeholder attributes the result to a function that never ran and "
                        "is indistinguishable downstream from a real one (P6, #380)."
                    )
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role='tool', name={msg.name!r}) has content=None, and "
                        "a tool result with no recorded return value has no established "
                        "Gemini representation: whether the provider distinguishes "
                        "{'result': null} from {'result': ''} is unmeasured here. It is not "
                        "coerced to '', because that would report 'the tool returned nothing' "
                        "as 'the tool returned the empty string' (P6, #380). Set content "
                        "explicitly at the call site."
                    )
                parts: list[dict[str, Any]] = [
                    {
                        "functionResponse": {
                            "name": msg.name,
                            "response": {"result": msg.content},
                        }
                    }
                ]
                contents.append({"role": "user", "parts": parts})
            elif msg.role == MessageRole.ASSISTANT:
                parts = []
                if msg.content is not None:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.name,
                                "args": cast(dict[str, Any], unwrap_immutable(tc.arguments)),
                            }
                        }
                    )
                if not parts:
                    raise UnmappableChatMessageError(
                        "ChatMessage(role='assistant') has content=None and no tool_calls, so "
                        "there is no part to send. Gemini requires a non-empty parts list, and "
                        "it is not backfilled with [{'text': ''}], because that would send an "
                        "assistant turn that said nothing as one that said the empty string "
                        "(P6, #380). Omit the message or give it content."
                    )
                contents.append({"role": "model", "parts": parts})
            else:
                if msg.content is None:
                    raise UnmappableChatMessageError(
                        f"ChatMessage(role={msg.role.value!r}) has content=None, so there is "
                        "no text to send. It is not coerced to '', because an absent turn and "
                        "an empty turn are different inputs to the model and would arrive as "
                        "the same bytes (P6, #380)."
                    )
                contents.append({"role": "user", "parts": [{"text": msg.content}]})

        gen_config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            gen_config["maxOutputTokens"] = request.max_tokens

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_config,
        }

        if system_texts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}

        if request.tools:
            # `parametersJsonSchema`, not `parameters`. `parameters` is Gemini's OpenAPI
            # `Schema` subset, which rejects keywords our tools really emit —
            # `additionalProperties` (every `extra="forbid"` params model) and `$defs`/`$ref`
            # (nested params models) — with a 400 for the whole request. The JSON Schema
            # field takes the advertised schema as it is, so nothing is flattened or
            # dropped here: the schema Gemini sees is the one the other three connectors
            # send (#1542's shared pass), in keeping with emitting the caller's value
            # faithfully rather than a lossy rewrite of it.
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parametersJsonSchema": cast(
                                dict[str, Any], unwrap_immutable(t.parameters)
                            ),
                        }
                        for t in request.tools
                    ]
                }
            ]

        return payload

    async def generate(self, request: LLMRequest) -> ModelResponse:
        """Generate response from Gemini REST endpoint."""
        model = self._requested_model(request)
        payload = self._build_payload(request)
        url = f"{self.base_url}/models/{model}:generateContent"
        headers: dict[str, str] = {
            "x-goog-api-key": self._require_api_key(),
            "Content-Type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None

        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise failed_status(
                    provider=_PROVIDER, model=model, status_code=resp.status_code, body=resp.text
                )
            data: dict[str, Any] = resp.json()
        except httpx.RequestError as exc:
            raise failed_request(provider=_PROVIDER, model=model, exc=exc) from exc
        except json.JSONDecodeError as exc:
            raise unusable_response(provider=_PROVIDER, model=model, detail=str(exc)) from exc
        finally:
            if should_close:
                await client.aclose()

        candidates = data.get("candidates", [])
        if not candidates:
            raise unusable_response(provider=_PROVIDER, model=model, detail="no candidates")

        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for i, part in enumerate(parts):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCallRequest(
                        id=f"call_{i}",
                        name=fc.get("name", ""),
                        arguments=parse_dict_payload(fc.get("args", {})),
                    )
                )

        content = "".join(text_parts) if text_parts else None
        usage_meta = data.get("usageMetadata")
        if usage_meta is None:
            # Gemini sent no usage at all: estimated and labelled, never read as 0 (#939).
            in_tokens, out_tokens, count_source = resolve_token_counts(
                request, None, None, reply=content, tool_calls=tool_calls
            )
            total_tokens = in_tokens + out_tokens
        else:
            in_tokens, out_tokens, total_tokens = _usage_counts(usage_meta)
            count_source = TokenCountSource.PROVIDER
        model_name = data.get("modelVersion", model)

        usage = TokenUsage(
            provider="gemini",
            model=model_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            count_source=count_source,
        )

        finish_reason = (
            FinishReason.TOOL_CALLS
            if tool_calls
            else self._map_finish_reason(candidate.get("finishReason"))
        )
        # P6: `model` is what the URL asked for, `modelVersion` what Gemini says it
        # ran — the alias case P6 names explicitly (`gemini-1.5-pro` served by
        # `gemini-1.5-pro-002`). Collapsing them erased it (#149).
        provenance = Provenance.primary(provider="gemini", model=model, served_model=model_name)

        return ModelResponse(
            content=content,
            tool_calls=tuple(tool_calls),
            usage=usage,
            finish_reason=finish_reason,
            model_name=model_name,
            provenance=provenance,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks from Gemini SSE endpoint."""
        model = self._requested_model(request)
        payload = self._build_payload(request)
        url = f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse"
        headers: dict[str, str] = {
            "x-goog-api-key": self._require_api_key(),
            "Content-Type": "application/json",
        }
        client = self._get_client()
        should_close = self._http_client is None

        try:
            async with client.stream(
                "POST", url, json=payload, headers=headers, timeout=self.timeout
            ) as resp:
                if resp.status_code != 200:
                    err_body = await resp.aread()
                    raise failed_status(
                        provider=_PROVIDER,
                        model=model,
                        status_code=resp.status_code,
                        body=err_body.decode("utf-8", errors="replace"),
                    )

                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    candidates = chunk_json.get("candidates", [])
                    delta_content: str | None = None
                    tool_calls: list[ToolCallRequest] = []
                    finish_reason: FinishReason | None = None

                    if candidates:
                        cand = candidates[0]
                        parts = cand.get("content", {}).get("parts", [])
                        text_list: list[str] = []
                        for i, p in enumerate(parts):
                            if "text" in p:
                                text_list.append(p["text"])
                            elif "functionCall" in p:
                                fc = p["functionCall"]
                                tool_calls.append(
                                    ToolCallRequest(
                                        id=f"call_{i}",
                                        name=fc.get("name", ""),
                                        arguments=parse_dict_payload(fc.get("args", {})),
                                    )
                                )
                        if text_list:
                            delta_content = "".join(text_list)
                        # `is not None`, not truthiness: an absent or null
                        # `finishReason` on a mid-stream candidate carries no claim, but
                        # `""` is a value Gemini reported and `generate` maps it to
                        # `UNKNOWN`. See the same change in `openai.py` (P6, #397).
                        raw_fr: object = cand.get("finishReason")
                        if raw_fr is not None:
                            finish_reason = self._map_finish_reason(str(raw_fr))

                    usage_meta = chunk_json.get("usageMetadata")
                    usage: TokenUsage | None = None
                    if usage_meta:
                        in_tok, out_tok, total_tok = _usage_counts(usage_meta)
                        model_name = chunk_json.get("modelVersion", model)
                        usage = TokenUsage(
                            provider="gemini",
                            model=model_name,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=total_tok,
                        )

                    if delta_content or tool_calls or usage or finish_reason:
                        yield StreamChunk(
                            delta_content=delta_content,
                            tool_calls=tuple(tool_calls),
                            usage=usage,
                            finish_reason=finish_reason,
                        )
        except httpx.RequestError as exc:
            raise failed_request(provider=_PROVIDER, model=model, exc=exc) from exc
        finally:
            if should_close:
                await client.aclose()
