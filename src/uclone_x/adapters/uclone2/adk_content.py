"""Bidirectional conversion between the Google ADK `Content` shape and `ChatMessage`.

Deliverable 1 of issue #367. The card asks for "seamless bidirectional transformation
between `google.genai.types.Content` (ADK) and `uclone_x.llm.models.ChatMessage`".

## Why this module does not import `google.genai`

Two measured reasons, both of which the card's literal wording does not survive:

1. **P5 forbids it.** [`docs/principles/details/p5-llm-token-management.md`](../../../../docs/principles/details/p5-llm-token-management.md)
   names the ban and names this package: "No module outside `uclone_x.llm.connectors.*`
   may import a provider SDK (`anthropic`, `openai`, `google.genai`, `ollama`, or
   equivalent)". `uclone_x.adapters.uclone2` is outside that namespace. The same
   principle also says translation belongs in an adapter layer, so the *package* is
   right and only the *import* is forbidden.
2. **The SDK is not installed in the shared `.venv`.** `google-genai>=0.1.0` is declared
   in `pyproject.toml` under the **optional** `llm` extra, and that extra is absent from
   the shared `.venv` (`import google.genai` →
   `ModuleNotFoundError: No module named 'google.genai'`; `anthropic` and `openai` are
   absent likewise, which is why every connector under `uclone_x.llm.connectors.*`
   speaks raw HTTP through `httpx` instead of an SDK). With
   `reportMissingImports = true` under Pyright strict, a hard `from google.genai import
   types` would fail the quality gate rather than merely fail at runtime.

   **It is absent, not unavailable.** `uv.lock` already resolves and pins
   `google-genai` at 2.21.0, so the extra is installable (`uv sync --extra llm`).
   The mirror is used at runtime to avoid a hard runtime dependency, and verified
   under `tests/` when the optional `llm` extra is available, in compliance with P5's
   test-only SDK import exception.

So the ADK side is represented by the local frozen mirror models below — `ADKContent`,
`ADKPart`, `ADKFunctionCall`, `ADKFunctionResponse` — whose field names are transcribed
from the `google.genai.types` public schema so that a caller holding the real SDK can
cross the boundary in one line:

```python
from google.genai import types  # only legal inside uclone_x.llm.connectors.*

native = types.Content.model_validate(adk_content.to_genai_payload())
mirror = ADKContent.from_genai_payload(native.model_dump(exclude_none=True))
```

**That interop line is verified under tests.** The field names were transcribed from the published
schema, and are now verified against the installed SDK in `tests/unit/test_adapters_uclone2_adk_content.py`
when the optional `llm` extra is available, ensuring the shape this module emits and accepts
matches the real SDK.

## The round-trip contract, including what is lossy

`ChatMessage` and ADK `Content` are not isomorphic, so "bidirectional" cannot mean
"identity in both directions". The asymmetry is deliberate and is asserted in tests:

* **`ChatMessage -> ADKContent -> ChatMessage` is identity, totally**, over every message
  this module accepts (see the rejection list below). Includes multi-line text, tool
  calls with arguments, tool results whose text happens to itself be JSON, and empty
  strings (which is why the text part is emitted on `content is not None` rather than on
  truthiness — `content=""` would otherwise come back as `None`).
* **`ADKContent -> ChatMessage -> ADKContent` is identity on a characterised subset**,
  and lossy outside it. Two losses, both named:
  1. **N text parts collapse to 1.** `ChatMessage.content` is a single `str`, so N text
     parts are joined with `"\n"` and the boundaries are gone. The reverse direction
     emits one text part.
  2. **A `FunctionResponse.response` that is not `{"result": <str|None>}` collapses.**
     `ChatMessage.content` is a `str`, so an arbitrary JSON response document is carried
     as its JSON text, and re-encoding wraps that text in the `{"result": ...}` envelope
     rather than restoring the original document.

  The `{"result": ...}` envelope is the convention already in use in
  `uclone_x.llm.connectors.gemini._build_payload`, kept here so the two agree.

  Making *both* directions total is not available: `{"result": "x"}` and the literal
  string `'{"result": "x"}'` cannot both round-trip through one `str` field without a
  marker, and a marker in the payload would be a channel invented by this adapter and
  visible to ADK. One direction was chosen to be total; it is the one whose inputs
  originate in this codebase.

## P6: what is rejected rather than coerced

Every shape below raises with the offending value in the message. None of them defaults,
drops, or coerces — that is the defect class #364 recorded against PR #358, where
`except Exception: return []` meant "nothing is missing".

* An ADK `role` that is neither `"user"` nor `"model"` (including `None`).
* A `MessageRole.SYSTEM` message: ADK carries system instructions out of band in
  `GenerateContentConfig.system_instruction`, not as a `Content` role. Coercing it to
  `"user"` would silently promote a system prompt into the dialogue.
* An ADK `FunctionCall` with no `id`: `ToolCallRequest.id` is required, and synthesising
  one would break correlation with the matching `FunctionResponse`.
* A `MessageRole.TOOL` message with no `name`: ADK `FunctionResponse.name` is required,
  and defaulting it (as `gemini._build_payload` does, with `msg.name or "tool"`) would
  attribute a result to a tool that never ran.
* A `Content` mixing `function_response` with text, carrying more than one
  `function_response`, or putting `function_call` under `"user"` / `function_response`
  under `"model"` — none has a single-`ChatMessage` representation.
* Any `ChatMessage` field with no ADK slot in the chosen role: `name` or `tool_call_id`
  on a non-`TOOL` message, `tool_calls` on a non-`ASSISTANT` message.
* An ADK part carrying anything other than exactly one of `text`, `function_call`,
  `function_response`. `extra="forbid"` on the mirror models makes an unmodelled ADK
  part field (`inline_data`, `code_execution_result`, ...) a loud `ValidationError` at
  `from_genai_payload` rather than a silently dropped part.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from uclone_x.core.immutable import ImmutableJsonMapping, unwrap_immutable
from uclone_x.errors import (
    ADKMalformedContentError,
    ADKUnmappedRoleError,
    ADKUnrepresentableMessageError,
)
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest

__all__ = [
    "ADK_ROLE_MODEL",
    "ADK_ROLE_USER",
    "TOOL_RESULT_KEY",
    "ADKContent",
    "ADKContentAdapter",
    "ADKFunctionCall",
    "ADKFunctionResponse",
    "ADKPart",
]

ADK_ROLE_USER = "user"
"""The ADK `Content.role` for a turn authored by the user *or* carrying a tool result."""

ADK_ROLE_MODEL = "model"
"""The ADK `Content.role` for a turn authored by the model."""

TOOL_RESULT_KEY = "result"
"""Envelope key for a scalar tool result inside `FunctionResponse.response`.

`FunctionResponse.response` is a JSON object while `ChatMessage.content` is a string, so
a string result needs a key to live under. `result` is the key
`uclone_x.llm.connectors.gemini._build_payload` already uses; duplicating the choice
rather than inventing a second one keeps a payload built by either path readable by both.
"""

# The mirror models below deliberately do NOT set `strict=True`, unlike the models in
# `uclone_x.llm.models`. Their whole job is to accept a foreign payload: under Pydantic
# strict mode a nested `dict` will not validate into a nested model, which would make
# `from_genai_payload` — the one method that exists to ingest a dict from outside —
# impossible. `frozen=True` (P2/P3 zero-copy immutability) and `extra="forbid"` (P6: an
# unmodelled ADK field fails loudly instead of vanishing) are both kept.
_ADK_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")


class ADKFunctionCall(BaseModel):
    """Mirror of `google.genai.types.FunctionCall`."""

    model_config = _ADK_MODEL_CONFIG

    name: str
    id: str | None = None
    args: ImmutableJsonMapping = Field(default_factory=dict)


class ADKFunctionResponse(BaseModel):
    """Mirror of `google.genai.types.FunctionResponse`."""

    model_config = _ADK_MODEL_CONFIG

    name: str
    id: str | None = None
    response: ImmutableJsonMapping = Field(default_factory=dict)


class ADKPart(BaseModel):
    """Mirror of `google.genai.types.Part`, narrowed to the three modelled payloads.

    The real `Part` is a union of many optional payloads. This mirror models the three
    the card names and forbids the rest, so an unsupported part is a `ValidationError`
    naming the field rather than a part that quietly disappeared from a conversation.
    """

    model_config = _ADK_MODEL_CONFIG

    text: str | None = None
    function_call: ADKFunctionCall | None = None
    function_response: ADKFunctionResponse | None = None

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> Self:
        present = [
            name
            for name, value in (
                ("text", self.text),
                ("function_call", self.function_call),
                ("function_response", self.function_response),
            )
            if value is not None
        ]
        if len(present) != 1:
            raise ValueError(
                "an ADK Part must carry exactly one of text, function_call, "
                f"function_response; got {present or ['none']}"
            )
        return self


class ADKContent(BaseModel):
    """Mirror of `google.genai.types.Content`."""

    model_config = _ADK_MODEL_CONFIG

    role: str | None = None
    parts: tuple[ADKPart, ...] = Field(default_factory=tuple)

    def to_genai_payload(self) -> dict[str, JsonValue]:
        """This content as a mapping intended for `google.genai.types.Content.model_validate`.

        **That compatibility is verified in tests.** The field names are transcribed
        from the `google.genai.types` published schema, and are verified against an installed
        SDK in `tests/unit/test_adapters_uclone2_adk_content.py` when the optional `llm`
        extra is present.

        `exclude_none=True` matters: the real `Part` has many optional fields this mirror
        does not model, and emitting `{"text": null}` alongside a `function_call` would
        describe a part that carries two payloads.

        `mode="json"` matters too: the default `mode="python"` emits `parts` as a `tuple`,
        which is not `JsonValue`, so the annotation on this method would be false. Note
        what this is *not* — a `tuple` serialises to a JSON array perfectly well, and the
        frozen `args`/`response` fields come out of a python-mode dump as plain `dict`s
        (measured), so `json.dumps` on a python-mode payload **succeeds**. An earlier
        draft of this docstring predicted `TypeError: Object of type mappingproxy is not
        JSON serializable`, which does not happen; it confused the field's type on the
        model with the dumped value's type. The real consequence is a `tuple` where the
        return type promises a list, which `test_genai_payload_is_json_serialisable`
        pins via `isinstance(payload["parts"], list)` and round-trip equality.
        """
        return cast(dict[str, JsonValue], self.model_dump(mode="json", exclude_none=True))

    @classmethod
    def from_genai_payload(cls, payload: Mapping[str, JsonValue]) -> Self:
        """Build a mirror from what `types.Content.model_dump(exclude_none=True)` is
        intended to produce.

        **That compatibility is verified in tests**, for the reason given on
        `to_genai_payload` and in the module docstring.

        Raises `pydantic.ValidationError` on any field this mirror does not model, which
        is the intended behaviour: a dropped part is worse than a refused conversion.
        """
        return cls.model_validate(dict(payload))


def _plain(mapping: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """A mutable, JSON-serialisable copy of a frozen mapping field."""
    return cast(dict[str, JsonValue], unwrap_immutable(dict(mapping)))


def _decode_tool_result(response: Mapping[str, JsonValue]) -> str | None:
    """`FunctionResponse.response` as a `ChatMessage.content` string.

    The `{"result": <str|None>}` envelope round-trips exactly; anything else is carried
    as its JSON text and does not survive re-encoding (see the module docstring).
    """
    plain = _plain(response)
    if list(plain) == [TOOL_RESULT_KEY]:
        only = plain[TOOL_RESULT_KEY]
        if only is None or isinstance(only, str):
            return only
    return json.dumps(plain, sort_keys=True, ensure_ascii=False)


def _to_tool_call(call: ADKFunctionCall) -> ToolCallRequest:
    if call.id is None:
        raise ADKMalformedContentError(
            f"ADK FunctionCall name={call.name!r} carries no id, and "
            "ToolCallRequest.id is required. An id is not synthesised here because it "
            "is the only thing correlating this call with its FunctionResponse; a made-up "
            "value would silently mis-pair a result with a call (P6)."
        )
    return ToolCallRequest(id=call.id, name=call.name, arguments=_plain(call.args))


def _to_function_call(call: ToolCallRequest) -> ADKFunctionCall:
    return ADKFunctionCall(name=call.name, id=call.id, args=_plain(call.arguments))


class ADKContentAdapter:
    """Stateless converter between `ADKContent` and `ChatMessage`.

    Every method is a `staticmethod`/`classmethod` and the class holds no state: it is a
    namespace, not an object with a lifecycle, so two callers cannot observe each other.
    """

    @staticmethod
    def to_chat_message(content: ADKContent) -> ChatMessage:
        """Convert one ADK `Content` into one `ChatMessage`.

        Raises:
            ADKUnmappedRoleError: `content.role` is neither `"user"` nor `"model"`.
            ADKMalformedContentError: the part mix has no single-`ChatMessage` form.
        """
        role = content.role
        if role not in (ADK_ROLE_USER, ADK_ROLE_MODEL):
            raise ADKUnmappedRoleError(
                f"ADK Content.role={role!r} has no MessageRole mapping. Mapped roles are "
                f"{ADK_ROLE_USER!r} and {ADK_ROLE_MODEL!r}; an unrecognised role is not "
                "defaulted to USER because that would silently reattribute a turn (P6)."
            )

        texts = [part.text for part in content.parts if part.text is not None]
        calls = [part.function_call for part in content.parts if part.function_call is not None]
        responses = [
            part.function_response for part in content.parts if part.function_response is not None
        ]
        joined = "\n".join(texts) if texts else None

        if role == ADK_ROLE_MODEL:
            if responses:
                raise ADKMalformedContentError(
                    f"ADK Content.role={ADK_ROLE_MODEL!r} carries "
                    f"{len(responses)} function_response part(s) "
                    f"(names={[r.name for r in responses]}); a tool result is authored by "
                    f"the caller and rides under role={ADK_ROLE_USER!r} in ADK."
                )
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=joined,
                tool_calls=tuple(_to_tool_call(call) for call in calls),
            )

        if calls:
            raise ADKMalformedContentError(
                f"ADK Content.role={ADK_ROLE_USER!r} carries {len(calls)} function_call "
                f"part(s) (names={[c.name for c in calls]}); a tool call is authored by the "
                f"model and rides under role={ADK_ROLE_MODEL!r} in ADK."
            )
        if not responses:
            return ChatMessage(role=MessageRole.USER, content=joined)
        if len(responses) > 1:
            raise ADKMalformedContentError(
                f"ADK Content carries {len(responses)} function_response parts "
                f"(names={[r.name for r in responses]}); ChatMessage holds one tool result "
                "(one name, one tool_call_id), so these must be split across one Content "
                "each rather than merged."
            )
        if texts:
            raise ADKMalformedContentError(
                f"ADK Content mixes a function_response (name="
                f"{responses[0].name!r}) with {len(texts)} text part(s); ChatMessage.content "
                "is already the tool result, so the text has nowhere to go and is not "
                "dropped."
            )
        answered = responses[0]
        return ChatMessage(
            role=MessageRole.TOOL,
            content=_decode_tool_result(answered.response),
            name=answered.name,
            tool_call_id=answered.id,
        )

    @staticmethod
    def to_adk_content(message: ChatMessage) -> ADKContent:
        """Convert one `ChatMessage` into one ADK `Content`.

        Raises:
            ADKUnmappedRoleError: the message is `MessageRole.SYSTEM`.
            ADKUnrepresentableMessageError: a populated field has no slot in the ADK
                shape for this role.
        """
        if message.role == MessageRole.SYSTEM:
            raise ADKUnmappedRoleError(
                "MessageRole.SYSTEM has no ADK Content.role. ADK carries a system prompt "
                "out of band in GenerateContentConfig.system_instruction; mapping it to "
                f"role={ADK_ROLE_USER!r} would promote a system instruction into the "
                "dialogue as if the user had said it (P6). Extract it before converting."
            )

        if message.role != MessageRole.TOOL and (
            message.name is not None or message.tool_call_id is not None
        ):
            raise ADKUnrepresentableMessageError(
                f"ChatMessage(role={message.role.value!r}) carries name={message.name!r} "
                f"tool_call_id={message.tool_call_id!r}; ADK has a slot for neither outside "
                "a FunctionResponse, and they are not dropped."
            )
        if message.role != MessageRole.ASSISTANT and message.tool_calls:
            raise ADKUnrepresentableMessageError(
                f"ChatMessage(role={message.role.value!r}) carries "
                f"{len(message.tool_calls)} tool_calls "
                f"(names={[c.name for c in message.tool_calls]}); in ADK a function_call is "
                f"authored by the model and belongs to role={ADK_ROLE_MODEL!r}."
            )

        if message.role == MessageRole.TOOL:
            if message.name is None:
                raise ADKUnrepresentableMessageError(
                    "ChatMessage(role='tool') has name=None, and ADK FunctionResponse.name "
                    "is required. It is not defaulted to a placeholder because a placeholder "
                    "attributes the result to a tool that never ran (P6); "
                    "uclone_x.llm.connectors.gemini._build_payload does default it, and that "
                    "is a defect there rather than a precedent."
                )
            envelope: dict[str, JsonValue] = {TOOL_RESULT_KEY: message.content}
            return ADKContent(
                role=ADK_ROLE_USER,
                parts=(
                    ADKPart(
                        function_response=ADKFunctionResponse(
                            name=message.name,
                            id=message.tool_call_id,
                            response=envelope,
                        )
                    ),
                ),
            )

        parts: list[ADKPart] = []
        if message.content is not None:
            parts.append(ADKPart(text=message.content))
        parts.extend(ADKPart(function_call=_to_function_call(c)) for c in message.tool_calls)
        adk_role = ADK_ROLE_MODEL if message.role == MessageRole.ASSISTANT else ADK_ROLE_USER
        return ADKContent(role=adk_role, parts=tuple(parts))

    @classmethod
    def to_chat_messages(cls, contents: Sequence[ADKContent]) -> tuple[ChatMessage, ...]:
        """Convert a multi-turn ADK transcript, preserving order."""
        return tuple(cls.to_chat_message(content) for content in contents)

    @classmethod
    def to_adk_contents(cls, messages: Sequence[ChatMessage]) -> tuple[ADKContent, ...]:
        """Convert a multi-turn `ChatMessage` transcript, preserving order.

        A `MessageRole.SYSTEM` message raises rather than being skipped: dropping it
        would hand ADK a transcript missing its instructions with no error to see.
        """
        return tuple(cls.to_adk_content(message) for message in messages)
