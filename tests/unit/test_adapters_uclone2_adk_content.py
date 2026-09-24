"""Tests for `ADKContentAdapter` (issue #367, deliverable 1).

The organising idea is that an adapter's contract is its round trip, so the A -> B
assertions are here only where they pin a mapping decision (which role, which envelope
key). The load-bearing assertions are:

* `ChatMessage -> ADKContent -> ChatMessage` identity, over every shape the adapter
  accepts, asserted on whole messages rather than on selected fields.
* `ADKContent -> ChatMessage -> ADKContent` identity over the characterised subset, and
  an explicit assertion of *what is lost* outside it — a documented lossy adapter is
  fine; a silently lossy one is a P6 defect.
* Every rejection: an unmapped role, part type or field shape must raise with the
  unmapped value in the message, never default or drop.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest
from pydantic import JsonValue, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from uclone_x.adapters.uclone2 import (
    ADK_ROLE_MODEL,
    ADK_ROLE_USER,
    TOOL_RESULT_KEY,
    ADKContent,
    ADKContentAdapter,
    ADKFunctionCall,
    ADKFunctionResponse,
    ADKPart,
)
from uclone_x.errors import (
    ADKConversionError,
    ADKMalformedContentError,
    ADKUnmappedRoleError,
    ADKUnrepresentableMessageError,
)
from uclone_x.llm.models import ChatMessage, MessageRole, ToolCallRequest

# ---------------------------------------------------------------------------------------
# Fixtures: the shapes that actually exercise the hard parts.
# ---------------------------------------------------------------------------------------

USER_TEXT = ChatMessage(role=MessageRole.USER, content="What is the weather in Seoul?")

ASSISTANT_TEXT = ChatMessage(role=MessageRole.ASSISTANT, content="Let me look that up.")

ASSISTANT_MULTILINE = ChatMessage(
    role=MessageRole.ASSISTANT,
    content="line one\nline two\n\nline four",
)

TOOL_CALL = ToolCallRequest(
    id="call-1",
    name="get_weather",
    arguments={"city": "Seoul", "units": "metric", "days": 3, "detailed": True},
)

TOOL_CALL_NO_ARGS = ToolCallRequest(id="call-2", name="get_time", arguments={})

ASSISTANT_TOOL_CALL_ONLY = ChatMessage(
    role=MessageRole.ASSISTANT,
    content=None,
    tool_calls=(TOOL_CALL,),
)

ASSISTANT_TEXT_AND_CALLS = ChatMessage(
    role=MessageRole.ASSISTANT,
    content="Checking two things.",
    tool_calls=(TOOL_CALL, TOOL_CALL_NO_ARGS),
)

TOOL_RESULT = ChatMessage(
    role=MessageRole.TOOL,
    content="18C, clear",
    name="get_weather",
    tool_call_id="call-1",
)

TOOL_RESULT_ERROR_PROSE = ChatMessage(
    role=MessageRole.TOOL,
    content="Tool 'get_weather' not found",
    name="get_weather",
    tool_call_id="call-1",
)

MULTI_TURN = (
    USER_TEXT,
    ASSISTANT_TEXT_AND_CALLS,
    TOOL_RESULT,
    ChatMessage(role=MessageRole.TOOL, content="09:41", name="get_time", tool_call_id="call-2"),
    ASSISTANT_MULTILINE,
    ChatMessage(role=MessageRole.USER, content="Thanks."),
)


# ---------------------------------------------------------------------------------------
# ChatMessage -> ADKContent -> ChatMessage identity (total).
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(USER_TEXT, id="user-text"),
        pytest.param(ChatMessage(role=MessageRole.USER, content=None), id="user-no-content"),
        pytest.param(ChatMessage(role=MessageRole.USER, content=""), id="user-empty-string"),
        pytest.param(ASSISTANT_TEXT, id="assistant-text"),
        pytest.param(ASSISTANT_MULTILINE, id="assistant-multiline"),
        pytest.param(ASSISTANT_TOOL_CALL_ONLY, id="assistant-tool-call-only"),
        pytest.param(ASSISTANT_TEXT_AND_CALLS, id="assistant-text-and-two-calls"),
        pytest.param(TOOL_RESULT, id="tool-result"),
        pytest.param(TOOL_RESULT_ERROR_PROSE, id="tool-result-prose-not-json"),
        pytest.param(
            ChatMessage(
                role=MessageRole.TOOL,
                content=None,
                name="get_weather",
                tool_call_id="call-1",
            ),
            id="tool-result-none-content",
        ),
        pytest.param(
            ChatMessage(
                role=MessageRole.TOOL,
                content="",
                name="get_weather",
                tool_call_id="call-1",
            ),
            id="tool-result-empty-string",
        ),
        pytest.param(
            ChatMessage(role=MessageRole.TOOL, content="42", name="t", tool_call_id=None),
            id="tool-result-no-call-id",
        ),
        # The adversarial case for the {"result": ...} envelope: a tool result whose
        # text is itself a JSON object. Forward always wraps, reverse unwraps only a
        # single str-valued `result` key, so this survives. A "parse the content if it
        # looks like JSON" forward path would corrupt exactly this message.
        pytest.param(
            ChatMessage(
                role=MessageRole.TOOL,
                content='{"result": "nested"}',
                name="echo",
                tool_call_id="call-9",
            ),
            id="tool-result-content-is-json-object",
        ),
        pytest.param(
            ChatMessage(
                role=MessageRole.TOOL,
                content='{"temperature_c": 18, "sky": "clear"}',
                name="get_weather",
                tool_call_id="call-1",
            ),
            id="tool-result-content-is-json-document",
        ),
    ],
)
def test_chat_message_round_trip_is_identity(message: ChatMessage) -> None:
    """A -> B -> A on whole messages, which is the half a one-way test omits."""
    assert ADKContentAdapter.to_chat_message(ADKContentAdapter.to_adk_content(message)) == message


def test_multi_turn_chat_round_trip_is_identity() -> None:
    """The mixed transcript, in order: text, calls, results, more text."""
    adk = ADKContentAdapter.to_adk_contents(MULTI_TURN)
    assert len(adk) == len(MULTI_TURN)
    assert ADKContentAdapter.to_chat_messages(adk) == MULTI_TURN


def test_tool_call_arguments_survive_round_trip_by_value() -> None:
    """Argument values keep their JSON types, not just their keys."""
    restored = ADKContentAdapter.to_chat_message(
        ADKContentAdapter.to_adk_content(ASSISTANT_TOOL_CALL_ONLY)
    )
    assert restored.tool_calls[0].arguments == {
        "city": "Seoul",
        "units": "metric",
        "days": 3,
        "detailed": True,
    }
    assert restored.tool_calls[0].arguments["days"] == 3
    assert restored.tool_calls[0].arguments["detailed"] is True


# ---------------------------------------------------------------------------------------
# ADKContent -> ChatMessage -> ADKContent identity, on the characterised subset.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            ADKContent(role=ADK_ROLE_USER, parts=(ADKPart(text="hello"),)),
            id="user-single-text",
        ),
        pytest.param(ADKContent(role=ADK_ROLE_USER, parts=()), id="user-no-parts"),
        pytest.param(
            ADKContent(role=ADK_ROLE_MODEL, parts=(ADKPart(text="hi"),)),
            id="model-single-text",
        ),
        pytest.param(
            ADKContent(
                role=ADK_ROLE_MODEL,
                parts=(
                    ADKPart(text="thinking"),
                    ADKPart(
                        function_call=ADKFunctionCall(
                            name="get_weather", id="call-1", args={"city": "Seoul"}
                        )
                    ),
                ),
            ),
            id="model-text-and-call",
        ),
        pytest.param(
            ADKContent(
                role=ADK_ROLE_USER,
                parts=(
                    ADKPart(
                        function_response=ADKFunctionResponse(
                            name="get_weather", id="call-1", response={TOOL_RESULT_KEY: "18C"}
                        )
                    ),
                ),
            ),
            id="user-function-response-result-str",
        ),
        pytest.param(
            ADKContent(
                role=ADK_ROLE_USER,
                parts=(
                    ADKPart(
                        function_response=ADKFunctionResponse(
                            name="get_weather", id="call-1", response={TOOL_RESULT_KEY: None}
                        )
                    ),
                ),
            ),
            id="user-function-response-result-null",
        ),
    ],
)
def test_adk_round_trip_is_identity_on_the_characterised_subset(content: ADKContent) -> None:
    assert ADKContentAdapter.to_adk_content(ADKContentAdapter.to_chat_message(content)) == content


def test_multiple_text_parts_collapse_and_the_loss_is_exactly_the_boundaries() -> None:
    """Documented loss 1. `ChatMessage.content` is one string, so N text parts join.

    Asserted rather than merely written down, so a future change to the join separator
    or to the collapse behaviour has to come past a test.
    """
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(ADKPart(text="first"), ADKPart(text="second"), ADKPart(text="third")),
    )
    message = ADKContentAdapter.to_chat_message(content)
    assert message.content == "first\nsecond\nthird"

    back = ADKContentAdapter.to_adk_content(message)
    assert back != content
    assert back.parts == (ADKPart(text="first\nsecond\nthird"),)
    assert len(content.parts) == 3
    assert len(back.parts) == 1


def test_non_envelope_function_response_collapses_and_the_loss_is_named() -> None:
    """Documented loss 2. An arbitrary response document becomes its JSON text."""
    response = {"temperature_c": 18, "sky": "clear"}
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(
            ADKPart(
                function_response=ADKFunctionResponse(
                    name="get_weather", id="call-1", response=response
                )
            ),
        ),
    )
    message = ADKContentAdapter.to_chat_message(content)
    assert message.role == MessageRole.TOOL
    assert message.content is not None
    assert json.loads(message.content) == response

    back = ADKContentAdapter.to_adk_content(message)
    assert back != content
    restored = back.parts[0].function_response
    assert restored is not None
    # The document is inside the envelope as text, not restored as a document.
    assert list(restored.response) == [TOOL_RESULT_KEY]
    assert json.loads(str(restored.response[TOOL_RESULT_KEY])) == response


def test_function_response_with_str_result_key_is_not_double_wrapped() -> None:
    """The one-way half of the envelope rule, pinning the key rather than the round trip."""
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(
            ADKPart(
                function_response=ADKFunctionResponse(
                    name="t", id="c", response={TOOL_RESULT_KEY: "bare"}
                )
            ),
        ),
    )
    assert ADKContentAdapter.to_chat_message(content).content == "bare"


# ---------------------------------------------------------------------------------------
# Role mapping, one-way, pinning which role goes where.
# ---------------------------------------------------------------------------------------


def test_role_mapping_directions() -> None:
    assert ADKContentAdapter.to_adk_content(USER_TEXT).role == ADK_ROLE_USER
    assert ADKContentAdapter.to_adk_content(ASSISTANT_TEXT).role == ADK_ROLE_MODEL
    # A tool result rides under "user" in ADK, distinguished by the part type only.
    assert ADKContentAdapter.to_adk_content(TOOL_RESULT).role == ADK_ROLE_USER

    assert (
        ADKContentAdapter.to_chat_message(
            ADKContent(role=ADK_ROLE_USER, parts=(ADKPart(text="x"),))
        ).role
        == MessageRole.USER
    )
    assert (
        ADKContentAdapter.to_chat_message(
            ADKContent(role=ADK_ROLE_MODEL, parts=(ADKPart(text="x"),))
        ).role
        == MessageRole.ASSISTANT
    )
    assert (
        ADKContentAdapter.to_chat_message(
            ADKContent(
                role=ADK_ROLE_USER,
                parts=(ADKPart(function_response=ADKFunctionResponse(name="t", id="c")),),
            )
        ).role
        == MessageRole.TOOL
    )


# ---------------------------------------------------------------------------------------
# P6: every unmapped shape raises, naming the unmapped value.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["system", "tool", "assistant", "MODEL", "", None])
def test_unmapped_adk_role_raises_naming_the_role(role: str | None) -> None:
    with pytest.raises(ADKUnmappedRoleError) as excinfo:
        ADKContentAdapter.to_chat_message(ADKContent(role=role, parts=(ADKPart(text="payload"),)))
    assert repr(role) in str(excinfo.value)
    assert "has no MessageRole mapping" in str(excinfo.value)


def test_system_message_raises_rather_than_becoming_a_user_turn() -> None:
    system = ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant.")
    with pytest.raises(ADKUnmappedRoleError) as excinfo:
        ADKContentAdapter.to_adk_content(system)
    assert "system_instruction" in str(excinfo.value)


def test_system_message_in_a_transcript_is_not_skipped() -> None:
    transcript = (
        ChatMessage(role=MessageRole.SYSTEM, content="Be brief."),
        USER_TEXT,
    )
    with pytest.raises(ADKUnmappedRoleError):
        ADKContentAdapter.to_adk_contents(transcript)


def test_function_call_without_id_raises_rather_than_synthesising_one() -> None:
    content = ADKContent(
        role=ADK_ROLE_MODEL,
        parts=(ADKPart(function_call=ADKFunctionCall(name="get_weather", id=None)),),
    )
    with pytest.raises(ADKMalformedContentError) as excinfo:
        ADKContentAdapter.to_chat_message(content)
    assert "get_weather" in str(excinfo.value)
    assert "carries no id" in str(excinfo.value)


def test_tool_message_without_name_raises_rather_than_defaulting_to_a_placeholder() -> None:
    message = ChatMessage(role=MessageRole.TOOL, content="18C", name=None, tool_call_id="c")
    with pytest.raises(ADKUnrepresentableMessageError) as excinfo:
        ADKContentAdapter.to_adk_content(message)
    assert "name=None" in str(excinfo.value)


def test_function_response_under_model_role_raises() -> None:
    content = ADKContent(
        role=ADK_ROLE_MODEL,
        parts=(ADKPart(function_response=ADKFunctionResponse(name="get_weather", id="c")),),
    )
    with pytest.raises(ADKMalformedContentError) as excinfo:
        ADKContentAdapter.to_chat_message(content)
    assert "get_weather" in str(excinfo.value)


def test_function_call_under_user_role_raises() -> None:
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(ADKPart(function_call=ADKFunctionCall(name="get_weather", id="c")),),
    )
    with pytest.raises(ADKMalformedContentError) as excinfo:
        ADKContentAdapter.to_chat_message(content)
    assert "get_weather" in str(excinfo.value)


def test_two_function_responses_in_one_content_raise_rather_than_losing_one() -> None:
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(
            ADKPart(function_response=ADKFunctionResponse(name="a", id="c1")),
            ADKPart(function_response=ADKFunctionResponse(name="b", id="c2")),
        ),
    )
    with pytest.raises(ADKMalformedContentError) as excinfo:
        ADKContentAdapter.to_chat_message(content)
    assert "'a'" in str(excinfo.value)
    assert "'b'" in str(excinfo.value)


def test_function_response_mixed_with_text_raises_rather_than_dropping_the_text() -> None:
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(
            ADKPart(text="here you go"),
            ADKPart(function_response=ADKFunctionResponse(name="get_weather", id="c")),
        ),
    )
    with pytest.raises(ADKMalformedContentError) as excinfo:
        ADKContentAdapter.to_chat_message(content)
    assert "get_weather" in str(excinfo.value)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            ChatMessage(role=MessageRole.USER, content="x", name="somebody"),
            id="user-with-name",
        ),
        pytest.param(
            ChatMessage(role=MessageRole.USER, content="x", tool_call_id="c"),
            id="user-with-tool-call-id",
        ),
        pytest.param(
            ChatMessage(role=MessageRole.ASSISTANT, content="x", name="somebody"),
            id="assistant-with-name",
        ),
        pytest.param(
            ChatMessage(role=MessageRole.ASSISTANT, content="x", tool_call_id="c"),
            id="assistant-with-tool-call-id",
        ),
    ],
)
def test_fields_with_no_adk_slot_raise_rather_than_being_dropped(message: ChatMessage) -> None:
    with pytest.raises(ADKUnrepresentableMessageError):
        ADKContentAdapter.to_adk_content(message)


def test_tool_calls_on_a_non_assistant_role_raise() -> None:
    message = ChatMessage(role=MessageRole.USER, content="x", tool_calls=(TOOL_CALL,))
    with pytest.raises(ADKUnrepresentableMessageError) as excinfo:
        ADKContentAdapter.to_adk_content(message)
    assert "get_weather" in str(excinfo.value)


def test_every_adapter_error_is_one_conversion_error_family() -> None:
    """A caller may catch one type without enumerating four."""
    for err in (
        ADKUnmappedRoleError,
        ADKMalformedContentError,
        ADKUnrepresentableMessageError,
    ):
        assert issubclass(err, ADKConversionError)


# ---------------------------------------------------------------------------------------
# Mirror-model validation: an unmodelled part fails loudly instead of vanishing.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no-payload"),
        pytest.param({"text": "x", "function_call": ADKFunctionCall(name="t")}, id="two-payloads"),
    ],
)
def test_part_must_carry_exactly_one_payload(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ADKPart(**kwargs)  # pyright: ignore[reportArgumentType]


def test_unmodelled_adk_part_field_is_refused_not_dropped() -> None:
    """`inline_data` is a real ADK part payload this mirror does not model.

    `extra="forbid"` makes ingesting one an error. Dropping it would hand the caller a
    conversation with an image silently missing.

    The assertion is on `errors()[i]["type"] == "extra_forbidden"` and not on the
    rendered message, because the rendered message is satisfied by both trees: Pydantic
    echoes the offending input into the text, so `"inline_data" in str(exc)` holds even
    under `extra="ignore"` (where the part is instead refused for carrying no payload at
    all). Mutation M5 escaped against that weaker assertion — it passed for the wrong
    reason — which is exactly the trap of an assertion that cannot tell the trees apart.
    """
    payload: dict[str, JsonValue] = {
        "role": ADK_ROLE_USER,
        "parts": [{"inline_data": {"mime_type": "image/png"}}],
    }
    with pytest.raises(ValidationError) as excinfo:
        ADKContent.from_genai_payload(payload)
    forbidden = [err for err in excinfo.value.errors() if err["type"] == "extra_forbidden"]
    assert forbidden, f"expected an extra_forbidden error, got {excinfo.value.errors()}"
    assert any("inline_data" in str(err["loc"]) for err in forbidden)


def test_genai_payload_shape_round_trips_through_the_mirror() -> None:
    """The shape this module emits is the shape it accepts.

    NOTE: this pins `to_genai_payload`/`from_genai_payload` against each other and
    against a hand-written ADK-shaped mapping. It does NOT verify agreement with
    `google.genai.types.Content`, which is not installed in the shared venv (the `llm`
    extra is optional and absent). See the module docstring.
    """
    payload: dict[str, JsonValue] = {
        "role": ADK_ROLE_MODEL,
        "parts": [
            {"text": "checking"},
            {"function_call": {"name": "get_weather", "id": "call-1", "args": {"city": "Seoul"}}},
        ],
    }
    mirror = ADKContent.from_genai_payload(payload)
    assert mirror.to_genai_payload() == payload

    # And the emitted payload never describes a part with two payloads.
    emitted_parts = mirror.to_genai_payload()["parts"]
    assert isinstance(emitted_parts, list)
    assert len(emitted_parts) == 2
    for part in emitted_parts:
        assert isinstance(part, dict)
        assert len(part) == 1


def test_mirror_models_are_frozen() -> None:
    call = ADKFunctionCall(name="t", id="c", args={"a": 1})
    with pytest.raises(ValidationError):
        call.name = "other"  # pyright: ignore[reportAttributeAccessIssue]
    # And the mapping field is a read-only view, not a dict the caller can still edit.
    with pytest.raises(TypeError):
        call.args["a"] = 2  # pyright: ignore[reportIndexIssue]


def test_genai_payload_is_json_serialisable() -> None:
    """The payload's annotation says `JsonValue`; assert it actually is one.

    What a python-mode `model_dump()` actually leaves behind is a `tuple` for `parts`,
    and nothing else: the frozen `args`/`response` fields come out as plain `dict`s
    (measured), so `json.dumps` on a python-mode payload **succeeds**. An earlier draft
    of this docstring claimed `mappingproxy` was left in place and that these "serialise
    to nothing"; both halves are false, and a `tuple` in fact serialises to a JSON array.

    So the assertion that does the work is not the `json.dumps` survival but the two
    below it: `json.loads(json.dumps(payload)) == payload` fails on a `tuple`, because a
    round trip turns it into a `list` that no longer compares equal, and
    `isinstance(payload["parts"], list)` fails directly. That is what catches mutation
    M10 (`mode="json"` removed) — not a serialisation error, which never occurs.
    """
    content = ADKContentAdapter.to_adk_content(ASSISTANT_TEXT_AND_CALLS)
    payload = content.to_genai_payload()
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["parts"], list)


def test_result_key_with_a_non_string_value_is_not_unwrapped() -> None:
    """The envelope unwraps only `{"result": <str|None>}`, and that bound matters.

    Unwrapping `{"result": 42}` to the string `"42"` would make the reverse direction
    re-encode it as `{"result": "42"}` — a number silently turned into a string. It
    falls through to JSON text instead, which is lossy in a way the caller can see.
    """
    content = ADKContent(
        role=ADK_ROLE_USER,
        parts=(
            ADKPart(
                function_response=ADKFunctionResponse(
                    name="count", id="c", response={TOOL_RESULT_KEY: 42}
                )
            ),
        ),
    )
    message = ADKContentAdapter.to_chat_message(content)
    assert message.content == '{"result": 42}'
    assert json.loads(message.content) == {TOOL_RESULT_KEY: 42}


# ---------------------------------------------------------------------------------------
# Why `compaction_ledger` cannot falsify the totality claim.
# ---------------------------------------------------------------------------------------

#: A minimal message per role that the adapter would accept if the role were mappable at
#: all, so the probe below measures *role* mappability and not a missing required field.
_MINIMAL_BY_ROLE: dict[MessageRole, ChatMessage] = {
    MessageRole.SYSTEM: ChatMessage(role=MessageRole.SYSTEM, content="x"),
    MessageRole.USER: ChatMessage(role=MessageRole.USER, content="x"),
    MessageRole.ASSISTANT: ChatMessage(role=MessageRole.ASSISTANT, content="x"),
    MessageRole.TOOL: ChatMessage(role=MessageRole.TOOL, content="x", name="t"),
}


def _roles_the_adapter_maps() -> set[MessageRole]:
    mapped: set[MessageRole] = set()
    for role, message in _MINIMAL_BY_ROLE.items():
        try:
            ADKContentAdapter.to_adk_content(message)
        except ADKConversionError:
            continue
        mapped.add(role)
    return mapped


def _roles_on_which_compaction_ledger_is_legal() -> set[MessageRole]:
    legal: set[MessageRole] = set()
    for role in MessageRole:
        try:
            ChatMessage(role=role, content="x", name="t", compaction_ledger=True)
        except ValidationError:
            continue
        legal.add(role)
    return legal


def test_compaction_ledger_cannot_falsify_the_round_trip_totality_claim() -> None:
    """The totality claim holds *because* these two role sets are disjoint.

    `ChatMessage.compaction_ledger` has no ADK counterpart, so a mappable role that
    permitted it would carry a field this adapter must either drop (P6 violation) or
    refuse (breaking the "identity, totally" claim). Neither happens today, and the
    reason is structural rather than lucky: the field's model validator confines it to
    `SYSTEM`, and `SYSTEM` is exactly the role `to_adk_content` refuses.

    **The assertion is on that inference, not on the refusal**, and the two futures that
    would break the claim are not equivalent. Both were mutated:

    * **`SYSTEM` becomes mappable** (harness M14: the `to_adk_content` `SYSTEM` guard
      disabled). Disjointness goes red — but so do the two existing raise-based `SYSTEM`
      tests, so a third `pytest.raises` would have caught this one too.
    * **`compaction_ledger`'s legality widens onto an already-mappable role** (M15: the
      `ChatMessage` validator amended to permit `ASSISTANT`). Disjointness goes red and
      **every raise-based `SYSTEM` test still passes** — measured, not argued: M15's only
      new failures are this test and `test_chat_message_rejects_compaction_ledger_on_non_system_roles`.
      `SYSTEM` still raises, so nothing about the refusal has changed; what changed is a
      validator in a different module that this file's other tests never touch.

    M15 is what this test is for. It is the future where the claim stops being true
    silently, and the only form of assertion that sees it is one that reads both sets.

    Both sets are asserted non-empty and every role is asserted probed before the
    disjointness check runs, so a probe that silently measured nothing cannot reach the
    emptiness that would satisfy it (§6.9 case 2).
    """
    assert set(_MINIMAL_BY_ROLE) == set(MessageRole), (
        "the probe does not cover every MessageRole; a role added since this was written "
        f"is unclassified: {set(MessageRole) - set(_MINIMAL_BY_ROLE)}"
    )

    mapped = _roles_the_adapter_maps()
    ledger_legal = _roles_on_which_compaction_ledger_is_legal()

    # Guards positioned before the disjointness comparison, so a broken probe fails here
    # rather than producing an empty intersection that reads as a pass.
    assert mapped, "probe measured zero mappable roles — it is not exercising the adapter"
    assert ledger_legal, "probe measured zero ledger-legal roles — it is not exercising ChatMessage"

    assert mapped.isdisjoint(ledger_legal), (
        "a role is now both mappable to ADK and permitted to carry compaction_ledger="
        f"True: {sorted(r.value for r in mapped & ledger_legal)}. `compaction_ledger` has "
        "no ADK slot, so the ChatMessage -> ADKContent -> ChatMessage identity claim in "
        "the module docstring is no longer total — either carry the field or reject it "
        "explicitly, and update that claim."
    )


# Alias preserving discovery under both names (Issue #383 Criterion 5).
test_compaction_ledger_invariant_cannot_leak_into_adk = (
    test_compaction_ledger_cannot_falsify_the_round_trip_totality_claim
)


# ---------------------------------------------------------------------------------------
# Totality invariant against newly added ChatMessage fields (#383).
# ---------------------------------------------------------------------------------------


def _derive_adapter_handled_fields() -> set[str]:
    """Derive ChatMessage fields handled by ADKContentAdapter.to_adk_content via AST inspection."""
    source = textwrap.dedent(inspect.getsource(ADKContentAdapter.to_adk_content))
    tree = ast.parse(source)
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "message"
    }


def _discover_non_default_value(field_name: str, field_info: FieldInfo) -> object:
    """Discover a non-default value for a ChatMessage field that can be legally instantiated."""
    default_val = field_info.get_default(call_default_factory=True)
    candidates: list[object] = []
    if default_val is not PydanticUndefined and default_val is not None:
        if isinstance(default_val, bool):
            candidates.append(not default_val)
        elif isinstance(default_val, (int, float)):
            candidates.append(default_val + 1)
        elif isinstance(default_val, str):
            candidates.append(default_val + "_non_default")
        elif isinstance(default_val, (tuple, list)):
            candidates.append(("probe_item",))
        elif isinstance(default_val, dict):
            candidates.append({"probe_key": True})
    for fallback in (True, "probe_val", 1, ("probe_item",), {"probe_key": True}):
        if fallback not in candidates and fallback != default_val:
            candidates.append(fallback)

    for candidate in candidates:
        for _role, base in _MINIMAL_BY_ROLE.items():
            kwargs: dict[str, object] = {**base.model_dump(), field_name: candidate}
            try:
                ChatMessage.model_validate(kwargs)
                return candidate
            except ValidationError:
                continue

    raise AssertionError(
        f"Unable to discover a constructible non-default value for unhandled field {field_name!r}"
    )


def test_adk_adapter_totality_invariant_against_newly_added_chat_message_fields() -> None:
    """Every ChatMessage field must be either handled by the adapter or refused if populated.

    Totality claim in ADKContentAdapter docstring: ChatMessage -> ADKContent -> ChatMessage
    is total identity. That claim depends on every field of ChatMessage being either:
    1. Handled by ADKContentAdapter (derived dynamically via AST inspection of
       ADKContentAdapter.to_adk_content), or
    2. Unhandled by the adapter, but structurally unreachable because the adapter refuses
       every MessageRole on which that unhandled field can carry a non-default value.

    Futures covered:
    - The committed narrow guard (test_compaction_ledger_cannot_falsify_the_round_trip_totality_claim)
      covers field-specific role-disjointness for compaction_ledger: specifically catching M14
      (SYSTEM becoming mappable in the adapter) and M15 (compaction_ledger widening to another role).
    - This general invariant test covers arbitrary newly-added fields to ChatMessage that are not
      handled by ADKContentAdapter. If a new field is added to ChatMessage without adding handling
      or refusal in ADKContentAdapter, this test catches it because construction of non-default
      values succeeds on mappable roles (e.g. USER, ASSISTANT, TOOL) that the adapter does not refuse.

    Probe floor assertions sit BEFORE any subtraction or comparison to prevent vacuous passes:
    - ChatMessage.model_fields count must be at least 5 (currently 6).
    - Discovered handled fields count must be at least 5 (role, content, name, tool_call_id, tool_calls).
    - Discovered non-default values must be constructible on at least one role.
    """
    all_fields = set(ChatMessage.model_fields)
    assert len(all_fields) >= 5, (
        f"ChatMessage introspection returned fewer than 5 fields: {all_fields}; "
        "introspection is broken"
    )

    handled_fields = _derive_adapter_handled_fields()
    assert len(handled_fields) >= 5, (
        f"Adapter inspection discovered fewer than 5 handled fields: {handled_fields}; "
        "adapter AST introspection is broken"
    )
    assert handled_fields.issubset(all_fields), (
        f"Adapter accesses fields not present on ChatMessage: {handled_fields - all_fields}"
    )

    unhandled_fields = all_fields - handled_fields

    for field_name in sorted(unhandled_fields):
        field_info = ChatMessage.model_fields[field_name]
        non_default_value = _discover_non_default_value(field_name, field_info)

        roles_accepting_non_default: dict[MessageRole, ChatMessage] = {}
        for role, base in _MINIMAL_BY_ROLE.items():
            kwargs: dict[str, object] = {**base.model_dump(), field_name: non_default_value}
            try:
                msg = ChatMessage.model_validate(kwargs)
                roles_accepting_non_default[role] = msg
            except ValidationError:
                continue

        assert roles_accepting_non_default, (
            f"Unhandled field {field_name!r} could not be legally constructed on any MessageRole "
            f"with discovered non-default value {non_default_value!r}"
        )

        for _role, message in roles_accepting_non_default.items():
            with pytest.raises(ADKConversionError):
                ADKContentAdapter.to_adk_content(message)


# ---------------------------------------------------------------------------------------
# SDK Conformity Verification (#381)
# ---------------------------------------------------------------------------------------


def test_adk_mirror_conforms_to_real_sdk() -> None:
    """Verify that ADKContent mirror types exactly match the real google.genai types.

    Permitted by the P5 test-only SDK import exception (Issue #381).
    Skips if the optional `llm` extra is absent ("Requires uv sync --extra llm").
    Fails loudly if the schemas diverge.
    """
    try:
        from google.genai import (  # pyright: ignore[reportMissingImports]
            types as genai_types,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError:
        pytest.skip("Requires uv sync --extra llm")

    # Construct a complete mirror content representing what we emit
    content = ADKContent(role=ADK_ROLE_USER, parts=(ADKPart(text="test text"),))
    # Note: google.genai.types.Content validates single parts.
    payload = content.to_genai_payload()

    # 1. Test SDK can validate the generated payload
    try:
        native = genai_types.Content.model_validate(payload)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except Exception as e:
        pytest.fail(f"Real google.genai SDK rejected the mirror payload: {e}")

    # 2. Test mirror can validate the SDK's dumped payload
    try:
        native_dump = native.model_dump(exclude_none=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        mirror = ADKContent.from_genai_payload(native_dump)  # pyright: ignore[reportUnknownArgumentType]
    except Exception as e:
        pytest.fail(f"Mirror ADKContent rejected the real google.genai payload: {e}")

    # 3. Verify round-trip equivalence
    assert mirror == content
