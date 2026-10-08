"""Each connector sends a message's images as images, and says when a model reads them (#2107).

A tool result's screenshot reaches the model in the form its provider reads: inside
Anthropic's `tool_result`, as Gemini `inlineData` beside the `functionResponse`, and on a
`user` message after the tool messages for OpenAI's and Ollama's chat formats, which have
no image on a tool message. An image whose bytes are gone is sent as a note, not dropped.
A message with no images is built exactly as before.

Whether a model reads images is read from the provider: Anthropic's listing
(`capabilities.image_input.supported`) and Ollama's `/api/show` (`vision`). No request
leaves the process: payloads are built directly and HTTP goes to an `httpx.MockTransport`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from uclone_x.llm.catalog import CatalogEntry, ListedImageInput
from uclone_x.llm.connectors import base as connector_base
from uclone_x.llm.connectors.anthropic import AnthropicConnector
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.connectors.ollama import OllamaConnector
from uclone_x.llm.connectors.openai import OpenAIConnector
from uclone_x.llm.connectors.vllm import VLLMConnector
from uclone_x.llm.models import (
    IMAGE_UNAVAILABLE_NOTE,
    ChatMessage,
    ImagePart,
    LLMRequest,
    MessageRole,
    ToolCallRequest,
)

Handler = Callable[[httpx.Request], httpx.Response]

_PNG = ImagePart.from_bytes(b"\x89PNG fake screenshot", "image/png")
_GONE = ImagePart(media_type="image/jpeg", digest="0" * 64)
"""An image read back from a record whose bytes could not be put back: `data` is None."""

_CALL = ToolCallRequest(id="call_1", name="look", arguments={})


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _tool_turn(*images: ImagePart, content: str = "Screenshot taken.") -> LLMRequest:
    """A user asks, the model calls `look`, the tool answers with `images`, the user goes on."""
    return LLMRequest(
        model="m",
        messages=(
            ChatMessage(role=MessageRole.USER, content="What is on screen?"),
            ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=(_CALL,)),
            ChatMessage(
                role=MessageRole.TOOL,
                content=content,
                tool_call_id="call_1",
                name="look",
                images=images,
            ),
            ChatMessage(role=MessageRole.USER, content="Describe it."),
        ),
    )


def _user_with(*images: ImagePart) -> LLMRequest:
    return LLMRequest(
        model="m",
        messages=(ChatMessage(role=MessageRole.USER, content="See this.", images=images),),
    )


def _anthropic() -> AnthropicConnector:
    return AnthropicConnector(api_key="k")


def _openai() -> OpenAIConnector:
    return OpenAIConnector(api_key="k")


def _gemini() -> GeminiConnector:
    return GeminiConnector(api_key="k")


def _ollama(handler: Handler | None = None) -> OllamaConnector:
    client = _client(handler) if handler is not None else None
    return OllamaConnector(base_url="http://localhost:11434", model="llava", http_client=client)


def _payload(
    connector: AnthropicConnector | OpenAIConnector | GeminiConnector | OllamaConnector,
    request: LLMRequest,
) -> dict[str, Any]:
    return connector._build_payload(request)  # pyright: ignore[reportPrivateUsage]


_DATA_URL = f"data:image/png;base64,{_PNG.data}"


# Anthropic
# ======================================================================================


def test_anthropic_puts_a_tool_results_image_inside_its_tool_result() -> None:
    """Killed by: src/uclone_x/llm/connectors/anthropic.py :: _with_images(msg.content, images=msg.images)
    Becomes: msg.content
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: source = {"type": "base64", "media_type": image.media_type, "data": image.data}
    Becomes: source = {"type": "base64", "media_type": "image/png", "data": ""}
    """
    payload = _payload(_anthropic(), _tool_turn(_PNG))

    tool_result = payload["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert tool_result["content"] == [
        {"type": "text", "text": "Screenshot taken."},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _PNG.data},
        },
    ]
    # The mark is on the tool result, never on the trailing user message.
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][3] == {"role": "user", "content": "Describe it."}


def test_anthropic_sends_a_user_messages_image_as_a_block_and_never_marks_it() -> None:
    """Killed by: src/uclone_x/llm/connectors/anthropic.py :: _with_images(msg.content, msg.images) if msg.images else msg.content
    Becomes: msg.content
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: stable = [i for i in stable if i not in user_turns]
    Becomes: stable = stable
    """
    request = LLMRequest(
        model="m",
        messages=(
            ChatMessage(role=MessageRole.USER, content="Hi."),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hello."),
            ChatMessage(role=MessageRole.USER, content="See this.", images=(_PNG,)),
        ),
    )

    payload = _payload(_anthropic(), request)

    user = payload["messages"][2]
    assert user == {
        "role": "user",
        "content": [
            {"type": "text", "text": "See this."},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _PNG.data},
            },
        ],
    }
    assert payload["messages"][1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_sends_the_note_for_an_image_with_no_bytes() -> None:
    """Killed by: src/uclone_x/llm/connectors/anthropic.py :: blocks.append({"type": "text", "text": IMAGE_UNAVAILABLE_NOTE})
    Becomes: pass
    """
    payload = _payload(_anthropic(), _tool_turn(_GONE))

    assert payload["messages"][2]["content"][0]["content"] == [
        {"type": "text", "text": "Screenshot taken."},
        {"type": "text", "text": IMAGE_UNAVAILABLE_NOTE},
    ]


def test_anthropic_leaves_out_a_blank_text_block_beside_an_image() -> None:
    """Anthropic refuses a text block with no non-whitespace text.

    Killed by: src/uclone_x/llm/connectors/anthropic.py :: if text.strip():
    Becomes: if True:
    """
    payload = _payload(_anthropic(), _tool_turn(_PNG, content=""))

    blocks = payload["messages"][2]["content"][0]["content"]
    assert [block["type"] for block in blocks] == ["image"]


def test_anthropic_without_images_sends_plain_strings_as_before() -> None:
    payload = _payload(_anthropic(), _tool_turn())

    assert payload["messages"][0] == {"role": "user", "content": "What is on screen?"}
    assert payload["messages"][2]["content"][0]["content"] == "Screenshot taken."


# OpenAI and vLLM (chat completions)
# ======================================================================================


def test_openai_sends_a_tool_results_image_on_one_user_message_after_the_tool_run() -> None:
    """Killed by: src/uclone_x/llm/connectors/openai.py :: tool_images.extend(_image_parts(msg.images))
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/base.py :: return " ".join(["Images returned by tool call", *said]) + ":"
    Becomes: return "Images:"
    """
    second = ToolCallRequest(id="call_2", name="look", arguments={})
    request = LLMRequest(
        model="m",
        messages=(
            ChatMessage(role=MessageRole.USER, content="Compare."),
            ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=(_CALL, second)),
            ChatMessage(
                role=MessageRole.TOOL,
                content="one",
                tool_call_id="call_1",
                name="look",
                images=(_PNG,),
            ),
            ChatMessage(role=MessageRole.TOOL, content="two", tool_call_id="call_2", name="look"),
            ChatMessage(role=MessageRole.USER, content="Which differs?"),
        ),
    )

    messages = _payload(_openai(), request)["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "user", "user"]
    assert messages[2] == {
        "role": "tool",
        "content": "one",
        "name": "look",
        "tool_call_id": "call_1",
    }
    assert messages[4] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Images returned by tool call call_1 (look):"},
            {"type": "image_url", "image_url": {"url": _DATA_URL}},
        ],
    }
    assert messages[5] == {"role": "user", "content": "Which differs?"}


def test_openai_sends_tool_images_after_a_run_that_ends_the_request() -> None:
    """Killed by: src/uclone_x/llm/connectors/openai.py :: if tool_images:
    Becomes: if False:
    """
    request = LLMRequest(model="m", messages=_tool_turn(_PNG).messages[:3])

    messages = _payload(_openai(), request)["messages"]

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"][-1] == {"type": "image_url", "image_url": {"url": _DATA_URL}}


def test_openai_sends_a_user_messages_image_as_content_parts() -> None:
    """Killed by: src/uclone_x/llm/connectors/openai.py :: m_dict["content"] = [*text, *_image_parts(msg.images)]
    Becomes: pass
    """
    messages = _payload(_openai(), _user_with(_PNG))["messages"]

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "See this."},
                {"type": "image_url", "image_url": {"url": _DATA_URL}},
            ],
        }
    ]


def test_openai_sends_the_note_for_an_image_with_no_bytes() -> None:
    """Killed by: src/uclone_x/llm/connectors/openai.py :: parts.append({"type": "text", "text": IMAGE_UNAVAILABLE_NOTE})
    Becomes: pass
    """
    messages = _payload(_openai(), _user_with(_GONE))["messages"]

    assert messages[0]["content"] == [
        {"type": "text", "text": "See this."},
        {"type": "text", "text": IMAGE_UNAVAILABLE_NOTE},
    ]


def test_openai_without_images_adds_no_message() -> None:
    messages = _payload(_openai(), _tool_turn())["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[0] == {"role": "user", "content": "What is on screen?"}


def test_vllm_reuses_the_openai_image_shape() -> None:
    connector = VLLMConnector(base_url="http://localhost:8000/v1", model="m")

    messages = _payload(connector, _tool_turn(_PNG))["messages"]

    assert messages[3]["content"][-1] == {"type": "image_url", "image_url": {"url": _DATA_URL}}


# Gemini
# ======================================================================================


def test_gemini_puts_a_tool_results_image_beside_its_function_response() -> None:
    """Killed by: src/uclone_x/llm/connectors/gemini.py :: parts.extend(_image_parts(msg.images))
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/gemini.py :: parts.append({"inlineData": {"mimeType": image.media_type, "data": image.data}})
    Becomes: parts.append({"inlineData": {"mimeType": "image/png", "data": ""}})
    """
    contents = _payload(_gemini(), _tool_turn(_PNG))["contents"]

    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "look", "response": {"result": "Screenshot taken."}}},
            {"inlineData": {"mimeType": "image/png", "data": _PNG.data}},
        ],
    }


def test_gemini_sends_a_user_messages_image_beside_its_text() -> None:
    """Killed by: src/uclone_x/llm/connectors/gemini.py :: {"role": "user", "parts": [{"text": msg.content}, *_image_parts(msg.images)]}
    Becomes: {"role": "user", "parts": [{"text": msg.content}]}
    """
    contents = _payload(_gemini(), _user_with(_PNG))["contents"]

    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "See this."},
                {"inlineData": {"mimeType": "image/png", "data": _PNG.data}},
            ],
        }
    ]


def test_gemini_sends_the_note_for_an_image_with_no_bytes() -> None:
    """Killed by: src/uclone_x/llm/connectors/gemini.py :: parts.append({"text": IMAGE_UNAVAILABLE_NOTE})
    Becomes: pass
    """
    contents = _payload(_gemini(), _user_with(_GONE))["contents"]

    assert contents[0]["parts"] == [{"text": "See this."}, {"text": IMAGE_UNAVAILABLE_NOTE}]


def test_gemini_without_images_is_unchanged() -> None:
    contents = _payload(_gemini(), _tool_turn())["contents"]

    assert contents[0] == {"role": "user", "parts": [{"text": "What is on screen?"}]}
    assert len(contents[2]["parts"]) == 1


# Ollama
# ======================================================================================


def test_ollama_sends_a_tool_results_image_on_a_user_message_after_the_tool_run() -> None:
    """Killed by: src/uclone_x/llm/connectors/ollama.py :: run_images.extend(images)
    Becomes: pass
    """
    messages = _payload(_ollama(), _tool_turn(_PNG))["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user", "user"]
    assert "images" not in messages[2]
    assert messages[3] == {
        "role": "user",
        "content": "Images returned by tool call call_1 (look):",
        "images": [_PNG.data],
    }
    assert messages[4] == {"role": "user", "content": "Describe it."}


def test_ollama_puts_a_user_messages_image_on_that_message() -> None:
    """Killed by: src/uclone_x/llm/connectors/ollama.py :: m_dict["images"] = images
    Becomes: pass
    """
    messages = _payload(_ollama(), _user_with(_PNG))["messages"]

    assert messages == [{"role": "user", "content": "See this.", "images": [_PNG.data]}]


def test_ollama_sends_the_note_for_an_image_with_no_bytes() -> None:
    """Killed by: src/uclone_x/llm/connectors/ollama.py :: if notes:
    Becomes: if False:
    Killed by: src/uclone_x/llm/connectors/ollama.py :: run_text.extend([tool_images_caption(msg), *notes])
    Becomes: run_text.extend([tool_images_caption(msg)])
    """
    user = _payload(_ollama(), _user_with(_GONE))["messages"]
    tool = _payload(_ollama(), _tool_turn(_GONE))["messages"]

    assert user == [{"role": "user", "content": f"See this.\n\n{IMAGE_UNAVAILABLE_NOTE}"}]
    assert tool[3] == {
        "role": "user",
        "content": f"Images returned by tool call call_1 (look):\n\n{IMAGE_UNAVAILABLE_NOTE}",
    }


def test_ollama_sends_a_user_message_of_images_alone() -> None:
    """A user message with no text and only images is sent, not refused (#2124 item 2).

    Killed by: src/uclone_x/llm/connectors/ollama.py :: text = [msg.content] if msg.content is not None else []
    Becomes: text = [msg.content]
    Killed by: src/uclone_x/llm/connectors/ollama.py :: msg.role is MessageRole.USER and msg.images
    Becomes: msg.role is MessageRole.USER and False
    """

    def alone(*images: ImagePart) -> LLMRequest:
        return LLMRequest(model="m", messages=(ChatMessage(role=MessageRole.USER, images=images),))

    assert _payload(_ollama(), alone(_GONE))["messages"] == [
        {"role": "user", "content": IMAGE_UNAVAILABLE_NOTE}
    ]
    assert _payload(_ollama(), alone(_PNG))["messages"] == [{"role": "user", "images": [_PNG.data]}]


def test_ollama_without_images_is_unchanged() -> None:
    messages = _payload(_ollama(), _tool_turn())["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert all("images" not in m for m in messages)


def _show(capabilities: list[str], seen: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"capabilities": capabilities})

    return handler


@pytest.mark.asyncio
async def test_ollama_reads_vision_from_api_show_and_keeps_the_answer() -> None:
    """Killed by: src/uclone_x/llm/connectors/ollama.py :: answer = isinstance(capabilities, list) and "vision" in cast(list[object], capabilities)
    Becomes: answer = isinstance(capabilities, list)
    Killed by: src/uclone_x/llm/connectors/ollama.py :: return held
    Becomes: pass
    """
    seen: list[httpx.Request] = []
    vision = _ollama(_show(["completion", "vision"], seen))
    text_only = _ollama(_show(["completion", "tools"], seen))

    assert await vision.accepts_images() is True
    assert await vision.accepts_images("llava") is True
    assert await text_only.accepts_images() is False
    assert len(seen) == 2
    assert str(seen[0].url) == "http://localhost:11434/api/show"
    assert json.loads(seen[0].content) == {"model": "llava"}


@pytest.mark.asyncio
async def test_ollama_answers_false_when_api_show_fails_and_asks_again_later() -> None:
    """Killed by: src/uclone_x/llm/connectors/ollama.py :: if not resp.is_success:
    Becomes: if False:
    """
    answers = [httpx.Response(404, json={"error": "model not found"})]

    def handler(request: httpx.Request) -> httpx.Response:
        if answers:
            return answers.pop(0)
        return httpx.Response(200, json={"capabilities": ["vision"]})

    connector = _ollama(handler)

    assert await connector.accepts_images() is False
    assert await connector.accepts_images() is True


@pytest.mark.asyncio
async def test_ollama_answers_false_when_the_daemon_cannot_be_reached() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await _ollama(handler).accepts_images() is False


# Whether a listed model reads images
# ======================================================================================


def _anthropic_listing(*items: dict[str, Any]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": list(items), "has_more": False})

    return handler


@pytest.mark.asyncio
async def test_anthropic_listing_marks_image_input_only_when_supported_is_true() -> None:
    """Killed by: src/uclone_x/llm/connectors/anthropic.py :: return cast(dict[str, Any], image_input).get("supported") is True
    Becomes: return True
    Killed by: src/uclone_x/llm/connectors/anthropic.py :: accepts_images=_reads_images(item),
    Becomes: accepts_images=False,
    """
    connector = AnthropicConnector(
        api_key="k",
        http_client=_client(
            _anthropic_listing(
                {"id": "sees", "capabilities": {"image_input": {"supported": True}}},
                {"id": "blind", "capabilities": {"image_input": {"supported": False}}},
                {"id": "says-nothing"},
                {"id": "odd", "capabilities": {"image_input": {"supported": "true"}}},
            )
        ),
    )

    entries = await connector.list_models()

    assert {e.id: e.accepts_images for e in entries} == {
        "sees": True,
        "blind": False,
        "says-nothing": False,
        "odd": False,
    }


@pytest.mark.asyncio
async def test_accepts_images_reads_the_listing_once_and_answers_from_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/llm/connectors/base.py :: LISTED_IMAGE_INPUT.remember(self.provider_name, listed)
    Becomes: pass
    Killed by: src/uclone_x/llm/connectors/base.py :: return known
    Becomes: pass
    """
    monkeypatch.setattr(connector_base, "LISTED_IMAGE_INPUT", ListedImageInput())
    calls: list[httpx.Request] = []
    listing = _anthropic_listing(
        {"id": "sees", "capabilities": {"image_input": {"supported": True}}},
        {"id": "blind"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return listing(request)

    connector = AnthropicConnector(api_key="k", model="sees", http_client=_client(handler))

    assert await connector.accepts_images() is True
    assert await connector.accepts_images("blind") is False
    assert await connector.accepts_images("not-listed") is False
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_accepts_images_answers_false_when_the_listing_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(connector_base, "LISTED_IMAGE_INPUT", ListedImageInput())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    connector = AnthropicConnector(api_key="k", model="sees", http_client=_client(handler))

    assert await connector.accepts_images() is False


@pytest.mark.asyncio
async def test_accepts_images_uses_a_listing_the_catalogue_already_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ListedImageInput()
    store.remember("openai", [CatalogEntry(id="gpt-x", accepts_images=True)])
    monkeypatch.setattr(connector_base, "LISTED_IMAGE_INPUT", store)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the listing was already read")

    connector = OpenAIConnector(api_key="k", model="gpt-x", http_client=_client(handler))

    assert await connector.accepts_images() is True


@pytest.mark.asyncio
async def test_mock_reads_images_only_when_told_to() -> None:
    assert await MockLLMConnector().accepts_images() is False
    assert await MockLLMConnector(reads_images=True).accepts_images() is True
