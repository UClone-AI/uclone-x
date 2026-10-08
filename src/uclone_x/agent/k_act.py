"""K + act: tools offered as text, and calls read from the reply's text (#2188, #2170).

The ``k_act`` tools module declares no tools to the provider. The model is told in words
which tools it has and how to call one, and it writes each call into its reply as
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``. The host reads those blocks
and runs them through the same dispatch every other module uses -- bind on call,
did-you-mean, the persona's range, the hooks -- so nothing here executes anything.

Where each part of the text goes (design §5.1):

- The **base tools** are described once, in the slow context of the system message
  (`text_tools_section`). They do not change within a session.
- A tool **host binding** adds for a user message is described once, appended to that user
  message (`text_tools_block`); a tool **bound on the call** that named it is described once,
  appended to that call's result. Either way the text enters history where it first applies
  and stays there, so each request only extends the one before it.
- The model's **reply** keeps its text as written, block included, so the next request
  repeats the bytes the model produced, never a re-rendered call. The calls read from it are
  kept beside it as structured calls (`tool_calls`), so the history stays well formed for
  every reader that pairs a call with its result. On the wire an assistant message carries
  only its text (`text_call_messages`), so a call is never shown to the model twice.

**A reply that cannot be read is refused** (owner ruling, 2026-10-04): a reply holding a
``<tool_call>`` block that does not read as one call is not run, the turn ends with
`TEXT_CALL_UNREADABLE_MESSAGE`, and there is no fallback to native tool calling within the
turn, which would change the tools layer mid-epoch.

The wording, the parser and the reference lines are the ones the `tool_strategy` eval's
arm K measured; the eval imports them from here, so it measures this code. That includes
two sentences that still say "through call" (written for the single-call arms); rewording
them is a prompt change to re-measure, not an edit to make here.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, cast

from uclone_x.core.immutable import unwrap_immutable
from uclone_x.errors import PlainRefusalError
from uclone_x.llm.models import (
    ChatMessage,
    MessageRole,
    ModelResponse,
    ToolCallRequest,
    ToolDefinition,
)

__all__ = [
    "TEXT_CALL_ACT",
    "TEXT_CALL_CLOSE",
    "TEXT_CALL_EXAMPLE_NAME",
    "TEXT_CALL_INTRO",
    "TEXT_CALL_OPEN",
    "TEXT_CALL_UNREADABLE_MESSAGE",
    "TEXT_TOOLS_ADDED_HEADER",
    "TEXT_TOOLS_BASE_HEADER",
    "TEXT_TOOLS_MORE",
    "TextCallStream",
    "TextCallStreamFilter",
    "TextToolCallUnreadableError",
    "parse_text_tool_calls",
    "runnable_text_calls",
    "text_call_messages",
    "text_call_response",
    "text_tool_calls",
    "text_tools_block",
    "text_tools_section",
    "tool_reference_line",
    "tool_reference_lines",
    "visible_reply_text",
]

TEXT_CALL_OPEN: Final = "<tool_call>"
TEXT_CALL_CLOSE: Final = "</tool_call>"
# R5 (#2190): the example's name and argument are placeholders that cannot be a tool's name
# (tool names have no "<", ">" or space). The fictional `mcp__example__lookup` it replaced
# was called as if it were a tool when the binding held no fitting one (T05, qwen3:14b).
TEXT_CALL_EXAMPLE_NAME: Final = "<tool name>"
TEXT_CALL_INTRO: Final = (
    "To use a tool below, write the call as text in exactly this form, and nothing else in "
    f'that reply: {TEXT_CALL_OPEN}{{"name": "{TEXT_CALL_EXAMPLE_NAME}", "arguments": '
    f'{{"<parameter>": "<value>"}}}}{TEXT_CALL_CLOSE}, with a listed tool\'s exact name and '
    "its parameters in place of the ones in angle brackets. The result comes back in the "
    "next message. Use a tool only when the request needs one."
)
# `act` (#2170): the first wording ("write the call") made qwen2.5:7b write bare
# `name{...}` with no <tool_call> block in 12 of 15 runs, so each sentence names the block.
TEXT_CALL_ACT: Final = (
    f"When a request needs several tools, make the calls one after another: as soon as a "
    f"result comes back, write the next {TEXT_CALL_OPEN} block. Never say you will use a "
    f"tool; write its {TEXT_CALL_OPEN} block instead. Reply in words only when every step "
    "the request asked for is done."
)
TEXT_TOOLS_BASE_HEADER: Final = "Tools available now:"
TEXT_TOOLS_MORE: Final = (
    "More tools exist. When the conversation needs them, the host adds them at the end of the "
    "user's message, in the same form. From then on they are available through call too."
)
TEXT_TOOLS_ADDED_HEADER: Final = "[Tools added] Now also available through call:"

#: The turn's answer when a reply's tool call cannot be read (owner ruling, 2026-10-04).
#: Plain words: what happened, that nothing ran, and what to do; no format, tag or class.
TEXT_CALL_UNREADABLE_MESSAGE: Final = (
    "The clone tried to use a tool, but wrote its request in a form that could not be "
    "read, so nothing was run and it did not answer. Send the message again to retry."
)


class TextToolCallUnreadableError(PlainRefusalError):
    """A reply's ``<tool_call>`` text that does not read as a call; the turn is refused.

    The message is always `TEXT_CALL_UNREADABLE_MESSAGE`: the reply's own text is the
    model's, not something to show a person verbatim, and stays in the log.
    """

    def __init__(self) -> None:
        super().__init__(TEXT_CALL_UNREADABLE_MESSAGE, reason_code="tool_call_unreadable")


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tool_reference_line(tool: ToolDefinition) -> str:
    """The first line describing `tool`: its name and description."""
    return f"- {tool.name}: {tool.description}"


def tool_reference_lines(tools: Sequence[ToolDefinition]) -> list[str]:
    """The plain-text description of `tools`: a name line and an args-schema line each."""
    lines: list[str] = []
    for tool in tools:
        lines.append(tool_reference_line(tool))
        lines.append(f"  args schema: {_dumps(unwrap_immutable(tool.parameters))}")
    return lines


def text_tools_section(
    tools: Sequence[ToolDefinition], *, act: bool = True, more: bool = True
) -> str:
    """The system text that offers `tools` (the base set) and says how to call one.

    `more` adds the sentence that more tools arrive on the user's message; it is left out
    when nothing can be bound. `act` is the instruction to keep calling until the request
    is done (the module always sends it; the eval can measure K without it).
    """
    intro = f"{TEXT_CALL_INTRO} {TEXT_CALL_ACT}" if act else TEXT_CALL_INTRO
    parts = [intro, "\n".join([TEXT_TOOLS_BASE_HEADER, *tool_reference_lines(tools)])]
    if more:
        parts.append(TEXT_TOOLS_MORE)
    return "\n\n".join(parts)


def text_tools_block(tools: Sequence[ToolDefinition]) -> str | None:
    """The block announcing `tools`, appended once where they first apply; None for none."""
    if not tools:
        return None
    return "\n".join([TEXT_TOOLS_ADDED_HEADER, *tool_reference_lines(tools)])


def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Tool calls written in Qwen's ``<tool_call>{json}</tool_call>`` form, in order.

    A block's body is read whole first, then one JSON object per line: granite3.3:8b under
    K (2026-10-04) wrote the object pretty-printed over several lines, which line-by-line
    reading lost as unparsed fragments. A block with no closing tag is read to the end. A
    line that does not parse is returned as ``{"name": None, "unparsed": ...}``.
    """
    calls: list[dict[str, Any]] = []
    for chunk in text.split(TEXT_CALL_OPEN)[1:]:
        body = chunk.split(TEXT_CALL_CLOSE)[0].strip()
        try:
            whole: object = json.loads(body)
        except json.JSONDecodeError:
            whole = None
        if isinstance(whole, dict):
            obj = cast("dict[str, Any]", whole)
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments")})
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed: object = json.loads(line)
            except json.JSONDecodeError:
                calls.append({"name": None, "unparsed": line[:200]})
                continue
            if isinstance(parsed, dict):
                obj = cast("dict[str, Any]", parsed)
                calls.append({"name": obj.get("name"), "arguments": obj.get("arguments")})
    return calls


def runnable_text_calls(content: str) -> list[tuple[str, object]]:
    """Each block `parse_text_tool_calls` read as JSON naming a tool, as ``(name, arguments)``.

    Lenient: a block it could not read is skipped. The eval scores with this; the turn
    refuses such a reply instead (`text_tool_calls`).
    """
    return [
        (cast("str", c["name"]), c.get("arguments") or {})
        for c in parse_text_tool_calls(content)
        if isinstance(c.get("name"), str) and c["name"]
    ]


def _arguments(value: object) -> dict[str, Any] | None:
    """A call's arguments as a mapping: absent is none, a JSON object's text is read."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = cast("object", json.loads(value))
        except json.JSONDecodeError:
            return None
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def text_tool_calls(content: str) -> tuple[ToolCallRequest, ...]:
    """The calls a reply wrote as ``<tool_call>`` blocks, in order; none without a block.

    Every block must read as one call naming a tool, with arguments that are an object.
    Ids are ``call_<i>`` in reply order, as the Ollama connector numbers native calls.

    Raises:
        TextToolCallUnreadableError: The reply holds a block, and some block, or the whole
            reply, does not read as a call.
    """
    if TEXT_CALL_OPEN not in content:
        return ()
    calls: list[ToolCallRequest] = []
    for index, found in enumerate(parse_text_tool_calls(content)):
        name = found.get("name")
        arguments = _arguments(found.get("arguments"))
        if not isinstance(name, str) or not name.strip() or arguments is None:
            raise TextToolCallUnreadableError()
        calls.append(ToolCallRequest(id=f"call_{index}", name=name.strip(), arguments=arguments))
    if not calls:
        raise TextToolCallUnreadableError()
    return tuple(calls)


def _call_block(call: ToolCallRequest) -> str:
    body = json.dumps(
        {"name": call.name, "arguments": unwrap_immutable(call.arguments)}, ensure_ascii=False
    )
    return f"{TEXT_CALL_OPEN}{body}{TEXT_CALL_CLOSE}"


def text_call_response(resp: ModelResponse) -> ModelResponse:
    """`resp` with the calls its text writes as `tool_calls`, its text kept as written.

    A provider that parsed a call out of the text itself (it should not: nothing is
    declared) is written back into the text as a block, so the reply's text is still the
    one record of the call the model sees again.

    Raises:
        TextToolCallUnreadableError: See `text_tool_calls`.
    """
    content = (resp.content or "") + "".join(_call_block(call) for call in resp.tool_calls)
    return resp.model_copy(update={"content": content, "tool_calls": text_tool_calls(content)})


def text_call_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """`messages` as a ``k_act`` request sends them: an assistant message as text only.

    A reply this module read already carries its calls in its text, so its structured
    copy is dropped. One written under another module (before a module change) carries no
    block; its calls are written into its text as blocks instead.
    """
    out: list[ChatMessage] = []
    for message in messages:
        if message.role is not MessageRole.ASSISTANT or not message.tool_calls:
            out.append(message)
            continue
        content = message.content or ""
        if TEXT_CALL_OPEN not in content:
            content += "".join(_call_block(call) for call in message.tool_calls)
        out.append(message.model_copy(update={"content": content, "tool_calls": ()}))
    return out


_BLOCK_RE: Final = re.compile(
    re.escape(TEXT_CALL_OPEN) + r".*?(?:" + re.escape(TEXT_CALL_CLOSE) + r"|\Z)", re.DOTALL
)


def _tag_prefix_at_end(text: str, tag: str) -> int:
    """The length of the longest end of `text` that a `tag` could still begin with."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


#: The shortest end of a reply read as a cut-off opening tag (``<tool``): a lone ``<`` or
#: ``<t`` at the end of a sentence is left alone.
_CUT_TAG_MIN: Final = len("<tool")


def _without_cut_tag(text: str) -> str:
    """`text` without an opening tag cut off before its ``>`` at its end (``<tool_call``)."""
    cut = _tag_prefix_at_end(text, TEXT_CALL_OPEN)
    return text[: len(text) - cut].rstrip() if cut >= _CUT_TAG_MIN else text


def visible_reply_text(content: str) -> str:
    """A reply's text as a person sees it: every ``<tool_call>`` block taken out, and an
    opening tag the reply ended in before its ``>``."""
    if TEXT_CALL_OPEN in content:
        content = _BLOCK_RE.sub("", content).strip()
    return _without_cut_tag(content)


class TextCallStreamFilter:
    """Takes ``<tool_call>`` blocks out of streamed text, across chunk boundaries.

    `feed` returns what may be shown now; text that could still be the start of a tag is
    held back until the next chunk decides it. `flush` returns what is held when the
    stream ends, unless it is inside a block (an unclosed block is never shown).
    """

    def __init__(self) -> None:
        self._held = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        text = self._held + chunk
        self._held = ""
        shown: list[str] = []
        while text:
            tag = TEXT_CALL_CLOSE if self._inside else TEXT_CALL_OPEN
            at = text.find(tag)
            if at >= 0:
                if not self._inside:
                    shown.append(text[:at])
                text = text[at + len(tag) :]
                self._inside = not self._inside
                continue
            keep = _tag_prefix_at_end(text, tag)
            if not self._inside:
                shown.append(text[: len(text) - keep])
            self._held = text[len(text) - keep :]
            break
        return "".join(shown)

    def flush(self) -> str:
        held, self._held = self._held, ""
        return "" if self._inside else _without_cut_tag(held)


_StreamCallback = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class TextCallStream:
    """A stream callback that passes on every event but shows no ``<tool_call>`` text.

    Wraps the turn's callback for one model invocation: each ``token`` event carries only
    the visible part of its text (none, when all of it is a call); every other event is
    passed on as it is. Call `flush` once the invocation returns.
    """

    def __init__(self, callback: _StreamCallback) -> None:
        self._callback = callback
        self._filter = TextCallStreamFilter()

    async def _send(self, event: str, data: Mapping[str, Any]) -> None:
        result = self._callback(event, dict(data))
        if asyncio.iscoroutine(result):
            await result

    async def __call__(self, event: str, data: dict[str, Any]) -> None:
        if event != "token":
            await self._send(event, data)
            return
        shown = self._filter.feed(str(data.get("content") or ""))
        if shown:
            await self._send(event, {**data, "content": shown})

    async def flush(self) -> None:
        shown = self._filter.flush()
        if shown:
            await self._send("token", {"content": shown})
