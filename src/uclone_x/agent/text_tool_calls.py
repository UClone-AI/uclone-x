"""Detection of tool calls a model emitted as message content instead of as tool calls.

## What this is for

A provider's chat response has two channels. `tool_calls` is the model addressing the
runtime: a structured invocation the connector parses, the hooks gate, and the registry
executes. `content` is the model addressing the user. The separation is not a transport
detail — it is the only evidence the runtime has about which of those two things the model
meant to do.

`qwen2.5-coder:14b` on Ollama puts one in the other. Asked to read `pyproject.toml`, it
returns `tool_calls == []` and this as `content` (#694, reproduced 2026-09-11):

```json
{ "name": "file_read",
  "arguments": { "path": "pyproject.toml", "max_lines": 10, "max_bytes": 45000,
                 "start_line": null, "end_line": null } }
```

Well-formed, valid arguments, a registered tool, wrong channel. `execute_turn` sees no
structured call, breaks out of the step loop, and returns that blob as the turn's answer.
Across a full 100-problem `frontier_live` run: `tool_calls_total: 0`,
`problems_with_zero_tool_calls: 100`, and a published capability figure of 1.1% for a model
that was trying to use tools the entire time.

This is per-model, not a missing capability. Ollama reports `capabilities: ['completion',
'tools', ...]` for `qwen2.5-coder:14b` and `qwen3:8b` alike, and both models' Ollama
templates instruct the model to wrap a call in `<tool_call>` tags; `qwen3:8b` does and
works, `qwen2.5-coder:14b` does not and so Ollama's own parser never sees a call to lift
into the structured field.

## What this deliberately does not do

**It returns names, not `ToolCallRequest`s.** There is nothing here a caller could
accidentally execute, and that is the point rather than an omission — see the "no
recovery" argument in the commit that introduced this module. A detector that handed back
a ready-made invocation would be one `await` away from making model prose an entry point
into the tool path.

**It is conservative on purpose.** It recognises a response that *is* a tool call, not one
that *mentions* one: the whole content must parse, after fence and tag stripping, as a
call object or a list of them, and every key must belong to the call shape. A detector
that scanned for embedded JSON would fire on an agent correctly quoting a tool schema back
to a user.

## What a false positive actually costs

Conservatism is worth paying for, but not for the reason an earlier version of this
docstring gave. It said a false positive "pushes a run toward `UNREACHABLE` and discards a
real measurement". **That is false about this code.** `_toolless_run_refusal` in
`evals/suites/frontier.py` decides purely from the zero-tool-call share; the text-emission
map only selects which cause sentence the refusal appends. No count produced here can
refuse a run, and a run refused because its tools never fired is refused with or without
this detector.

What a false positive does cost:

* `TurnResult.text_emitted_tool_calls` is non-empty on a healthy turn, against a field
  description that reads it as proof work was discarded. The field is the runtime surface
  a caller sees, and it would be asserting something untrue about a turn that went fine.
* On a genuinely toolless run, the refusal's stated cause flips from "the agent answered
  from its weights" to the #694 attribution, sending a triager to the wrong fix.
* `text_emitted_tool_calls_total` in the eval report is inflated.

Those are misdiagnosis costs, not measurement-loss costs, and they are why the shape check
is a whitelist rather than "has a `name`".

## The detector does have false positives today

`parameters` is in `_CALL_KEYS`, so a tool *definition* is call-shaped: `{"name":
"file_read", "parameters": {...}}` — a correct answer to "what arguments does file_read
take?" — is counted, as is a bare `{"name": "file_read"}` and a JSON list of definitions.
What currently saves the OpenAI-style definition echo (`{"type": "function", "function":
{"name": ..., "description": ...}}`) is only that `description` falls outside `_CALL_KEYS`.

**That is accidental, and it is not intended to be load-bearing.** It is recorded here
because it is the current behaviour and a reader is entitled to know why the echo case
passes, not because the key set was designed as a definition filter. The designed fix is
to require an argument-carrying key: a real call has `arguments`/`args`, a definition has
`parameters`. That is a behaviour change with its own tests, so it is a follow-up rather
than a line smuggled in here — see #694's review thread. Until then, do not add a key to
`_CALL_KEYS` on the assumption that the definition cases are already excluded, and do not
remove `description`'s exclusion by adding it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, cast

#: Keys a tool-call object may carry. A mapping with any key outside this set is data that
#: happens to have a `name`, not an invocation — `{"name": "file_read", "size": 10}` is a
#: row about a tool, and counting it would report a discarded invocation where there was
#: none.
#:
#: This set does **not** separate a call from a tool *definition*: `parameters` is here, so
#: a definition is call-shaped and is counted. See the module docstring for what that costs
#: and for the narrowing that would fix it.
_CALL_KEYS: frozenset[str] = frozenset(
    {"name", "arguments", "parameters", "args", "type", "function", "id", "tool_call_id"}
)

#: ```json ... ``` and friends. Models that emit a call as prose usually dress it as a code
#: block, and the fence is not part of the JSON.
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)

#: `<tool_call>{...}</tool_call>`, the wrapper both qwen templates ask for. Reaching
#: `content` at all means the provider did not lift it into the structured field, which is
#: the same defect with a different surface.
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)


def _candidate_payloads(content: str) -> list[str]:
    """The JSON texts worth parsing out of one response body.

    At most one un-tagged candidate: the whole content, fences removed. Tagged candidates
    are taken separately because a model may emit several `<tool_call>` blocks in one
    message and each is its own attempt.
    """
    tagged = [m.group("body") for m in _TOOL_CALL_TAG_RE.finditer(content)]
    if tagged:
        return tagged
    fenced = _FENCE_RE.match(content)
    return [fenced.group("body") if fenced else content]


def _name_of(obj: Any) -> str | None:
    """The tool name a candidate object names, or `None` if it is not call-shaped."""
    if not isinstance(obj, Mapping):
        return None
    mapping = cast(Mapping[str, Any], obj)
    keys = {str(k) for k in mapping}
    if not keys or not keys <= _CALL_KEYS:
        return None
    # OpenAI's nesting: {"type": "function", "function": {"name": ..., "arguments": ...}}.
    inner = mapping.get("function")
    if isinstance(inner, Mapping):
        return _name_of(inner)
    name = mapping.get("name")
    return name if isinstance(name, str) and name else None


def detect_text_emitted_tool_calls(
    content: str | None,
    registered_tool_names: Iterable[str],
) -> tuple[str, ...]:
    """Names of registered tools this response body invoked in the wrong channel.

    Empty for every well-behaved response, including one that merely talks about a tool.
    Membership in `registered_tool_names` is the load-bearing condition: a model quoting
    some other system's API in JSON is not this defect, and without the registry check the
    count would measure JSON-shaped prose rather than discarded invocations.
    """
    if not content or not content.strip():
        return ()
    registered = {str(n) for n in registered_tool_names}
    if not registered:
        return ()

    found: list[str] = []
    for payload in _candidate_payloads(content):
        text = payload.strip()
        if not text or text[0] not in "{[":
            continue
        try:
            parsed: Any = json.loads(text)
        except (ValueError, TypeError):
            continue
        objects: list[Any] = cast(list[Any], parsed) if isinstance(parsed, list) else [parsed]
        for obj in objects:
            name = _name_of(obj)
            if name is not None and name in registered:
                found.append(name)
    return tuple(found)


__all__ = ["detect_text_emitted_tool_calls"]
