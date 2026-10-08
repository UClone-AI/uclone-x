"""Case-skill routing for drawing turns: which drawing guidance fits this message.

Any clone offered `generate_image` on a turn is routed (#2091); the routing was first
measured on the Artist.

One system prompt cannot serve a two-word brief, a detailed description, a pencil sketch
"with nothing else" and "make it night" at once: guidance that makes a vague brief rich
makes a precise one wander. Measured on qwen3:8b, giving the model the one piece of
guidance that fits the request raised both richness and fidelity. This module decides
which piece fits, and hands it over as a turn section after the latest user message --
never in the system prompt or a tool description, so the cached request prefix is the
same with and without it.

How the case is decided:

* **Follow-up** is read from conversation state, not wording: the history holds a
  successful `generate_image` result (`is_follow_up`). It gets `art-iterative-edit`, and
  `art-genre-vocab` too when the message asks for more detail.
* **A first turn** uses grounded extraction (`extract_request_facts`): one structured
  call quotes the medium, any request for simplicity, and the concrete details the
  person wrote. Code keeps only quotes that occur verbatim in the message
  (`grounded_facts`) -- without that check a small model invents media and details --
  and `route_first_turn` decides from what is left.

The skill texts live in the runtime skill store (`ucx-agent-skills/art-*`), pinned in
`SHIPPED_SKILL_PINS` like every shipped skill, and are read from the agent's registry
(`case_skill_texts`). They are written so the model has nothing to paste: no slot labels, no
`no X` phrase (the only `no ...` string is the real tag `no humans`), no quality tags
(the call-time fill appends them), and each culture's traditional vocabulary on its own
line, because one mixed line had the model dress a kimono request in hanbok.

The caller decides what a failure costs. `BaseAgent` treats any failure as "no skill":
the turn runs as it would have without this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from uclone_x.errors import UCloneXError
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ModelResponse
from uclone_x.memory.models import fold_name
from uclone_x.skills.models import ROUTED_SKILL_TAG, SkillStatus
from uclone_x.skills.protocols import SkillRegistryProtocol

__all__ = [
    "CASE_SKILL_NAMES",
    "ROUTED_SKILL_TAG",
    "CaseSkillExtractionError",
    "RequestFacts",
    "asks_for_more_detail",
    "case_skill_section",
    "case_skill_texts",
    "extract_request_facts",
    "grounded_facts",
    "is_follow_up",
    "latest_span_message",
    "route_first_turn",
    "route_follow_up",
]

CASE_SKILL_HEADER = "[Drawing Case Skills]"
"""Persona-neutral: every clone offered `generate_image` reads it, not only the Artist."""

#: Grounded attributes at or above this count make a request a detailed specification.
LITERAL_ATTRIBUTES = 4

CASE_SKILL_NAMES: tuple[str, ...] = (
    "art-brief-expansion",
    "art-literal-spec",
    "art-medium-restraint",
    "art-iterative-edit",
    "art-genre-vocab",
)
"""The case skills, each a prompt-only package in the runtime skill store (#1865)."""


def case_skill_texts(skills: SkillRegistryProtocol | None) -> dict[str, str]:
    """The text of each case skill the registry holds active, by name.

    Read from the skill store, where each is approved and pinned like any shipped skill.
    A case skill the store does not hold (a store that was not loaded, or a package whose
    bytes no longer match its pin) is absent, and routing skips it.
    """
    if skills is None:
        return {}
    texts: dict[str, str] = {}
    for name in CASE_SKILL_NAMES:
        skill = skills.get(name)
        if skill is not None and skill.manifest.status is SkillStatus.ACTIVE:
            texts[name] = skill.instructions_markdown.strip()
    return texts


_MORE_DETAIL = re.compile(
    r"디테일|화려|자세히|풍부|더 많이|\bmore detail|\bdetailed\b|\bricher\b|\belaborate",
    re.IGNORECASE,
)


def asks_for_more_detail(message: str) -> bool:
    """Whether a follow-up asks for a richer picture (then genre vocabulary helps)."""
    return bool(_MORE_DETAIL.search(message))


#: A line of a room span that opens a message: `[<sender id>]: `, as the room renders it.
_SPAN_SENDER = re.compile(r"^\[(?P<sender>[^\]\n]+)\]: ", re.MULTILINE)


def latest_span_message(span: str, person_names: Sequence[str] = ()) -> str:
    """The latest user message in a room span, without its sender, or `span` when it has none.

    A room hands a seat every message it has not seen, one `[sender]: text` line each,
    the latest last. A room turn's request is what the user asked for (#1865): in a room,
    another clone's comment must not steer routing or image-set planning. When the span
    holds messages from a human participant (`person_names`, falling back to `"user"`),
    the latest one of those is returned. If no message from a person is found in the
    span, it falls back to the latest message.
    """
    starts = list(_SPAN_SENDER.finditer(span))
    if not starts:
        return span

    def _is_person(sender: str) -> bool:
        s = sender.strip()
        if not s:
            return False
        targets = person_names or ("user",)
        folded = fold_name(s)
        return any(fold_name(p) == folded for p in targets)

    for i in range(len(starts) - 1, -1, -1):
        match = starts[i]
        sender = match.group("sender")
        if _is_person(sender):
            content_start = match.end()
            content_end = starts[i + 1].start() if i + 1 < len(starts) else len(span)
            return span[content_start:content_end].strip("\r\n")

    return span[starts[-1].end() :]


def is_follow_up(history: Sequence[ChatMessage]) -> bool:
    """Whether this conversation already holds an image this clone drew.

    Read from state: a `generate_image` tool result that reports a saved image
    (`relative_url`). A failed call leaves no image, so it is not a follow-up.
    """
    return any(
        m.role is MessageRole.TOOL
        and m.name == "generate_image"
        and '"relative_url"' in (m.content or "")
        for m in history
    )


class CaseSkillExtractionError(UCloneXError):
    """The extraction reply was not the facts object the schema asks for."""


@dataclass(frozen=True)
class RequestFacts:
    """What the person wrote, as quotes: a medium, a request for simplicity, details."""

    medium_quote: str = ""
    minimal_quote: str = ""
    attribute_quotes: tuple[str, ...] = ()


_EXTRACTION_INSTRUCTIONS = """\
Quote facts from an image request. Copy words EXACTLY as they appear in the request; never \
add anything the user did not write. Answer JSON only.
medium_quote: the exact words where the user asks the image to be drawn in an art medium or \
technique (watercolor, pencil, ink, lineart, pixel art, oil painting, monochrome, manga, \
croquis...), or an empty string if the user did not ask for one. An object in the scene \
(someone holding a sketchbook) or a clothing color is NOT a medium.
minimal_quote: the exact words where the user asks for simplicity, empty space, only one \
thing, no background or nothing extra; an empty string if none.
attribute_quotes: the exact words for each concrete visual detail the user wrote: hair, \
eyes, clothing, colors, pose, framing or camera angle, background objects, lighting. Not \
the subject noun, not moods. An empty list if none."""

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "medium_quote": {"type": "string"},
        "minimal_quote": {"type": "string"},
        "attribute_quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["medium_quote", "minimal_quote", "attribute_quotes"],
}


def _parse_facts(response: ModelResponse) -> RequestFacts:
    try:
        raw = json.loads(response.content or "")
    except ValueError as exc:
        raise CaseSkillExtractionError("extraction reply is not JSON") from exc
    if not isinstance(raw, dict):
        raise CaseSkillExtractionError("extraction reply is not an object")
    fields: dict[str, Any] = {str(k): v for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    attributes: Any = fields.get("attribute_quotes") or []
    if not isinstance(attributes, list):
        raise CaseSkillExtractionError("attribute_quotes is not a list")
    medium: Any = fields.get("medium_quote") or ""
    minimal: Any = fields.get("minimal_quote") or ""
    return RequestFacts(
        medium_quote=medium if isinstance(medium, str) else "",
        minimal_quote=minimal if isinstance(minimal, str) else "",
        attribute_quotes=tuple(a for a in attributes if isinstance(a, str)),  # pyright: ignore[reportUnknownVariableType]
    )


async def extract_request_facts(
    generate: Callable[[LLMRequest], Awaitable[ModelResponse]],
    message: str,
    *,
    model: str | None = None,
    context_window: int | None = None,
) -> RequestFacts:
    """Quote the medium, simplicity and details `message` states, in one structured call.

    The quotes are as the model gave them; `grounded_facts` keeps only the true ones.

    Raises:
        CaseSkillExtractionError: the reply is not a facts object.
        StructuredOutputUnsupportedError: the connector cannot take a schema.
        Any provider error from `generate`, unchanged.
    """
    request = LLMRequest(
        model=model,
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content=_EXTRACTION_INSTRUCTIONS),
            ChatMessage(role=MessageRole.USER, content=message),
        ),
        temperature=0.0,
        max_tokens=300,
        thinking=False,
        response_schema=EXTRACTION_SCHEMA,
        context_window=context_window,
        auto_compact=False,
    )
    return _parse_facts(await generate(request))


def _grounded(quote: str, message: str) -> bool:
    text = quote.strip().lower()
    return bool(text) and text not in ("null", "none") and text in message.lower()


def grounded_facts(facts: RequestFacts, message: str) -> RequestFacts:
    """`facts` less every quote that does not occur verbatim (ignoring case) in `message`."""
    return RequestFacts(
        medium_quote=facts.medium_quote if _grounded(facts.medium_quote, message) else "",
        minimal_quote=facts.minimal_quote if _grounded(facts.minimal_quote, message) else "",
        attribute_quotes=tuple(a for a in facts.attribute_quotes if _grounded(a, message)),
    )


def route_first_turn(facts: RequestFacts) -> tuple[str, ...]:
    """The case skills for a first-turn request, from its grounded facts.

    A medium or a request for simplicity is respected first, with literal translation
    when the person also gave many details. Many details alone mean translate, not
    invent. Anything else is a brief to expand.
    """
    detailed = len(facts.attribute_quotes) >= LITERAL_ATTRIBUTES
    if facts.medium_quote or facts.minimal_quote:
        return (
            ("art-medium-restraint", "art-literal-spec") if detailed else ("art-medium-restraint",)
        )
    if detailed:
        return ("art-literal-spec",)
    return ("art-brief-expansion", "art-genre-vocab")


def route_follow_up(message: str) -> tuple[str, ...]:
    """The case skills for a message that follows an image already drawn."""
    if asks_for_more_detail(message):
        return ("art-iterative-edit", "art-genre-vocab")
    return ("art-iterative-edit",)


def case_skill_section(names: Sequence[str], texts: Mapping[str, str]) -> str:
    """The turn-context section carrying the named case skills, or `""` for none.

    `texts` is `case_skill_texts` of the agent's registry; a name it lacks is skipped.
    """
    names = [name for name in names if name in texts]
    if not names:
        return ""
    parts = [
        CASE_SKILL_HEADER,
        "Guidance the runtime picked for the latest message. What the person explicitly "
        "asked for always wins over it.",
    ]
    parts.extend(f"[{name}]\n{texts[name]}" for name in names)
    return "\n\n".join(parts)
