"""Case-skill routing for the Artist: which drawing guidance fits this message.

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

The skill texts are written so the model has nothing to paste: no slot labels, no
`no X` phrase (the only `no ...` string is the real tag `no humans`), no quality tags
(the call-time fill appends them), and each culture's traditional vocabulary on its own
line, because one mixed line had the model dress a kimono request in hanbok.

The caller decides what a failure costs. `BaseAgent` treats any failure as "no skill":
the turn runs as it would have without this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from uclone_x.errors import UCloneXError
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ModelResponse

__all__ = [
    "CASE_SKILLS",
    "CASE_SKILL_PERSONAS",
    "CaseSkillExtractionError",
    "RequestFacts",
    "asks_for_more_detail",
    "case_skill_section",
    "extract_request_facts",
    "grounded_facts",
    "is_follow_up",
    "route_first_turn",
    "route_follow_up",
]

CASE_SKILL_PERSONAS = frozenset({"artist"})
"""Personas whose turns are routed. Only the one the measurement covered."""

CASE_SKILL_HEADER = "[Artist Case Skills]"

#: Grounded attributes at or above this count make a request a detailed specification.
LITERAL_ATTRIBUTES = 4

_BRIEF_EXPANSION = """\
If the latest message asks for an image and names only a subject or a place, you are the \
art director, so decide every visual detail the person left open. A prompt that only restates \
their words is too thin. What they did write stays exactly as written.
Step 1 - Keep the subject and every detail the person gave. The subject's defining object \
appears as a tag. A knight wears armor and holds a weapon, a hacker has a laptop or \
holographic screens, a chef holds a knife or pan, a mage holds a staff or book. Choose the \
root tag from the subject. A woman or girl is 1girl. A man, boy or old man is 1boy, and an \
old man also gets old man. An animal alone is the animal tag plus no humans. A landscape \
with nobody in it is no humans, scenery.
Step 2 - Then decide, in this order. The root, and solo for one person. Hair colour, length \
and style. Eyes and expression. Three or more clothing items with colour or material, plus \
the role's prop. One pose. Shot size and camera angle (full body, cowboy shot, upper body, \
portrait; from below, from side, from above). Four or more objects that belong to the \
place. Time, weather and light source (sunset, overcast, lantern light, rim light, \
backlighting). Colour palette and mood (warm tones, muted colors, serene).
Step 3 - Vary your choices between images, and make them agree with each other, so warm \
lighting goes with a warm palette.
Write 25-40 real Danbooru tags, comma-separated, tags only. Put anything to leave out in \
negative_prompt as plain tags. Set the shape with aspect_ratio, never in the prompt. Use \
3:4 for a standing figure, 16:9 for wide scenery or action, 1:1 for a portrait.
After the image, tell the person in one sentence which details you chose, so they can \
change any of them."""

_LITERAL_SPEC = """\
If the latest message asks for an image and already describes it in detail, translate it \
exactly and invent nothing.
Step 1 - Turn every phrase the person wrote into a tag and keep all of them. Colour words \
are exact.
은발 = silver hair, 백발 = white hair, 금발 = blonde hair, 흑발 = black hair, 갈색 머리 = brown \
hair, 빨간 머리 = red hair, 분홍 머리 = pink hair
단발 = short hair, bob cut; 장발 = long hair; 트윈테일 = twintails; 포니테일 = ponytail
붉은 눈 = red eyes, 파란 눈 = blue eyes, 초록 눈 = green eyes, 금색 눈 = yellow eyes
Never swap a colour for a nearby one. Silver is not blonde.
Step 2 - Choose the root from the subject. 소녀, 여자 and 여인 are 1girl. 소년, 남자, 아저씨, \
노인 and other men are 1boy, and 노인 adds old man, 수염 adds beard. Never write 1girl for a man.
Step 3 - Add only what was left open and cannot conflict, such as shot size and camera angle, and \
one light tag if the person gave none. Clothing, props, people, colours and background \
objects stay exactly as the person wrote them, with nothing added. A plain or white \
background stays plain.
Step 4 - Order the tags as root, character, clothing, pose, framing, place, light.
Real Danbooru tags only, comma-separated. Set the shape with aspect_ratio, never in the \
prompt."""

_MEDIUM_RESTRAINT = """\
If the latest message asks for an image in a named medium, or asks for something simple or \
for something to be left out, respect that before anything else.
Use these medium tags as written here.
흑백, 모노크롬 = monochrome, greyscale
잉크 = ink (medium), traditional media
스케치 = sketch; 연필 = sketch, graphite (medium)
수채화 = watercolor (medium), traditional media
유화 = oil painting (medium)
선화 = lineart
When the person asks for something simple, an empty background or one thing only \
(단순하게, 배경 없이, 하나만), use simple background or white background and add nothing \
beyond what they named. Extra people, props, lighting effects, colours and mood tags stay \
out.
A monochrome or sketch request also leaves out colour names, glow and cinematic lighting.
What is left out goes into negative_prompt as plain tags, for example color, extra people, \
detailed background. The prompt itself lists only what is drawn.
The root still follows the subject. 노인 어부 is 1boy, old man; 고양이 한 마리 is cat, no \
humans. Use 1girl only for a female subject.
Keep the prompt short, 8-15 tags, in this order. The root, the details the person gave, \
the action, the medium tags, the background tag."""

_ITERATIVE_EDIT = """\
If the latest message changes an image you already made in this conversation, start from \
your previous prompt. Keep the character's identity tags (hair, eyes, body, signature \
outfit) unless the person changes them. Decide which kind of change it is, then rewrite the \
prompt rather than appending a word.
- A new place or scene (카페 장면, 공장에서). Replace the old place, pose, framing and \
lighting. Build the new place from four or more concrete objects, with a pose, light \
source and mood that fit it, and remove the old place's tags.
- A time or weather change (밤으로, 비 오게). Change the time tag and make every light and \
palette tag agree, for example night, dark sky, moonlight or street lamps, cool tones, and \
remove sunlight and golden hour.
- A style or medium change (수채화 느낌). Add the medium tags, for example watercolor \
(medium), traditional media, and a fitting palette, and keep the content.
- A request for more detail (디테일 채워, 더 화려하게, 더 자세히). Enrich every part the \
person left open, with clothing materials and accessories, four or more objects in the \
place, a precise light source, atmosphere (dust, light rays, steam, rain), depth of field \
and a palette, 30-40 tags in all. Name actual objects, since generic filler such as \
detailed background adds nothing.
- One attribute (눈은 초록). Change that tag only.
After the image, say in one sentence what you changed."""

_GENRE_VOCAB = """\
Tags to draw on when the latest message asks for an image. Pick a few that fit, never all \
of them, and vary them between images. Each culture's clothing and places stay on their \
own line and are never mixed.
- Fantasy. castle ruins, floating island, library, runes, crystal, magic circle, glowing \
particles, cloak, circlet, gauntlets, spellbook, staff, dragon, torchlight
- Japanese traditional (기모노, 사무라이, 신사). kimono, japanese clothes, obi, yukata, haori, \
hakama, katana, sheathed sword, kanzashi, paper lantern, torii, shrine, tatami, cherry \
blossoms, falling petals
- Korean traditional (한복, 사극, 한옥). hanbok, korean clothes, norigae, hair stick, east asian \
architecture, tiled roof, stone wall, pine tree, mist
- Chinese traditional (무협, 치파오, 한푸). hanfu, chinese clothes, china dress, chinese \
architecture, pagoda, red lantern, bamboo forest, lotus, mist
- Sci-fi and cyberpunk. neon lights, holographic interface, cyborg, visor, techwear, hooded \
jacket, glowing circuits, cityscape, skyscraper, wet street, monitor, cable, drone, \
spacecraft interior
- Daily life (카페, 학교, 집). cafe, wooden table, cup, latte art, window, plant, bookshelf, \
classroom, desk, chalkboard, bedroom, bed, curtains, sunlight
- Scenery. mountain, snow, pine tree, cabin, chimney, lake, reflection, cloud, starry sky, \
aurora, sunset, horizon, waves, beach, field, flower field
- Action. motion blur, speed lines, dynamic angle, from below, debris, sparks, fire, wind, \
floating hair"""

CASE_SKILLS: dict[str, str] = {
    "art-brief-expansion": _BRIEF_EXPANSION,
    "art-literal-spec": _LITERAL_SPEC,
    "art-medium-restraint": _MEDIUM_RESTRAINT,
    "art-iterative-edit": _ITERATIVE_EDIT,
    "art-genre-vocab": _GENRE_VOCAB,
}
"""The case skills by name. Each text is written to be read, not pasted."""

_MORE_DETAIL = re.compile(
    r"디테일|화려|자세히|풍부|더 많이|\bmore detail|\bdetailed\b|\bricher\b|\belaborate",
    re.IGNORECASE,
)


def asks_for_more_detail(message: str) -> bool:
    """Whether a follow-up asks for a richer picture (then genre vocabulary helps)."""
    return bool(_MORE_DETAIL.search(message))


def is_follow_up(history: Sequence[ChatMessage]) -> bool:
    """Whether this conversation already holds an image the Artist drew.

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


def case_skill_section(names: Sequence[str]) -> str:
    """The turn-context section carrying the named case skills, or `""` for none."""
    if not names:
        return ""
    parts = [
        CASE_SKILL_HEADER,
        "Guidance the runtime picked for the latest message. What the person explicitly "
        "asked for always wins over it.",
    ]
    parts.extend(f"[{name}]\n{CASE_SKILLS[name]}" for name in names)
    return "\n\n".join(parts)
