"""Image-set planning: N varied images from one request, one prompt per image.

Asked for "5 different scenes", a local model calls `generate_image(prompt=..., count=5)`:
one prompt, five seeds, five near-identical pictures. Measured on qwen3:8b, the fix
that held on every run was to have the model fill a small JSON plan under a schema --
what the images share and what differs per image -- and to build the prompts in code.

Three parts, each usable alone:

* `detect_image_set` -- a deterministic reading of the user's message: does it ask for
  N (2..10) *varied* images? It does not fire for seed variations of one picture.
* `plan_image_set` -- one structured-output call (`LLMRequest.response_schema`), plus
  one repair turn when entries are not tags, then `assemble_prompts`.
* `image_set_note` -- the turn-context section that hands the prompts to the model.

The caller decides what a failure costs. `BaseAgent` treats every failure here as "no
plan": the turn runs exactly as it would have without this module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uclone_x.errors import UCloneXError
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole, ModelResponse
from uclone_x.tools.builtin.image_set_intent import (
    MAX_IMAGES,
    detect_image_set,
    wants_varied_locations,
)

__all__ = [
    "IMAGE_SET_PERSONAS",
    "MAX_IMAGES",
    "QUALITY_TAGS",
    "ImageSetPlan",
    "ImageSetPlanError",
    "ImageVariant",
    "assemble_prompts",
    "bad_entries",
    "detect_image_set",
    "image_set_note",
    "plan_image_set",
    "plan_schema",
    "repeated_locations",
]

IMAGE_SET_PERSONAS = frozenset({"artist"})
"""Personas whose turns are planned. Only the one the measurement covered."""

QUALITY_TAGS = ("masterpiece", "best quality", "newest")
"""Appended to every assembled prompt, last: the first tags cut when CLIP truncates."""


class ImageVariant(BaseModel):
    """The tags one image has that the others do not."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    subject: tuple[str, ...] = ()
    action: tuple[str, ...] = ()
    expression: tuple[str, ...] = ()
    location: tuple[str, ...] = ()
    camera: tuple[str, ...] = ()

    def tags(self) -> tuple[str, ...]:
        return (*self.subject, *self.action, *self.expression, *self.location, *self.camera)


class ImageSetPlan(BaseModel):
    """What the model filled in: shared tags, style tags, and one variant per image."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    shared: tuple[str, ...] = ()
    style: tuple[str, ...] = ()
    variants: tuple[ImageVariant, ...] = Field(default=())


class ImageSetPlanError(UCloneXError):
    """No usable plan came back. The message is for logs, not for the person."""


def plan_schema(count: int) -> dict[str, Any]:
    """The JSON Schema the planning call's reply must satisfy, for `count` images."""
    tags: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    fields = ("subject", "action", "expression", "location", "camera")
    variant = {
        "type": "object",
        "properties": dict.fromkeys(fields, tags),
        "required": list(fields),
    }
    return {
        "type": "object",
        "properties": {
            "shared": tags,
            "style": tags,
            "variants": {
                "type": "array",
                "items": variant,
                "minItems": count,
                "maxItems": count,
            },
        },
        "required": ["shared", "style", "variants"],
    }


_PLAN_INSTRUCTIONS = """\
You plan a set of {count} images for an anime image model (Illustrious-XL) that reads \
English Danbooru tags. Fill the JSON fields with English Danbooru tags only: short \
lowercase tags, no sentences, no non-English text, no aspect ratios, no field names.

- shared: tags every image has in common. When the same character appears in every \
image, put the count tag (1girl, 1boy, solo), hair, eyes, clothing and accessories here. \
Leave it empty when every image shows a different subject.
- style: medium, art style and lighting common to all images.
- variants: exactly {count} entries, each clearly different from the others:
  - subject: who is in this image, only when the subject changes between images \
(count tag first, then hair, eyes, clothing); otherwise empty.
  - action: pose or action.
  - expression: facial expression.
  - location: background. When the user fixed the location, repeat the same tags in \
every entry.
  - camera: framing and angle (full body, upper body, from above, close-up, ...).

Keep every detail the user wrote. Invent specific details for anything missing."""

_VARIED_LOCATIONS = """

The user asked for different scenes. Every variant needs its own location: a different \
place in each entry, chosen to suit the subject and the request. Put no background tags \
in shared or style."""

_REPAIR = (
    "These entries are not valid English Danbooru tags: {entries}. Return the same JSON "
    "with every entry rewritten as a short English Danbooru tag (e.g. 'black hair', "
    "'mole under eye', 'cherry blossoms'), and exactly {count} variants. Keep all meanings."
)

_SAME_LOCATION_REPAIR = (
    "Every variant has to be a different place, and some share one. Give each variant its "
    "own location tags, all different, and move background tags out of shared and style."
)

_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_LABEL_SHAPED = re.compile(
    r"^(?:hair|eyes|outfit|prop|root|expression|key_object|subject|action|location|camera"
    r"|style|shared)_|^[a-z_ ]+:\s|^\d+:\d+$",
    re.I,
)
_ROOT_TAG = re.compile(r"^(?:\d\+?(?:girl|boy|other)s?|solo|no humans|multiple (?:girls|boys))$")


def _entries(plan: ImageSetPlan) -> list[str]:
    out = [*plan.shared, *plan.style]
    for variant in plan.variants:
        out.extend(variant.tags())
    return out


def bad_entries(plan: ImageSetPlan) -> list[str]:
    """Entries that are not tags: non-ASCII, field-label shaped, aspect ratios, or prose."""
    return [
        entry
        for entry in _entries(plan)
        if _NON_ASCII.search(entry.strip())
        or _LABEL_SHAPED.search(entry.strip())
        or len(entry.split()) > 4
    ]


def repeated_locations(plan: ImageSetPlan) -> bool:
    """Whether two variants of `plan` have the same location, or one has none."""
    places = [frozenset(_normal(t) for t in v.location) for v in plan.variants]
    return any(not place for place in places) or len(set(places)) != len(places)


def _without_shared_places(plan: ImageSetPlan) -> ImageSetPlan:
    """`plan` with shared and style tags that name a variant's place removed.

    Told to keep backgrounds out of shared, qwen3:8b still puts "forest" there beside
    variant locations "castle gate" and "tavern", and every image then asks for both. A
    shared tag whose words all appear in one variant location tag is taken as that place.
    """
    places = [set(_normal(t).split()) for v in plan.variants for t in v.location]

    def keep(tags: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(t for t in tags if not any(set(_normal(t).split()) <= p for p in places))

    return plan.model_copy(update={"shared": keep(plan.shared), "style": keep(plan.style)})


def _normal(tag: str) -> str:
    # Danbooru spells tags with underscores; the prompt wants the spaced form the
    # text encoders were trained on, and one spelling so duplicates collapse.
    return " ".join(tag.replace("_", " ").split()).lower()


def assemble_prompts(plan: ImageSetPlan) -> list[str]:
    """One comma-separated prompt per variant, differing tags first.

    CLIP reads 75 tokens and drops the rest, so the order is the priority: count tags
    (1girl, solo -- two tokens that decide the whole composition), then the tags that make
    this image different from the others, then what all images share, then style, then
    quality. A variant tag every variant carries (a fixed location repeated in each) is
    not a difference, and moves back to the shared part. Tags appear once per prompt.
    """
    variant_sets = [{_normal(t) for t in v.tags()} for v in plan.variants]
    common: set[str] = variant_sets[0].intersection(*variant_sets[1:]) if variant_sets else set()
    prompts: list[str] = []
    for variant in plan.variants:
        own = [t for t in variant.tags() if _normal(t) not in common]
        repeated = [t for t in variant.tags() if _normal(t) in common]
        ordered = [*own, *repeated, *plan.shared, *plan.style, *QUALITY_TAGS]
        roots = [t for t in ordered if _ROOT_TAG.match(_normal(t))]
        seen: set[str] = set()
        tags: list[str] = []
        for tag in (*roots, *ordered):
            key = _normal(tag)
            if key and key not in seen:
                seen.add(key)
                tags.append(key)
        prompts.append(", ".join(tags))
    return prompts


def _parse(response: ModelResponse) -> ImageSetPlan:
    try:
        return ImageSetPlan.model_validate(json.loads(response.content or ""))
    except (ValueError, ValidationError) as exc:
        raise ImageSetPlanError(f"planning reply is not a plan: {type(exc).__name__}") from exc


def _strip_non_ascii(plan: ImageSetPlan) -> ImageSetPlan:
    def keep(tags: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(t for t in tags if not _NON_ASCII.search(t))

    return ImageSetPlan(
        shared=keep(plan.shared),
        style=keep(plan.style),
        variants=tuple(
            ImageVariant(
                subject=keep(v.subject),
                action=keep(v.action),
                expression=keep(v.expression),
                location=keep(v.location),
                camera=keep(v.camera),
            )
            for v in plan.variants
        ),
    )


def _with_earlier(message: str, earlier: str) -> str:
    """The request, preceded by the recent conversation when there is one.

    "철수 다양한 포즈 5장" names a character whose hair and eyes were settled earlier (or
    read from a character sheet); planned from the message alone, the set would invent
    new ones and the note would tell the model to draw those.
    """
    if not earlier.strip():
        return message
    return (
        "Earlier in this conversation (keep the details it settles about characters and "
        "places; plan only what the request below asks for):\n"
        f"{earlier.strip()}\n\nRequest: {message}"
    )


async def plan_image_set(
    generate: Callable[[LLMRequest], Awaitable[ModelResponse]],
    message: str,
    count: int,
    *,
    model: str | None = None,
    context_window: int | None = None,
    earlier: str = "",
) -> list[str]:
    """Plan `count` images for `message` and return one prompt per image.

    `generate` is the model call -- the agent's own, so the token budget is charged. At
    most two calls: the plan, and one repair turn when the plan has entries that are not
    tags or the wrong number of variants. Entries still non-ASCII after the repair are
    dropped; a prompt holding them would reach the image model as mojibake.

    Raises:
        ImageSetPlanError: the reply is not a plan, or still has the wrong number of
            variants after the repair.
        StructuredOutputUnsupportedError: the connector cannot take a schema.
        Any provider error from `generate`, unchanged.
    """
    count = min(count, MAX_IMAGES)
    vary_locations = wants_varied_locations(message)
    schema = plan_schema(count)
    messages: list[ChatMessage] = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=_PLAN_INSTRUCTIONS.format(count=count)
            + (_VARIED_LOCATIONS if vary_locations else ""),
        ),
        ChatMessage(role=MessageRole.USER, content=_with_earlier(message, earlier)),
    ]

    def request() -> LLMRequest:
        return LLMRequest(
            model=model,
            messages=tuple(messages),
            temperature=0.7,
            max_tokens=1500,
            thinking=False,
            response_schema=schema,
            context_window=context_window,
            auto_compact=False,
        )

    first = await generate(request())
    try:
        plan = _parse(first)
    except ImageSetPlanError:
        # A small model sometimes breaks the JSON once (seen live on qwen3:8b); one
        # fresh attempt, not a repair turn, since there is no plan to point at.
        second = await generate(request())
        plan = _parse(second)
        first = second
    bad = bad_entries(plan)
    same_place = vary_locations and repeated_locations(plan)
    if bad or same_place or len(plan.variants) != count:
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=first.content or ""))
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=" ".join(
                    part
                    for part in (
                        _REPAIR.format(
                            entries=json.dumps(bad[:30], ensure_ascii=False), count=count
                        )
                        if bad or len(plan.variants) != count
                        else "",
                        _SAME_LOCATION_REPAIR if same_place else "",
                    )
                    if part
                ),
            )
        )
        plan = _parse(await generate(request()))
    if len(plan.variants) != count:
        raise ImageSetPlanError(f"plan has {len(plan.variants)} variants, {count} were asked")
    plan = _strip_non_ascii(plan)
    if vary_locations:
        plan = _without_shared_places(plan)
    prompts = assemble_prompts(plan)
    if len(set(prompts)) != len(prompts):
        raise ImageSetPlanError("plan has variants that do not differ")
    return prompts


def image_set_note(prompts: Sequence[str]) -> str:
    """The turn-context section that hands the planned prompts to the model."""
    listing = "\n".join(f"{i}. {p}" for i, p in enumerate(prompts, 1))
    return (
        "[Image Set Plan]\n"
        f"The user asked for {len(prompts)} different images, and the runtime planned one "
        f"prompt per image:\n{listing}\n"
        f"Call generate_image once with prompts set to exactly these {len(prompts)} prompts, "
        "in this order. Do not use count. You may set negative_prompt, aspect_ratio and "
        "style.\n"
        f"prompts={json.dumps(list(prompts), ensure_ascii=False)}"
    )
