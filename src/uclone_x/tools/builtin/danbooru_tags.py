"""Deterministic Danbooru tag normalization, run at call time before the fill.

A `danbooru` prompt written by a small model carries tags that are almost right:
`golden hair` for `blonde hair`, `twin tail` for `twintails`, a `no text` that the image
model reads as *text*, and the same tag twice. `normalize_danbooru_tags` fixes the form of
those tags and nothing else, with no model call:

1. A tag is rewritten only when it is a known near-miss of the same meaning, from the
   curated `NEAR_MISS_TAGS` map below. Matching is on the whole tag, ignoring case, with
   `_` read as a space. An unknown tag passes unchanged.
2. A weighted or bracketed tag (`(golden hair:1.2)`) is the writer's own syntax and is
   left as written.
3. A `no X` or `without X` tag leaves the prompt and `X` goes to the negative prompt,
   because a positive `no text` draws text. Real Danbooru tags that begin with `no`
   (`no humans`, `no shoes`, ...) are in `REAL_NO_TAGS` and stay. When the model ignores
   negative prompts, nothing moves.
4. A negative prompt that was given is only appended to, never rewritten.
5. Exact repeats are removed, keeping the first.
6. Every change is one plain sentence (P6).

The map is small and curated in the repository on purpose: validating every tag against
the full Danbooru tag list needs a downloaded dataset, and is a separate step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from uclone_x.tools.builtin.media_registry import (
    ModelProfile,
    PromptFamily,
    PromptFill,
    fill_prompt_defaults,
    names_term,
)

_COLORS = (
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
    "aqua",
)

#: Near-miss spelling -> the Danbooru tag of the same meaning. Keys are lower case with
#: spaces for `_`. Only form changes belong here: an entry that would change what is
#: drawn (a different colour, a dropped side or count) does not.
NEAR_MISS_TAGS: dict[str, str] = {
    "golden hair": "blonde hair",
    "gold hair": "blonde hair",
    "blond hair": "blonde hair",
    "gray hair": "grey hair",
    "gray eyes": "grey eyes",
    "twin tail": "twintails",
    "twin tails": "twintails",
    "twin-tails": "twintails",
    "twintail": "twintails",
    "pony tail": "ponytail",
    "side pony tail": "side ponytail",
    "bob hair": "bob cut",
    "bob haircut": "bob cut",
    "long hairs": "long hair",
    "short hairs": "short hair",
    "1man": "1boy",
    "1 man": "1boy",
    "one man": "1boy",
    "1 boy": "1boy",
    "1woman": "1girl",
    "1 woman": "1girl",
    "one woman": "1girl",
    "1 girl": "1girl",
    "dot under eye": "mole under eye",
    "cat ear": "cat ears",
    "animal ear": "animal ears",
    "looking at the viewer": "looking at viewer",
    "smiling": "smile",
    "full-body": "full body",
    "upper-body": "upper body",
    "cowboy-shot": "cowboy shot",
    **{f"{color} eye": f"{color} eyes" for color in _COLORS},
}

#: Danbooru tags that begin with `no` and mean what they say: they stay in the prompt.
REAL_NO_TAGS = frozenset(
    {
        "no humans",
        "no pupils",
        "no nose",
        "no mouth",
        "no eyes",
        "no sclera",
        "no headwear",
        "no shoes",
        "no socks",
        "no legwear",
        "no gloves",
    }
)

_NEGATION = re.compile(r"^(?:no|without)\s+(\S.*)$", re.IGNORECASE)


def _key(tag: str) -> str:
    """How a tag is compared: lower case, `_` read as a space, spaces collapsed."""
    return " ".join(tag.replace("_", " ").lower().split())


def _is_weighted(tag: str) -> bool:
    return any(c in tag for c in "()[]{}")


@dataclass(frozen=True)
class TagNormalization:
    """A normalized prompt, the terms it moved to the negative, and what changed."""

    prompt: str
    moved_to_negative: tuple[str, ...]
    changes: tuple[str, ...]


def normalize_danbooru_tags(prompt: str, *, move_negations: bool = True) -> TagNormalization:
    """Fix the form of ``prompt``'s tags by the rules in this module's docstring.

    The prompt is returned as given when nothing changed. When something did, the kept
    tags are joined with `, ` in their original order.
    """
    tags = [t.strip() for line in prompt.splitlines() for t in line.split(",")]
    tags = [t for t in tags if t]
    kept: list[str] = []
    seen: set[str] = set()
    moved: list[str] = []
    changes: list[str] = []
    for tag in tags:
        weighted = _is_weighted(tag)
        current = tag if weighted else NEAR_MISS_TAGS.get(_key(tag), tag)
        negation = None if weighted else _NEGATION.match(current.replace("_", " "))
        if move_negations and negation and _key(current) not in REAL_NO_TAGS:
            term = negation.group(1).strip()
            moved.append(term)
            changes.append(f"Moved '{tag}' from the prompt to the negative prompt as '{term}'.")
            continue
        seen_key = current if weighted else _key(current)
        if seen_key in seen:
            changes.append(f"Removed a repeated tag: '{tag}'.")
            continue
        if current != tag:
            changes.append(f"Changed the tag '{tag}' to the Danbooru tag '{current}'.")
        seen.add(seen_key)
        kept.append(current)
    if not changes:
        return TagNormalization(prompt, (), ())
    return TagNormalization(", ".join(kept), tuple(moved), tuple(changes))


def prepare_prompt(prompt: str, negative_prompt: str, profile: ModelProfile) -> PromptFill:
    """Normalize a `danbooru` prompt's tags, then fill what it left open.

    `fill_prompt_defaults` runs on the normalized prompt with the negative as it was
    given, so an empty negative still gets the profile's default. Terms moved out of the
    prompt are then appended to the resulting negative, skipping any it already names;
    its own text is never rewritten. Other families go straight to the fill.
    """
    if profile.family != PromptFamily.DANBOORU:
        return fill_prompt_defaults(prompt, negative_prompt, profile)
    normal = normalize_danbooru_tags(prompt, move_negations=not profile.suppress_negative)
    fill = fill_prompt_defaults(normal.prompt, negative_prompt, profile)
    negative = fill.negative_prompt
    extra: list[str] = []
    for term in normal.moved_to_negative:
        if negative and names_term(negative, term):
            continue
        negative = ", ".join(t for t in (negative, term) if t)
        extra.append(term)
    changes = list(normal.changes) + list(fill.changes)
    if extra and fill.negative_prompt != negative:
        changes.append(f"Added to the negative prompt: {', '.join(extra)}.")
    return PromptFill(fill.prompt, negative, tuple(changes))
