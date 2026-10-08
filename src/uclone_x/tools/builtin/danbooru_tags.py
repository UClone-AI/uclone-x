"""Deterministic Danbooru tag normalization, run at call time before the fill.

A `danbooru` prompt written by a small model carries tags that are almost right:
`golden hair` for `blonde hair`, `twin tail` for `twintails`, a `no text` that the image
model reads as *text*, and the same tag twice. `normalize_danbooru_tags` fixes the form of
those tags and nothing else, with no model call:

1. A tag is rewritten only when it is a known near-miss of the same meaning, from the
   curated `NEAR_MISS_TAGS` map below. Matching is on the whole tag, ignoring case, with
   `_` read as a space. Known Korean tags are translated into standard Danbooru tags.
   An untranslated tag containing non-Latin script is dropped; an unknown tag in Latin
   script passes unchanged (#1865).
2. A weighted or bracketed tag (`(golden hair:1.2)`) is the writer's own syntax and is
   left as written, except that a weighted tag containing non-Latin script has its inner
   term translated if known or dropped if untranslated (#1865).
3. A `no X` or `without X` tag leaves the prompt and `X` goes to the negative prompt,
   because a positive `no text` draws text. Real Danbooru tags that begin with `no`
   (`no humans`, `no shoes`, ...) are in `REAL_NO_TAGS` and stay. When the model ignores
   negative prompts, nothing moves.
4. A negative prompt that was given is only appended to, never rewritten.
5. Exact repeats are removed, keeping the first.
6. Every change is one plain sentence (P6).

The map is small and curated in the repository on purpose: validating every tag against
the full Danbooru tag list needs that list checked in, and Danbooru publishes no licence
for its tag data. Its terms of service (https://danbooru.donmai.us/terms_of_service) grant
a licence only in the other direction, from uploaders to Danbooru for what they submit
(#1865). The entries that are here cite where they were read.
"""

from __future__ import annotations

import re
import unicodedata
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
    # Danbooru's own active aliases of `no ...` tags (#1865), read 2026-09-28 from
    # https://danbooru.donmai.us/tag_aliases.json?search[consequent_name_matches]=no_*&search[status]=active
    # and, for the headwear and legwear ones, search[antecedent_name]=no_hat / no_headwear /
    # no_legwear. The alias is the site's ruling that the two names are one tag. `no hat`
    # and `no headwear` have no posts of their own; the image model learned the tag they
    # point to. Aliases onto tags about bodies or underwear are left out on purpose.
    "no hat": "missing headwear",
    "no headwear": "missing headwear",
    "no legwear": "missing legwear",
    "no glasses": "no eyewear",
    "no capelet": "no cape",
    "no cloak": "no cape",
    "no horn": "no horns",
    "no human": "no humans",
    "no line-art": "no lineart",
    "lineless": "no lineart",
    "no sign": "no symbol",
    # Korean translations for common Danbooru tags (#1865).
    "금발": "blonde hair",
    "은발": "silver hair",
    "백발": "white hair",
    "흑발": "black hair",
    "갈색 머리": "brown hair",
    "갈색머리": "brown hair",
    "빨간 머리": "red hair",
    "빨간머리": "red hair",
    "적발": "red hair",
    "분홍 머리": "pink hair",
    "분홍머리": "pink hair",
    "파란 머리": "blue hair",
    "파란머리": "blue hair",
    "청발": "blue hair",
    "초록 머리": "green hair",
    "초록머리": "green hair",
    "녹발": "green hair",
    "단발": "short hair",
    "단발머리": "short hair",
    "장발": "long hair",
    "트윈테일": "twintails",
    "트윈 테일": "twintails",
    "포니테일": "ponytail",
    "포니 테일": "ponytail",
    "생머리": "straight hair",
    "파란 눈": "blue eyes",
    "파란눈": "blue eyes",
    "벽안": "blue eyes",
    "빨간 눈": "red eyes",
    "빨간눈": "red eyes",
    "적안": "red eyes",
    "초록 눈": "green eyes",
    "초록눈": "green eyes",
    "녹안": "green eyes",
    "갈색 눈": "brown eyes",
    "갈색눈": "brown eyes",
    "검은 눈": "black eyes",
    "검은눈": "black eyes",
    "흑안": "black eyes",
    "금색 눈": "yellow eyes",
    "금안": "yellow eyes",
    "보라색 눈": "purple eyes",
    "자안": "purple eyes",
    "소녀": "1girl",
    "소년": "1boy",
    "여자": "1girl",
    "여인": "1girl",
    "남자": "1boy",
    "솔로": "solo",
    "메이드복": "maid outfit",
    "흑백 메이드복": "black and white maid outfit",
    "교복": "school uniform",
    "세일러복": "sailor suit",
    "기모노": "kimono",
    "한복": "hanbok",
    "수영복": "swimsuit",
    "비키니": "bikini",
    "드레스": "dress",
    "정장": "suit",
    "안경": "glasses",
    "모자": "hat",
    "리본": "ribbon",
    "고양이 귀": "cat ears",
    "고양이귀": "cat ears",
    "토끼 귀": "rabbit ears",
    "토끼귀": "rabbit ears",
    "동물 귀": "animal ears",
    "동물귀": "animal ears",
    "날개": "wings",
    "꼬리": "tail",
    "흰 배경": "white background",
    "하얀 배경": "white background",
    "단색 배경": "simple background",
    "전신": "full body",
    "상반신": "upper body",
    "미소": "smile",
    "스탠딩": "standing",
}

#: Danbooru tags that begin with `no` and mean what they say: they stay in the prompt.
#: Each is a live general tag with posts, read 2026-09-28 from
#: https://danbooru.donmai.us/tags.json?search[name_matches]=no_*&search[category]=0&search[order]=count
#: (#1865). Tags about bodies or underwear are left out on purpose; one of those written as
#: `no X` still moves X to the negative prompt, which is the safer reading.
REAL_NO_TAGS = frozenset(
    {
        "no humans",
        "no pupils",
        "no nose",
        "no mouth",
        "no eyes",
        "no sclera",
        "no shoes",
        "no socks",
        "no gloves",
        "no lineart",
        "no jacket",
        "no eyewear",
        "no eyebrows",
        "no mask",
        "no blindfold",
        "no eyepatch",
        "no mole",
        "no hair ornament",
        "no symbol",
        "no coat",
        "no armor",
        "no cape",
        "no hairband",
        "no detached sleeves",
        "no hair bow",
        "no emblem",
        "no horns",
        "no print",
        "no scar",
        "no headgear",
        "no earrings",
        "no ahoge",
        "no animal ears",
        "no hands",
        "no choker",
        "no ears",
        "no cardigan",
        "no scarf",
        "no vest",
        "no tattoo",
        "no neckwear",
    }
)

_NEGATION = re.compile(r"^(?:no|without)\s+(\S.*)$", re.IGNORECASE)
_WEIGHT_RE = re.compile(r"^([(\[{]+)(.*?)(:[0-9.]+)?([)\]}]+)$")


def _has_non_latin(text: str) -> bool:
    """Whether ``text`` contains any letter outside the Latin alphabet."""
    return any(
        unicodedata.category(ch).startswith("L")
        and not unicodedata.name(ch, "").startswith("LATIN")
        for ch in text
    )


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
        if weighted and _has_non_latin(tag):
            m = _WEIGHT_RE.match(tag)
            if m:
                prefix, inner, weight, suffix = m.groups()
                inner_trans = NEAR_MISS_TAGS.get(_key(inner.strip()))
                if inner_trans and not _has_non_latin(inner_trans):
                    current = f"{prefix}{inner_trans}{weight or ''}{suffix}"
        if _has_non_latin(current):
            changes.append(f"Dropped a tag that is not Latin script: '{tag}'.")
            continue
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
            if _has_non_latin(tag):
                changes.append(f"Translated the tag '{tag}' to the Danbooru tag '{current}'.")
            else:
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
