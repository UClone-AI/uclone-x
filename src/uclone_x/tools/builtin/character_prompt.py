"""One prompt for one or more characters, from their sheets (P9).

Kept apart from `character.py` so a caller with no story loaded can compose: `show_self`
(clone-self-and-scenes §4.2, #2017) draws the clone with the same composer as a story
character, and importing the agent package must not load the story package (#2012).
"""

from __future__ import annotations

from typing import Any


def compose_character_prompt(
    loaded_chars: list[dict[str, Any]], scene_context: str | None
) -> dict[str, Any]:
    """One prompt for the characters in `loaded_chars`, each a sheet as `get` returns it."""
    # Derive Root Subject Tag
    females = sum(1 for c in loaded_chars if c.get("gender") == "female")
    males = sum(1 for c in loaded_chars if c.get("gender") == "male")
    others = len(loaded_chars) - females - males

    root_tags: list[str] = []
    if females > 0 and males == 0 and others == 0:
        root_tags.append("1girl" if females == 1 else f"{females}girls")
    elif males > 0 and females == 0 and others == 0:
        root_tags.append("1boy" if males == 1 else f"{males}boys")
    elif females > 0 and males > 0:
        f_tag = "1girl" if females == 1 else f"{females}girls"
        m_tag = "1boy" if males == 1 else f"{males}boys"
        root_tags.extend([f_tag, m_tag])
    else:
        root_tags.append(f"{len(loaded_chars)}people")

    if len(loaded_chars) == 1:
        root_tags.append("solo")

    # Compose Danbooru tags per character with attribute segregation
    char_danbooru_blocks: list[str] = []
    for c in loaded_chars:
        ctags = c.get("danbooru_tags", "").strip().rstrip(",")
        if ctags:
            char_danbooru_blocks.append(f"{ctags}")

    # Combine prompt
    prompt_parts: list[str] = [", ".join(root_tags)]
    if char_danbooru_blocks:
        prompt_parts.append(", ".join(char_danbooru_blocks))

    if scene_context:
        prompt_parts.append(scene_context.strip())

    # Quality tags
    prompt_parts.append("masterpiece, newest, high quality, cinematic lighting")
    composed_danbooru_prompt = ", ".join(p for p in prompt_parts if p)

    # Collect negative tags
    neg_set: set[str] = set()
    for c in loaded_chars:
        cneg = c.get("negative_tags", "")
        if cneg:
            for t in cneg.split(","):
                cleaned_t = t.strip()
                if cleaned_t:
                    neg_set.add(cleaned_t)

    base_neg = "worst quality, bad anatomy, deformed, bad hands, animal, blurry, text, watermark"
    for t in base_neg.split(","):
        neg_set.add(t.strip())

    composed_negative = ", ".join(sorted(neg_set))

    # Recommendation for aspect ratio
    rec_aspect = "16:9" if len(loaded_chars) >= 2 else "3:4"

    return {
        "status": "success",
        "action": "compose",
        "characters": loaded_chars,
        "composed_danbooru_prompt": composed_danbooru_prompt,
        "composed_negative_prompt": composed_negative,
        "recommended_aspect_ratio": rec_aspect,
        "seeds": [c.get("base_seed") for c in loaded_chars if c.get("base_seed") is not None],
        "guidance": (
            "Multi-character prompt composed. To avoid attribute bleeding in Danbooru/SDXL models, "
            "ensure distinct character features and avoid conflicting color keywords. "
            f"Recommended aspect ratio: {rec_aspect}."
        ),
    }
