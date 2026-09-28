"""Danbooru tag normalization at call time: form only, and every change said (#1828).

One test per rule in `danbooru_tags`'s docstring, then the composition with
`fill_prompt_defaults` and the dispatcher that runs it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from uclone_x.tools.builtin.danbooru_tags import (
    normalize_danbooru_tags,
    prepare_prompt,
)
from uclone_x.tools.builtin.image import (
    GenerateImageParams,
    GenerateImageTool,
    ImageGenerationResult,
    ImagePipelineDispatcher,
)
from uclone_x.tools.builtin.media_registry import (
    ModelProfile,
    PromptFamily,
    fill_prompt_defaults,
)
from uclone_x.tools.models import ToolContext


def _danbooru(*, suppress: bool = False) -> ModelProfile:
    return ModelProfile(
        model_id="danbooru-test",
        display_name="Danbooru Test",
        family=PromptFamily.DANBOORU,
        default_negative="worst quality, text, watermark",
        suppress_negative=suppress,
    )


def test_a_near_miss_becomes_the_danbooru_tag_and_is_reported() -> None:
    """Rule 1.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::         current = tag if weighted else NEAR_MISS_TAGS.get(_key(tag), tag)
    Becomes:         current = tag
    """
    result = normalize_danbooru_tags("1girl, Golden_Hair, twin tail, smile")

    assert result.prompt == "1girl, blonde hair, twintails, smile"
    assert result.changes == (
        "Changed the tag 'Golden_Hair' to the Danbooru tag 'blonde hair'.",
        "Changed the tag 'twin tail' to the Danbooru tag 'twintails'.",
    )


def test_an_unchanged_prompt_is_returned_exactly_as_given() -> None:
    """Nothing to fix: the prompt keeps its own spacing and newlines.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::     if not changes:
    Becomes:     if False:
    """
    prompt = "1girl,  solo\nsilver hair, anime_style"
    result = normalize_danbooru_tags(prompt)

    assert result.prompt == prompt
    assert result.changes == ()
    assert result.moved_to_negative == ()


def test_a_weighted_tag_is_left_as_written() -> None:
    """Rule 2.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::     return any(c in tag for c in "()[]{}")
    Becomes:     return False
    """
    result = normalize_danbooru_tags("1girl, (golden hair:1.2), [no text], no (logo:1.3)")

    assert result.prompt == "1girl, (golden hair:1.2), [no text], no (logo:1.3)"
    assert result.changes == ()


def test_a_no_tag_moves_to_the_negative_and_a_real_no_tag_stays() -> None:
    """Rule 3.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::         if move_negations and negation and _key(current) not in REAL_NO_TAGS:
    Becomes:         if move_negations and negation:
    """
    result = normalize_danbooru_tags("cat, no humans, no text, without_signature, No_Shoes")

    assert result.prompt == "cat, no humans, No_Shoes"
    assert result.moved_to_negative == ("text", "signature")
    assert result.changes == (
        "Moved 'no text' from the prompt to the negative prompt as 'text'.",
        "Moved 'without_signature' from the prompt to the negative prompt as 'signature'.",
    )


def test_nothing_moves_when_the_model_ignores_negatives() -> None:
    """Rule 3: a moved term would be lost, so a `no X` stays where the person put it.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::     normal = normalize_danbooru_tags(prompt, move_negations=not profile.suppress_negative)
    Becomes:     normal = normalize_danbooru_tags(prompt)
    """
    fill = prepare_prompt("1girl, no text, masterpiece", "", _danbooru(suppress=True))

    assert fill.prompt == "1girl, no text, masterpiece"
    assert fill.negative_prompt == ""


def test_a_given_negative_is_only_appended_to() -> None:
    """Rule 4: the given text stays first and unchanged; a term it names is not repeated.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::         if negative and names_term(negative, term):
    Becomes:         if False:
    """
    fill = prepare_prompt("1girl, no text, no signature, masterpiece", "Text,  lowres", _danbooru())

    assert fill.negative_prompt == "Text,  lowres, signature"
    assert "Added to the negative prompt: signature." in fill.changes


def test_an_empty_negative_still_gets_the_default_then_the_moved_terms() -> None:
    """The fill sees the negative as given, so the profile default is not skipped.

    `text` is in the default: it was named by `no text`, but once moved out of the
    prompt the fill no longer strips it, and it is not added twice.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::     fill = fill_prompt_defaults(normal.prompt, negative_prompt, profile)
    Becomes:     fill = fill_prompt_defaults(prompt, negative_prompt, profile)
    """
    fill = prepare_prompt("1girl, no text, no logo", "", _danbooru())

    assert fill.prompt.startswith("1girl, masterpiece")
    assert fill.negative_prompt == "worst quality, text, watermark, logo"


def test_an_exact_repeat_is_removed_keeping_the_first() -> None:
    """Rule 5, after rule 1: `golden hair` repeats `blonde hair` once respelled.

    Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::         if seen_key in seen:
    Becomes:         if False:
    """
    result = normalize_danbooru_tags("blonde hair, long hair, Long_Hair, golden hair, smile")

    assert result.prompt == "blonde hair, long hair, smile"
    assert result.changes == (
        "Removed a repeated tag: 'Long_Hair'.",
        "Removed a repeated tag: 'golden hair'.",
    )


def test_no_tag_the_person_wrote_is_lost() -> None:
    """Form only: every tag's meaning ends in the prompt or the negative."""
    prompt = "1girl, gold hair, pony tail, no text, red eye, cafe, cafe, (smile:1.1), no humans"
    fill = prepare_prompt(prompt, "", _danbooru())
    tags = {t.strip() for t in fill.prompt.split(",")} | {
        t.strip() for t in fill.negative_prompt.split(",")
    }

    for expected in (
        "1girl",
        "blonde hair",
        "ponytail",
        "text",
        "red eyes",
        "cafe",
        "(smile:1.1)",
        "no humans",
    ):
        assert expected in tags


def test_other_families_go_straight_to_the_fill() -> None:
    """Killed by: src/uclone_x/tools/builtin/danbooru_tags.py ::     if profile.family != PromptFamily.DANBOORU:
    Becomes:     if False:
    """
    prose = ModelProfile(
        model_id="flux-test", display_name="FLUX Test", family=PromptFamily.NATURAL_PROSE
    )
    prompt = "A girl with golden hair, no text on the sign"

    assert prepare_prompt(prompt, "", prose) == fill_prompt_defaults(prompt, "", prose)


def _gen_result() -> ImageGenerationResult:
    return ImageGenerationResult(
        image_bytes=b"png",
        seed=1,
        engine_name="mock",
        device_info="test",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )


@pytest.mark.asyncio
async def test_the_engine_receives_the_normalized_prompt_and_the_result_says_why(
    tmp_path: Path,
) -> None:
    """The dispatcher's local path runs the normalization before the fill.

    Killed by: src/uclone_x/tools/builtin/image.py ::         fill = prepare_prompt(prompt, negative_prompt, profile)
    Becomes:         fill = fill_prompt_defaults(prompt, negative_prompt, profile)
    """
    engine = AsyncMock()
    engine.is_available.return_value = True
    engine.generate.return_value = _gen_result()
    dispatcher = ImagePipelineDispatcher(
        remote_engine=engine, comfy_engine=engine, local_engine=engine
    )
    profile = _danbooru()
    dispatcher.get_active_profile = lambda: profile  # type: ignore[method-assign]
    tool = GenerateImageTool(dispatcher=dispatcher)
    context = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)

    result = await tool.run(
        GenerateImageParams(prompt="1girl, golden hair, no logo, masterpiece"), context
    )

    sent = engine.generate.call_args.kwargs
    assert sent["prompt"] == "1girl, blonde hair, masterpiece"
    assert sent["negative_prompt"] == "worst quality, text, watermark, logo"
    assert (
        "Changed the tag 'golden hair' to the Danbooru tag 'blonde hair'."
        in result["prompt_changes"]
    )
