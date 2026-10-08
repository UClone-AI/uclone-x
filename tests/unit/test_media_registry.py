"""Unit tests for ModelRegistry, ModelProfile, and PromptFamily resolution (P0/P8/P9)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from pydantic import ValidationError

from uclone_x.tools.builtin.image import (
    GenerateImageTool,
    ImageGenerationResult,
    ImagePipelineDispatcher,
    resolve_aspect_dimensions,
)
from uclone_x.tools.builtin.media_registry import (
    DANBOORU_QUALITY_TAGS,
    NEGATIVE_NOT_USED,
    ModelProfile,
    ModelRegistry,
    PromptFamily,
    fill_prompt_defaults,
)


def test_prompt_family_values() -> None:
    """PromptFamily enum members have expected string values."""
    assert PromptFamily.DANBOORU == "danbooru"
    assert PromptFamily.NATURAL_PROSE == "prose"
    assert PromptFamily.HYBRID == "hybrid"
    assert PromptFamily.GENERIC == "generic"


def test_model_profile_strict_validation() -> None:
    """ModelProfile enforces strict validation and forbids unknown fields."""
    profile = ModelProfile(
        model_id="test_model",
        display_name="Test Model",
        filename="test.safetensors",
        family=PromptFamily.DANBOORU,
    )
    assert profile.model_id == "test_model"
    assert profile.family == PromptFamily.DANBOORU
    assert profile.width == 1024
    assert profile.steps == 20

    with pytest.raises(ValidationError):
        ModelProfile(
            model_id="bad",
            display_name="Bad",
            family=PromptFamily.DANBOORU,
            unknown_attribute=123,  # type: ignore[call-arg]
        )


def test_default_registry_loads_shipped_models() -> None:
    """Default registry correctly loads shipped default_models.yaml."""
    registry = ModelRegistry()

    # anillustrious_v4 lookup
    profile_by_id = registry.resolve("anillustrious_v4")
    assert profile_by_id.model_id == "anillustrious_v4"
    assert profile_by_id.family == PromptFamily.DANBOORU
    assert profile_by_id.skill_name is None

    # lookup by filename
    profile_by_file = registry.resolve("anillustrious_v4.safetensors")
    assert profile_by_file.model_id == "anillustrious_v4"

    # flux-2-klein-base-4b lookup
    flux_profile = registry.resolve("flux-2-klein-base-4b")
    assert flux_profile.family == PromptFamily.NATURAL_PROSE
    assert flux_profile.skill_name is None


def test_sidecar_metadata_takes_highest_priority(tmp_path: Path) -> None:
    """A co-located <name>.meta.json sidecar file overrides registry configurations."""
    ckpt_file = tmp_path / "custom_checkpoint.safetensors"
    ckpt_file.write_bytes(b"dummy")

    sidecar_file = tmp_path / "custom_checkpoint.meta.json"
    sidecar_data = {
        "model_id": "sidecar_custom",
        "display_name": "Sidecar Discovered Model",
        "family": "danbooru",
        "width": 832,
        "height": 1216,
        "default_negative": "low quality, bad hands",
    }
    sidecar_file.write_text(json.dumps(sidecar_data), encoding="utf-8")

    registry = ModelRegistry()
    profile = registry.resolve(str(ckpt_file))

    assert profile.model_id == "sidecar_custom"
    assert profile.family == PromptFamily.DANBOORU
    assert profile.width == 832
    assert profile.height == 1216
    assert profile.default_negative == "low quality, bad hands"


def test_user_config_overrides_shipped_defaults(tmp_path: Path) -> None:
    """User configuration file overrides shipped default profiles."""
    user_config_file = tmp_path / "user_models.yaml"
    user_data = {
        "models": [
            {
                "model_id": "anillustrious_v4",
                "display_name": "User Overridden Illustrious",
                "filename": "anillustrious_v4.safetensors",
                "family": "danbooru",
                "steps": 28,
                "cfg": 6.5,
                "default_negative": "custom negative",
            }
        ]
    }
    user_config_file.write_text(yaml.dump(user_data), encoding="utf-8")

    registry = ModelRegistry(user_config_path=user_config_file)
    profile = registry.resolve("anillustrious_v4")

    assert profile.display_name == "User Overridden Illustrious"
    assert profile.steps == 28
    assert profile.cfg == 6.5
    assert profile.default_negative == "custom negative"


def test_heuristic_detection_for_unregistered_checkpoints() -> None:
    """Heuristic filename patterns detect anime or flux families when not in registry."""
    registry = ModelRegistry()

    # Anime heuristic
    anime_prof = registry.resolve("my_special_pony_diffusion_v6.safetensors")
    assert anime_prof.family == PromptFamily.DANBOORU
    assert anime_prof.skill_name is None

    # Flux heuristic
    flux_prof = registry.resolve("custom_flux_schnell_v1.safetensors")
    assert flux_prof.family == PromptFamily.NATURAL_PROSE
    assert flux_prof.skill_name is None
    assert flux_prof.steps == 4
    assert flux_prof.cfg == 1.0


def test_fallback_profile_for_unknown_model_or_none() -> None:
    """Unknown checkpoint without heuristics falls back to universal profile."""
    registry = ModelRegistry()

    none_profile = registry.resolve(None)
    assert none_profile.model_id == "generic_fallback"
    assert none_profile.family == PromptFamily.GENERIC
    assert none_profile.skill_name is None

    unknown_profile = registry.resolve("totally_unrecognized_checkpoint_xyz.ckpt")
    assert unknown_profile.model_id == "generic_fallback"


def test_graceful_handling_of_malformed_yaml(tmp_path: Path) -> None:
    """Corrupted or invalid YAML files do not crash the registry."""
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("not: valid: yaml: [", encoding="utf-8")

    registry = ModelRegistry(default_config_path=bad_yaml)
    profile = registry.resolve(None)
    assert profile.model_id == "generic_fallback"


def test_resolve_aspect_dimensions_with_danbooru_profile() -> None:
    """A registered SDXL-class profile renders at SDXL's ~1MP buckets, not below 768².

    Illustrious was trained without images under 768x768; the old DANBOORU branch gave
    it 576x768 for 3:4, below its own training floor.

    Killed by: src/uclone_x/tools/builtin/image.py :: "3:4": (896, 1152),
    Becomes: "3:4": (576, 768),
    """
    danbooru_profile = ModelProfile(
        model_id="anime_test",
        display_name="Anime Test",
        family=PromptFamily.DANBOORU,
    )

    assert resolve_aspect_dimensions("3:4", profile=danbooru_profile) == (896, 1152)
    assert resolve_aspect_dimensions("9:16", profile=danbooru_profile) == (768, 1344)
    assert resolve_aspect_dimensions("16:9", profile=danbooru_profile) == (1344, 768)
    assert resolve_aspect_dimensions("4:3", profile=danbooru_profile) == (1152, 896)
    assert resolve_aspect_dimensions("1:1", profile=danbooru_profile) == (1024, 1024)

    # Standard fallback without profile keeps the legacy table
    assert resolve_aspect_dimensions("3:4") == (576, 768)


@pytest.mark.asyncio
async def test_dispatcher_injects_default_negative_prompt_when_empty() -> None:
    """Dispatcher automatically populates empty negative_prompt from active model profile."""
    mock_engine = AsyncMock()
    mock_engine.is_available.return_value = True
    mock_engine.generate.return_value = ImageGenerationResult(
        image_bytes=b"png",
        seed=1,
        engine_name="mock",
        device_info="test",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )

    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_engine,
        comfy_engine=mock_engine,
        local_engine=mock_engine,
    )

    # Calling dispatch with empty negative prompt
    await dispatcher.dispatch(
        prompt="1girl, warrior",
        negative_prompt="",
        aspect_ratio="1:1",
        seed=123,
        style="anime",
    )

    # Verify that mock_engine.generate was called with the active profile's default negative prompt
    active_profile = dispatcher.get_active_profile()
    assert active_profile.default_negative != ""

    mock_engine.generate.assert_awaited_once()
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["negative_prompt"] == active_profile.default_negative


def test_the_description_names_neither_the_active_model_nor_its_family() -> None:
    """The tools layer stays the same whatever checkpoint is active (layering §5.1, #1723).

    Killed by: src/uclone_x/tools/builtin/image.py :: parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION]
    Becomes: parts = [self.GEMINI_BASE_DESCRIPTION if gemini else self.BASE_DESCRIPTION, self.active_profile().model_id]
    """
    dispatcher = ImagePipelineDispatcher()
    tool = GenerateImageTool(dispatcher=dispatcher)

    profile = dispatcher.get_active_profile()
    text = tool.description

    assert profile.model_id not in text
    assert "prompt family" not in text
    assert "media-prompt" not in text


# ------------------------------------------------ fill_prompt_defaults (owner ruling, #1723)


def _danbooru(default_negative: str = "worst quality, text, watermark, blurry") -> ModelProfile:
    return ModelProfile(
        model_id="danbooru-test",
        display_name="Danbooru Test",
        family=PromptFamily.DANBOORU,
        default_negative=default_negative,
    )


def test_a_negative_the_model_gave_is_passed_on_unchanged() -> None:
    """Rule 3: an explicit negative is never merged with the profile's defaults.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::         return PromptFill(effective_prompt, given_negative, tuple(changes))
    Becomes:         pass
    """
    fill = fill_prompt_defaults("1girl, solo, masterpiece", "animal, extra limbs", _danbooru())

    assert fill.negative_prompt == "animal, extra limbs"
    assert fill.changes == ()


def test_quality_tags_are_added_only_to_a_prompt_without_any() -> None:
    """Rule 2: the danbooru quality tags fill an open slot, and say so.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::     if profile.family == PromptFamily.DANBOORU and not has_quality_tag(effective_prompt):
    Becomes:     if profile.family == PromptFamily.DANBOORU:
    Killed by: src/uclone_x/tools/builtin/media_registry.py ::         changes.append(f"Added quality tags the prompt did not have: {DANBOORU_QUALITY_TAGS}.")
    Becomes:         pass
    """
    open_slot = fill_prompt_defaults("1girl, solo", "x", _danbooru())
    assert open_slot.prompt == f"1girl, solo, {DANBOORU_QUALITY_TAGS}"
    assert any(DANBOORU_QUALITY_TAGS in change for change in open_slot.changes)

    stated = fill_prompt_defaults("1girl, solo, best quality", "x", _danbooru())
    assert stated.prompt == "1girl, solo, best quality"
    assert stated.changes == ()


def test_a_weighted_or_score_quality_tag_counts_as_stated() -> None:
    """Rule 2: `(masterpiece:1.2)` and a Pony `score_9` are the prompt's own quality.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::             name = _tag_name(raw).replace("_", " ")
    Becomes:             name = raw.strip().lower()
    Killed by: src/uclone_x/tools/builtin/media_registry.py ::             if name in QUALITY_TAG_NAMES or name.startswith("score "):
    Becomes:             if name in QUALITY_TAG_NAMES:
    """
    for prompt in ("1girl, (Masterpiece:1.2)", "score_9, score_8_up, 1girl"):
        assert fill_prompt_defaults(prompt, "x", _danbooru()).prompt == prompt


def test_a_quality_tag_on_its_own_line_counts_as_stated() -> None:
    """Rule 2: tags split by a newline are tags too, so the defaults are not added twice.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::     for line in prompt.splitlines():
    Becomes:     for line in [prompt]:
    """
    prompt = "1girl, solo\nmasterpiece"
    fill = fill_prompt_defaults(prompt, "x", _danbooru())

    assert fill.prompt == prompt
    assert fill.changes == ()


def test_an_underscored_quality_tag_counts_as_stated() -> None:
    """Rule 2: `best_quality` is the Danbooru spelling of `best quality`.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::             name = _tag_name(raw).replace("_", " ")
    Becomes:             name = _tag_name(raw)
    """
    prompt = "1girl, best_quality"
    fill = fill_prompt_defaults(prompt, "x", _danbooru())

    assert fill.prompt == prompt
    assert fill.changes == ()


def test_the_prompt_is_never_translated_reordered_or_cut() -> None:
    """Rule 1: the person's words come first and whole; fills only ever append.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::         effective_prompt = ", ".join(t for t in (effective_prompt, DANBOORU_QUALITY_TAGS) if t)
    Becomes:         effective_prompt = ", ".join(t for t in (DANBOORU_QUALITY_TAGS, effective_prompt) if t)
    """
    korean = "  갑옷을 입은 전사, 1girl  "
    fill = fill_prompt_defaults(korean, "", _danbooru())
    assert fill.prompt.startswith("갑옷을 입은 전사, 1girl, ")

    generic = ModelProfile(
        model_id="g", display_name="g", family=PromptFamily.GENERIC, default_negative="blurry"
    )
    assert fill_prompt_defaults("갑옷을 입은 전사", "", generic).prompt == "갑옷을 입은 전사"


def test_the_default_negative_drops_what_the_prompt_asks_for() -> None:
    """Rule 4: a diagram asking for text labels does not get `text` in its negative.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::     kept = [t for t in defaults if not names_term(prompt, t)]
    Becomes:     kept = defaults
    Killed by: src/uclone_x/tools/builtin/media_registry.py ::         changes.append(f"No negative prompt was given, so this one was used: {effective_negative}.")
    Becomes:         pass
    """
    fill = fill_prompt_defaults("flowchart, text labels, masterpiece", "", _danbooru())

    assert fill.negative_prompt == "worst quality, watermark, blurry"
    assert fill.changes == (
        "No negative prompt was given, so this one was used: worst quality, watermark, blurry.",
    )


def test_a_term_inside_another_word_is_not_a_request_for_it() -> None:
    """Rule 4 matches whole words: `context` does not ask for `text`.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::     return re.search(pattern, prompt.lower()) is not None
    Becomes:     return term.lower() in prompt.lower()
    """
    fill = fill_prompt_defaults("1girl, historical context, masterpiece", "", _danbooru())

    assert fill.negative_prompt == "worst quality, text, watermark, blurry"


def test_a_negative_on_a_model_that_ignores_it_is_reported_unused() -> None:
    """Rule 5: prose and suppress_negative models get no negative, and never silently.

    Killed by: src/uclone_x/tools/builtin/media_registry.py ::             changes.append(NEGATIVE_NOT_USED)
    Becomes:             pass
    Killed by: src/uclone_x/tools/builtin/media_registry.py ::     if profile.suppress_negative or profile.family == PromptFamily.NATURAL_PROSE:
    Becomes:     if profile.suppress_negative:
    """
    prose = ModelProfile(
        model_id="flux-test",
        display_name="FLUX Test",
        family=PromptFamily.NATURAL_PROSE,
        default_negative="blurry",
    )

    given = fill_prompt_defaults("A cinematic photo of a knight", "worst quality", prose)
    assert given.prompt == "A cinematic photo of a knight"
    assert given.negative_prompt == ""
    assert given.changes == (NEGATIVE_NOT_USED,)

    empty = fill_prompt_defaults("A photo", "", prose)
    assert empty.negative_prompt == ""
    assert empty.changes == ()


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
async def test_the_tool_result_lists_what_was_filled_in(tmp_path: Path) -> None:
    """P6: what the defaults added reaches the model in the tool result, single and batch.

    Killed by: src/uclone_x/tools/builtin/image.py ::                     prompt_changes=fill.changes,
    Becomes:                     prompt_changes=(),
    """
    from uclone_x.tools.builtin.image import GenerateImageParams
    from uclone_x.tools.models import ToolContext

    mock_engine = AsyncMock()
    mock_engine.is_available.return_value = True
    mock_engine.generate.return_value = _gen_result()
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_engine, comfy_engine=mock_engine, local_engine=mock_engine
    )
    profile = _danbooru()
    dispatcher.get_active_profile = lambda own=None: profile  # type: ignore[method-assign]
    tool = GenerateImageTool(dispatcher=dispatcher)
    context = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)

    single = await tool.run(GenerateImageParams(prompt="1girl, solo"), context)
    batch = await tool.run(GenerateImageParams(prompts=["1girl", "1boy, masterpiece"]), context)

    expected = list(fill_prompt_defaults("1girl, solo", "", profile).changes)
    assert expected and single["prompt_changes"] == expected
    assert mock_engine.generate.call_args_list[0].kwargs["prompt"] == (
        f"1girl, solo, {DANBOORU_QUALITY_TAGS}"
    )
    assert [img["prompt_changes"] for img in batch["images"]] == [
        list(fill_prompt_defaults("1girl", "", profile).changes),
        list(fill_prompt_defaults("1boy, masterpiece", "", profile).changes),
    ]


@pytest.mark.asyncio
async def test_each_sidecar_records_the_filled_prompt_and_what_was_added(tmp_path: Path) -> None:
    """The image's metadata keeps the prompt the engine drew from, not only the model's (#1865).

    The sidecar records the filled prompt and negative prompt, the fill's changes, and the
    tags it added, as tags; the tool result carries the added tags for the conversation
    (once, at the top, when every picture of a batch gained the same ones).

    Killed by: src/uclone_x/tools/builtin/image.py ::             **fill_record,
    Becomes:             **{},
    Killed by: src/uclone_x/tools/builtin/image.py ::                     **batch_fill,
    Becomes:                     **{},
    Killed by: src/uclone_x/tools/builtin/image.py ::                     filled_prompt=fill.prompt,
    Becomes:                     filled_prompt=None,
    """
    from uclone_x.tools.builtin.image import GenerateImageParams
    from uclone_x.tools.models import ToolContext

    mock_engine = AsyncMock()
    mock_engine.is_available.return_value = True
    mock_engine.generate.return_value = _gen_result()
    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_engine, comfy_engine=mock_engine, local_engine=mock_engine
    )
    profile = _danbooru()
    dispatcher.get_active_profile = lambda own=None: profile  # type: ignore[method-assign]
    tool = GenerateImageTool(dispatcher=dispatcher)
    context = ToolContext(agent_id="a", session_id="s", workspace_root=tmp_path)

    single = await tool.run(GenerateImageParams(prompt="1girl, solo"), context)
    batch = await tool.run(GenerateImageParams(prompts=["1girl", "1boy"]), context)

    fill = fill_prompt_defaults("1girl, solo", "", profile)
    from uclone_x.tools.base import artifact_path_from_url
    from uclone_x.tools.builtin.image import sidecar_path_for

    def _sidecar_of(url: object) -> dict[str, object]:
        rel = artifact_path_from_url(url)
        assert rel is not None
        side: dict[str, object] = json.loads((tmp_path / sidecar_path_for(rel)).read_text())
        return side

    sidecar = _sidecar_of(single["relative_url"])
    assert sidecar["prompt"] == "1girl, solo"
    assert sidecar["filled_prompt"] == fill.prompt
    assert sidecar["filled_negative_prompt"] == fill.negative_prompt
    assert sidecar["prompt_changes"] == list(fill.changes)
    quality = [t.strip() for t in DANBOORU_QUALITY_TAGS.split(",")]
    assert sidecar["prompt_added"] == quality == single["prompt_added"]
    assert single["negative_added"] == [t.strip() for t in fill.negative_prompt.split(",")]

    # Both pictures gained the same tags, so the batch says them once, at the top (#2013).
    assert batch["prompt_added"] == quality
    for img in batch["images"]:
        assert "prompt_added" not in img
        side = _sidecar_of(img["relative_url"])
        assert side["filled_prompt"] == f"{img['prompt']}, {DANBOORU_QUALITY_TAGS}"
        assert side["prompt_added"] == quality


@pytest.mark.asyncio
async def test_dispatcher_flux_profile_suppresses_negative() -> None:
    """Dispatcher with a FLUX profile strips negative_prompt even if provided."""
    mock_engine = AsyncMock()
    mock_engine.is_available.return_value = True
    mock_engine.generate.return_value = ImageGenerationResult(
        image_bytes=b"flux_png",
        seed=42,
        engine_name="mock_flux",
        device_info="test",
        duration_seconds=0.1,
        width=1024,
        height=1024,
    )

    registry = ModelRegistry()
    flux_profile = ModelProfile(
        model_id="flux-custom",
        display_name="FLUX Custom",
        family=PromptFamily.NATURAL_PROSE,
        default_negative="",
        suppress_negative=True,
    )
    registry.register(flux_profile)

    dispatcher = ImagePipelineDispatcher(
        remote_engine=mock_engine,
        comfy_engine=mock_engine,
        local_engine=mock_engine,
        registry=registry,
    )
    # Force active profile resolution to the flux profile
    dispatcher.get_active_profile = lambda own=None: flux_profile  # type: ignore[method-assign]

    await dispatcher.dispatch(
        prompt="A photo of a mountain",
        negative_prompt="blurry, ugly",
        aspect_ratio="1:1",
        seed=42,
        style="photorealistic",
    )

    mock_engine.generate.assert_awaited_once()
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["negative_prompt"] == ""
