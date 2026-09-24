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
    ModelProfile,
    ModelRegistry,
    PromptFamily,
    optimize_prompts,
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
        skill_name="media-prompt-danbooru",
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
            skill_name="media-prompt-danbooru",
            unknown_attribute=123,  # type: ignore[call-arg]
        )


def test_default_registry_loads_shipped_models() -> None:
    """Default registry correctly loads shipped default_models.yaml."""
    registry = ModelRegistry()

    # anillustrious_v4 lookup
    profile_by_id = registry.resolve("anillustrious_v4")
    assert profile_by_id.model_id == "anillustrious_v4"
    assert profile_by_id.family == PromptFamily.DANBOORU
    assert profile_by_id.skill_name == "media-prompt-danbooru"

    # lookup by filename
    profile_by_file = registry.resolve("anillustrious_v4.safetensors")
    assert profile_by_file.model_id == "anillustrious_v4"

    # flux-2-klein-base-4b lookup
    flux_profile = registry.resolve("flux-2-klein-base-4b")
    assert flux_profile.family == PromptFamily.NATURAL_PROSE
    assert flux_profile.skill_name == "media-prompt-flux"


def test_sidecar_metadata_takes_highest_priority(tmp_path: Path) -> None:
    """A co-located <name>.meta.json sidecar file overrides registry configurations."""
    ckpt_file = tmp_path / "custom_checkpoint.safetensors"
    ckpt_file.write_bytes(b"dummy")

    sidecar_file = tmp_path / "custom_checkpoint.meta.json"
    sidecar_data = {
        "model_id": "sidecar_custom",
        "display_name": "Sidecar Discovered Model",
        "family": "danbooru",
        "skill_name": "media-prompt-danbooru",
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
                "skill_name": "media-prompt-danbooru",
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
    assert anime_prof.skill_name == "media-prompt-danbooru"

    # Flux heuristic
    flux_prof = registry.resolve("custom_flux_schnell_v1.safetensors")
    assert flux_prof.family == PromptFamily.NATURAL_PROSE
    assert flux_prof.skill_name == "media-prompt-flux"
    assert flux_prof.steps == 4
    assert flux_prof.cfg == 1.0


def test_fallback_profile_for_unknown_model_or_none() -> None:
    """Unknown checkpoint without heuristics falls back to universal profile."""
    registry = ModelRegistry()

    none_profile = registry.resolve(None)
    assert none_profile.model_id == "generic_fallback"
    assert none_profile.family == PromptFamily.GENERIC
    assert none_profile.skill_name == "media-prompt-generic"

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
    """Aspect ratio resolution adapts to Danbooru anime native resolutions."""
    danbooru_profile = ModelProfile(
        model_id="anime_test",
        display_name="Anime Test",
        family=PromptFamily.DANBOORU,
        skill_name="media-prompt-danbooru",
    )

    # 3:4 portrait resolution aligned to 64px
    assert resolve_aspect_dimensions("3:4", profile=danbooru_profile) == (576, 768)
    # 9:16 portrait
    assert resolve_aspect_dimensions("9:16", profile=danbooru_profile) == (512, 896)
    # 16:9 landscape
    assert resolve_aspect_dimensions("16:9", profile=danbooru_profile) == (896, 512)
    # 1:1 square
    assert resolve_aspect_dimensions("1:1", profile=danbooru_profile) == (768, 768)

    # Standard fallback without profile
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


def test_generate_image_tool_dynamic_description() -> None:
    """GenerateImageTool dynamically decorates tool description with active model prompt family."""
    dispatcher = ImagePipelineDispatcher()
    tool = GenerateImageTool(dispatcher=dispatcher)

    profile = dispatcher.get_active_profile()
    assert f"Active Model: '{profile.model_id}'" in tool.description
    assert f"({profile.family.value} prompt family)" in tool.description
    assert "Danbooru tags" in tool.description or "media-prompt" in tool.description


def test_optimize_prompts_flux_suppresses_negative() -> None:
    """Prose/FLUX models suppress negative prompts completely."""
    flux_profile = ModelProfile(
        model_id="flux-test",
        display_name="FLUX Test",
        family=PromptFamily.NATURAL_PROSE,
        skill_name="media-prompt-flux",
        default_negative="",
        suppress_negative=True,
    )

    pos, neg = optimize_prompts(
        prompt="A cinematic photo of a knight",
        negative_prompt="worst quality, blurry",
        profile=flux_profile,
    )
    assert pos == "A cinematic photo of a knight"
    assert neg == ""

    # Even with empty input, remains empty
    _, empty_neg = optimize_prompts(
        prompt="A photo",
        negative_prompt="",
        profile=flux_profile,
    )
    assert empty_neg == ""


def test_optimize_prompts_danbooru_merges_negative() -> None:
    """Danbooru models automatically merge user negative tags with default defense tags."""
    danbooru_profile = ModelProfile(
        model_id="danbooru-test",
        display_name="Danbooru Test",
        family=PromptFamily.DANBOORU,
        skill_name="media-prompt-danbooru",
        default_negative="worst quality, bad anatomy, deformed, bad hands",
    )

    # Empty user negative -> uses default negative
    pos, neg = optimize_prompts(
        prompt="1girl, solo",
        negative_prompt="",
        profile=danbooru_profile,
    )
    assert pos == "1girl, solo"
    assert neg == "worst quality, bad anatomy, deformed, bad hands"

    # User supplied some negative tags -> deduplicates and merges
    pos, neg = optimize_prompts(
        prompt="1girl, solo",
        negative_prompt="animal, bad anatomy, extra limbs",
        profile=danbooru_profile,
    )
    assert pos == "1girl, solo"
    # user tags first, followed by missing default tags
    assert "animal" in neg
    assert "extra limbs" in neg
    assert "worst quality" in neg
    assert "bad hands" in neg
    assert "deformed" in neg
    # Ensure no duplicate "bad anatomy"
    assert neg.count("bad anatomy") == 1


def test_optimize_prompts_generic_fallback() -> None:
    """Generic models keep user negative if provided, or fallback to default."""
    generic_profile = ModelProfile(
        model_id="generic-test",
        display_name="Generic Test",
        family=PromptFamily.GENERIC,
        skill_name="media-prompt-generic",
        default_negative="worst quality, blurry",
    )

    # Empty -> fallback
    _, neg_empty = optimize_prompts("prompt", "", generic_profile)
    assert neg_empty == "worst quality, blurry"

    # Provided -> retained as-is
    _, neg_provided = optimize_prompts("prompt", "custom negative tag", generic_profile)
    assert neg_provided == "custom negative tag"


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
        skill_name="media-prompt-flux",
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
    dispatcher.get_active_profile = lambda: flux_profile  # type: ignore[method-assign]

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
